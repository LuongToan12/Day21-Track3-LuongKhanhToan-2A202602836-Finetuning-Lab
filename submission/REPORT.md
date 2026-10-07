# Lab 21 — Evaluation Report

**Họ tên**: Lương Khánh Toàn  **MSSV**: 2A202602836  **Ngày**: 2026-10-07  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB khả dụng)`

---

## 1. Setup & Cấu hình thí nghiệm

| Thành phần | Cấu hình thực tế | Ghi chú & Cơ sở lựa chọn |
|---|---|---|
| **Dataset** | 250 ticket CSKH tiếng Việt → JSON triage 4 trường | Gồm `intent`, `urgency`, `product`, `sentiment` theo định dạng chuẩn |
| **Train / val split** | 225 / 25 mẫu | Phân chia cố định với `seed=42` để đảm bảo tính tái lập (reproducibility) |
| **`max_length`** | 1024 (p95 thực tế là 98 tokens) | Suggested theo p95 là 256; giữ 1024 theo mặc định tier T4 để an toàn cho prompt dài |
| **`MASK_MODE`** | `assistant-only` | Chỉ tính loss trên câu trả lời của assistant, che toàn bộ prompt và system prompt |
| **Epochs / max_steps** | 2.0 epochs / 30 steps | Tính theo công thức `planned_steps(225, T4, 2.0) = 30 steps`, áp dụng chung cho cả 4 run |

**Template có giữ khối `<think>` không?**  
Có. Kết quả kiểm tra từ `results/template_check.json` xác nhận: `verdict: "reasoning preserved — safe to train on traces"`. Khối suy luận `<think>` và thẻ đóng `</think>` được render nguyên vẹn trong template ChatML của model, không bị nuốt hay cắt xén ngầm.

---

## 2. Bằng chứng Mask (NB1 Mask Proof)

| Chỉ số kiểm định | Giá trị đo được | Đánh giá |
|---|---|---|
| `supervised_fraction` | 0.4149 (41.49%) | Đạt chuẩn (< 0.95), chứng minh prompt đã được che triệt để |
| Câu trả lời nằm trong loss | True | Khẳng định nhãn mục tiêu được tính gradient |
| Câu hỏi KHÔNG nằm trong loss | True | Khẳng định mô hình không học vẹt cách chép lại prompt |

**Đoạn trích chuỗi được tính loss (Supervised Preview):**

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

*Nhận xét:* Phần loss chỉ bắt đầu từ sau khối `</think>` và bao trọn chuỗi JSON mục tiêu cùng token kết thúc `<|im_end|>`. Câu hỏi và system prompt được gán `IGNORE_INDEX (-100)` hoàn toàn.

---

## 3. Ba Baseline (NB2 — Đo và Đóng Băng TRƯỚC Khi Huấn Luyện)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| **(a) base + naive prompt** | 0.000 | 0.724 | 0.000 | 11331.0 |
| **(b) base + optimized prompt** | 0.760 | 0.724 | 1.000 | 3775.0 |
| **(c) LoRA fine-tune** | 0.815 | 0.710 | 1.000 | 1450.0 |

**(b) có thật sự mạnh hơn (a) không?**  
Có, baseline (b) vượt trội hoàn toàn baseline (a) trên cả độ chính xác tác vụ (`target`: 0.000 → 0.760), độ tuân thủ định dạng JSON (`format`: 0.000 → 1.000), và giảm độ trễ suy luận xuống gần 3 lần (11331 ms → 3775 ms). Khi chưa có schema hướng dẫn, base model trả lời bằng văn bản tự do dài dòng và không tạo ra JSON hợp lệ. Prompt tối ưu đã ép model dừng sinh đúng lúc sau khi hoàn thành JSON.  
Tệp `OPTIMIZED_PROMPT` được giữ nguyên vẹn (`sha256 = 719e74d3b6232053`), không bị làm yếu đi để tâng bốc bản fine-tune.

---

## 4. Giải Phẫu Cấu Hình Sai (NB4 Misconfiguration Autopsy)

Bảng đối chứng 4 run với **cùng ngân sách bước huấn luyện (30 steps)** và đo đạc trên cùng tập dữ liệu:

| Run | vị trí adapter | r | alpha | trainable params | LR | train loss (NB4) | **target (NB5 §4)** | train time (s) | peak VRAM (GB) |
|---|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (12 proj) | 16 | 32 | 32,464,896 | 1e-4 | 0.0549 | **0.815** | 995.5 | 12.07 |
| `attn_only` | q,v (2 proj) | 283 | 566 | 32,456,704 | 1e-4 | **0.0531** | 0.770 | 888.9 | 12.09 |
| `wrong_lr` | text-linear (12 proj) | 16 | 32 | 32,464,896 | 1e-5 | 0.0903 | 0.280 | 1021.3 | 12.08 |
| `qlora` | text-linear (12 proj) | 16 | 32 | 32,464,896 | 1e-4 | 0.0670 | 0.765 | 1084.7 | **7.15** |

### Phân tích chi tiết ba câu hỏi giải phẫu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Trên tập target, `attn_only` thua `correct` một cách rõ ràng (0.770 so với 0.815), mặc dù ngân sách tham số huấn luyện là tương đương nhau (32,456,704 so với 32,464,896, sai lệch chỉ 0.025%). Thứ tự này hoàn toàn ngược lại so với thứ tự theo `train loss` (nơi `attn_only` đạt loss 0.0531 thấp hơn 0.0549 của `correct`). Hiện tượng này chứng minh rằng rank cao (r=283) tập trung ở ít lớp chỉ giúp mô hình ghi nhớ (memorize) tập train tốt hơn, nhưng thiếu khả năng khái quát hóa. Vị trí gắn adapter trải rộng trên toàn bộ các lớp linear (`all-linear` text decoder) mới là đòn bẩy quyết định chất lượng biểu diễn, chứ không phải độ lớn của rank.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Khi hạ Learning Rate xuống 10 lần (1e-5, mức thường dùng cho Full Fine-tuning), `train loss` giảm rất chậm và dừng lại ở 0.0903, khiến điểm target sụp đổ xuống mức 0.280. Nếu một kỹ sư chỉ nhìn vào loss mà không biết tham số LR, họ rất dễ lầm tưởng rằng mô hình bị underfitting do dữ liệu quá khó, hoặc cho rằng rank r=16 là quá nhỏ và vội vàng tăng rank. Thực tế nguyên nhân cốt lõi là ma trận LoRA khởi tạo bằng 0 cần một bước nhảy gradient lớn hơn (thang ~10x so với full FT) để thích ứng nhanh trong không gian biểu diễn con.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
QLoRA 4-bit giúp cắt giảm mạnh mẽ bộ nhớ GPU từ 12.07 GB xuống 7.15 GB (tiết kiệm tới 40.8% VRAM). Tuy nhiên, cái giá phải trả là điểm target tụt từ 0.815 xuống 0.765 (ngang ngửa baseline b) và tốc độ huấn luyện chậm hơn do chi phí dequantize on-the-fly (1084.7 s so với 995.5 s). Kết quả đo đạc thực nghiệm này hoàn toàn ủng hộ khuyến nghị chính thức từ nhà sản xuất Qwen3.5: sai số lượng tử hóa 4-bit làm suy giảm đáng kể năng lực xử lý ngôn ngữ chi tiết, và nếu VRAM đủ đáp ứng (như 16GB trên T4), sử dụng 16-bit LoRA trực tiếp luôn mang lại hiệu quả vượt trội.

---

## 5. Phán Quyết Cổng Hồi Quy (NB5 Verdict)

* **Kết quả kiểm định**: **PASSED**
* **Chênh lệch mục tiêu (`target Δ`)**: **+0.055** (vượt baseline b từ 0.760 lên 0.815)
* **Chênh lệch tổng quát (`regression Δ`)**: **-0.014** (suy giảm 1.4%, nằm an toàn trong ngưỡng dung sai cho phép 2.0%)
* **Tỷ lệ suy luận hợp lệ (`valid_trace_rate`)**: **0.000** (do prompt mục tiêu yêu cầu JSON thuần, không kích hoạt chế độ thinking)

### Diễn giải phán quyết:
Mô hình fine-tune `correct` đã vượt qua cổng kiểm định hồi quy một cách thuyết phục. Về mặt mục tiêu, mô hình cải thiện 5.5% độ chính xác so với prompt được tối ưu hóa công phu nhất, chứng minh rằng tri thức phân loại miền nghiệp vụ đã được cô đọng hiệu quả vào trọng số LoRA. Đồng thời, về mặt năng lực tổng quát, điểm số trên 15 câu hỏi phổ thông chỉ giảm nhẹ từ 0.724 xuống 0.710 (Δ = -0.014, hoàn toàn nằm trong biên an toàn 0.020). Điều này khẳng định hiện tượng quên thảm họa (catastrophic forgetting) không xảy ra nghiêm trọng ở mức thiết lập tham số chuẩn này. Đặc biệt, bản fine-tune đạt được điều này với prompt hệ thống cực ngắn ("Phân loại ticket sau"), giúp giảm độ trễ suy luận từ 3775 ms xuống 1450 ms (nhanh hơn 2.6 lần), mang lại lợi ích kinh tế và vận hành to lớn khi đưa vào hệ thống production phục vụ hàng triệu người dùng.

---

## 6. Đánh Giá Định Tính (Qualitative Evaluation)

Dưới đây là 5 ca kiểm thử thực tế trên tập đánh giá mục tiêu, bao gồm cả ca thắng lẫn ca thua:

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Đánh giá chi tiết |
|---|---|---|---|---|---|
| **1** | "Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt." | `intent`: doi_tra, `urgency`: cao, `product`: chuột không dây, `sentiment`: tich_cuc | Đoán `sentiment`: trung_tinh | Đoán `sentiment`: tich_cuc, đầy đủ 4 khóa | ✅ **FT thắng**: Fine-tune phân biệt chính xác sắc thái khen "Shop hỗ trợ tốt" là tích cực dù khách muốn đổi trả. |
| **2** | "Alo shop, mình đặt máy xay sinh tố mã đơn OD126693. Muốn đổi. Đã 3 ngày rồi. Bực mình." | `intent`: doi_tra, `urgency`: trung_binh, `product`: máy xay sinh tố, `sentiment`: tieu_cuc | Nhầm `urgency`: cao do từ "Bực mình" | Nhận diện đúng `urgency`: trung_binh ("Đã 3 ngày rồi") | ✅ **FT thắng**: Fine-tune tách bạch được yếu tố khẩn cấp thời gian và cảm xúc tiêu cực của khách hàng. |
| **3** | "Cho mình hỏi, mình đặt đèn bàn LED mã đơn VN339109. Vỡ khi nhận. Gấp. Shop xem giúp." | `intent`: san_pham_loi, `urgency`: cao, `product`: đèn bàn LED, `sentiment`: trung_tinh | Đoán đúng toàn bộ 4 trường | Nhầm `sentiment`: tieu_cuc | ❌ **FT thua**: Fine-tune bị thiên kiến (bias) khi thấy "Vỡ khi nhận" liền áp đặt sentiment tiêu cực, trong khi lời lẽ khách trung tính. |
| **4** | "Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi." | `intent`: san_pham_loi, `urgency`: thap, `product`: nồi chiên không dầu, `sentiment`: trung_tinh | Đoán đúng `intent`: san_pham_loi | Nhầm `intent`: hoi_thong_tin do cụm "Cho tôi hỏi" | ❌ **FT thua**: Cụm từ cuối câu làm nhiễu biểu diễn của adapter, khiến nó ưu tiên intent hỏi thông tin thay vì phản ánh lỗi sản phẩm. |
| **5** | "Xin chào, mình đặt balo laptop mã đơn DH863123. Đổi size. Hỏi cho biết thôi. Lần cuối mua ở đây." | `intent`: doi_tra, `urgency`: thap, `product`: balo laptop, `sentiment`: tieu_cuc | Nhầm `urgency`: cao do câu dọa "Lần cuối..." | Nhận diện đúng `urgency`: thap nhờ "Hỏi cho biết thôi" | ✅ **FT thắng**: Adapter bắt trúng ngữ cảnh mức độ ưu tiên xử lý thực tế thay vì bị đánh lạc hướng bởi câu than phiền. |

**Mẫu chung ở các ca FT thua:**  
Các ca mô hình fine-tune thất bại thường xuất hiện khi văn bản chứa các tín hiệu ngữ nghĩa xung đột ở các vị trí khác nhau (ví dụ: mô tả lỗi ở đầu câu nhưng lại kết thúc bằng mẫu câu hỏi chung chung ở cuối câu, hoặc tình huống khách phàn nàn nhưng thái độ ngôn từ vẫn lịch sự). Adapter LoRA có xu hướng gán trọng số quá mức vào các token mang tính cảm xúc mạnh hoặc các mẫu câu định khuôn ở cuối câu, dẫn đến hiện tượng phân loại nhầm cục bộ.

---

## 7. Kết Luận & Bài Học Kinh Nghiệm

### Kết luận tổng kết:
Bản fine-tune LoRA `correct` hoàn toàn xứng đáng được triển khai vào môi trường sản xuất thực tế. Nó không chỉ vượt qua mốc chuẩn khắt khe của baseline prompt tối ưu về mặt chất lượng nghiệp vụ (tăng 5.5% độ chính xác trường), mà còn tạo ra lợi thế cạnh tranh áp đảo về chi phí vận hành: giảm hơn 60% số lượng token đầu vào do không cần nhồi nhét prompt ví dụ phức tạp, từ đó hạ độ trễ từ 3775 ms xuống 1450 ms và tăng thông lượng phục vụ lên 2.6 lần. 

Đòn bẩy thực sự quyết định thành công của lab này không nằm ở việc cố gắng nâng rank LoRA lên các con số khổng lồ, mà nằm ở: (1) **Tính đúng đắn tuyệt đối của Loss Masking** — đảm bảo mô hình chỉ học những gì cần sinh ra; (2) **Chiến lược đặt adapter toàn diện (`all-linear`)** bao phủ trọn vẹn không gian biểu diễn của text decoder; và (3) **Quy mô Learning Rate đúng thang đo** (~10x mức full FT). Khi ba yếu tố nền tảng này được thiết lập chuẩn xác trong "vùng không hối tiếc", mô hình đạt trạng thái hội tụ tối ưu mà không cần tinh chỉnh tham số phức tạp hay tốn kém tài nguyên tính toán.

### Ba điều tôi học được:
1. **Mask Loss là sinh mệnh của SFT**: Không bao giờ được tin tưởng mù quáng vào các cờ mặc định của thư viện (như `assistant_only_loss`). Cần phải luôn giải mã ngược các token trong vùng tính loss để kiểm chứng bằng mắt và bằng assert rằng prompt đã bị loại trừ 100%.
2. **So sánh công bằng đòi hỏi cùng ngân sách**: So sánh `attn_only` và `all-linear` ở cùng rank là một sai lầm bản chất vì chúng khác nhau về số lượng tham số. Chỉ khi cân bằng tham số huấn luyện bằng `matched_rank`, ta mới thấy rõ vị trí đặt adapter quan trọng hơn độ lớn của rank.
3. **Prompt Engineering là một mốc chuẩn bắt buộc**: Không thể tuyên bố một mô hình fine-tune thành công nếu nó chỉ thắng một prompt ngây thơ. Việc đo đạc baseline prompt tối ưu trước khi huấn luyện giúp người kỹ sư giữ được sự liêm chính khoa học và tránh lãng phí tài nguyên cho những bài toán vốn chỉ cần giải quyết bằng kỹ thuật prompt tốt.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
- Bổ sung 2–5% dữ liệu replay đa nhiệm (General QA) vào tập huấn luyện để triệt tiêu hoàn toàn mức tụt 1.4% của regression score, đưa `regression Δ` về mức >= 0.000.
- Thực hiện kiểm chứng thêm với kỹ thuật DoRA (Weight-Decomposed Low-Rank Adaptation) để phân tích sự phân tách giữa hướng và độ lớn vector trọng số trên bài toán trích xuất JSON tiếng Việt.

---

## Phụ Lục — Điểm Thưởng Đã Thực Hiện

- [x] **B1: NB6 Merge & Hot-swap**: Đã thực hiện kiểm chứng gộp trọng số `W = W₀ + (α/r)·BA`, xác nhận điểm target sau merge không suy giảm (`delta = 0.0000`, lưu tại `results/merge_check.json`), và kiểm tra tính năng nạp hoán đổi adapter linh hoạt trên cùng một base model trong bộ nhớ.
