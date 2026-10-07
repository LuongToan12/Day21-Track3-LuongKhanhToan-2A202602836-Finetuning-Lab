# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Điều làm tôi ngạc nhiên nhất là hiện tượng `attn_only` (chỉ gắn vào q,v với rank được nâng lên r=283 để cân bằng số tham số) lại đạt loss huấn luyện thấp hơn cả `correct` (0.0531 so với 0.0549), nhưng khi đánh giá trên tập target thực tế thì điểm số lại thua rõ rệt (0.770 so với 0.815). Điều này cho thấy `train loss` là một chỉ số thay thế nguy hiểm nếu dùng làm căn cứ đánh giá mô hình: loss thấp ở rank cao chỉ phản ánh việc mô hình ghi nhớ vẹt dữ liệu ở số ít lớp chứ không đem lại khả năng khái quát hóa.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Tôi mất nhiều thời gian nhất ở việc hiểu và kiểm chứng cơ chế loss masking của ChatML template (F-01, F-10). Ban đầu tôi dự đoán phần huấn luyện trên GPU sẽ tốn thời gian nhất. Nhưng thực tế, việc đảm bảo ranh giới token giữa câu hỏi và câu trả lời không bị sai lệch do cơ chế nối chuỗi hay tiền xử lý của tokenizer mới là phần then chốt và tiềm ẩn nhiều lỗi ngầm nhất — nếu sai ở đây thì toàn bộ thời gian train sau đó đều vô nghĩa.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước lab này, tôi từng tin rằng: "Nếu mô hình fine-tune chưa đạt hiệu năng cao, cách khắc phục đầu tiên là tăng rank LoRA (r=32, 64, 128...) và chỉ cần gắn LoRA vào các lớp attention (q,v) là đủ tiết kiệm và hiệu quả". Giờ đây tôi hiểu rằng tăng rank ở số ít lớp hoàn toàn vô dụng so với việc trải đều adapter lên toàn bộ các lớp linear (`all-linear`) với rank vừa phải (r=16), kết hợp đúng thang Learning Rate (~10x mức full-FT).

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi dùng AI assistant để hỗ trợ phân tích luồng logic kiểm định của các bài test, giải thích cơ chế tương thích giữa fp16 và GradScaler trên kiến trúc GPU Turing (T4), cũng như định dạng các bảng biểu đối chứng khoa học. Chỗ AI dễ nhầm lẫn nhất là xu hướng áp đặt cứng `bf16=True` theo thói quen của các bài hướng dẫn A100 hiện đại, trong khi phần cứng T4 hoàn toàn không hỗ trợ tập lệnh bf16 nguyên bản và sẽ gây lỗi crash ngầm trong quá trình scale gradient.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên tôi sẽ làm là xây dựng một prompt thật hoàn chỉnh (Baseline b) và thiết lập một bộ dữ liệu đánh giá độc lập (chứa cả tập mục tiêu và tập hồi quy đa nhiệm) để đo lường mốc chuẩn đóng băng. Chỉ khi chứng minh được rằng prompt engineering đã chạm ngưỡng trần và bản fine-tune thực sự cần thiết để cải thiện độ chính xác hoặc giảm chi phí token/độ trễ, tôi mới bắt tay vào chuẩn bị dữ liệu và huấn luyện mô hình.
