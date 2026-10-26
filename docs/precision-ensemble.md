# Pipeline Ensemble Precision

## Luồng suy luận

1. YOLO quét các lát cắt vuông, hợp nhất box bằng NMS.
2. Big phân loại một vùng vuông lớn chứa nhiều tâm box nhất; không có box thì cắt giữa ảnh.
3. Small phân loại từng vùng box (padding 25%, tối thiểu 128 px) và lấy trung bình xác suất. Không có box thì phân loại ảnh cắt giữa.
4. Ensemble áp dụng một quy tắc chung cho tất cả lớp:
   - Cùng nhãn: giữ nhãn chung; lấy confidence cao hơn, bằng nhau chọn Big.
   - Khác nhãn: so `precision_big[nhãn Big]` với `precision_small[nhãn Small]`; chọn dự đoán có precision cao hơn, bằng nhau chọn Big.

Confidence được trả về là điểm softmax của nhánh được chọn (Small dùng trung bình các box), không phải precision và không phải xác suất đúng đã hiệu chuẩn của ensemble. Không trộn vector xác suất Big/Small và không có luật ưu tiên riêng theo bệnh. YOLO quyết định vùng đầu vào, không ghi đè nhãn ensemble.

## Bảng precision đang dùng

`app/inference/data/precision_by_class.csv` được đưa vào mã nguồn để ứng dụng có thể chạy độc lập với thư mục evaluation. Đường dẫn có thể thay bằng `CAFE_PRECISION_TABLE` trong `.env`; đường dẫn tương đối tính từ gốc repository.

Precision của một lớp = số dự đoán đúng lớp đó / tổng số lần nhánh dự đoán lớp đó. Hai nhánh có hai bảng riêng; không dùng Macro Precision. Các giá trị CSV nằm trong khoảng 0–1.

| Lớp | Big (%) | Small (%) |
|---|---:|---:|
| Anthracnose | 92,59 | 84,62 |
| Corticium | 97,83 | 88,24 |
| Mealy | 75,76 | 64,10 |
| Miner | 96,15 | 100,00 |
| Nematode | 96,00 | 95,83 |
| Phoma | 100,00 | 100,00 |
| Rust | 95,45 | 97,30 |
| Healthy | 100,00 | 100,00 |

Ví dụ Big nói Nematode, Small nói Rust: so 96,00% với 97,30%, chọn Rust của Small, kể cả confidence hiện tại của Big cao hơn. Nếu Big nói Phoma, Small nói Miner thì cả hai precision đều 100%, chọn Phoma của Big.

Bảng hiện tại lấy từ `evaluation/thực-tế-ptn/precision_by_class.csv`, trên toàn bộ 342 ảnh đã có nhãn của `Data/anh-train-thuc-te-va-thi-nghiem/test`. Bảng được tính từ chính tập đánh giá (in-sample), không dùng kiểm tra chéo. Kết quả khảo sát là 328/342 ảnh đúng (95,91%), sai 14 ảnh; đây không phải kết quả trên test độc lập hay cam kết độ chính xác cho ảnh tải lên. Nguồn và SHA-256 checkpoint được lưu trong `precision_metadata.json` cạnh CSV.

Khi phục vụ API, ảnh mới không có nhãn thật: ứng dụng chỉ tra bảng đã lưu, không tính lại precision từ ảnh của người dùng và không dùng lịch sử chẩn đoán làm nhãn thật.

## Cập nhật bảng khi thay model hoặc bộ đánh giá

1. Chạy `evaluation/thực-tế-ptn/run_evaluation.py` với đúng checkpoint Big/Small dự định triển khai để xuất `precision_by_class.csv`.
2. Chép CSV mới vào `app/inference/data/precision_by_class.csv`, hoặc trỏ `CAFE_PRECISION_TABLE` đến CSV đó. Giữ nguyên header `class,precision_big,precision_small`; mỗi lớp xuất hiện đúng một lần, tên lớp phải khớp checkpoint, precision hữu hạn trong khoảng 0–1. Thứ tự dòng không bắt buộc.
3. Cập nhật thông tin nguồn, số mẫu và dấu vân tay checkpoint trong hồ sơ triển khai/`precision_metadata.json`. Metadata đi kèm chỉ mô tả bảng mặc định; dùng CSV tùy chỉnh cần quản lý hồ sơ riêng. Ứng dụng kiểm tra cấu trúc/tên lớp/giá trị, không tự chứng minh CSV được tạo bởi checkpoint nào.
4. Khởi động lại server để nạp bảng mới. Thiếu file, lớp bị lặp, thiếu/thừa lớp hoặc giá trị sai sẽ báo lỗi; không tự quay về luật cũ.

Thay checkpoint có cùng tên lớp vẫn cần đo lại precision. Hai bảng đi kèm bộ model, không được chọn lại theo từng yêu cầu API.

## Top-2, cờ phân vân và giao diện

- Khi bất đồng, nhãn phụ là dự đoán của nhánh còn lại. Điểm Top-1/Top-2 có thể đến từ hai nhánh khác nhau và không tạo thành một phân phối xác suất; nhãn được chọn có thể có confidence thấp hơn nhãn phụ.
- Khi đồng thuận, nhãn phụ là lớp có điểm cao nhất khác nhãn chung trong vector của nhánh được chọn.
- Cờ phân vân bật khi confidence < 0,65 hoặc trị tuyệt đối chênh lệch hai điểm < 0,20. Cờ này không thay đổi nhãn đã chọn.
- Giao diện chỉ hiển thị nhãn phụ khi cờ phân vân bật và điểm phụ nội bộ ≥ 0,10, trừ khi cấu hình luôn hiển thị Top-2. Khi ẩn, API trả `top2_result: null`.
- API không trả `top2_confidence` trong kết quả chẩn đoán, `details` hoặc lịch sử. Nhãn Top-2 và cờ phân vân vẫn được giữ. Điểm phụ vẫn dùng nội bộ để quyết định hiển thị; giao diện và ảnh chú thích không hiển thị điểm phụ.
- Biểu đồ chính hiển thị vector điểm của nhánh được chọn. Hai biểu đồ Big/Small vẫn thể hiện điểm riêng từng nhánh, không vẽ trung bình 55/45.

## API trace

`details.method` nhận một trong: `agreement`, `precision_big`, `precision_small`, `precision_tie_big`. `details.selected_branch` là `big` hoặc `small`. Mỗi nhánh có `pred_class`, `confidence`, `precision` theo lớp dự đoán và `all_probs`. Khi bất đồng, so `details.big_branch.precision` với `details.small_branch.precision`. Các tên lớp trong nhánh và `all_probs` giữ đúng tên checkpoint; `result.name` là mã bệnh chuẩn hóa của API.

Kết quả cuối và trạng thái chỉ nằm ở cấp ngoài response, không lặp trong `details`. Số box suy ra từ `detected_boxes.length`; `small_branch.box_predictions[].id` trỏ đến box YOLO cùng `id` để tra `xyxy`. API không trả lại tọa độ crop của cùng box; ảnh cắt Small thực tế vẫn có padding như mô tả ở trên. Schema OpenAPI định nghĩa cụ thể các trường thay vì nhận dictionary tùy ý.
