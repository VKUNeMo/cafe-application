# Luồng dự đoán

1. YOLO phát hiện các vùng tổn thương.
2. Cắt vùng lá vuông và phân loại bằng ConvNeXt V2-T (nhánh Big).
3. Nếu confidence của top 1 ≤ 75%, hiển thị thêm lớp có xác suất cao thứ hai. Trên 75% chỉ hiển thị top 1.

Ngưỡng duy nhất nằm ở `TOP2_CONF_THRESHOLD` trong `app/inference/config.py`. Không dùng Small, bảng precision, khoảng cách hai điểm hoặc điểm tối thiểu của top 2.

`app/inference/pipeline.py` xử lý suy luận và ảnh chú thích. `app/services/prediction_service.py` xử lý toàn bộ yêu cầu: kiểm tra ảnh, gọi pipeline, ánh xạ thông tin bệnh, lưu ảnh/lịch sử và trả response API. Response giữ `method: "big_only"` và `selected_branch: "big"` để các client hiện tại tiếp tục hoạt động.

Chạy `python -m unittest discover -s tests -v` để kiểm tra xếp hạng Big, ngưỡng 75%, API và lịch sử mà không cần nạp checkpoint thật.
