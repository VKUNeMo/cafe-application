# Vận hành local và dữ liệu

## Quy trình chạy

1. Tạo môi trường Python 3.13, cài dependencies theo README.
2. Chép checkpoint, tạo `.env` và secret.
3. Nếu có dữ liệu cũ, chạy migration trước server.
4. Chạy `python -m scripts.check_environment --load-models` và `python -m pip check`.
5. Chạy Uvicorn một worker, thử `/ping`, giao diện và một ảnh lá hợp lệ.

Log ghi ra terminal với thời gian, mức độ và tên module. Không dùng reload cho phiên ổn định vì mỗi lần sửa Python sẽ nạp lại model. `/ping` kiểm tra kết nối sau khi nạp model; không đo chất lượng dự đoán.

## GPU

Bản cài chuẩn dùng CPU. Để dùng NVIDIA GPU, tạo môi trường riêng và cài bộ `torch==2.12.0`, `torchvision==0.27.0` có CUDA phù hợp phần cứng/driver theo hướng dẫn của nhà cung cấp. Không dùng index CPU cho môi trường CUDA.

Kiểm tra `python -c "import torch; print(torch.cuda.is_available())"`, đặt `CAFE_DEVICE=cuda` và chạy công cụ kiểm tra với `--load-models`. Nếu bộ phiên bản này không có wheel phù hợp, dùng CPU hoặc kiểm chứng tương thích riêng trước khi đổi phiên bản. CUDA chưa nằm trong kiểm chứng local của bản bàn giao.

## Sao lưu và khôi phục

- Dừng server để tránh sao chép JSON giữa lúc ghi.
- Sao lưu toàn bộ `CAFE_STORAGE_DIR` gồm users, histories, uploads; lưu secret riêng ở nơi có kiểm soát quyền truy cập.
- Khôi phục storage rồi đặt `CAFE_STORAGE_DIR` tương ứng. Giữ secret cũ nếu cần duy trì token chưa hết hạn.
- Khởi động, đăng nhập, kiểm tra lịch sử và một ảnh cũ. Sao lưu riêng checkpoint để tái lập môi trường.

Không sửa users bằng tay hoặc đưa lên GitHub. Không chạy migration trong lúc server đang ghi. Nếu migration bị ngắt giữa chừng, nguồn vẫn còn; chọn đích trống mới để chạy lại.

## Kiểm tra bàn giao

Chạy lint/tests, xem `git status` và `git ls-files`; snapshot bàn giao không chứa `.env`, checkpoint hoặc dữ liệu người dùng. `.gitignore` không xóa nội dung đã commit trước đây khỏi lịch sử Git. Đợt tái cấu trúc chỉ bỏ theo dõi dữ liệu runtime trong snapshot mới, không viết lại lịch sử.
