# Ghi nhận kiểm chứng

Ngày kiểm chứng: **16/10/2026**.

Phần kiểm chứng ngày 16/10 bên dưới là ghi nhận lịch sử của đợt bàn giao. Bộ test và file constraints được nhắc đến không còn đi kèm; các lệnh cài đặt và kiểm tra hiện hành nằm trong README. CI hiện kiểm tra dependency, lint và cú pháp, không chạy pytest.

## Kiểm tra sau tinh gọn — 19/10/2026

- Windows/Python 3.13 CPU, dùng môi trường và ba checkpoint đã có; không cài lại môi trường mới.
- OpenAPI trước/sau khớp hoàn toàn, bao gồm tọa độ mặc định Tây Nguyên.
- Suy luận thật trên hai ảnh local: toàn bộ response suy luận, bao gồm ảnh base64, khớp trước/sau sau khi bỏ trường thời gian xử lý. Đây là kiểm tra tương thích, không phải đánh giá độ chính xác.
- Kiểm tra API bằng công cụ tạm local và storage riêng: đăng ký/đăng nhập, phân quyền, tọa độ, ghi/đọc lịch sử, bản đồ, tải ảnh, ảnh hỏng, dữ liệu `croods` cũ và dọn ảnh khi ghi lịch sử thất bại đều đạt. Trong các ca API này, suy luận được thay bằng kết quả cố định để kiểm tra luồng nghiệp vụ độc lập.
- Ruff, kiểm tra cú pháp Python và `pip check` trong `.venv-check` đều đạt. Không chạy GitHub Actions hoặc kiểm tra CUDA trong phiên này.

## Môi trường

- Windows, Python 3.13.7 64 bit, CPU.
- Tạo môi trường ảo mới, cài PyTorch `2.12.0+cpu`, torchvision `0.27.0+cpu`, sau đó cài `requirements-dev.txt`.
- Kiểm tra một bản sao mã nguồn chỉ chứa các file thuộc snapshot bàn giao; không có model, users/history hoặc ảnh cũ trong snapshot ban đầu.
- Checkpoint được cấp qua đường dẫn bên ngoài snapshot, tương ứng cách đối tác nhận model riêng. Cấu hình thử được nạp từ `.env` tạo dựa trên `.env.example`.
- Các phiên bản dependencies đã giải quyết được ghi tại `requirements/constraints.txt`; kiểm tra resolver bằng dry-run offline không yêu cầu thay đổi gói đã cài.

## Kết quả

| Kiểm tra | Kết quả |
| --- | --- |
| `python -m pip check` trong môi trường mới | Không có dependencies bị xung đột |
| `python -m ruff check .` | Đạt |
| Tests API, cấu hình, lưu trữ, tọa độ và migration | **44 passed, 1 deselected**; test yêu cầu checkpoint được tách riêng |
| Đối chiếu suy luận trước/sau với ba ảnh local cố định | Kết quả chuẩn hóa khớp hoàn toàn trên cả môi trường cũ và mới |
| Lifespan và API với checkpoint thật trong snapshot sạch | Nạp model, `/ping`, giao diện, tài nguyên tĩnh, `/docs`, `/redoc` hoạt động |
| Dự đoán thật, lưu/đọc lịch sử và tải lại ảnh | Đạt |
| Ảnh hỏng sau khi đã nạp Ultralytics | Trả `400`, không kích hoạt cài plugin HEIF |
| Kiểm tra cú pháp hai module JavaScript bằng Node.js | Đạt |
| `git diff --check` | Đạt |
| Kiểm tra Git index với models/storage/users/history/ảnh tải lên | Không còn file runtime hoặc checkpoint được theo dõi |

Đối chiếu suy luận kiểm tra nhãn, confidence, Top-2, cờ phân vân, bounding boxes và kết quả từng nhánh. Không so sánh thời gian xử lý hoặc nội dung nhị phân ảnh hiển thị. Ba ảnh đối chiếu là dữ liệu local, không được đưa vào repository.

Tests trong bản sao dùng `--basetemp` trỏ đến một thư mục thử nghiệm mới bên trong workspace vì quyền truy cập thư mục Temp mặc định trên máy kiểm chứng bị hạn chế. Lệnh tests thông thường trong README áp dụng cho môi trường local/CI có quyền ghi thư mục tạm.

## Giới hạn xác nhận

- Chưa chạy trên Linux, macOS hoặc CUDA. Workflow GitHub Actions đã được bổ sung cho Windows/Linux; chưa chạy workflow từ GitHub trong phiên bàn giao này.
- Kiểm tra giao diện bao gồm phục vụ HTML/CSS/JS và cú pháp JavaScript; chưa tự động hóa thao tác camera, GPS hoặc tương tác trình duyệt.
- Không đo độ chính xác trên tập đánh giá hoặc benchmark tải; đợt này giữ thuật toán hiện tại và kiểm tra tính tương thích sau tái cấu trúc.
- Test `integration` đi kèm là tùy chọn, bật bằng `CAFE_RUN_INTEGRATION=1`; kiểm chứng model thật trong phiên này sử dụng luồng API đầy đủ và bộ ảnh đối chiếu nêu trên.
