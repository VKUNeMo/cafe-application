# Ghi nhận kiểm chứng

## Thu gọn response chẩn đoán — 26/10/2026

- Kết quả cuối, confidence, nhãn phụ và trạng thái chỉ trả một lần ở cấp ngoài. `details` có schema cụ thể, không cho thêm các trường lặp; precision được đưa vào từng nhánh. Bỏ `num_boxes` và tọa độ crop lặp lại; các box liên kết bằng `id`.
- **12 tests đạt**: thêm kiểm tra schema từ chối trường lặp, precision nằm đúng nhánh, dự đoán box tra tọa độ qua `id`, giữ nguyên confidence/nhãn và bảo toàn dữ liệu lịch sử. Hai ví dụ API được kiểm tra bằng schema mới; Ruff và cú pháp Python/JavaScript đạt. Không chạy lại model vì logic suy luận không đổi.
- Giao diện đếm box từ `detected_boxes.length`; tài liệu nêu rõ các đường dẫn response cũ đã được thay thế. Các ghi nhận schema cũ bên dưới là lịch sử.

## Ẩn confidence Top-2 trong API — 26/10/2026

- Bỏ `top2_confidence` khỏi schema/response chẩn đoán, `details` và response lịch sử, kể cả bản ghi cũ. Giữ nhãn Top-2, confidence nhãn chính và các điều kiện phân vân; điểm phụ và dữ liệu lưu trữ vẫn được giữ nội bộ.
- Giao diện và ảnh chú thích mới không hiển thị phần trăm Top-2. Hai bản API documentation và tài liệu pipeline đã cập nhật.
- **10 tests đạt**, gồm 7 test precision và 3 test API/lịch sử dùng kết quả suy luận giả lập: kiểm tra field bị loại, nhãn/confidence chính giữ nguyên và bản ghi cũ không bị sửa. Ruff, cú pháp Python và JavaScript đạt. Không chạy lại model vì thay đổi chỉ ở lớp trình bày/response.

## Ensemble Precision — 26/10/2026

- Pipeline ứng dụng dùng bảng precision toàn tập của Big/Small thay bộ luật riêng theo bệnh. Quy tắc: cùng nhãn lấy confidence cao hơn; khác nhãn so precision tại hai nhãn dự đoán; precision bằng nhau chọn Big. Không chạy kiểm tra chéo trong ứng dụng.
- CSV mặc định khớp bảng xuất từ `evaluation/thực-tế-ptn`. SHA-256 của cả ba checkpoint trong `models/` khớp checkpoint của lần đánh giá 342 ảnh. Nguồn bảng và dấu vân tay checkpoint nằm trong `app/inference/data/precision_metadata.json`.
- Đối chiếu trên xác suất đã lưu của **342 ảnh**: nhãn ensemble, method, confidence, nhãn/điểm phụ và cờ phân vân của ứng dụng đều khớp với evaluation. Đây là đối chiếu logic trên cache, không phải chạy lại YOLO/Big/Small cho cả 342 ảnh.
- Suy luận thật CPU trên một ảnh Healthy local: nạp model và CSV, hai nhánh/nhãn cuối khớp mẫu đánh giá, vẽ overlay thành công. Kiểm tra bộ chuyển response API với kết quả này: schema, nhánh được chọn và trace precision đúng; không ghi lịch sử hoặc ảnh lên storage.
- **7 kiểm thử tự động** kiểm tra đồng thuận, chọn Big/Small theo precision, precision bằng nhau và CSV sai/thiếu. Chạy bằng `python -m unittest discover -s tests -v`, không cần checkpoint. CI đã thêm bước chạy các test này.
- Ruff, cú pháp Python, cú pháp hai module JavaScript và kiểm tra hiển thị biểu đồ theo nhánh được chọn đều đạt. Chưa kiểm thử giao diện trong trình duyệt hoặc chạy workflow từ GitHub, Linux/CUDA hay benchmark tải.
- README, hai bản API documentation, kiến trúc và tài liệu [Ensemble Precision](inference.md) đã cập nhật. Các xác nhận tương thích với thuật toán cũ dưới đây là lịch sử, không áp dụng cho nhãn ensemble sau lần đổi thuật toán này.

Bảng được tính từ nhãn thật của chính tập đánh giá (in-sample). Accuracy 95,91% (328/342), Macro F1 95,47% là kết quả khảo sát trên tập đó, không phải kiểm chứng test độc lập hoặc độ chính xác cho mọi ảnh người dùng.

## Ghi nhận lịch sử trước khi đổi thuật toán

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
