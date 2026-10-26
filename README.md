# Coffee Leaf Application

Ứng dụng nhận diện bệnh lá cà phê qua ảnh, cung cấp **REST API bằng FastAPI** và **giao diện web** trên cùng server. Pipeline kết hợp YOLO phát hiện vùng bệnh, hai nhánh Big/Small và **Ensemble Precision theo lớp**: cùng nhãn thì giữ nhãn chung và confidence cao hơn; khác nhãn thì chọn dự đoán có precision cao hơn, bằng nhau chọn Big.

Bảng precision mặc định nằm trong `app/inference/data/precision_by_class.csv`, được đo trên toàn bộ 342 ảnh có nhãn của bộ đánh giá hiện tại. Ứng dụng tra bảng này khi nhận ảnh mới, không tính precision từ ảnh tải lên. Xem [logic, nguồn bảng, Top-2 và cách cập nhật](docs/precision-ensemble.md).

API trả kết quả cuối một lần ở cấp ngoài; `details` chỉ chứa diễn giải chọn nhánh, box YOLO và kết quả riêng của Big/Small. Precision nằm trong từng nhánh. Xem cấu trúc mới và hướng dẫn đổi các đường dẫn response cũ trong [API documentation](API_DOCUMENTATION.md).

Repository chứa mã nguồn, giao diện và dữ liệu mô tả bệnh. **Ba checkpoint được bàn giao riêng**; `models/` được Git bỏ qua hoàn toàn.

## Mục lục

- [Ensemble Precision](docs/precision-ensemble.md)
- [Yêu cầu](#yêu-cầu)
- [Cài đặt và chạy](#cài-đặt-và-chạy)
- [Cấu hình](#cấu-hình)
- [Sử dụng API](#sử-dụng-api) (Xem chi tiết tại [API_DOCUMENTATION.md](API_DOCUMENTATION.md))
- [Phát triển](#phát-triển)
- [Dữ liệu và sao lưu](#dữ-liệu-và-sao-lưu)
- [Khắc phục lỗi](#khắc-phục-lỗi)

## Yêu cầu

- Python **3.13** 64 bit, Git, dung lượng cho môi trường PyTorch và model. Riêng ba checkpoint khoảng 700 MB.
- CPU chạy được toàn bộ pipeline; GPU NVIDIA là tùy chọn và cần bộ PyTorch/CUDA phù hợp.
- Internet cho lần cài dependencies. Giao diện lấy font từ Google Fonts; khi không có mạng sẽ dùng font dự phòng.
- Chạy **một worker** vì users/history lưu bằng JSON. Bản bàn giao tập trung chạy local.

Môi trường đã kiểm chứng và các giới hạn được ghi trong [validation.md](docs/validation.md). Các hướng dẫn dưới đây thực hiện từ thư mục gốc repository.

`requirements.txt` khóa phiên bản dependencies trực tiếp để chạy ứng dụng. Dependencies gián tiếp được pip giải quyết theo yêu cầu của các gói, không khóa riêng.

## Cài đặt và chạy

### 1. Clone và tạo môi trường ảo

Thay `<REPOSITORY_URL>` bằng URL GitHub do bên bàn giao cung cấp.

**Windows PowerShell:**

```powershell
git clone <REPOSITORY_URL> cafe-application
cd cafe-application
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Linux/macOS:**

```bash
git clone <REPOSITORY_URL> cafe-application
cd cafe-application
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Cài dependencies

Windows/Linux dùng CPU có thể cài trước bộ PyTorch CPU để giảm dung lượng tải:

```text
python -m pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

Trên macOS, bỏ qua lệnh dùng index CPU và cài trực tiếp `python -m pip install -r requirements.txt`, sau đó chạy `python -m pip check`. Cần kiểm chứng các bước này trên máy đích; xem [ghi nhận kiểm chứng](docs/validation.md).

### 3. Chép model nhận riêng

Tạo thư mục `models` ở gốc dự án và chép chính xác ba file:

```text
models/
├── best-yolov26.pt
├── best-big-last.pth
└── best-small-last.pth
```

Phải dùng bộ checkpoint tương thích do bên bàn giao cung cấp. Đổi tên một checkpoint khác không bảo đảm chạy đúng. Không cần Git LFS và không tải checkpoint từ repository này.

### 4. Tạo cấu hình local

PowerShell:

```powershell
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Linux/macOS:

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Chép chuỗi vừa tạo vào `CAFE_AUTH_SECRET` trong `.env`, giữ ổn định giữa các lần chạy. `.env` tự động được nạp và được Git bỏ qua. Mẫu cấu hình chọn CPU.

### 5. Khởi động

```text
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Startup kiểm tra checkpoint, dữ liệu bệnh học và bảng precision, sau đó nạp pipeline một lần. CSV thiếu lớp, trùng lớp hoặc có giá trị không hợp lệ sẽ làm startup báo lỗi. Đợi terminal báo `Application startup complete`, rồi mở:

| Địa chỉ                  | Mục đích                                  |
| --------------------------- | -------------------------------------------- |
| http://127.0.0.1:8000/      | Giao diện tải/chụp ảnh và xem kết quả |
| http://127.0.0.1:8000/docs  | Swagger UI để thử API                     |
| http://127.0.0.1:8000/redoc | Tài liệu API                               |
| http://127.0.0.1:8000/ping  | Trả `{"message":"pong"}`                  |

Dừng bằng `Ctrl+C`. Lần sau chỉ cần kích hoạt môi trường ảo và chạy lại Uvicorn. Lệnh cũ `python -m uvicorn main:app` và `python main.py` vẫn hoạt động. Khi phát triển có thể dùng `--reload` thay cho `--workers 1`; mỗi lần reload sẽ nạp lại model.

## Cấu hình

Biến môi trường tiến trình được ưu tiên hơn `.env`. Đường dẫn tương đối tính theo thư mục gốc repository, không theo vị trí terminal.

| Biến                    | Mặc định khi chưa đặt                   | Ý nghĩa                                                               |
| ------------------------ | --------------------------------------------- | ----------------------------------------------------------------------- |
| `CAFE_AUTH_SECRET`     | Sinh ngẫu nhiên theo tiến trình           | Secret ký token; đặt cố định để token còn hợp lệ sau restart |
| `CAFE_MODEL_DIR`       | `models`                                    | Thư mục chứa checkpoint                                              |
| `CAFE_PRECISION_TABLE` | `app/inference/data/precision_by_class.csv` | CSV precision từng lớp, phải tương ứng checkpoint Big/Small       |
| `CAFE_STORAGE_DIR`     | `storage`                                   | Users, histories và uploads                                            |
| `CAFE_DEVICE`          | `auto`                                      | `auto`: CUDA nếu có, còn lại CPU; hoặc ép `cpu`, `cuda`     |
| `CAFE_CORS_ORIGINS`    | Rỗng                                         | Origin của frontend ngoài server, phân cách bằng dấu phẩy        |

Ví dụ CORS: `http://localhost:3000,https://frontend.example.com`. Giao diện đi kèm cùng origin nên không cần đặt. `cuda` trên máy không có CUDA sẽ báo lỗi khởi động. Muốn dùng GPU NVIDIA, cài bộ PyTorch/torchvision có CUDA tương thích phần cứng và đặt `CAFE_DEVICE=cuda`; không dùng index CPU ở bước cài PyTorch. CUDA chưa được kiểm chứng trong bản bàn giao.

## Sử dụng API

| Phương thức và URL                 | Chức năng                                            |
| -------------------------------------- | ------------------------------------------------------ |
| `GET /ping`                          | Kiểm tra kết nối sau startup                        |
| `POST /predictor/predict`            | Multipart `file`, tùy chọn `coords`, `user_id` |
| `POST /auth/register`                | `phone`, `password`, tùy chọn `username`       |
| `POST /auth/login`                   | Nhận token và `user_id`                            |
| `GET /user/{user_id}`                | Hồ sơ tài khoản hiện tại                         |
| `GET /histories/{user_id}`           | Lịch sử tài khoản hiện tại                       |
| `GET /histories/{history_id}/detail` | Chi tiết một lần dự đoán                         |
| `GET /histories/map`                 | Các điểm tọa độ của tài khoản hiện tại      |
| `GET /image-cafe-input/{filename}`   | Ảnh thuộc lịch sử tài khoản hiện tại           |

Không gửi token sẽ dùng tài khoản khách dùng chung **`default_user`**. Giao diện hiện dùng tài khoản này. Với tài khoản đăng nhập, gửi `Authorization: Bearer <token>`; `user_id` trong form có thể bỏ qua, nếu gửi phải khớp token. Token có thời hạn 7 ngày.

### PowerShell

```powershell
curl.exe http://127.0.0.1:8000/ping
curl.exe -X POST http://127.0.0.1:8000/predictor/predict -F "file=@C:/images/leaf.jpg" -F "coords=14.0583,108.2772"

$body = @{ phone = '0900000001'; password = 'mat-khau-thu-nghiem'; username = 'Doi tac' } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/auth/register -Method Post -ContentType 'application/json' -Body $body
$login = Invoke-RestMethod http://127.0.0.1:8000/auth/login -Method Post -ContentType 'application/json' -Body $body
$headers = @{ Authorization = "Bearer $($login.token)" }
Invoke-RestMethod "http://127.0.0.1:8000/histories/$($login.user_id)" -Headers $headers
curl.exe -X POST http://127.0.0.1:8000/predictor/predict -H "Authorization: Bearer $($login.token)" -F "file=@C:/images/leaf.jpg"
```

### Linux/macOS

```bash
curl http://127.0.0.1:8000/ping
curl -X POST http://127.0.0.1:8000/predictor/predict -F 'file=@/path/to/leaf.jpg' -F 'coords=14.0583,108.2772'
curl -X POST http://127.0.0.1:8000/auth/register -H 'Content-Type: application/json' -d '{"phone":"0900000001","password":"mat-khau-thu-nghiem","username":"Doi tac"}'
curl -X POST http://127.0.0.1:8000/auth/login -H 'Content-Type: application/json' -d '{"phone":"0900000001","password":"mat-khau-thu-nghiem"}'
# Thay bằng token và user_id nhận từ response đăng nhập.
TOKEN='<token>'
USER_ID='<user_id>'
curl "http://127.0.0.1:8000/histories/$USER_ID" -H "Authorization: Bearer $TOKEN"
curl -X POST http://127.0.0.1:8000/predictor/predict -H "Authorization: Bearer $TOKEN" -F 'file=@/path/to/leaf.jpg'
```

Ảnh hỗ trợ JPG/JPEG, PNG, WebP, BMP, tối đa 12 MiB và 20 triệu pixel; có kiểm tra độ nét, độ sáng và vùng lá. `coords` nhận `lat,long` hoặc JSON `{"lat":14.0583,"long":108.2772}`. Khi bỏ qua, dùng mặc định `14.0583,108.2772` và vẫn tạo điểm bản đồ; đây không phải xác nhận vị trí chụp thực tế.

Response có `history_user_id`, `coords`, `result`, `confidence`, `top2_result`, `show_top2`, `is_uncertain`, tên file ảnh gốc (`image_url`), tên file ảnh khoanh vùng bệnh (`annotated_image_url`) và `details`. Top-2 chỉ có nội dung khi chính sách pipeline cho phép. Schema đầy đủ ở `/docs`. Lỗi kiểm định ảnh trả `400` với `error_code`/`message`; vượt giới hạn trả `413`, tọa độ sai trả `422`, lỗi xử lý trả `500`.

## Phát triển

```text
app/
  main.py          # FastAPI factory, lifespan, middleware
  api/             # Router và dependencies xác thực
  core/            # Settings, hashing và token
  schemas/         # Request/response
  services/        # Nghiệp vụ và điều phối dự đoán
  repositories/    # JSON, ảnh và dữ liệu bệnh học
  inference/       # Classifier, detection, tiền xử lý, ensemble, visualization
  resources/       # diseases.json: dữ liệu bệnh học cố định
  static/          # Giao diện; api.js tập trung HTTP
scripts/           # Chuyển dữ liệu từ cấu trúc cũ (tùy chọn)
docs/              # Kiến trúc và ghi nhận kiểm chứng
models/            # Checkpoint gửi riêng; Git bỏ qua
storage/           # Dữ liệu phát sinh (storage/uploads, users.json, histories.json); Git bỏ qua
```

```text
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m compileall -q app main.py
python -m unittest discover -s tests -v
```

GitHub Actions kiểm tra dependencies, cú pháp Python và bộ test precision trên Windows/Linux. Bộ test trong `tests/` kiểm tra chọn nhánh và bảng CSV, không cần checkpoint hoặc huấn luyện lại mô hình. Suy luận thật cần kiểm chứng riêng bằng model nhận từ bên bàn giao. Sau khi sửa nghiệp vụ, kiểm tra dự đoán, lưu/đọc lịch sử và tải ảnh qua API. Xem [kiến trúc](docs/architecture.md) và [ghi nhận kiểm chứng](docs/validation.md).

## Dữ liệu và sao lưu

Ứng dụng lưu ảnh vật lý vào thư mục **`storage/uploads/`** (được cấu hình qua `CAFE_STORAGE_DIR/uploads`). Khi client yêu cầu xem ảnh, hệ thống phục vụ qua **API route ảo** `/image-cafe-input/{filename}` sau khi đã kiểm tra quyền sở hữu ảnh của user. Clone mới không chứa tài khoản hoặc lịch sử.
.env, model, users/history, ảnh tải lên và môi trường ảo được Git bỏ qua. Khi sao lưu, dừng server rồi sao chép toàn bộ `CAFE_STORAGE_DIR` (users, histories, uploads); giữ secret và checkpoint riêng. Khi khôi phục, đặt lại storage và secret cũ, khởi động rồi kiểm tra đăng nhập, lịch sử và ảnh.

## Khắc phục lỗi

| Hiện tượng                                   | Cách xử lý                                                                                     |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `Activate.ps1` bị chặn                      | Dùng `.\.venv\Scripts\python.exe` thay cho `python`; không bắt buộc activate              |
| Thiếu checkpoint                               | Kiểm tra tên file và `CAFE_MODEL_DIR`; startup sẽ báo đường dẫn bị thiếu             |
| Lỗi `state_dict`, kiến trúc hoặc số lớp | Nhận lại đúng bộ trọng số; đổi tên file không giải quyết model không tương thích |
| Không có CUDA/hết VRAM                       | Đặt `CAFE_DEVICE=cpu` rồi restart; kiểm tra biến môi trường ghi đè `.env`           |
| Lỗi PyTorch/torchvision                        | Tạo lại `.venv`, cài đúng bộ phiên bản và chạy `python -m pip check`                |
| Cổng 8000 đang dùng                          | Dừng tiến trình cũ hoặc đổi `--port 8001` và địa chỉ truy cập                       |
| Không ghi được dữ liệu                    | Kiểm tra quyền ghi của `CAFE_STORAGE_DIR` và dung lượng đĩa                             |
| Token hỏng sau restart                         | Đặt secret cố định; đăng nhập lại nếu secret đã đổi                                 |
| Không thấy lịch sử cũ                      | Chuyển dữ liệu bằng script, kiểm tra storage và tài khoản đang dùng                     |
| PowerShell lỗi Unicode                         | Chạy `$env:PYTHONUTF8='1'` trước Python; lưu cấu hình bằng UTF-8                         |
| Camera/vị trí bị chặn                       | Dùng localhost và cấp quyền trình duyệt; truy cập qua mạng có thể cần HTTPS            |

Giới hạn hiện tại: JSON một worker, tài khoản khách dùng chung, giao diện chưa có luồng đăng nhập; thời gian suy luận phụ thuộc phần cứng.
