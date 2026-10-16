# Coffee Leaf API

Ứng dụng dùng bản `CoffeePipeline` ở thư mục `pipeline/` của application, đồng bộ từ `../pipeline/`. Ba checkpoint được nạp từ `models/`. Checkpoint Small chuẩn nằm ở `small_best_model-rust-only-real-canonical.pth`; bản `small_best_model-rust-only-real.pth` có hash khác và không được dùng.

## Chạy ứng dụng

```powershell
cd cafe-application
python -m pip install -r requirements.txt
$env:CAFE_AUTH_SECRET = "thay-bang-chuoi-bi-mat-on-dinh"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Đặt `CAFE_CORS_ORIGINS` thành danh sách origin cách nhau bằng dấu phẩy khi giao diện chạy ở domain khác. Không đặt biến này thì chỉ các yêu cầu cùng origin được trình duyệt cho phép. Chạy một worker khi còn dùng `users.json` và `histories.json`; cần database cho nhiều worker.

## API

| Endpoint                                      | Mục đích                                                    |
| --------------------------------------------- | -------------------------------------------------------------- |
| `GET /ping`                                 | Kiểm tra kết nối                                            |
| `POST /predictor/predict`                   | Dự đoán từ multipart `file` (bắt buộc), `coords` (tùy chọn) |
| `POST /auth/register`, `POST /auth/login` | Đăng ký, nhận Bearer token                                 |
| `GET /user/{user_id}`                       | Xem hồ sơ của token hiện tại                              |
| `GET /histories/{user_id}`                  | Xem lịch sử của token hiện tại                            |
| `GET /histories/{history_id}/detail`        | Chi tiết một lần dự đoán                                 |
| `GET /histories/map`                        | Các điểm có tọa độ của người dùng hiện tại        |
| `GET /image-cafe-input/{filename}`          | Tải ảnh thuộc lịch sử của người dùng hiện tại       |

Không gửi token sẽ dùng tài khoản khách `default_user`; `user_id` kiểu cũ vẫn được nhận nhưng lịch sử luôn lưu dưới `default_user`. Trường `history_user_id` trong response cho biết ID thực tế được dùng để lưu lịch sử. Với tài khoản đã đăng nhập, gửi `Authorization: Bearer <token>` và `user_id` đúng với token. Ảnh tải lên tối đa 12 MB và 20 triệu pixel. Trường form-data `coords` nhận `lat,long` (ví dụ `14.0583,108.2772`, dùng dấu chấm thập phân) hoặc chuỗi JSON `{"lat":14.0583,"long":108.2772}`. Không gửi, để trống hoặc gửi `null` sẽ dùng mặc định `14.0583,108.2772` (Tây Nguyên) và vẫn ghi điểm lên bản đồ; giá trị sai định dạng nhận 422. Lịch sử cũ có trường `croods` vẫn được đọc và trả về dưới tên `coords`. API trả `top2_result` khi chính sách `show_top2` của pipeline cho phép. Chi tiết hợp đồng xem tại `/docs`.

Ảnh được lưu tại `image-cafe-input/`. Tài khoản đã đăng nhập phải gửi Bearer token khi tải ảnh; giao diện web hiện dùng tài khoản khách. Đặt `CAFE_AUTH_SECRET` cố định để token còn hợp lệ sau khi khởi động lại.
