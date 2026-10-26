# 📚 TÀI LIỆU KỸ THUẬT API - COFFEE LEAF APPLICATION (HỆ THỐNG CHẨN ĐOÁN BỆNH LÁ CÀ PHÊ)

Tài liệu này dành cho đội ngũ phát triển Frontend, Mobile App hoặc bên thứ ba cần tích hợp hệ thống API nhận diện và chẩn đoán bệnh lá cà phê qua mô hình AI (YOLO + Big).

---

## 1. THÔNG TIN CHUNG (OVERVIEW)

- **Base URL (Local)**: `http://127.0.0.1:8000` (hoặc domain triển khai)
- **Kiến trúc**: RESTful API, chuẩn JSON & Multipart Form Data
- **Interactive Documentation**:
  - **Swagger UI (Thử trực tiếp)**: `http://127.0.0.1:8000/docs`
  - **ReDoc**: `http://127.0.0.1:8000/redoc`
- **Cơ chế xác thực (Authentication)**:
  - Header: `Authorization: Bearer <access_token>`
  - **Chế độ khách (Guest Mode)**: Nếu **không truyền token**, API sẽ tự động gán vào tài khoản khách mặc định (`default_user`). Ứng dụng client có thể gọi API chẩn đoán ngay mà không bắt buộc người dùng phải đăng nhập trước.

---

## 2. DANH MỤC CÁC LỚP BỆNH (DISEASE TAXONOMY)

Hệ thống chẩn đoán 8 lớp đối tượng trên lá cà phê:

| Mã lớp (idx) | Tên tiếng Anh (name) | Tên bệnh tiếng Việt (disease) |
| :------------: | :--------------------- | :-------------------------------- |
|     `0`     | `healthy`            | Lá Khỏe Mạnh                   |
|     `1`     | `miner`              | Sâu Đục Lá                   |
|     `2`     | `rust`               | Bệnh Gỉ Sắt Cà Phê           |
|     `3`     | `phoma`              | Bệnh Đốm Lá Phoma             |
|     `4`     | `corticium`          | Bệnh Nấm Hồng                  |
|     `5`     | `nematode`           | Bệnh Tuyến Trùng               |
|     `6`     | `anthracnose`        | Bệnh Thán Thư                  |
|     `7`     | `mealy`              | Rệp Sáp Gây Hại               |

---

## Logic Big

- Chỉ dùng YOLO và nhánh Big; không nạp/chạy Small và không tra bảng precision.
- Top 1 và top 2 lấy theo xác suất giảm dần của Big.
- Confidence ≤ 0.75: `show_top2` và `is_uncertain` là `true`, trả nhãn top 2.
- Confidence > 0.75: `show_top2` và `is_uncertain` là `false`, `top2_result: null`.
- Không có điều kiện margin hay ngưỡng confidence tối thiểu của top 2.
- `details`: `method: "big_only"`, `selected_branch: "big"`, `detected_boxes`, `big_branch`, `elapsed_seconds`.
- `big_branch`: `pred_class`, `confidence`, `all_probs`. Không còn `small_branch` hoặc `precision`.
- API không trả điểm confidence top 2. Lịch sử cũ được giữ nguyên.


## 3. CHI TIẾT CÁC ENDPOINTS

### 3.1. Kiểm tra trạng thái hệ thống (Health Check)

Kiểm tra API server đã khởi động và nạp xong model AI.

- **URL**: `GET /ping`
- **Headers**: Không yêu cầu
- **Phản hồi mẫu (200 OK)**:

```json
{
  "message": "pong"
}
```

---

### 3.2. Chẩn đoán bệnh lá cà phê (Core AI API)

Gửi ảnh chụp phiến lá cà phê để phân tích, phát hiện vùng đốm bệnh và trả về tên bệnh, độ tin cậy, phác đồ điều trị cùng ảnh vẽ bounding box.

- **URL**: `POST /predictor/predict`
- **Content-Type**: `multipart/form-data`
- **Headers**:
  - `Authorization: Bearer <token>` *(Tùy chọn: nếu có thì lưu lịch sử vào tài khoản, nếu không thì lưu vào `default_user`)*

#### Tham số Form-data:

| Tham số              | Kiểu dữ liệu |  Bắt buộc  | Mặc định            | Mô tả                                                                                                                                               |
| :-------------------- | :-------------: | :-----------: | :--------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`file`**    |  File (Binary)  | **Có** | -                      | File ảnh định dạng `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp` (tối đa 12 MB).                                                           |
| **`coords`**  |     String     |    Không    | `"14.0583,108.2772"` | Tọa độ GPS chụp ảnh: chuỗi `"vĩ_độ,kinh_độ"` (ví dụ: `"14.0583,108.2772"`) hoặc chuỗi JSON `{"lat": 14.0583, "long": 108.2772}`. |
| **`user_id`** |     String     |    Không    | `null`               | ID tài khoản người dùng (nếu có token thì để trống hoặc gửi khớp với token).                                                           |

#### Cơ chế kiểm định ảnh tự động (Image Quality Validation):

Trước khi đưa vào model AI, ảnh sẽ được kiểm tra chất lượng tự động:

- Nếu ảnh không đạt (kích thước quá nhỏ, quá tối, cháy sáng, không phải lá cà phê), API trả về **HTTP 400 Bad Request** kèm mã lỗi và hướng dẫn chụp lại. Hiện không kiểm tra độ sắc nét/độ mờ và không trả lỗi `IMAGE_BLURRY`.
- Các mã lỗi kiểm định:
  - `RESOLUTION_TOO_LOW`: Ảnh quá nhỏ (< 180x180 px hoặc < 40.000 pixels).
  - `IMAGE_TOO_DARK`: Ảnh chụp trong tối, thiếu sáng.
  - `IMAGE_OVEREXPOSED`: Ảnh chụp ngược sáng hoặc chói lóa ánh mặt trời.
  - `NOT_A_COFFEE_LEAF`: Ảnh nền đất/sàn nhà, tỷ lệ màu sắc mô lá thực vật < 15%.

#### Phản hồi lỗi kiểm định mẫu (400 Bad Request):

```json
{
  "success": false,
  "error_code": "IMAGE_TOO_DARK",
  "message": "Ảnh quá tối (độ sáng 8.5/255 < 35.0). Vui lòng bật đèn flash hoặc chụp ở nơi có đủ ánh sáng ban ngày.",
  "details": {
    "mean_brightness": 8.5,
    "threshold": 35.0
  }
}
```

#### Phản hồi chẩn đoán thành công (200 OK):

```json
{
  "history_user_id": "default_user",
  "coords": {
    "lat": 14.0583,
    "long": 108.2772
  },
  "result": {
    "idx": 2,
    "name": "rust",
    "disease": "Bệnh Gỉ Sắt Cà Phê"
  },
  "confidence": 0.945,
  "top2_result": null,
  "show_top2": false,
  "is_uncertain": false,
  "differential_note": "",
  "image_url": "e4b9c1d2f3a44b9c8d1e2f3a4b5c6d7e.jpg",
  "annotated_image_url": "e4b9c1d2f3a44b9c8d1e2f3a4b5c6d7e_annotated.jpg",
  "details": {
    "method": "big_only",
    "selected_branch": "big",
    "detected_boxes": [],
    "big_branch": {
      "pred_class": "Rust",
      "confidence": 0.945,
      "all_probs": {
        "Anthracnose": 0.005,
        "Corticium": 0.005,
        "Mealy": 0.005,
        "Miner": 0.005,
        "Nematode": 0.01,
        "Phoma": 0.02,
        "Rust": 0.945,
        "Healthy": 0.005
      }
    },
    "elapsed_seconds": 3.2
  }
}
```

*Ý nghĩa các trường quan trọng:*

- `result`: Thông tin bệnh chẩn đoán (`idx`, `name`, `disease`). *Lưu ý: Không trả về phác đồ và nguyên nhân (dành riêng cho luồng chuyên gia).*
- `confidence`: Điểm của nhánh được chọn, từ `0.0` đến `1.0`; không phải precision theo lớp hay xác suất đúng đã hiệu chuẩn của ensemble.
- `image_url`: Tên file ảnh gốc (ví dụ: `e4b9c1d2f3a44b9c8d1e2f3a4b5c6d7e.jpg`). Để tải hoặc hiển thị ảnh, truyền tên file này vào endpoint `GET /image-cafe-input/{filename}`.
- `annotated_image_url`: Tên file ảnh đã được khoanh vùng nhận diện vết bệnh (ví dụ: `e4b9c1d2f3a44b9c8d1e2f3a4b5c6d7e_annotated.jpg`). Để tải hoặc hiển thị ảnh, truyền tên file này vào endpoint `GET /image-cafe-input/{filename}`.
- `show_top2`: Nếu `true`, nghĩa là triệu chứng có khả năng trùng lặp với bệnh thứ hai (`top2_result`), giao diện nên hiển thị thêm cảnh báo phân biệt cho nông dân.

---

### 3.3. Tải lại ảnh đã chẩn đoán (Image Endpoint)

> **Lưu ý quan trọng về đường dẫn:**
> - `/image-cafe-input/{filename}` là **URL Endpoint ảo** (API Route) của hệ thống, **không phải** là thư mục vật lý `/image-cafe-input/` trên ổ cứng.
> - Thư mục lưu trữ vật lý thực tế của file ảnh là **`storage/uploads/{filename}`** (được quản lý bởi cấu hình `Settings.upload_dir` trong `app/core/config.py`).
> - Khi gọi tới endpoint này, server sẽ tự động kiểm tra quyền sở hữu ảnh (`owns_image`) của người dùng trước khi đọc file từ thư mục vật lý `storage/uploads/` và trả về.

- **URL**: `GET /image-cafe-input/{filename}`
- **Headers**:
  - `Authorization: Bearer <token>` *(Bắt buộc nếu ảnh thuộc tài khoản có đăng nhập. Đối với tài khoản mặc định `default_user`, có thể truy cập trực tiếp).*
- **Phản hồi**: File Binary ảnh (`image/jpeg` hoặc `image/png`).
- **Mã lỗi thường gặp**:
  - `404 Not Found`: Không tìm thấy ảnh hoặc tài khoản hiện tại không có quyền xem ảnh của tài khoản khác.

---

### 3.4. Xác thực người dùng (Auth Endpoints)

#### 3.4.1. Đăng ký tài khoản

- **URL**: `POST /auth/register`
- **Content-Type**: `application/json`
- **Request Body**:

```json
{
  "phone": "0912345678",
  "password": "mat_khau_bao_mat_123",
  "username": "Nguyen Van A"
}
```

- **Phản hồi (200 OK)**:

```json
{
  "status": 200,
  "message": "Đăng ký thành công",
  "user_id": "usr_9f8c1b2a3d4e5f6g"
}
```

#### 3.4.2. Đăng nhập

- **URL**: `POST /auth/login`
- **Content-Type**: `application/json`
- **Request Body**:

```json
{
  "phone": "0912345678",
  "password": "mat_khau_bao_mat_123"
}
```

- **Phản hồi (200 OK)**:

```json
{
  "status": 200,
  "message": "Đăng nhập thành công",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user_id": "usr_9f8c1b2a3d4e5f6g",
  "username": "Nguyen Van A"
}
```

*Lưu ý: Token có hiệu lực 7 ngày.*

---

### 3.5. Hồ sơ & Lịch sử người dùng

#### 3.5.1. Xem hồ sơ người dùng

- **URL**: `GET /user/{user_id}`
- **Headers**: `Authorization: Bearer <token>`
- **Phản hồi (200 OK)**:

```json
{
  "user_id": "usr_9f8c1b2a3d4e5f6g",
  "phone": "0912345678",
  "username": "Nguyen Van A",
  "created_at": "2026-10-22T08:30:00"
}
```

#### 3.5.2. Lấy danh sách lịch sử chẩn đoán

- **URL**: `GET /histories/{user_id}`
- **Headers**: `Authorization: Bearer <token>`
- **Phản hồi (200 OK)**:

```json
[
  {
    "id": "his_1a2b3c4d5e",
    "user_id": "usr_9f8c1b2a3d4e5f6g",
    "disease_name": "Bệnh Gỉ Sắt Cà Phê",
    "confidence": 0.9425,
    "image_url": "/image-cafe-input/e4b9c1d2f3a44b9c8d1e2f3a4b5c6d7e.jpg",
    "coords": {
      "lat": 14.0583,
      "long": 108.2772
    },
    "created_at": "2026-10-22T08:35:10"
  }
]
```

#### 3.5.3. Xem chi tiết 1 lần chẩn đoán

- **URL**: `GET /histories/{history_id}/detail`
- **Headers**: `Authorization: Bearer <token>`
- **Phản hồi (200 OK)**: Trả về chi tiết bản ghi kèm nguyên nhân và phác đồ điều trị bệnh.

#### 3.5.4. Lấy dữ liệu tọa độ cho bản đồ dịch tễ

- **URL**: `GET /histories/map`
- **Headers**: `Authorization: Bearer <token>` (hoặc không truyền để lấy điểm của khách)
- **Phản hồi (200 OK)**:

```json
[
  {
    "id": "his_1a2b3c4d5e",
    "disease_name": "Bệnh Gỉ Sắt Cà Phê",
    "lat": 14.0583,
    "long": 108.2772
  }
]
```

---

## 4. CODE MẪU TÍCH HỢP (CODE EXAMPLES)

### 4.1. cURL

```bash
# 1. Chẩn đoán dạng khách vãng lai
curl -X POST "http://127.0.0.1:8000/predictor/predict" \
     -F "file=@/duong_dan_den_anh/la_ca_phe.jpg" \
     -F "coords=14.0583,108.2772"

# 2. Chẩn đoán có Token người dùng
curl -X POST "http://127.0.0.1:8000/predictor/predict" \
     -H "Authorization: Bearer <token_o_day>" \
     -F "file=@/duong_dan_den_anh/la_ca_phe.jpg" \
     -F "coords=14.0583,108.2772"
```

### 4.2. JavaScript (Fetch API / React / React Native / Vue)

```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('coords', '14.0583,108.2772');

const response = await fetch('http://127.0.0.1:8000/predictor/predict', {
  method: 'POST',
  headers: {
    // Nếu có đăng nhập:
    // 'Authorization': `Bearer ${userToken}`
  },
  body: formData
});

const data = await response.json();
if (response.ok) {
  console.log('Mã bệnh:', data.result.idx);
  console.log('Tên mã:', data.result.name);
  console.log('Tên bệnh tiếng Việt:', data.result.disease);
  console.log('Độ tin cậy:', (data.confidence * 100).toFixed(1) + '%');
  // Hiển thị ảnh có bounding box
  document.getElementById('resultImage').src = `/image-cafe-input/${data.annotated_image_url}`;
} else {
  alert(`Lỗi (${data.error_code}): ${data.message}`);
}
```

### 4.3. Python (Requests)

```python
import requests

url = "http://127.0.0.1:8000/predictor/predict"
files = {"file": open("leaf_test.jpg", "rb")}
data = {"coords": "14.0583,108.2772"}
headers = {} # {"Authorization": f"Bearer {token}"}

response = requests.post(url, files=files, data=data, headers=headers)
if response.status_code == 200:
    res = response.json()
    print("Mã bệnh:", res["result"]["idx"])
    print("Tên mã:", res["result"]["name"])
    print("Bệnh:", res["result"]["disease"])
    print("Độ tin cậy:", res["confidence"])
else:
    print("Lỗi:", response.json())
```

---

## 5. MÃ TRẠNG THÁI HTTP (HTTP STATUS CODES)

|             Mã lỗi             | Ý nghĩa                                | Xử lý                                                                        |
| :------------------------------: | :--------------------------------------- | :----------------------------------------------------------------------------- |
|       **`200 OK`**       | Xử lý thành công                     | Đọc dữ liệu JSON trả về.                                                 |
|  **`400 Bad Request`**  | Ảnh không đạt tiêu chuẩn           | Hiển thị thông báo `message` để người dùng chụp lại theo gợi ý. |
|  **`401 Unauthorized`**  | Token hết hạn hoặc không hợp lệ    | Điều hướng về màn hình đăng nhập.                                    |
|   **`403 Forbidden`**   | Không có quyền truy cập tài nguyên | Kiểm tra lại `user_id` và token.                                          |
|   **`404 Not Found`**   | Không tìm thấy ảnh hoặc bản ghi    | Kiểm tra lại `filename` hoặc `id`.                                      |
| **`422 Unprocessable`** | Tham số truyền sai định dạng        | Kiểm tra lại kiểu dữ liệu form-data / JSON.                               |
| **`500 Internal Error`** | Lỗi server nội bộ                     | Kiểm tra lại log server và file checkpoint model.                           |
