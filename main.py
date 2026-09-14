# -*- coding: utf-8 -*-
"""
FASTAPI WEB APPLICATION - COFFEE LEAF DISEASE CLASSIFIER & API
==============================================================
Hệ thống nhận diện bệnh lá cà phê chuẩn hóa kiến trúc theo coung21/cf-api:
# - Pipeline: YOLOv8s (Fast Square Slicing) + ResNet Big (Width Crop) + ResNet Small (Individual Soft-Voting) + Scale-Aware Ensemble v2.
- Kiến trúc Modular:
  + /predictor/predict: Dự đoán và trả về phác đồ điều trị từ data.json
  + /histories: Quản lý lịch sử chẩn đoán và bản đồ dịch tễ
  + /auth: Đăng ký, đăng nhập tài khoản
  + /user: Quản lý thông tin người dùng
  + /ping: Healthcheck
- Frontend Web: Tích hợp đầy đủ tại / (HTML, JS, CSS)
"""

import sys
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Thiết lập đường dẫn import
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from routes import init_routes
from services.predictor_service import get_pipeline
from routes.predictor import predict_leaf

# Khởi tạo thư mục ảnh lưu trữ
IMAGE_INPUT_DIR = APP_DIR / "image-cafe-input"
IMAGE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = APP_DIR / "static"

app = FastAPI(
    title="Coffee Leaf Disease API & Web Application",
    description="Hệ thống nhận diện bệnh lá cà phê sử dụng YOLOv8s + ResNet Ensemble v2 (Chuẩn cf-api)",
    version="2.0.0"
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Đăng ký các router chuẩn cf-api (/predictor, /histories, /auth, /user)
init_routes(app)

# Phục vụ thư mục tĩnh
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/image-cafe-input", StaticFiles(directory=str(IMAGE_INPUT_DIR)), name="images")


@app.on_event("startup")
async def startup_event():
    """Tự động preload mô hình khi khởi động server."""
    print("🌿 Khởi động server: Đang nạp mô hình...")
    get_pipeline()


@app.get("/ping")
async def ping():
    """Endpoint kiểm tra kết nối tương thích cf-api."""
    return JSONResponse(content={"message": "pong"})


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Phục vụ trang giao diện web chính."""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


# Alias cho đường dẫn cũ để đảm bảo tính tương thích ngược (Backward Compatibility)
@app.post("/api/predict")
async def legacy_predict(file: UploadFile = File(...), user_id: str = Form(None), croods: str = Form(None)):
    """Chuyển tiếp yêu cầu từ endpoint cũ sang /predictor/predict."""
    return await predict_leaf(file=file, user_id=user_id, croods=croods)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
