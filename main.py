# -*- coding: utf-8 -*-
"""
FASTAPI WEB APPLICATION - COFFEE LEAF DISEASE CLASSIFIER & API
==============================================================
Hệ thống nhận diện bệnh lá cà phê chuẩn hóa kiến trúc theo coung21/cf-api:
- Pipeline: YOLOv8s (Fast Square Slicing) + Big/Small Classifier + Ensemble v5.
- Kiến trúc Modular:
  + /predictor/predict: Dự đoán và trả về phác đồ điều trị từ data.json
  + /histories: Quản lý lịch sử chẩn đoán và bản đồ dịch tễ
  + /auth: Đăng ký, đăng nhập tài khoản
  + /user: Quản lý thông tin người dùng
  + /ping: Healthcheck
- Frontend Web: Tích hợp đầy đủ tại / (HTML, JS, CSS)
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

# Thiết lập đường dẫn import (Độc lập trong cafe-application)
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from routes import init_routes
from services.predictor_service import get_pipeline
from services.auth_service import get_current_user_id
from services.history_service import owns_image

# Khởi tạo thư mục ảnh lưu trữ
IMAGE_INPUT_DIR = APP_DIR / "image-cafe-input"
IMAGE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = APP_DIR / "static"


@asynccontextmanager
async def lifespan(application: FastAPI):
    await run_in_threadpool(get_pipeline)
    yield


app = FastAPI(
    title="Coffee Leaf Disease API & Web Application",
    description="Hệ thống nhận diện bệnh lá cà phê sử dụng YOLOv8s + Big/Small Ensemble v5",
    version="3.0.0",
    lifespan=lifespan,
)

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get("CAFE_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    """Vô hiệu hóa cache trình duyệt đối với các tài nguyên tĩnh trong quá trình phát triển."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# Đăng ký các router chuẩn cf-api (/predictor, /histories, /auth, /user)
init_routes(app)

# Phục vụ thư mục tĩnh
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/ping")
async def ping():
    """Endpoint kiểm tra kết nối tương thích cf-api."""
    return JSONResponse(content={"message": "pong"})


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Phục vụ trang giao diện web chính."""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/image-cafe-input/{filename}")
async def serve_image(
    filename: str, current_user_id: str = Depends(get_current_user_id)
):
    image_path = IMAGE_INPUT_DIR / filename
    if (
        Path(filename).name != filename
        or not image_path.is_file()
        or not await owns_image(current_user_id, filename)
    ):
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh.")
    return FileResponse(image_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
