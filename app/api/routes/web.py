"""Web assets, health check and authorized image downloads."""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from app.api.dependencies import get_current_user_id
from app.core.config import get_settings
from app.services.history_service import owns_image

router = APIRouter()


@router.get("/ping")
async def ping():
    return {"message": "pong"}


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_index():
    return HTMLResponse(get_settings().static_dir.joinpath("index.html").read_text(encoding="utf-8"))


@router.get("/image-cafe-input/{filename}")
async def serve_image(filename: str, current_user_id: str = Depends(get_current_user_id)):
    """Phục vụ file ảnh qua URL ảo /image-cafe-input/{filename}.
    
    Lưu ý: Ảnh vật lý được lưu trữ tại `storage/uploads/` (cấu hình qua Settings.upload_dir).
    Endpoint này kiểm tra quyền sở hữu ảnh (owns_image) trước khi trả về file.
    """
    image_path = get_settings().upload_dir / filename
    if Path(filename).name != filename or not image_path.is_file() or not await owns_image(current_user_id, filename):
        raise HTTPException(status_code=404, detail="Không tìm thấy ảnh.")
    return FileResponse(image_path)
