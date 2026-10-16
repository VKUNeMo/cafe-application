"""Multipart image validation and backwards compatible coordinate parsing."""

import io
import json
import math
from pathlib import Path

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

# Capture Pillow's decoder before the inference stack can install global plugins.
# Invalid supported uploads must not trigger Ultralytics' optional HEIF installer.
_open_image = Image.open

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_COORDINATES = {"lat": 14.0583, "long": 108.2772}


def parse_coordinates(raw_coordinates: str | None) -> dict[str, float]:
    if not raw_coordinates or raw_coordinates.strip().lower() in {"", "null", "undefined", "string", "none"}:
        return DEFAULT_COORDINATES.copy()
    try:
        if raw_coordinates.lstrip().startswith("{"):
            value = json.loads(raw_coordinates)
            parts = [value["lat"], value["long"]]
        else:
            parts = raw_coordinates.split(",")
            if len(parts) != 2:
                raise ValueError
        if any(isinstance(part, bool) for part in parts):
            raise ValueError
        latitude, longitude = (float(str(part).strip()) for part in parts)
        if not (math.isfinite(latitude) and math.isfinite(longitude)):
            raise ValueError
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(
            status_code=422,
            detail='coords phải là lat,long (ví dụ: 14.0583,108.2772) hoặc JSON {"lat":14.0583,"long":108.2772}; có thể bỏ trống để dùng mặc định.',
        ) from error
    return {"lat": round(latitude, 6), "long": round(longitude, 6)}


async def read_image(file: UploadFile) -> Image.Image:
    filename = file.filename or ""
    if Path(filename).suffix.lower() not in VALID_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ ảnh JPG, PNG, WebP hoặc BMP.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Ảnh phải nhỏ hơn 12 MB.")
    try:
        with _open_image(io.BytesIO(content)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Ảnh vượt quá 20 triệu pixel.")
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=400, detail="Không đọc được dữ liệu ảnh.") from error
