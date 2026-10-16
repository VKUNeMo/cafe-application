import io
import json
import logging
import math
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from schemas import PredictionResponse
from services.history_service import add_history
from services.auth_service import get_current_user_id
from services.image_validator import validate_image_quality
from services.predictor_service import predict_coffee_leaf

router = APIRouter(prefix="/predictor", tags=["Predictor"])
IMAGE_INPUT_DIR = Path(__file__).resolve().parent.parent / "image-cafe-input"
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_COORDINATES = {"lat": 14.0583, "long": 108.2772}
logger = logging.getLogger(__name__)


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
        raise HTTPException(status_code=422, detail="coords phải là lat,long (ví dụ: 14.0583,108.2772) hoặc JSON {\"lat\":14.0583,\"long\":108.2772}; có thể bỏ trống để dùng mặc định.") from error
    return {"lat": round(latitude, 6), "long": round(longitude, 6)}


async def read_image(file: UploadFile) -> Image.Image:
    filename = file.filename or ""
    if Path(filename).suffix.lower() not in VALID_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ ảnh JPG, PNG, WebP hoặc BMP.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Ảnh phải nhỏ hơn 12 MB.")
    try:
        with Image.open(io.BytesIO(content)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Ảnh vượt quá 20 triệu pixel.")
            source.load()
            return source.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=400, detail="Không đọc được dữ liệu ảnh.") from error


@router.post("/predict", response_model=PredictionResponse)
async def predict_leaf(
    file: UploadFile = File(...),
    user_id: str | None = Form(None, examples=[None], description="Tùy chọn: ID người dùng."),
    coords: str | None = Form(
        None,
        examples=["14.0583,108.2772"],
        description="Tùy chọn: lat,long hoặc JSON có lat/long; bỏ trống dùng tọa độ mặc định.",
    ),
    current_user_id: str = Depends(get_current_user_id),
):
    clean_user_id = (
        user_id.strip()
        if user_id and user_id.strip().lower() not in {"", "string", "null", "undefined"}
        else None
    )
    if current_user_id != "default_user" and clean_user_id and clean_user_id != current_user_id:
        raise HTTPException(status_code=403, detail="Không thể lưu chẩn đoán cho người dùng khác.")
    try:
        image = await read_image(file)
    finally:
        await file.close()

    quality = await run_in_threadpool(validate_image_quality, image)
    if not quality["is_valid"]:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error_code": quality["error_code"],
                "message": quality["message"],
                "details": quality["details"],
            },
        )
    if coords:
        coordinates = parse_coordinates(coords)
    else:
        coordinates = DEFAULT_COORDINATES.copy()
    image_path: Path | None = None
    try:
        prediction = await predict_coffee_leaf(image)
        IMAGE_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"{uuid.uuid4().hex}.jpg"
        image_path = IMAGE_INPUT_DIR / filename
        await run_in_threadpool(image.save, image_path, "JPEG", quality=90)
        image_url = f"/image-cafe-input/{filename}"

        top2_result = prediction["top2_result"]
        history = {
            "user_id": current_user_id,
            "image_url": image_url,
            "result": prediction["result"]["idx"],
            "disease_name": prediction["result"]["disease"],
            "confidence": prediction["confidence"],
            "top2_name": top2_result["name"] if top2_result else "",
            "top2_confidence": prediction["top2_confidence"],
            "is_uncertain": prediction["is_uncertain"],
            "differential_note": prediction["differential_note"],
            "coords": coordinates,
        }
        await add_history(history)
        return {
            **prediction,
            "image_url": image_url,
            "coords": coordinates,
            "history_user_id": current_user_id,
        }
    except Exception as error:
        if image_path is not None:
            image_path.unlink(missing_ok=True)
        logger.exception("Không thể chẩn đoán ảnh")
        raise HTTPException(status_code=500, detail="Không thể xử lý ảnh lúc này.") from error
