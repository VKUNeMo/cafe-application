"""Prediction workflow: validate, infer, store image and history."""

import logging
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.repositories.images import delete_image, save_image

from .history_service import add_history
from .image_validator import validate_image_quality
from .predictor_service import predict_coffee_leaf
from .uploads import DEFAULT_COORDINATES, parse_coordinates, read_image

logger = logging.getLogger(__name__)


async def diagnose(file: UploadFile, user_id: str | None, coords: str | None, current_user_id: str):
    clean_user_id = (
        user_id.strip() if user_id and user_id.strip().lower() not in {"", "string", "null", "undefined"} else None
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
        filename = f"{uuid.uuid4().hex}.jpg"
        image_path = await run_in_threadpool(save_image, image, filename)
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
            delete_image(image_path)
        logger.exception("Không thể chẩn đoán ảnh")
        raise HTTPException(status_code=500, detail="Không thể xử lý ảnh lúc này.") from error
