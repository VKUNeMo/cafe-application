"""Prediction workflow: validate, infer, store image and history."""

import logging
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from PIL import Image

from app.repositories.images import delete_image, save_image

from .history_service import add_history
from .image_validator import validate_image_quality
from .predictor_service import predict_coffee_leaf
from .uploads import parse_coordinates, read_image

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
    coordinates = parse_coordinates(coords)
    image_path: Path | None = None
    annotated_path: Path | None = None
    try:
        prediction = await predict_coffee_leaf(image)
        stem = uuid.uuid4().hex
        filename = f"{stem}.jpg"
        annotated_filename = f"{stem}_annotated.jpg"

        image_path = await run_in_threadpool(save_image, image, filename)
        image_url = filename

        annotated_pil = prediction.get("annotated_image")
        if isinstance(annotated_pil, Image.Image):
            annotated_path = await run_in_threadpool(save_image, annotated_pil, annotated_filename)
            annotated_url = annotated_filename
        elif isinstance(annotated_pil, str) and annotated_pil:
            annotated_url = Path(annotated_pil).name
        else:
            annotated_url = filename

        top2_result = prediction["top2_result"]
        history = {
            "user_id": current_user_id,
            "image_url": image_url,
            "annotated_image_url": annotated_url,
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
            "history_user_id": current_user_id,
            "coords": coordinates,
            "result": prediction["result"],
            "confidence": prediction["confidence"],
            "top2_result": top2_result,
            "show_top2": prediction["show_top2"],
            "is_uncertain": prediction["is_uncertain"],
            "differential_note": prediction["differential_note"],
            "image_url": image_url,
            "annotated_image_url": annotated_url,
            "details": prediction["details"],
        }
    except Exception as error:
        if image_path is not None:
            delete_image(image_path)
        if annotated_path is not None:
            delete_image(annotated_path)
        logger.exception("Không thể chẩn đoán ảnh")
        raise HTTPException(status_code=500, detail="Không thể xử lý ảnh lúc này.") from error
