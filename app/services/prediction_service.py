"""Validate uploads, run inference, and save prediction results."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.repositories.images import delete_image, save_image
from app.repositories.knowledge import get_disease_knowledge

from .history_service import add_history
from .image_validator import validate_image_quality
from .uploads import parse_coordinates, read_image

logger = logging.getLogger(__name__)
_inference_lock = asyncio.Lock()


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
        async with _inference_lock:
            prediction = await run_in_threadpool(_predict_sync, image)
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


def _predict_sync(image: Image.Image) -> dict[str, Any]:
    from app.inference import get_pipeline

    prediction = get_pipeline().predict(image, return_visual=True)
    show_top2 = bool(prediction["show_top2"])
    top2_class = prediction["top2_class"] if show_top2 else None
    top2_result = get_disease_knowledge_by_name(top2_class) if top2_class else None
    top2_confidence = round(float(prediction["top2_confidence"]), 4) if show_top2 else 0.0

    boxes = [
        {
            "id": index,
            "xyxy": [round(float(value), 1) for value in box["xyxy"]],
            "confidence": round(float(box["conf"]), 3),
            "class_name": box["class_name"],
        }
        for index, box in enumerate(prediction["detected_boxes"], start=1)
    ]
    result = get_disease_knowledge_by_name(prediction["pred_class"])
    top2_disease = top2_result["disease"] if top2_result else ""
    if show_top2 and top2_disease:
        pred_disease = result["disease"]
        differential_note = (
            f"Hệ thống nghiêng về {pred_disease}, nhưng nhận thấy đặc điểm "
            f"rất gần với {top2_disease}. "
            f"Khuyến nghị kiểm tra thêm mặt dưới lá hoặc biểu hiện cành trước khi dùng thuốc."
        )
    else:
        differential_note = ""

    return {
        "result": result,
        "confidence": round(float(prediction["confidence"]), 4),
        "top2_result": top2_result,
        "top2_confidence": top2_confidence,
        "show_top2": show_top2,
        "is_uncertain": show_top2,
        "differential_note": differential_note,
        "annotated_image": prediction["annotated_image"],
        "original_image": image,
        "details": {
            "method": "big_only",
            "selected_branch": "big",
            "detected_boxes": boxes,
            "big_branch": {
                **prediction["big_branch"],
                "confidence": round(float(prediction["confidence"]), 4),
                "all_probs": {name: round(float(value), 4) for name, value in prediction["big_branch"]["all_probs"].items()},
            },
            "elapsed_seconds": prediction["elapsed_seconds"],
        },
    }


def get_disease_knowledge_by_name(class_name: str) -> dict[str, Any]:
    record = get_disease_knowledge()[class_name.casefold()]
    return {
        "idx": record["idx"],
        "name": record["name"],
        "disease": record["disease"],
    }
