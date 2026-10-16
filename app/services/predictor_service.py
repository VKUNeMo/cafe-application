from __future__ import annotations

import asyncio
import base64
import io
from typing import TYPE_CHECKING, Any

from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.repositories.knowledge import get_disease_knowledge

if TYPE_CHECKING:
    from app.inference.pipeline import CoffeePipeline

_inference_lock = asyncio.Lock()


def get_pipeline() -> CoffeePipeline:
    # Import the ML stack only when startup/inference actually needs it.
    from app.inference import get_pipeline as get_cached_pipeline

    pipeline = get_cached_pipeline()
    missing = {name.casefold() for name in pipeline.class_names} - get_disease_knowledge().keys()
    if missing:
        raise ValueError(f"Thiếu dữ liệu bệnh học cho lớp: {sorted(missing)}")
    return pipeline


def get_disease_knowledge_by_name(class_name: str) -> dict[str, Any]:
    return get_disease_knowledge()[class_name.casefold()]


def pil_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _predict_sync(image: Image.Image) -> dict[str, Any]:
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
    small_box_predictions = [
        {
            "id": box["id"],
            "crop_box": [int(value) for value in box["crop_box"]],
            "pred_class": box["pred_class"],
            "confidence": round(float(box["prob"]), 3),
        }
        for box in prediction["small_branch"]["box_preds"]
    ]

    def branch_details(branch: dict[str, Any]) -> dict[str, Any]:
        return {
            "pred_class": branch["pred_class"],
            "confidence": round(float(branch["confidence"]), 4),
            "all_probs": {name: round(float(value), 4) for name, value in branch["all_probs"].items()},
        }

    return {
        "result": get_disease_knowledge_by_name(prediction["pred_class"]),
        "confidence": round(float(prediction["confidence"]), 4),
        "top2_result": top2_result,
        "top2_confidence": top2_confidence,
        "show_top2": show_top2,
        "is_uncertain": bool(prediction["is_uncertain"]),
        "differential_note": prediction["differential_note"] if show_top2 else "",
        "annotated_image": pil_to_base64(prediction["annotated_image"]),
        "original_image": pil_to_base64(image),
        "details": {
            "pred_class": prediction["pred_class"],
            "confidence": round(float(prediction["confidence"]), 4),
            "top2_class": top2_class or "None",
            "top2_confidence": top2_confidence,
            "is_uncertain": bool(prediction["is_uncertain"]),
            "differential_note": prediction["differential_note"] if show_top2 else "",
            "method": prediction["method"],
            "num_boxes": len(boxes),
            "detected_boxes": boxes,
            "big_branch": branch_details(prediction["big_branch"]),
            "small_branch": {
                **branch_details(prediction["small_branch"]),
                "box_predictions": small_box_predictions,
            },
            "elapsed_seconds": prediction["elapsed_seconds"],
        },
    }


async def predict_coffee_leaf(image: Image.Image) -> dict[str, Any]:
    async with _inference_lock:
        return await run_in_threadpool(_predict_sync, image)
