from __future__ import annotations

import asyncio
from typing import Any

from PIL import Image
from starlette.concurrency import run_in_threadpool

from app.repositories.knowledge import get_disease_knowledge

_inference_lock = asyncio.Lock()


def get_disease_knowledge_by_name(class_name: str) -> dict[str, Any]:
    record = get_disease_knowledge()[class_name.casefold()]
    return {
        "idx": record["idx"],
        "name": record["name"],
        "disease": record["disease"],
    }


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


async def predict_coffee_leaf(image: Image.Image) -> dict[str, Any]:
    async with _inference_lock:
        return await run_in_threadpool(_predict_sync, image)
