# -*- coding: utf-8 -*-
"""
PREDICTOR SERVICE
=================
Service phụ trách suy luận mô hình cà phê:
- Nạp CoffeePipeline chuẩn Ensemble v4 (YOLOv8s + ResNet Big + ResNet Small).
- Thực thi quy trình chẩn đoán đa tầng kèm Top-2 Differential Diagnosis và Cờ phân vân lâm sàng.
- Ghép nối dữ liệu bệnh học từ data.json theo chuẩn cf-api.
"""

import sys
import io
import json
import base64
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image

SERVICES_DIR = Path(__file__).resolve().parent
APP_DIR = SERVICES_DIR.parent
PROJECT_DIR = APP_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from pipeline.pipeline import CoffeePipeline

DATA_JSON_PATH = APP_DIR / "data.json"

# Nạp cơ sở tri thức bệnh học từ data.json
with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
    DISEASE_KNOWLEDGE = json.load(f)


_pipeline_instance: Optional[CoffeePipeline] = None


def get_pipeline() -> CoffeePipeline:
    """Singleton quản lý nạp mô hình CoffeePipeline duy nhất một lần."""
    global _pipeline_instance
    if _pipeline_instance is None:
        print("🚀 [PredictorService] Đang nạp CoffeePipeline Ensemble v4 (8 lớp)...")
        _pipeline_instance = CoffeePipeline()
        print(f"✅ [PredictorService] Pipeline sẵn sàng! Danh sách lớp: {_pipeline_instance.class_names}")
    return _pipeline_instance


def pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Chuyển đổi PIL Image thành chuỗi base64."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_disease_knowledge_by_name(class_name: str) -> Dict[str, Any]:
    """Tìm thông tin bệnh trong data.json theo tên lớp (khớp case-insensitive)."""
    if not class_name or class_name == "None":
        return None
    clean_name = class_name.lower().strip()
    for item in DISEASE_KNOWLEDGE:
        if item["name"].lower().strip() == clean_name:
            return item
    # Fallback nếu không khớp chính xác
    return {
        "idx": 99,
        "name": clean_name,
        "disease": f"Bệnh {class_name}",
        "cause": f"Bệnh hại lá cà phê ({class_name}).",
        "solution": ["Kiểm tra trực tiếp tại vườn và tham khảo ý kiến chuyên gia nông nghiệp."]
    }


async def predict_coffee_leaf(img_pil: Image.Image) -> Dict[str, Any]:
    """
    Quy trình chẩn đoán toàn diện:
    1. YOLOv8s Slicing BBoxes
    2. ResNet Big (Width Crop)
    3. ResNet Small (BBox Crop)
    4. Ensemble v4 (Top-1 + Top-2 + Cờ phân vân)
    5. Ghép nối data.json chuẩn cf-api
    """
    pipeline = get_pipeline()

    # Chạy suy luận toàn bộ qua pipeline
    pred_res = pipeline.predict(img_pil, return_visual=True)

    final_class = pred_res["pred_class"]
    final_conf = float(pred_res["confidence"])
    top2_class = pred_res["top2_class"]
    top2_conf = float(pred_res["top2_confidence"])
    is_uncertain = pred_res["is_uncertain"]
    diff_note = pred_res["differential_note"]

    # Khớp nối tri thức bệnh học
    disease_obj = get_disease_knowledge_by_name(final_class)
    top2_disease_obj = get_disease_knowledge_by_name(top2_class) if top2_class != "None" else None

    # Chuẩn hóa danh sách detected boxes cho API frontend
    boxes_info = []
    for i, db in enumerate(pred_res.get("detected_boxes", [])):
        boxes_info.append({
            "id": i + 1,
            "xyxy": [round(float(v), 1) for v in db["xyxy"]],
            "confidence": round(float(db.get("conf", 1.0)), 3),
            "class_name": db.get("class_name", "lesion")
        })

    # Chuẩn hóa small box predictions
    small_box_preds = []
    for bp in pred_res.get("small_branch", {}).get("box_preds", []):
        small_box_preds.append({
            "id": bp.get("id", 1),
            "crop_box": [int(v) for v in bp.get("crop_box", [])],
            "pred_class": bp.get("pred_class", ""),
            "confidence": round(float(bp.get("prob", 0.0)), 3)
        })

    annotated_b64 = pil_to_base64(pred_res["annotated_image"]) if pred_res.get("annotated_image") else ""
    original_b64 = pil_to_base64(img_pil)

    return {
        # Cấu trúc chuẩn tương thích cf-api
        "result": disease_obj,
        "confidence": round(final_conf, 4),
        "top2_result": top2_disease_obj,
        "top2_confidence": round(top2_conf, 4),
        "is_uncertain": is_uncertain,
        "differential_note": diff_note,
        "annotated_image": annotated_b64,
        "original_image": original_b64,
        "details": {
            "pred_class": final_class,
            "confidence": round(final_conf, 4),
            "top2_class": top2_class,
            "top2_confidence": round(top2_conf, 4),
            "is_uncertain": is_uncertain,
            "differential_note": diff_note,
            "method": pred_res.get("method", "ensemble_v4"),
            "num_boxes": len(boxes_info),
            "detected_boxes": boxes_info,
            "big_branch": {
                "pred_class": pred_res["big_branch"]["pred_class"],
                "confidence": round(float(pred_res["big_branch"]["confidence"]), 4),
                "all_probs": {k: round(float(v), 4) for k, v in pred_res["big_branch"]["all_probs"].items()}
            },
            "small_branch": {
                "pred_class": pred_res["small_branch"]["pred_class"],
                "confidence": round(float(pred_res["small_branch"]["confidence"]), 4),
                "all_probs": {k: round(float(v), 4) for k, v in pred_res["small_branch"]["all_probs"].items()},
                "box_predictions": small_box_preds
            },
            "elapsed_seconds": pred_res.get("elapsed_seconds", 0.0)
        }
    }
