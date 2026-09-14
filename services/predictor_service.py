# -*- coding: utf-8 -*-
"""
PREDICTOR SERVICE
=================
Service phụ trách suy luận mô hình cà phê:
- Nạp NewCoffeePipeline (setting_handle_nema: ResNet Big 448x448 + ResNet Small Patch Nema + YOLO).
- Thực thi quy trình 5 bước Ensemble v2 (Scale-Aware + Nematode Guard).
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

from pipeline.new_pipeline import NewCoffeePipeline

DATA_JSON_PATH = APP_DIR / "data.json"

# Load knowledge base data.json
with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
    DISEASE_KNOWLEDGE = json.load(f)


_pipeline_instance: Optional[NewCoffeePipeline] = None


def get_pipeline() -> NewCoffeePipeline:
    """Singleton quản lý nạp mô hình NewCoffeePipeline duy nhất một lần."""
    global _pipeline_instance
    if _pipeline_instance is None:
        yolo_path = PROJECT_DIR / "Model" / "model-sau-thực-địa" / "setting-yolo-add-image-realistic" / "yolo-v8-augment-640-height-image.pt"
        big_path = PROJECT_DIR / "Model" / "model-sau-thực-địa" / "setting-later-healthy" / "big_healthy_best_model.pth"
        small_path = PROJECT_DIR / "Model" / "model-sau-thực-địa" / "setting-later-healthy" / "small_healthy_best_model.pth"

        print("🚀 [PredictorService] Đang nạp Pipeline 7 lớp (Kèm Healthy)...")
        print(f"   + YOLO: {yolo_path.name}")
        print(f"   + ResNet Big: {big_path.name}")
        print(f"   + ResNet Small: {small_path.name}")

        _pipeline_instance = NewCoffeePipeline(
            base_dir=str(PROJECT_DIR),
            yolo_path=str(yolo_path),
            big_resnet_path=str(big_path),
            small_resnet_path=str(small_path),
            setting_name="setting_healthy"
        )
        print(f"✅ [PredictorService] Pipeline sẵn sàng! Danh sách lớp: {_pipeline_instance.class_names}")
    return _pipeline_instance


def draw_results_on_image(
    img_pil: Image.Image,
    detected_boxes: list,
    crop_box_big=None,
    small_processed=None,
    final_class: str = "",
    final_prob: float = 0.0
) -> Image.Image:
    """Vẽ bounding boxes và các lát cắt trực quan lên ảnh."""
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    # 1. Vẽ YOLO Boxes (Đỏ)
    for i, db in enumerate(detected_boxes):
        dx1, dy1, dx2, dy2 = map(int, db['xyxy'])
        cv2.rectangle(img_bgr, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
        c = db.get('conf', 1.0)
        cv2.putText(
            img_bgr, f"#{i+1} ({c:.2f})", (dx1, max(20, dy1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2
        )

    # 2. Vẽ Vùng Crop Big (Cyan)
    if crop_box_big is not None:
        bx1, by1, bx2, by2 = map(int, crop_box_big)
        cv2.rectangle(img_bgr, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
        lbl = "Big Square Crop"
        cv2.putText(img_bgr, lbl, (bx1 + 8, by1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    # 3. Vẽ Vùng Crop Small (Vàng)
    if small_processed:
        for idx, pb in enumerate(small_processed):
            sx1, sy1, sx2, sy2 = map(int, pb['crop_box'])
            cv2.rectangle(img_bgr, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)
            cv2.putText(img_bgr, f"S#{idx+1}", (sx1 + 3, min(H - 5, sy2 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 4. Vẽ banner kết quả phía trên ảnh
    if final_class:
        banner_h = 48
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (W, banner_h), (27, 67, 50), -1)
        img_bgr = cv2.addWeighted(overlay, 0.85, img_bgr, 0.15, 0)
        text = f"{final_class.upper()} ({final_prob*100:.1f}%)"
        cv2.putText(img_bgr, text, (15, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, (82, 183, 136), 2)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Chuyển đổi PIL Image thành base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_disease_knowledge_by_name(class_name: str) -> Dict[str, Any]:
    """Tìm thông tin bệnh trong data.json theo tên lớp (khớp case-insensitive)."""
    clean_name = class_name.lower().strip()
    for item in DISEASE_KNOWLEDGE:
        if item["name"].lower().strip() == clean_name:
            return item
    # Fallback nếu không khớp chính xác
    return {
        "idx": 99,
        "name": clean_name,
        "disease": class_name,
        "cause": f"Bệnh lá cà phê ({class_name})",
        "solution": ["Kiểm tra trực tiếp tại vườn và tham khảo ý kiến chuyên gia nông nghiệp."]
    }


async def predict_coffee_leaf(img_pil: Image.Image) -> Dict[str, Any]:
    """
    Hàm thực thi toàn bộ pipeline chuẩn:
    1. YOLO Detect boxes
    2. ResNet Big predict
    3. ResNet Small predict
    4. Ensemble v2 (Scale-Aware + Nematode Guard)
    5. Khớp nối data.json chuẩn cf-api
    """
    pipeline = get_pipeline()

    # 1. Phát hiện vết bệnh qua YOLO
    detected = pipeline.detect_boxes(img_pil)

    # 2. Suy luận nhánh Big
    res_big = pipeline.predict_big(img_pil, detected_boxes=detected)

    # 3. Suy luận nhánh Small
    res_small = pipeline.predict_small(img_pil, detected_boxes=detected)

    # 4. Ensemble 5 bước mới
    res_ensemble = pipeline.predict_ensemble(
        img_pil,
        big_pred=res_big,
        small_pred=res_small,
        detected_boxes=detected
    )

    final_class = res_ensemble["pred_class"]
    final_conf = float(res_ensemble["pred_prob"])

    # 5. Vẽ ảnh chú thích trực quan
    annotated_img = draw_results_on_image(
        img_pil,
        detected_boxes=detected,
        crop_box_big=res_big.get("crop_box"),
        small_processed=res_small.get("processed_boxes"),
        final_class=final_class,
        final_prob=final_conf
    )

    # 6. Tìm đối tượng dữ liệu tương ứng trong data.json
    disease_obj = get_disease_knowledge_by_name(final_class)

    # Chuẩn hóa danh sách boxes cho frontend
    boxes_info = []
    for i, db in enumerate(detected):
        boxes_info.append({
            "id": i + 1,
            "xyxy": [round(float(v), 1) for v in db["xyxy"]],
            "confidence": round(float(db.get("conf", 1.0)), 3)
        })

    # Chuẩn hóa small box predictions
    small_box_preds = []
    for bp in res_small.get("box_preds", []):
        small_box_preds.append({
            "crop_box": list(map(int, bp["crop_box"])),
            "pred_class": bp["pred_class"],
            "confidence": round(float(bp["prob"]), 3)
        })

    return {
        # Cấu trúc chuẩn theo cf-api
        "result": disease_obj,
        "confidence": round(final_conf, 4),
        "annotated_image": pil_to_base64(annotated_img),
        "original_image": pil_to_base64(img_pil),
        "details": {
            "pred_class": final_class,
            "method": res_ensemble.get("method", "scale_aware_ensemble"),
            "is_macro_box": res_ensemble.get("is_macro_box", False),
            "max_side_ratio": round(float(res_ensemble.get("max_side_ratio", 0.0)), 3),
            "num_boxes": len(detected),
            "detected_boxes": boxes_info,
            "big_branch": {
                "class": res_big["pred_class"],
                "confidence": round(float(res_big["pred_prob"]), 4),
                "all_probs": {k: round(float(v), 4) for k, v in res_big["all_probs"].items()}
            },
            "small_branch": {
                "class": res_small["pred_class"],
                "confidence": round(float(res_small["pred_prob"]), 4),
                "all_probs": {k: round(float(v), 4) for k, v in res_small["all_probs"].items()},
                "box_predictions": small_box_preds
            }
        }
    }
