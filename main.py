# -*- coding: utf-8 -*-
"""
FASTAPI WEB APPLICATION - COFFEE LEAF DISEASE CLASSIFIER
=========================================================
Ứng dụng web nhận diện bệnh lá cà phê sử dụng FastAPI.
Pipeline: YOLOv8s + ResNet Big + ResNet Small + Ensemble (7 lớp: 6 bệnh + Healthy)
Không sử dụng database — ảnh được lưu trực tiếp vào folder image-cafe-input/
"""

import sys
import io
import base64
import uuid
from pathlib import Path
from datetime import datetime
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# PATH SETUP: Thêm thư mục cha vào sys.path để import pipeline
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent                        # cafe-application/
PROJECT_DIR = APP_DIR.parent                                      # e:\Thạc Sĩ\Project\Cafe
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pipeline.new_pipeline import NewCoffeePipeline

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
IMAGE_INPUT_DIR = APP_DIR / "image-cafe-input"
IMAGE_INPUT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = APP_DIR / "static"

# Disease info cho frontend
DISEASE_INFO = {
    "Healthy": {
        "vi": "Khỏe mạnh",
        "desc": "Lá cà phê khỏe mạnh, không có dấu hiệu bệnh.",
        "color": "#52b788"
    },
    "Rust": {
        "vi": "Bệnh Gỉ Sắt",
        "desc": "Do nấm Hemileia vastatrix gây ra. Xuất hiện các đốm bột vàng cam ở mặt dưới lá.",
        "color": "#e76f51"
    },
    "Phoma": {
        "vi": "Bệnh Phoma",
        "desc": "Do nấm Phoma spp. gây ra. Lá bị cháy nâu từ mép lá, lan vào trong.",
        "color": "#9b2226"
    },
    "Miner": {
        "vi": "Sâu Đục Lá",
        "desc": "Do ấu trùng sâu đục lá Leucoptera coffeella. Tạo đường hầm ngoằn ngoèo trên lá.",
        "color": "#bb3e03"
    },
    "Corticium": {
        "vi": "Bệnh Corticium",
        "desc": "Do nấm Corticium koleroga gây ra. Lá xuất hiện các mảng nâu đen lớn.",
        "color": "#6d4c41"
    },
    "Mealy": {
        "vi": "Rệp Sáp",
        "desc": "Côn trùng rệp sáp bám trên lá, tiết ra chất sáp trắng bao phủ bề mặt lá.",
        "color": "#adb5bd"
    },
    "Nematode": {
        "vi": "Tuyến Trùng",
        "desc": "Do tuyến trùng Meloidogyne spp. gây hại rễ, ảnh hưởng lên lá gây vàng úa héo.",
        "color": "#e9c46a"
    }
}

# ---------------------------------------------------------------------------
# PIPELINE SINGLETON
# ---------------------------------------------------------------------------
_pipeline_instance = None

def get_pipeline() -> NewCoffeePipeline:
    """Khởi tạo pipeline 1 lần duy nhất (singleton)."""
    global _pipeline_instance
    if _pipeline_instance is None:
        print("🚀 Đang khởi tạo Pipeline (setting_healthy - 7 lớp)...")
        _pipeline_instance = NewCoffeePipeline(
            setting_name="setting_healthy",
            base_dir=str(PROJECT_DIR)
        )
        print(f"✅ Pipeline sẵn sàng! Classes: {_pipeline_instance.class_names}")
    return _pipeline_instance

# ---------------------------------------------------------------------------
# VISUALIZATION HELPERS
# ---------------------------------------------------------------------------
def draw_results_on_image(
    img_pil: Image.Image,
    detected_boxes: list,
    crop_box_big=None,
    small_processed=None,
    final_class: str = "",
    final_prob: float = 0.0
) -> Image.Image:
    """Vẽ bounding boxes và kết quả lên ảnh."""
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    H, W = img_bgr.shape[:2]

    # 1. Vẽ YOLO Boxes (Màu đỏ)
    for db in detected_boxes:
        dx1, dy1, dx2, dy2 = map(int, db['xyxy'])
        cv2.rectangle(img_bgr, (dx1, dy1), (dx2, dy2), (0, 0, 255), 2)
        c = db['conf']
        cv2.putText(img_bgr, f"YOLO: {c:.2f}", (dx1, min(H - 10, dy2 + 18)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # 2. Vẽ Vùng Crop Big (Cyan)
    if crop_box_big is not None:
        bx1, by1, bx2, by2 = crop_box_big
        cv2.rectangle(img_bgr, (bx1, by1), (bx2, by2), (255, 255, 0), 3)
        lbl = "Big Union Crop" if len(detected_boxes) > 0 else "Center Square Crop"
        cv2.putText(img_bgr, lbl, (bx1 + 5, by1 + 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    # 3. Vẽ Vùng Crop Small (Vàng)
    if small_processed:
        for idx, pb in enumerate(small_processed):
            sx1, sy1, sx2, sy2 = pb['crop_box']
            cv2.rectangle(img_bgr, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)
            cv2.putText(img_bgr, f"Small#{idx+1}", (sx1 + 3, max(20, sy1 - 5)),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # 4. Vẽ banner kết quả cuối cùng phía trên ảnh
    if final_class:
        banner_h = 45
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (0, 0), (W, banner_h), (27, 67, 50), -1)
        img_bgr = cv2.addWeighted(overlay, 0.85, img_bgr, 0.15, 0)
        text = f"{final_class} ({final_prob*100:.1f}%)"
        cv2.putText(img_bgr, text, (15, 32),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (82, 183, 136), 2)

    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Chuyển PIL Image thành chuỗi base64."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ---------------------------------------------------------------------------
# FASTAPI APPLICATION
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Coffee Leaf Disease Detection",
    description="Hệ thống nhận diện bệnh lá cà phê sử dụng YOLOv8s + ResNet Ensemble",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/image-cafe-input", StaticFiles(directory=str(IMAGE_INPUT_DIR)), name="images")

# ---------------------------------------------------------------------------
# STARTUP EVENT
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    """Pre-load pipeline khi server khởi động."""
    get_pipeline()

# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve trang HTML chính."""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "classes": pipeline.class_names,
        "num_classes": pipeline.num_classes,
        "device": str(pipeline.device)
    }


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    """
    Nhận ảnh upload, chạy pipeline đầy đủ, trả kết quả JSON.
    Ảnh được lưu vào image-cafe-input/ để thu thập data train.
    """
    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File phải là ảnh (jpg, jpeg, png)")

    try:
        # Đọc ảnh
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")

        # Lưu ảnh vào image-cafe-input/ với timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        original_name = file.filename or "unknown.jpg"
        # Sanitize filename
        safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in original_name)
        saved_filename = f"{timestamp}_{unique_id}_{safe_name}"
        saved_path = IMAGE_INPUT_DIR / saved_filename
        with open(saved_path, "wb") as f:
            f.write(contents)

        # Chạy pipeline
        pipeline = get_pipeline()

        # 1. YOLO Detection
        detected = pipeline.detect_boxes(img_pil)

        # 2. Nhánh Big
        res_big = pipeline.predict_big(img_pil, detected_boxes=detected)

        # 3. Nhánh Small
        res_small = pipeline.predict_small(img_pil, detected_boxes=detected)

        # 4. Ensemble
        res_ensemble = pipeline.predict_ensemble(
            img_pil,
            big_pred=res_big,
            small_pred=res_small,
            detected_boxes=detected,
            conf_thresh=0.70
        )

        # Vẽ kết quả lên ảnh
        annotated_img = draw_results_on_image(
            img_pil,
            detected_boxes=detected,
            crop_box_big=res_big.get('crop_box'),
            small_processed=res_small.get('processed_boxes'),
            final_class=res_ensemble['pred_class'],
            final_prob=res_ensemble['pred_prob']
        )

        # Build response
        # Danh sách boxes detected
        boxes_info = []
        for i, db in enumerate(detected):
            boxes_info.append({
                "id": i + 1,
                "xyxy": [round(v, 1) for v in db['xyxy']],
                "confidence": round(db['conf'], 3)
            })

        # Small box predictions
        small_box_preds = []
        for bp in res_small.get('box_preds', []):
            small_box_preds.append({
                "crop_box": list(bp['crop_box']),
                "pred_class": bp['pred_class'],
                "confidence": round(bp['prob'], 3)
            })

        # Disease info cho class được dự đoán
        pred_class = res_ensemble['pred_class']
        disease_detail = DISEASE_INFO.get(pred_class, {
            "vi": pred_class,
            "desc": "",
            "color": "#666"
        })

        response = {
            "success": True,
            "saved_image": saved_filename,
            "image_size": {"width": img_pil.size[0], "height": img_pil.size[1]},

            # Kết quả Ensemble
            "prediction": {
                "class": pred_class,
                "class_vi": disease_detail["vi"],
                "description": disease_detail["desc"],
                "color": disease_detail["color"],
                "confidence": round(res_ensemble['pred_prob'], 4),
                "method": res_ensemble['method']
            },

            # Chi tiết Big
            "big_branch": {
                "class": res_big['pred_class'],
                "confidence": round(res_big['pred_prob'], 4),
                "all_probs": {k: round(v, 4) for k, v in res_big['all_probs'].items()},
                "crop_box": list(res_big.get('crop_box', [])),
                "has_boxes": res_big.get('has_boxes', False)
            },

            # Chi tiết Small
            "small_branch": {
                "class": res_small['pred_class'],
                "confidence": round(res_small['pred_prob'], 4),
                "all_probs": {k: round(v, 4) for k, v in res_small['all_probs'].items()},
                "has_boxes": res_small.get('has_boxes', False),
                "box_predictions": small_box_preds
            },

            # YOLO Boxes
            "detected_boxes": boxes_info,
            "num_boxes": len(detected),

            # Ảnh kết quả (base64)
            "annotated_image": pil_to_base64(annotated_img),
            "original_image": pil_to_base64(img_pil)
        }

        return JSONResponse(content=response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
