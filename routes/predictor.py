# -*- coding: utf-8 -*-
"""
ROUTER: PREDICTOR
=================
Triển khai endpoint POST /predictor/predict theo đúng chuẩn của coung21/cf-api:
- Nhận file (multipart/form-data), user_id, croods ("lat,long").
- Thực thi NewCoffeePipeline (Scale-Aware + Nematode Guard v2).
- Tự động lưu lịch sử chẩn đoán.
- Trả về cấu trúc JSON chuẩn:
  {
      "result": {
          "idx": int,
          "name": str,
          "disease": str,
          "cause": str,
          "solution": [str, ...]
      },
      "confidence": float,
      "image_url": str,
      ...
  }
"""

import io
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from services.predictor_service import predict_coffee_leaf
from services.history_service import add_history
from services.image_validator import validate_image_quality

router = APIRouter(prefix="/predictor", tags=["Predictor"])

APP_DIR = Path(__file__).resolve().parent.parent
IMAGE_INPUT_DIR = APP_DIR / "image-cafe-input"
IMAGE_INPUT_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/predict")
async def predict_leaf(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    croods: Optional[str] = Form(None),
    coords: Optional[str] = Form(None)
):
    """
    Dự đoán bệnh lá cà phê với file ảnh đầu vào và tọa độ GPS tùy chọn.
    Đáp ứng chuẩn interface của cf-api:
    - Input: file (ảnh), user_id (mã người dùng), croods/coords (chuỗi "lat,long")
    - Output: result {idx, name, disease, cause, solution}, confidence, image_url
    """
    # 1. Kiểm tra định dạng file
    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    fname = (file.filename or "").lower()
    is_valid_type = (file.content_type and file.content_type.startswith("image/")) or any(fname.endswith(ext) for ext in valid_exts)
    if not is_valid_type:
        raise HTTPException(status_code=400, detail="Tệp tải lên phải là hình ảnh (jpg, jpeg, png, webp)")

    try:
        contents = await file.read()
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")

        # 2. Kiểm tra chất lượng ảnh đầu vào (Tiền xử lý kiểm định)
        val_res = validate_image_quality(img_pil)
        if not val_res["is_valid"]:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error_code": val_res["error_code"],
                    "message": val_res["message"],
                    "details": val_res["details"]
                }
            )

        # 3. Lưu ảnh vào image-cafe-input/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        orig_name = file.filename or "leaf.jpg"
        safe_name = "".join(c if c.isalnum() or c in ".-_" else "_" for c in orig_name)
        saved_filename = f"{timestamp}_{unique_id}_{safe_name}"
        saved_path = IMAGE_INPUT_DIR / saved_filename

        with open(saved_path, "wb") as f:
            f.write(contents)

        image_url = f"/image-cafe-input/{saved_filename}"

        # 4. Phân tích và chuẩn hóa tọa độ GPS nếu có
        lat, long_val = 14.0583, 108.2772  # Mặc định Tây Nguyên
        gps_raw = coords or croods
        if gps_raw:
            try:
                parts = gps_raw.split(",")
                if len(parts) >= 2:
                    p_lat = float(parts[0].strip())
                    p_long = float(parts[1].strip())
                    if -90.0 <= p_lat <= 90.0 and -180.0 <= p_long <= 180.0:
                        lat, long_val = round(p_lat, 6), round(p_long, 6)
            except Exception:
                pass

        # 5. Thực thi suy luận qua Pipeline v2
        pred_res = await predict_coffee_leaf(img_pil)

        disease_info = pred_res["result"]
        confidence = pred_res["confidence"]

        # 5. Lưu vào lịch sử chẩn đoán
        top2_res = pred_res.get("top2_result")
        history_record = {
            "user_id": user_id or "default_user",
            "image_url": image_url,
            "result": disease_info.get("idx", 0),
            "disease_name": disease_info.get("disease", ""),
            "confidence": confidence,
            "top2_name": top2_res.get("disease", "") if top2_res else "",
            "top2_confidence": pred_res.get("top2_confidence", 0.0),
            "is_uncertain": pred_res.get("is_uncertain", False),
            "differential_note": pred_res.get("differential_note", ""),
            "croods": {
                "lat": lat,
                "long": long_val
            }
        }
        await add_history(history_record)

        # 6. Trả về Response chuẩn hóa 100% theo cf-api kèm mở rộng Top-2 và phân vân
        response_payload = {
            "result": disease_info,
            "confidence": confidence,
            "top2_result": pred_res.get("top2_result"),
            "top2_confidence": pred_res.get("top2_confidence", 0.0),
            "is_uncertain": pred_res.get("is_uncertain", False),
            "differential_note": pred_res.get("differential_note", ""),
            "image_url": image_url,
            "annotated_image": pred_res["annotated_image"],
            "original_image": pred_res["original_image"],
            "details": pred_res["details"]
        }

        return JSONResponse(content=response_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý hình ảnh: {str(e)}")
