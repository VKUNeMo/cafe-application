"""Prediction HTTP endpoint."""

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.dependencies import get_current_user_id
from app.schemas.prediction import PredictionResponse
from app.services.prediction_service import diagnose
from app.services.uploads import DEFAULT_COORDINATES

router = APIRouter(prefix="/predictor", tags=["Predictor"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_leaf(
    file: UploadFile = File(...),
    user_id: str | None = Form(None, examples=[None], description="Tùy chọn: ID người dùng."),
    coords: str | None = Form(
        f"{DEFAULT_COORDINATES['lat']},{DEFAULT_COORDINATES['long']}",
        examples=["14.0583,108.2772"],
        description="Tùy chọn: lat,long hoặc JSON có lat/long; mặc định là 14.0583,108.2772 ở Tây Nguyên.",
    ),
    current_user_id: str = Depends(get_current_user_id),
):
    return await diagnose(file, user_id, coords, current_user_id)
