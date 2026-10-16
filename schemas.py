from typing import Any

from pydantic import BaseModel


class DiseaseInfo(BaseModel):
    idx: int
    name: str
    disease: str
    cause: str
    solution: list[str]


class PredictionResponse(BaseModel):
    history_user_id: str
    coords: dict[str, float]
    result: DiseaseInfo
    confidence: float
    top2_result: DiseaseInfo | None
    top2_confidence: float
    show_top2: bool
    is_uncertain: bool
    differential_note: str
    image_url: str
    annotated_image: str
    original_image: str
    details: dict[str, Any]
