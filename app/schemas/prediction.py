from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DiseaseInfo(BaseModel):
    idx: int
    name: str
    disease: str


class DetectionBox(BaseModel):
    id: int
    xyxy: list[float] = Field(min_length=4, max_length=4)
    confidence: float
    class_name: str


class BranchPrediction(BaseModel):
    pred_class: str
    confidence: float
    all_probs: dict[str, float]


class PredictionDetails(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["big_only"]
    selected_branch: Literal["big"]
    detected_boxes: list[DetectionBox]
    big_branch: BranchPrediction
    elapsed_seconds: float


class PredictionResponse(BaseModel):
    history_user_id: str
    coords: dict[str, float]
    result: DiseaseInfo
    confidence: float
    top2_result: DiseaseInfo | None
    show_top2: bool
    is_uncertain: bool
    differential_note: str
    image_url: str
    annotated_image_url: str
    details: PredictionDetails
