# -*- coding: utf-8 -*-
"""
PIPELINE MODULE
===============
Xuất các thành phần cốt lõi của Pipeline Ensemble v4.
"""

from pipeline.pipeline import CoffeePipeline, CoffeeClassifier, ConvNeXtCoffeeClassifier
from pipeline.config import (
    PipelineConfig,
    get_config,
    DEFAULT_CLASS_NAMES,
    YOLO_MODEL_PATH,
    BIG_MODEL_PATH,
    SMALL_MODEL_PATH
)

_cached_pipeline = None

def get_pipeline(device=None) -> CoffeePipeline:
    """Singleton helper trả về đối tượng CoffeePipeline duy nhất."""
    global _cached_pipeline
    if _cached_pipeline is None:
        _cached_pipeline = CoffeePipeline(device=device)
    return _cached_pipeline

__all__ = [
    "CoffeePipeline",
    "CoffeeClassifier",
    "ConvNeXtCoffeeClassifier",
    "get_pipeline",
    "PipelineConfig",
    "get_config",
    "DEFAULT_CLASS_NAMES"
]
