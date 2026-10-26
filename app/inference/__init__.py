"""Cached application inference pipeline."""

from app.inference.pipeline import CoffeePipeline

_cached_pipeline = None


def get_pipeline(device=None) -> CoffeePipeline:
    """Singleton helper trả về đối tượng CoffeePipeline duy nhất."""
    from app.repositories.knowledge import get_disease_knowledge

    global _cached_pipeline
    if _cached_pipeline is None:
        import torch

        from app.core.config import get_settings

        settings = get_settings()
        settings.validate_models()
        if device is None and settings.device != "auto":
            if settings.device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CAFE_DEVICE=cuda nhưng CUDA không khả dụng. Chọn cpu hoặc auto.")
            device = torch.device(settings.device)
        _cached_pipeline = CoffeePipeline(device=device)
    missing = {name.casefold() for name in _cached_pipeline.class_names} - get_disease_knowledge().keys()
    if missing:
        raise ValueError(f"Thiếu dữ liệu bệnh học cho lớp: {sorted(missing)}")
    return _cached_pipeline
