"""Storage for uploaded prediction images; public URLs are independent of paths."""

from pathlib import Path

from PIL import Image

from app.core.config import get_settings


def save_image(image: Image.Image, filename: str) -> Path:
    directory = get_settings().upload_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    try:
        image.save(path, "JPEG", quality=90)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def delete_image(path: Path) -> None:
    path.unlink(missing_ok=True)
