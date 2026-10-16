"""Application settings. Relative paths are resolved against the repository root."""

import os
import secrets
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_FILENAMES = ("best-yolov26.pt", "best-big-last.pth", "best-small-last.pth")


def configured_path(name: str, default: str) -> Path:
    value = Path(os.environ.get(name) or default).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_DIR / value).resolve()


@dataclass(frozen=True)
class Settings:
    project_dir: Path = PROJECT_DIR
    model_dir: Path = field(default_factory=lambda: configured_path("CAFE_MODEL_DIR", "models"))
    storage_dir: Path = field(default_factory=lambda: configured_path("CAFE_STORAGE_DIR", "storage"))
    auth_secret: str = field(default_factory=lambda: os.environ.get("CAFE_AUTH_SECRET") or secrets.token_urlsafe(32))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            origin.strip() for origin in os.environ.get("CAFE_CORS_ORIGINS", "").split(",") if origin.strip()
        )
    )
    device: str = field(default_factory=lambda: os.environ.get("CAFE_DEVICE", "auto").strip().lower())

    def __post_init__(self):
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("CAFE_DEVICE phải là auto, cpu hoặc cuda.")

    @property
    def users_file(self) -> Path:
        return self.storage_dir / "users.json"

    @property
    def histories_file(self) -> Path:
        return self.storage_dir / "histories.json"

    @property
    def upload_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def disease_file(self) -> Path:
        return PROJECT_DIR / "app" / "resources" / "diseases.json"

    @property
    def static_dir(self) -> Path:
        return PROJECT_DIR / "app" / "static"

    def validate_models(self) -> None:
        missing = [str(self.model_dir / name) for name in MODEL_FILENAMES if not (self.model_dir / name).is_file()]
        if missing:
            raise FileNotFoundError("Thiếu checkpoint. Chép model nhận riêng vào CAFE_MODEL_DIR: " + ", ".join(missing))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(PROJECT_DIR / ".env", override=False)
    return Settings()
