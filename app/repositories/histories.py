"""JSON history persistence."""

from typing import Any

from app.core.config import get_settings

from .json_store import load_json, save_json


def _load_histories() -> list[dict[str, Any]]:
    return load_json(get_settings().histories_file, list)


def _save_histories(histories: list[dict[str, Any]]) -> None:
    save_json(get_settings().histories_file, histories)
