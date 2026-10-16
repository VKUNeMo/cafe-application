from typing import Any

from app.core.config import get_settings

from .json_store import load_json, save_json


def _load_users() -> dict[str, Any]:
    return load_json(get_settings().users_file, dict)


def _save_users(users: dict[str, Any]) -> None:
    save_json(get_settings().users_file, users)
