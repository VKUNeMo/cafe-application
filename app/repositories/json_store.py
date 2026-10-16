"""JSON storage with atomic replacement. Supported with one application worker."""

import json
import os
import uuid
from pathlib import Path


def load_json(path: Path, expected_type: type):
    if not path.exists():
        return expected_type()
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, expected_type):
        raise ValueError(f"Dữ liệu JSON không hợp lệ: {path.name}")
    return value


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.stem}-{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
