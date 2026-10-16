"""Read-only disease knowledge, shared by inference and history."""

import json
from functools import lru_cache

from app.core.config import get_settings


@lru_cache(maxsize=1)
def get_disease_knowledge() -> dict:
    records = json.loads(get_settings().disease_file.read_text(encoding="utf-8"))
    return {record["name"].casefold(): record for record in records}
