# -*- coding: utf-8 -*-
"""
HISTORY SERVICE
===============
Dịch vụ quản lý lịch sử chẩn đoán:
- Lưu trữ lịch sử cục bộ tại histories.json (không cần cài đặt database bên ngoài).
- Cung cấp các API truy vấn lịch sử theo user_id, xem chi tiết theo ID, và bản đồ dịch tễ map.
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

SERVICES_DIR = Path(__file__).resolve().parent
APP_DIR = SERVICES_DIR.parent
HISTORIES_FILE = APP_DIR / "histories.json"
DATA_JSON_PATH = APP_DIR / "data.json"
_write_lock = asyncio.Lock()


def _normalize_history(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = record.copy()
    normalized["coords"] = normalized.pop("coords", None) or normalized.pop("croods", None)
    return normalized


def _load_histories() -> List[Dict[str, Any]]:
    """Đọc toàn bộ lịch sử từ file JSON."""
    if not HISTORIES_FILE.exists():
        return []
    with open(HISTORIES_FILE, "r", encoding="utf-8") as f:
        histories = json.load(f)
    if not isinstance(histories, list):
        raise ValueError("Dữ liệu lịch sử không hợp lệ.")
    return histories


def _save_histories(histories: List[Dict[str, Any]]) -> None:
    """Lưu danh sách lịch sử vào file JSON."""
    temp_path = HISTORIES_FILE.with_name(f".histories-{uuid.uuid4().hex}.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(histories, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, HISTORIES_FILE)
    finally:
        temp_path.unlink(missing_ok=True)


async def add_history(history_data: Dict[str, Any]) -> str:
    """Thêm bản ghi chẩn đoán và trả về ID."""
    async with _write_lock:
        histories = _load_histories()

        record_id = str(uuid.uuid4())
        coords = history_data.get("coords")

        record = {
            "id": record_id,
            "user_id": str(history_data.get("user_id") or "default_user"),
            "image_url": str(history_data.get("image_url", "")),
            "result": history_data.get("result"),  # idx hoặc tên bệnh
            "disease_name": history_data.get("disease_name", ""),
            "disease": history_data.get("disease_name", ""),
            "confidence": float(history_data.get("confidence", 0.0)),
            "top2_name": str(history_data.get("top2_name", "")),
            "top2_confidence": float(history_data.get("top2_confidence", 0.0)),
            "is_uncertain": bool(history_data.get("is_uncertain", False)),
            "differential_note": str(history_data.get("differential_note", "")),
            "coords": coords,
            "created_at": datetime.now().isoformat()
        }

        histories.insert(0, record)  # Mới nhất lên đầu
        _save_histories(histories)
        return record_id


async def get_histories_by_user_id(user_id: str) -> List[Dict[str, Any]]:
    """Lấy danh sách các lần chẩn đoán của một user_id."""
    histories = _load_histories()
    return [_normalize_history(h) for h in histories if str(h.get("user_id")) == str(user_id)]


async def owns_image(user_id: str, filename: str) -> bool:
    image_url = f"/image-cafe-input/{filename}"
    return any(
        record.get("image_url") == image_url
        for record in await get_histories_by_user_id(user_id)
    )


async def get_history_by_id(history_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Lấy chi tiết một lần chẩn đoán kèm phác đồ điều trị đầy đủ từ data.json."""
    histories = _load_histories()
    target = next((h for h in histories if str(h.get("id")) == str(history_id) and str(h.get("user_id")) == user_id), None)
    if not target:
        return None

    # Ghép thông tin chi tiết từ data.json
    try:
        with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
            data_kb = json.load(f)
    except Exception:
        data_kb = []

    res_val = target.get("result")
    matching_obj = None
    if isinstance(res_val, int):
        matching_obj = next((item for item in data_kb if item["idx"] == res_val), None)
    elif isinstance(res_val, str):
        matching_obj = next((item for item in data_kb if item["name"].lower() == res_val.lower()), None)

    # Ghép thông tin Top-2 nếu có
    top2_name = target.get("top2_name", "")
    top2_obj = None
    if top2_name:
        top2_obj = next((item for item in data_kb if item["name"].lower() == top2_name.lower() or item["disease"].lower() == top2_name.lower()), None)

    return {
        "id": target["id"],
        "user_id": target.get("user_id"),
        "image_url": target["image_url"],
        "confidence": target["confidence"],
        "coords": _normalize_history(target)["coords"],
        "result": matching_obj or {"idx": 99, "name": str(res_val), "disease": target.get("disease_name", ""), "cause": "", "solution": []},
        "top2_result": top2_obj,
        "top2_name": top2_name,
        "top2_confidence": target.get("top2_confidence", 0.0),
        "is_uncertain": target.get("is_uncertain", False),
        "differential_note": target.get("differential_note", ""),
        "created_at": target["created_at"]
    }


async def get_histories_map(user_id: str) -> List[Dict[str, Any]]:
    """Trả về danh sách tọa độ và kết quả bệnh để trực quan hóa bản đồ dịch tễ."""
    histories = await get_histories_by_user_id(user_id)
    map_points = []
    for h in histories:
        coords = h.get("coords")
        if coords and "lat" in coords and "long" in coords:
            map_points.append({
                "id": h["id"],
                "image_url": h["image_url"],
                "result": h["result"],
                "disease_name": h.get("disease_name", ""),
                "confidence": h["confidence"],
                "coords": coords,
                "created_at": h["created_at"]
            })
    return map_points
