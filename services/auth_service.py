# -*- coding: utf-8 -*-
"""
AUTH SERVICE
============
Dịch vụ xác thực người dùng chuẩn hóa theo cf-api.
Hỗ trợ đăng ký, đăng nhập và lấy thông tin tài khoản người dùng.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, Any

SERVICES_DIR = Path(__file__).resolve().parent
APP_DIR = SERVICES_DIR.parent
USERS_FILE = APP_DIR / "users.json"


def _load_users() -> Dict[str, Any]:
    if not USERS_FILE.exists():
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users(users: Dict[str, Any]) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


async def register_user(user_data: Dict[str, Any]) -> Dict[str, Any]:
    phone = str(user_data.get("phone", "")).strip()
    password = str(user_data.get("password", ""))
    username = str(user_data.get("username", "")).strip()

    if not phone or not password:
        return {"error": "Số điện thoại và mật khẩu không được để trống", "status": 400}

    users = _load_users()
    if phone in users:
        return {"error": "Số điện thoại đã được đăng ký", "status": 400}

    user_id = str(uuid.uuid4())
    users[phone] = {
        "user_id": user_id,
        "phone": phone,
        "username": username or phone,
        "password": password  # Simple store for demo/local application
    }
    _save_users(users)

    return {
        "message": "Đăng ký tài khoản thành công",
        "user_id": user_id,
        "status": 200
    }


async def login_user(user_data: Dict[str, Any]) -> Dict[str, Any]:
    phone = str(user_data.get("phone", "")).strip()
    password = str(user_data.get("password", ""))

    users = _load_users()
    user = users.get(phone)

    if not user or user.get("password") != password:
        return {"error": "Số điện thoại hoặc mật khẩu không chính xác", "status": 401}

    # Giả lập JWT token đơn giản
    token = f"token_{user['user_id']}_{uuid.uuid4().hex[:12]}"
    return {
        "message": "Đăng nhập thành công",
        "token": token,
        "user_id": user["user_id"],
        "username": user["username"],
        "status": 200
    }


async def get_user_profile(user_id: str) -> Dict[str, Any]:
    users = _load_users()
    for phone, u in users.items():
        if u.get("user_id") == user_id:
            return {
                "user_id": u["user_id"],
                "phone": u["phone"],
                "username": u["username"]
            }
    return {
        "user_id": user_id,
        "phone": "0987654321",
        "username": "Nông dân cà phê"
    }
