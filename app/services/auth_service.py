import asyncio
import uuid
from typing import Any

from app.core.security import _create_token, _hash_password, _verify_password
from app.repositories.users import _load_users, _save_users

_write_lock = asyncio.Lock()


async def register_user(user_data: dict[str, Any]) -> dict[str, Any]:
    phone = str(user_data.get("phone", "")).strip()
    password = str(user_data.get("password", ""))
    username = str(user_data.get("username") or "").strip()

    if not phone or len(password) < 8:
        return {"error": "Cần số điện thoại và mật khẩu ít nhất 8 ký tự.", "status": 400}

    async with _write_lock:
        users = _load_users()
        if phone in users:
            return {"error": "Số điện thoại đã được đăng ký.", "status": 400}
        user_id = str(uuid.uuid4())
        users[phone] = {
            "user_id": user_id,
            "phone": phone,
            "username": username or phone,
            "password_hash": _hash_password(password),
        }
        _save_users(users)
    return {"message": "Đăng ký tài khoản thành công.", "user_id": user_id, "status": 200}


async def login_user(user_data: dict[str, Any]) -> dict[str, Any]:
    phone = str(user_data.get("phone", "")).strip()
    password = str(user_data.get("password", ""))
    user = _load_users().get(phone)
    if not user or not _verify_password(password, user.get("password_hash", "")):
        return {"error": "Số điện thoại hoặc mật khẩu không chính xác.", "status": 401}
    return {
        "message": "Đăng nhập thành công.",
        "token": _create_token(user["user_id"]),
        "user_id": user["user_id"],
        "username": user["username"],
        "status": 200,
    }


async def get_user_profile(user_id: str) -> dict[str, Any] | None:
    for user in _load_users().values():
        if user.get("user_id") == user_id:
            return {
                "user_id": user["user_id"],
                "phone": user["phone"],
                "username": user["username"],
            }
    return None
