import asyncio
import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Header, HTTPException

USERS_FILE = Path(__file__).resolve().parent.parent / "users.json"
TOKEN_SECRET = (os.environ.get("CAFE_AUTH_SECRET") or secrets.token_urlsafe(32)).encode()
TOKEN_LIFETIME_SECONDS = 7 * 24 * 60 * 60
PASSWORD_ITERATIONS = 260_000
_write_lock = asyncio.Lock()


def _load_users() -> dict[str, Any]:
    if not USERS_FILE.exists():
        return {}
    with open(USERS_FILE, "r", encoding="utf-8") as file:
        users = json.load(file)
    if not isinstance(users, dict):
        raise ValueError("Dữ liệu người dùng không hợp lệ.")
    return users


def _save_users(users: dict[str, Any]) -> None:
    temp_path = USERS_FILE.with_name(f".users-{uuid.uuid4().hex}.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(users, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, USERS_FILE)
    finally:
        temp_path.unlink(missing_ok=True)


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt, stored_digest = encoded.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt), int(iterations)
        )
        return hmac.compare_digest(digest, bytes.fromhex(stored_digest))
    except (ValueError, AttributeError):
        return False


def _create_token(user_id: str) -> str:
    payload = json.dumps(
        {"sub": user_id, "exp": int(time.time()) + TOKEN_LIFETIME_SECONDS},
        separators=(",", ":"),
    ).encode()
    encoded_payload = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    signature = hmac.new(TOKEN_SECRET, encoded_payload.encode(), hashlib.sha256).hexdigest()
    return f"{encoded_payload}.{signature}"


def _verify_token(token: str) -> str:
    try:
        encoded_payload, signature = token.split(".", 1)
        expected = hmac.new(TOKEN_SECRET, encoded_payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError
        payload = json.loads(base64.urlsafe_b64decode(encoded_payload + "=" * (-len(encoded_payload) % 4)))
        if int(payload["exp"]) <= time.time():
            raise ValueError
        user_id = str(payload["sub"])
        if not user_id:
            raise ValueError
        return user_id
    except (ValueError, KeyError, TypeError, binascii.Error, UnicodeDecodeError):
        raise HTTPException(status_code=401, detail="Token không hợp lệ hoặc đã hết hạn.")


def get_current_user_id(authorization: str | None = Header(None)) -> str:
    if authorization is None:
        return "default_user"
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Cần Bearer token hợp lệ.")
    return _verify_token(token)


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
