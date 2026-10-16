"""Password hashing and signed bearer tokens."""

import base64
import binascii
import hashlib
import hmac
import json
import secrets
import time

from fastapi import HTTPException

from .config import get_settings

TOKEN_LIFETIME_SECONDS = 7 * 24 * 60 * 60
PASSWORD_ITERATIONS = 260_000


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PASSWORD_ITERATIONS)
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt, stored_digest = encoded.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(digest, bytes.fromhex(stored_digest))
    except (ValueError, AttributeError):
        return False


def _create_token(user_id: str) -> str:
    payload = json.dumps(
        {"sub": user_id, "exp": int(time.time()) + TOKEN_LIFETIME_SECONDS},
        separators=(",", ":"),
    ).encode()
    encoded_payload = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    signature = hmac.new(get_settings().auth_secret.encode(), encoded_payload.encode(), hashlib.sha256).hexdigest()
    return f"{encoded_payload}.{signature}"


def _verify_token(token: str) -> str:
    try:
        encoded_payload, signature = token.split(".", 1)
        expected = hmac.new(get_settings().auth_secret.encode(), encoded_payload.encode(), hashlib.sha256).hexdigest()
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
