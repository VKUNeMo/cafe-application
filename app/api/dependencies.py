"""Authentication dependencies shared by routes."""

from fastapi import Header, HTTPException

from app.core.security import _verify_token


def get_current_user_id(authorization: str | None = Header(None)) -> str:
    if authorization is None:
        return "default_user"
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Cần Bearer token hợp lệ.")
    return _verify_token(token)
