from typing import Optional

from pydantic import BaseModel


class RegisterRequest(BaseModel):
    phone: str
    password: str
    username: Optional[str] = None


class LoginRequest(BaseModel):
    phone: str
    password: str
