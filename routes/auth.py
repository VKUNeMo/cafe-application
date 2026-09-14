# -*- coding: utf-8 -*-
"""
ROUTER: AUTH
============
Triển khai các endpoint xác thực người dùng theo chuẩn coung21/cf-api:
- POST /auth/register
- POST /auth/login
"""

from pydantic import BaseModel
from typing import Optional
from fastapi import APIRouter
from fastapi.responses import JSONResponse

from services.auth_service import register_user, login_user

router = APIRouter(prefix="/auth", tags=["Auth"])


class RegisterRequest(BaseModel):
    phone: str
    password: str
    username: Optional[str] = None


class LoginRequest(BaseModel):
    phone: str
    password: str


@router.post("/register")
async def register(req: RegisterRequest):
    """Đăng ký tài khoản người dùng mới."""
    res = await register_user(req.dict())
    status_code = res.get("status", 200)
    return JSONResponse(content=res, status_code=status_code)


@router.post("/login")
async def login(req: LoginRequest):
    """Đăng nhập hệ thống và cấp token."""
    res = await login_user(req.dict())
    status_code = res.get("status", 200)
    return JSONResponse(content=res, status_code=status_code)
