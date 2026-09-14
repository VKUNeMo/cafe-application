# -*- coding: utf-8 -*-
"""
ROUTER: USER
============
Triển khai endpoint lấy thông tin người dùng theo chuẩn coung21/cf-api:
- GET /user/{user_id}
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from services.auth_service import get_user_profile

router = APIRouter(prefix="/user", tags=["User"])


@router.get("/{user_id}")
async def get_user(user_id: str):
    """Lấy thông tin tài khoản của user."""
    profile = await get_user_profile(user_id)
    return JSONResponse(content=profile)
