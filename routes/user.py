# -*- coding: utf-8 -*-
"""
ROUTER: USER
============
Triển khai endpoint lấy thông tin người dùng theo chuẩn coung21/cf-api:
- GET /user/{user_id}
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from services.auth_service import get_user_profile
from services.auth_service import get_current_user_id

router = APIRouter(prefix="/user", tags=["User"])


@router.get("/{user_id}")
async def get_user(user_id: str, current_user_id: str = Depends(get_current_user_id)):
    """Lấy thông tin tài khoản của user."""
    if user_id != current_user_id or user_id == "default_user":
        raise HTTPException(status_code=403, detail="Không thể xem hồ sơ này.")
    profile = await get_user_profile(user_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng.")
    return JSONResponse(content=profile)
