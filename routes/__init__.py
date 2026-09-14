# -*- coding: utf-8 -*-
"""
ROUTES INITIALIZATION
=====================
Tập trung đăng ký tất cả các Router theo cấu trúc coung21/cf-api:
- predictor_router: /predictor
- history_router:   /histories
- auth_router:      /auth
- user_router:      /user
"""

from fastapi import FastAPI

from .predictor import router as predictor_router
from .history import router as history_router
from .auth import router as auth_router
from .user import router as user_router


def init_routes(app: FastAPI) -> None:
    """Đăng ký tất cả các router vào ứng dụng FastAPI."""
    app.include_router(predictor_router)
    app.include_router(history_router)
    app.include_router(auth_router)
    app.include_router(user_router)
