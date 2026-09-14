# -*- coding: utf-8 -*-
"""
ROUTER: HISTORY
===============
Triển khai các endpoint quản lý lịch sử chẩn đoán theo chuẩn coung21/cf-api:
- GET /histories/{user_id}
- GET /histories/{history_id}/detail
- GET /histories/map
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from services.history_service import (
    get_histories_by_user_id,
    get_history_by_id,
    get_histories_map
)

router = APIRouter(prefix="/histories", tags=["History"])


@router.get("/map")
async def get_map():
    """Lấy danh sách các điểm chẩn đoán có tọa độ để vẽ bản đồ dịch tễ."""
    points = await get_histories_map()
    return JSONResponse(content=points)


@router.get("/{history_id}/detail")
async def get_detail(history_id: str):
    """Lấy chi tiết một lần chẩn đoán kèm phác đồ điều trị và nguyên nhân từ data.json."""
    record = await get_history_by_id(history_id)
    if not record:
        raise HTTPException(status_code=404, detail="Không tìm thấy bản ghi lịch sử này")
    return JSONResponse(content=record)


@router.get("/{user_id}")
async def get_user_histories(user_id: str):
    """Lấy toàn bộ lịch sử chẩn đoán của một người dùng cụ thể."""
    histories = await get_histories_by_user_id(user_id)
    return JSONResponse(content=histories)
