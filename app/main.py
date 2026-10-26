"""FastAPI application factory and startup lifecycle."""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.api.routes import init_routes
from app.api.routes.web import router as web_router
from app.core.config import get_settings
from app.repositories.knowledge import get_disease_knowledge

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    if not os.environ.get("CAFE_AUTH_SECRET"):
        logger.warning("CAFE_AUTH_SECRET chưa được đặt; token sẽ hết hiệu lực sau khi khởi động lại.")
    settings.validate_models()
    get_disease_knowledge()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    from app.inference import get_pipeline

    await run_in_threadpool(get_pipeline)
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title="Coffee Leaf Disease API & Web Application",
        description="Nhận diện bệnh lá cà phê bằng YOLO và Big.",
        version="3.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def add_no_cache_headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    init_routes(application)
    application.include_router(web_router)
    application.mount("/static", StaticFiles(directory=str(settings.static_dir)), name="static")
    return application


app = create_app()
