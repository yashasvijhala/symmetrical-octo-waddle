from fastapi import APIRouter

from forecasting_service.api.routes.health import router as health_router
from forecasting_service.api.routes.platform import router as platform_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(platform_router)
