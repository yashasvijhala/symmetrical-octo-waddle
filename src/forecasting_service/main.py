from contextlib import asynccontextmanager

from fastapi import FastAPI

from forecasting_service import __version__
from forecasting_service.api.router import api_router
from forecasting_service.config import Settings, get_settings
from forecasting_service.runtime import Runtime


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    service_runtime = Runtime(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        service_runtime.executor.shutdown(wait=False, cancel_futures=False)

    app = FastAPI(
        title="Forecasting Service",
        summary="Train, backtest, register, and serve panel time-series forecasts.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.runtime = service_runtime
    app.state.settings = settings
    app.include_router(api_router, prefix=settings.api_prefix)

    @app.get("/", include_in_schema=False)
    async def service_metadata() -> dict[str, str]:
        return {
            "name": "forecasting-service",
            "version": __version__,
            "docs": "/docs",
        }

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("forecasting_service.main:app", host="0.0.0.0", port=8000, reload=False)
