from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from forecasting_service import __version__
from forecasting_service.api.router import api_router
from forecasting_service.config import Settings, get_settings
from forecasting_service.runtime import DispatchError, Runtime


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    service_runtime = Runtime(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        service_runtime.close()

    app = FastAPI(
        title="Forecasting Service",
        summary="Train, backtest, register, and serve panel time-series forecasts.",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.runtime = service_runtime
    app.state.settings = settings
    app.include_router(api_router, prefix=settings.api_prefix)

    @app.exception_handler(DispatchError)
    async def dispatch_unavailable(_: Request, exc: DispatchError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

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
