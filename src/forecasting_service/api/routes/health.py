from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/health")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


@router.get("/live", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    """Report whether the API process is alive."""

    return HealthResponse()


@router.get("/ready", response_model=HealthResponse)
async def readiness(request: Request) -> HealthResponse:
    """Report whether configured dependencies are ready.

    The local durable store must be initialized for the process to be ready.
    """

    request.app.state.runtime.store.ping()
    return HealthResponse()
