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
    """Report whether PostgreSQL and object storage are reachable."""

    request.app.state.runtime.store.ping()
    request.app.state.runtime.objects.ping()
    return HealthResponse()
