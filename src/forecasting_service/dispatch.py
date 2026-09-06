from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from forecasting_service.config import Settings


class JobDispatcher(Protocol):
    def submit(
        self,
        *,
        kind: str,
        tenant_id: str,
        resource_id: str,
        job_id: str,
        requires_gpu: bool,
    ) -> str: ...

    def cancel(self, run_id: str) -> None: ...

    def close(self) -> None: ...


class HatchetDispatcher:
    """Thin API-side adapter; task declarations are loaded only when first used."""

    def submit(
        self,
        *,
        kind: str,
        tenant_id: str,
        resource_id: str,
        job_id: str,
        requires_gpu: bool,
    ) -> str:
        from hatchet_sdk.types.priority import Priority

        from forecasting_service.worker import JobInput, get_registry

        registry = get_registry()
        task = registry.task(kind=kind, requires_gpu=requires_gpu)
        run = task.run(
            JobInput(tenant_id=tenant_id, resource_id=resource_id, job_id=job_id),
            wait_for_result=False,
            additional_metadata={
                "tenant_id": tenant_id,
                "job_id": job_id,
                "resource_id": resource_id,
            },
            priority=Priority.HIGH if kind == "forecast" else Priority.MEDIUM,
        )
        return run.workflow_run_id

    def cancel(self, run_id: str) -> None:
        from forecasting_service.worker import get_registry

        get_registry().hatchet.runs.cancel(run_id)

    def close(self) -> None:
        return None


class TestDispatcher:
    """Deterministic in-process adapter used exclusively by the test environment."""

    def __init__(
        self,
        max_workers: int,
        run_experiment: Callable[[str, str], None],
        run_forecast: Callable[[str, str], None],
    ) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="forecast-test"
        )
        self._handlers = {"experiment": run_experiment, "forecast": run_forecast}

    def submit(
        self,
        *,
        kind: str,
        tenant_id: str,
        resource_id: str,
        job_id: str,
        requires_gpu: bool,
    ) -> str:
        del tenant_id, requires_gpu
        self._executor.submit(self._handlers[kind], resource_id, job_id)
        return f"test:{job_id}"

    def cancel(self, run_id: str) -> None:
        del run_id

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)


def create_dispatcher(
    settings: Settings,
    run_experiment: Callable[[str, str], None],
    run_forecast: Callable[[str, str], None],
) -> JobDispatcher:
    if settings.environment == "test":
        return TestDispatcher(settings.test_max_workers, run_experiment, run_forecast)
    return HatchetDispatcher()
