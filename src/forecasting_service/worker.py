from __future__ import annotations

import argparse
import atexit
import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from functools import lru_cache
from typing import Any, Literal, cast

from hatchet_sdk import Context, Hatchet
from hatchet_sdk.runnables.workflow import BaseWorkflow, Standalone
from hatchet_sdk.types.concurrency import ConcurrencyExpression, ConcurrencyLimitStrategy
from hatchet_sdk.types.idempotency import StatusBasedIdempotencyConfig
from pydantic import BaseModel, ConfigDict

from forecasting_service.config import Settings, get_settings

JobKind = Literal["experiment", "forecast"]
WorkerKind = Literal["cpu", "gpu", "all"]


class JobInput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tenant_id: str
    resource_id: str
    job_id: str


Task = Standalone[JobInput, dict[str, str]]


@dataclass(frozen=True)
class WorkerRegistry:
    hatchet: Hatchet
    experiment_cpu: Task
    experiment_gpu: Task
    forecast_cpu: Task
    forecast_gpu: Task

    def task(self, *, kind: str, requires_gpu: bool) -> Task:
        if kind == "experiment":
            return self.experiment_gpu if requires_gpu else self.experiment_cpu
        if kind == "forecast":
            return self.forecast_gpu if requires_gpu else self.forecast_cpu
        raise ValueError(f"unsupported job kind: {kind}")

    def tasks_for(self, worker_kind: WorkerKind) -> list[Task]:
        if worker_kind == "cpu":
            return [self.experiment_cpu, self.forecast_cpu]
        if worker_kind == "gpu":
            return [self.experiment_gpu, self.forecast_gpu]
        return [
            self.experiment_cpu,
            self.forecast_cpu,
            self.experiment_gpu,
            self.forecast_gpu,
        ]


def _task_options(settings: Settings) -> dict[str, Any]:
    return {
        "input_validator": JobInput,
        "execution_timeout": timedelta(seconds=settings.job_execution_timeout_seconds),
        "schedule_timeout": timedelta(seconds=settings.job_schedule_timeout_seconds),
        "retries": settings.job_retries,
        "backoff_factor": 2.0,
        "backoff_max_seconds": 300,
        "concurrency": ConcurrencyExpression(
            expression="input.tenant_id",
            max_runs=settings.tenant_max_concurrent_jobs,
            limit_strategy=ConcurrencyLimitStrategy.GROUP_ROUND_ROBIN,
            name="forecasting-tenant-jobs",
            is_tenant_scoped=True,
        ),
        "idempotency": StatusBasedIdempotencyConfig(
            key_expression="input.job_id", fallback_ttl=timedelta(days=7)
        ),
    }


@lru_cache
def get_registry() -> WorkerRegistry:
    settings = get_settings()
    hatchet = Hatchet()
    options = _task_options(settings)

    @hatchet.task(name="forecast-train-cpu", **options)
    def experiment_cpu(job: JobInput, context: Context) -> dict[str, str]:
        return _execute("experiment", job, context)

    @hatchet.task(name="forecast-train-gpu", **options)
    def experiment_gpu(job: JobInput, context: Context) -> dict[str, str]:
        return _execute("experiment", job, context)

    @hatchet.task(name="forecast-predict-cpu", **options)
    def forecast_cpu(job: JobInput, context: Context) -> dict[str, str]:
        return _execute("forecast", job, context)

    @hatchet.task(name="forecast-predict-gpu", **options)
    def forecast_gpu(job: JobInput, context: Context) -> dict[str, str]:
        return _execute("forecast", job, context)

    return WorkerRegistry(
        hatchet=hatchet,
        experiment_cpu=experiment_cpu,
        experiment_gpu=experiment_gpu,
        forecast_cpu=forecast_cpu,
        forecast_gpu=forecast_gpu,
    )


@lru_cache
def _worker_runtime():
    from forecasting_service.runtime import Runtime

    service = Runtime(get_settings(), dispatch_jobs=False)
    atexit.register(service.close)
    return service


def _execute(kind: JobKind, job: JobInput, context: Context) -> dict[str, str]:
    service = _worker_runtime()

    def cancelled() -> bool:
        return context.is_cancelled

    service.store.update("jobs", job.job_id, retry_count=context.retry_count)
    try:
        if kind == "experiment":
            service.run_experiment(job.resource_id, job.job_id, cancelled)
        else:
            service.run_forecast(job.resource_id, job.job_id, cancelled)
    except Exception:
        settings = get_settings()
        collection = "experiments" if kind == "experiment" else "forecasts"
        if context.is_cancelled or service.store.get("jobs", job.job_id)["state"] == "cancelled":
            service.store.update(collection, job.resource_id, state="cancelled")
            service.store.update(
                "jobs", job.job_id, state="cancelled", stage="cancelled", progress=100
            )
            return {"job_id": job.job_id, "resource_id": job.resource_id}
        if context.retry_count < settings.job_retries:
            service.store.update(collection, job.resource_id, state="queued")
            service.store.update(
                "jobs",
                job.job_id,
                state="queued",
                stage="retry_scheduled",
                progress=0,
            )
        raise
    return {"job_id": job.job_id, "resource_id": job.resource_id}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run durable forecasting workers")
    parser.add_argument("--kind", choices=("cpu", "gpu", "all"), default="cpu")
    parser.add_argument("--name", help="Hatchet worker name; defaults to host, process, and kind")
    parser.add_argument("--slots", type=int, help="Override configured worker slots")
    return parser


def main() -> None:
    args = _parser().parse_args()
    settings = get_settings()
    kind: WorkerKind = args.kind
    configured_slots = {
        "cpu": settings.cpu_worker_slots,
        "gpu": settings.gpu_worker_slots,
        "all": settings.cpu_worker_slots + settings.gpu_worker_slots,
    }
    slots = args.slots or configured_slots[kind]
    if slots < 1:
        raise SystemExit("--slots must be at least 1")
    name = args.name or f"forecast-{kind}-{socket.gethostname()}-{os.getpid()}"
    registry = get_registry()
    registry.hatchet.worker(
        name,
        slots=slots,
        workflows=cast(list[BaseWorkflow[Any]], registry.tasks_for(kind)),
    ).start()


if __name__ == "__main__":
    main()
