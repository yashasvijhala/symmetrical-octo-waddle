from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from forecasting_service import worker as worker_module


@dataclass
class FakeTask:
    name: str
    options: dict[str, Any]
    function: Any


@dataclass
class FakeHatchet:
    tasks: list[FakeTask] = field(default_factory=list)

    def task(self, *, name: str, **options: Any):
        def decorate(function: Any) -> FakeTask:
            task = FakeTask(name, options, function)
            self.tasks.append(task)
            return task

        return decorate


def test_worker_registry_routes_resources_and_applies_durability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_module.get_registry.cache_clear()
    monkeypatch.setattr(worker_module, "Hatchet", FakeHatchet)

    registry = worker_module.get_registry()

    assert [task.name for task in registry.tasks_for("cpu")] == [
        "forecast-train-cpu",
        "forecast-predict-cpu",
    ]
    assert [task.name for task in registry.tasks_for("gpu")] == [
        "forecast-train-gpu",
        "forecast-predict-gpu",
    ]
    for task in registry.tasks_for("all"):
        fake_task = cast(Any, task)
        assert fake_task.options["retries"] == worker_module.JOB_RETRIES
        assert fake_task.options["concurrency"].max_runs == worker_module.TENANT_MAX_CONCURRENT_JOBS
        assert fake_task.options["concurrency"].is_tenant_scoped is True
        assert fake_task.options["idempotency"].key_expression == "input.job_id"

    with pytest.raises(ValueError, match="unsupported job kind"):
        registry.task(kind="profile", requires_gpu=False)

    worker_module.get_registry.cache_clear()
