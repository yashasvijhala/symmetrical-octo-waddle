import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl

from forecasting_service.config import Settings
from forecasting_service.modeling import (
    forecast_autogluon,
    forecast_lightgbm,
    train_autogluon,
    train_lightgbm,
)
from forecasting_service.schemas import DatasetManifest, ExperimentCreate, ForecastCreate
from forecasting_service.store import Store


class Runtime:
    def __init__(self, settings: Settings) -> None:
        self.store = Store(
            settings.state_dir,
            settings.database_url,
            settings.max_upload_bytes,
            settings.database_pool_min_size,
            settings.database_pool_max_size,
        )
        self.executor = ThreadPoolExecutor(
            max_workers=settings.max_workers, thread_name_prefix="forecast"
        )
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()
        self._recover_interrupted_jobs()

    def submit_experiment(self, experiment_id: str, job_id: str) -> None:
        self.executor.submit(self._run_experiment, experiment_id, job_id)

    def submit_forecast(self, forecast_id: str, job_id: str) -> None:
        self.executor.submit(self._run_forecast, forecast_id, job_id)

    def cancel(self, job_id: str) -> None:
        with self._lock:
            self._cancelled.add(job_id)
        self.store.update("jobs", job_id, state="cancelled", stage="cancelled")

    def _check_cancelled(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._cancelled:
                raise CancelledError

    def _run_experiment(self, experiment_id: str, job_id: str) -> None:
        try:
            self._stage(job_id, "validating", 5)
            experiment = self.store.get("experiments", experiment_id)
            dataset = self.store.get("datasets", experiment["dataset_id"])
            version = dataset["versions"][str(experiment["dataset_version"])]
            manifest = DatasetManifest.model_validate(version["manifest"])
            frame = pl.read_parquet(version["normalized_path"])
            self._check_cancelled(job_id)
            self._stage(job_id, "backtesting", 20)
            config = ExperimentCreate.model_validate(experiment["config"])
            model_id = self.store.new_id("mdl")
            if config.model_policy in {"autogluon", "high_accuracy"}:
                artifact = self.store.path("artifacts", model_id)
                result = train_autogluon(frame, manifest, config, artifact)
            else:
                artifact = self.store.path("artifacts", f"{model_id}.joblib")
                result = train_lightgbm(frame, manifest, config, artifact)
            self._check_cancelled(job_id)
            self._stage(job_id, "packaging", 90)
            model = self.store.create(
                "models",
                {
                    "id": model_id,
                    "tenant_id": experiment["tenant_id"],
                    "experiment_id": experiment_id,
                    "dataset_id": experiment["dataset_id"],
                    "dataset_version": experiment["dataset_version"],
                    "state": "ready",
                    "stage": "candidate",
                    **result,
                },
            )
            self.store.update(
                "experiments",
                experiment_id,
                state="succeeded",
                model_id=model_id,
                metrics=result.get("metrics", {}),
                folds=result.get("folds", []),
                leaderboard=result.get("leaderboard", []),
            )
            self.store.update(
                "jobs",
                job_id,
                state="succeeded",
                stage="succeeded",
                progress=100,
                result_id=model["id"],
            )
        except CancelledError:
            self.store.update("experiments", experiment_id, state="cancelled")
        except Exception as exc:  # persisted for API clients; worker must not disappear silently
            self.store.update("experiments", experiment_id, state="failed", error=str(exc))
            self.store.update(
                "jobs", job_id, state="failed", stage="failed", error=str(exc), progress=100
            )

    def _run_forecast(self, forecast_id: str, job_id: str) -> None:
        try:
            self._stage(job_id, "validating", 10)
            request_record = self.store.get("forecasts", forecast_id)
            request = ForecastCreate.model_validate(request_record["request"])
            model = self.store.get("models", request.model_id)
            dataset_id = request.dataset_id or model["dataset_id"]
            dataset = self.store.get("datasets", dataset_id)
            version = request.dataset_version if request.dataset_id else model["dataset_version"]
            dataset_version = dataset["versions"][str(version)]
            frame = pl.read_parquet(dataset_version["normalized_path"])
            self._stage(job_id, "predicting", 40)
            if model["engine"] == "autogluon":
                rows = forecast_autogluon(
                    Path(model["artifact_path"]), frame, request.future_covariates
                )
            else:
                rows = forecast_lightgbm(
                    Path(model["artifact_path"]),
                    frame,
                    request.horizon,
                    request.future_covariates,
                    request.new_items,
                )
            output = self.store.path("predictions", f"{forecast_id}.json")
            output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            summary = {"rows": len(rows), "items": len({row["item_id"] for row in rows})}
            self.store.update(
                "forecasts",
                forecast_id,
                state="succeeded",
                output_path=str(output),
                summary=summary,
            )
            self.store.update(
                "jobs",
                job_id,
                state="succeeded",
                stage="succeeded",
                progress=100,
                result_id=forecast_id,
            )
        except Exception as exc:
            self.store.update("forecasts", forecast_id, state="failed", error=str(exc))
            self.store.update(
                "jobs", job_id, state="failed", stage="failed", error=str(exc), progress=100
            )

    def _stage(self, job_id: str, stage: str, progress: int) -> None:
        self._check_cancelled(job_id)
        self.store.update("jobs", job_id, state="running", stage=stage, progress=progress)

    def monitoring(self, tenant_id: str, model_id: str) -> dict[str, Any]:
        self.store.owned("models", model_id, tenant_id)
        forecasts = self.store.list("forecasts", tenant_id, model_id=model_id, state="succeeded")
        actual_records = self.store.list("actuals", tenant_id, model_id=model_id)
        predicted: dict[tuple[str, str], float] = {}
        for forecast in forecasts:
            for row in json.loads(Path(forecast["output_path"]).read_text(encoding="utf-8")):
                predicted[(row["item_id"], row["timestamp"])] = float(
                    row.get("mean", row.get("0.5", 0))
                )
        pairs = []
        for record in actual_records:
            for point in record["points"]:
                key = (point["item_id"], point["timestamp"])
                if key in predicted:
                    pairs.append((float(point["value"]), predicted[key]))
        if not pairs:
            return {"matched_points": 0, "status": "awaiting_actuals"}
        errors = [forecast - actual for actual, forecast in pairs]
        absolute = [abs(value) for value in errors]
        denominator = sum(abs(actual) for actual, _ in pairs)
        return {
            "matched_points": len(pairs),
            "mae": sum(absolute) / len(absolute),
            "wape": sum(absolute) / denominator if denominator else sum(absolute) / len(absolute),
            "bias": sum(errors) / denominator if denominator else sum(errors) / len(errors),
        }

    def _recover_interrupted_jobs(self) -> None:
        interrupted = [
            *self.store.list("jobs", state="queued"),
            *self.store.list("jobs", state="running"),
        ]
        for job in interrupted:
            resource_id = job.get("resource_id", "")
            self.store.update(
                "jobs",
                job["id"],
                state="queued",
                stage="recovered_after_restart",
                progress=0,
                retry_count=int(job.get("retry_count", 0)) + 1,
            )
            if resource_id.startswith("exp_"):
                self.executor.submit(self._run_experiment, resource_id, job["id"])
            elif resource_id.startswith("fc_"):
                self.executor.submit(self._run_forecast, resource_id, job["id"])


class CancelledError(Exception):
    pass
