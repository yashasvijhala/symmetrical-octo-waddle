import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import polars as pl

from forecasting_service.config import Settings
from forecasting_service.dispatch import JobDispatcher, create_dispatcher
from forecasting_service.modeling import (
    forecast_autogluon,
    forecast_lightgbm,
    train_autogluon,
    train_lightgbm,
)
from forecasting_service.schemas import DatasetManifest, ExperimentCreate, ForecastCreate
from forecasting_service.store import NotFoundError, Store


class Runtime:
    def __init__(self, settings: Settings, *, dispatch_jobs: bool = True) -> None:
        self.store = Store(
            settings.state_dir,
            settings.database_url,
            settings.max_upload_bytes,
            settings.database_pool_min_size,
            settings.database_pool_max_size,
        )
        self.dispatcher: JobDispatcher | None = (
            create_dispatcher(settings, self.run_experiment, self.run_forecast)
            if dispatch_jobs
            else None
        )

    def submit_experiment(self, experiment_id: str, job_id: str) -> None:
        experiment = self.store.get("experiments", experiment_id)
        config = ExperimentCreate.model_validate(experiment["config"])
        self._submit(
            kind="experiment",
            tenant_id=experiment["tenant_id"],
            resource_id=experiment_id,
            job_id=job_id,
            requires_gpu=config.model_policy in {"autogluon", "high_accuracy"},
        )

    def submit_forecast(self, forecast_id: str, job_id: str) -> None:
        forecast = self.store.get("forecasts", forecast_id)
        model = self.store.get("models", forecast["model_id"])
        self._submit(
            kind="forecast",
            tenant_id=forecast["tenant_id"],
            resource_id=forecast_id,
            job_id=job_id,
            requires_gpu=model["engine"] == "autogluon",
        )

    def _submit(
        self,
        *,
        kind: str,
        tenant_id: str,
        resource_id: str,
        job_id: str,
        requires_gpu: bool,
    ) -> None:
        if self.dispatcher is None:
            raise RuntimeError("job dispatch is disabled in this process")
        try:
            run_id = self.dispatcher.submit(
                kind=kind,
                tenant_id=tenant_id,
                resource_id=resource_id,
                job_id=job_id,
                requires_gpu=requires_gpu,
            )
        except Exception as exc:
            message = f"job dispatch failed: {exc}"
            collection = "experiments" if kind == "experiment" else "forecasts"
            self.store.update(collection, resource_id, state="failed", error=message)
            self.store.update(
                "jobs", job_id, state="failed", stage="dispatch_failed", error=message
            )
            raise DispatchError(message) from exc
        self.store.update(
            "jobs",
            job_id,
            orchestrator="test" if run_id.startswith("test:") else "hatchet",
            orchestrator_run_id=run_id,
            queue="gpu" if requires_gpu else "cpu",
        )

    def cancel(self, job_id: str) -> None:
        job = self.store.get("jobs", job_id)
        run_id = job.get("orchestrator_run_id")
        if run_id and self.dispatcher is not None and job.get("orchestrator") == "hatchet":
            self.dispatcher.cancel(run_id)
        self.store.update("jobs", job_id, state="cancelled", stage="cancelled", progress=100)

    def _check_cancelled(self, job_id: str, cancelled: Callable[[], bool] | None = None) -> None:
        if (cancelled is not None and cancelled()) or self.store.get("jobs", job_id)[
            "state"
        ] == "cancelled":
            raise CancelledError

    def run_experiment(
        self,
        experiment_id: str,
        job_id: str,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        try:
            experiment = self.store.get("experiments", experiment_id)
            if experiment["state"] == "succeeded":
                return
            self._stage(job_id, "validating", 5, cancelled)
            dataset = self.store.get("datasets", experiment["dataset_id"])
            version = dataset["versions"][str(experiment["dataset_version"])]
            manifest = DatasetManifest.model_validate(version["manifest"])
            frame = pl.read_parquet(version["normalized_path"])
            self._check_cancelled(job_id, cancelled)
            self._stage(job_id, "backtesting", 20, cancelled)
            config = ExperimentCreate.model_validate(experiment["config"])
            model_id = experiment.get("model_id") or self.store.new_id("mdl")
            if "model_id" not in experiment:
                self.store.update("experiments", experiment_id, model_id=model_id)
            if config.model_policy in {"autogluon", "high_accuracy"}:
                artifact = self.store.path("artifacts", model_id)
                result = train_autogluon(frame, manifest, config, artifact)
            else:
                artifact = self.store.path("artifacts", f"{model_id}.joblib")
                result = train_lightgbm(frame, manifest, config, artifact)
            self._check_cancelled(job_id, cancelled)
            self._stage(job_id, "packaging", 90, cancelled)
            try:
                model = self.store.get("models", model_id)
            except NotFoundError:
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
                error=None,
            )
            self.store.update(
                "jobs",
                job_id,
                state="succeeded",
                stage="succeeded",
                progress=100,
                result_id=model["id"],
                error=None,
            )
        except CancelledError:
            self.store.update("experiments", experiment_id, state="cancelled")
        except Exception as exc:  # persisted for API clients; worker must not disappear silently
            self.store.update("experiments", experiment_id, state="failed", error=str(exc))
            self.store.update(
                "jobs", job_id, state="failed", stage="failed", error=str(exc), progress=100
            )
            raise

    def run_forecast(
        self,
        forecast_id: str,
        job_id: str,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        try:
            request_record = self.store.get("forecasts", forecast_id)
            if request_record["state"] == "succeeded":
                return
            self._stage(job_id, "validating", 10, cancelled)
            request = ForecastCreate.model_validate(request_record["request"])
            model = self.store.get("models", request.model_id)
            dataset_id = request.dataset_id or model["dataset_id"]
            dataset = self.store.get("datasets", dataset_id)
            version = request.dataset_version if request.dataset_id else model["dataset_version"]
            dataset_version = dataset["versions"][str(version)]
            frame = pl.read_parquet(dataset_version["normalized_path"])
            self._stage(job_id, "predicting", 40, cancelled)
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
            self._check_cancelled(job_id, cancelled)
            output = self.store.path("predictions", f"{forecast_id}.json")
            output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            summary = {"rows": len(rows), "items": len({row["item_id"] for row in rows})}
            self.store.update(
                "forecasts",
                forecast_id,
                state="succeeded",
                output_path=str(output),
                summary=summary,
                error=None,
            )
            self.store.update(
                "jobs",
                job_id,
                state="succeeded",
                stage="succeeded",
                progress=100,
                result_id=forecast_id,
                error=None,
            )
        except CancelledError:
            self.store.update("forecasts", forecast_id, state="cancelled")
        except Exception as exc:
            self.store.update("forecasts", forecast_id, state="failed", error=str(exc))
            self.store.update(
                "jobs", job_id, state="failed", stage="failed", error=str(exc), progress=100
            )
            raise

    def _stage(
        self,
        job_id: str,
        stage: str,
        progress: int,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self._check_cancelled(job_id, cancelled)
        self.store.update(
            "jobs", job_id, state="running", stage=stage, progress=progress, error=None
        )

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

    def close(self) -> None:
        if self.dispatcher is not None:
            self.dispatcher.close()
        self.store.close()


class CancelledError(Exception):
    pass


class DispatchError(RuntimeError):
    pass
