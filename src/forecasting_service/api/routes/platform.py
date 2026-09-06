import hashlib
import hmac
import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, RedirectResponse

from forecasting_service.data import canonicalize, infer_frequency, profile_file
from forecasting_service.object_store import ObjectRef
from forecasting_service.runtime import Runtime
from forecasting_service.schemas import (
    ActualsCreate,
    DatasetCreate,
    DatasetManifest,
    ExperimentCreate,
    ForecastCreate,
    IdResponse,
    PromotionRequest,
)
from forecasting_service.store import IdempotencyConflictError, NotFoundError

router = APIRouter()
IdempotencyKey = Annotated[
    str | None, Header(alias="Idempotency-Key", min_length=1, max_length=200)
]


def runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def tenant_identity(
    request: Request,
    x_tenant_id: Annotated[
        str | None, Header(alias="X-Tenant-ID", min_length=1, max_length=100)
    ] = None,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key", max_length=500)] = None,
) -> str:
    settings = request.app.state.settings
    if settings.api_keys:
        if not x_tenant_id or not x_api_key:
            raise HTTPException(status_code=401, detail="X-Tenant-ID and X-API-Key are required")
        expected = settings.api_keys.get(x_tenant_id)
        if expected is None or not hmac.compare_digest(expected, x_api_key):
            raise HTTPException(status_code=401, detail="invalid API credentials")
        return x_tenant_id
    if settings.environment == "production":
        raise HTTPException(status_code=503, detail="production API keys are not configured")
    return x_tenant_id or "local"


Tenant = Annotated[str, Depends(tenant_identity)]


def fingerprint(body: Any) -> str:
    canonical = json.dumps(
        body.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def create_idempotent(
    request: Request,
    records: list[tuple[str, dict[str, Any]]],
    tenant_id: str,
    scope: str,
    key: str | None,
    body: Any,
) -> tuple[dict[str, Any], bool]:
    try:
        return runtime(request).store.create_idempotent(
            records, tenant_id, scope, key, fingerprint(body)
        )
    except IdempotencyConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def owned(request: Request, collection: str, resource_id: str, tenant_id: str) -> dict[str, Any]:
    try:
        return runtime(request).store.owned(collection, resource_id, tenant_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def dispatch_required(
    request: Request, collection: str, resource: dict[str, Any], created: bool
) -> bool:
    """Treat the PostgreSQL job record as an outbox for safe idempotent redispatch."""

    store = runtime(request).store
    job = store.get("jobs", resource["job_id"])
    if not created and not job.get("orchestrator_run_id"):
        store.update(collection, resource["id"], state="queued", error=None)
        store.update(
            "jobs",
            job["id"],
            state="queued",
            stage="redispatching",
            progress=0,
            error=None,
        )
    return created or not job.get("orchestrator_run_id")


@router.post("/datasets", response_model=IdResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(
    body: DatasetCreate,
    request: Request,
    tenant_id: Tenant,
    idempotency_key: IdempotencyKey = None,
) -> IdResponse:
    store = runtime(request).store
    dataset_id = store.new_id("ds")
    dataset, _ = create_idempotent(
        request,
        [
            (
                "datasets",
                {
                    "id": dataset_id,
                    "tenant_id": tenant_id,
                    "name": body.name,
                    "description": body.description,
                    "state": "draft",
                    "versions": {},
                },
            )
        ],
        tenant_id,
        "create_dataset",
        idempotency_key,
        body,
    )
    return IdResponse(id=dataset["id"], state=dataset["state"])


@router.get("/datasets")
def list_datasets(request: Request, tenant_id: Tenant) -> list[dict[str, Any]]:
    return runtime(request).store.list("datasets", tenant_id)


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, request: Request, tenant_id: Tenant) -> dict[str, Any]:
    return owned(request, "datasets", dataset_id, tenant_id)


@router.post("/datasets/{dataset_id}/upload-url")
def upload_intent(dataset_id: str, request: Request, tenant_id: Tenant) -> dict[str, str]:
    owned(request, "datasets", dataset_id, tenant_id)
    return {
        "method": "POST",
        "url": f"/v1/datasets/{dataset_id}/upload",
        "content_type": "multipart/form-data",
        "note": (
            "Use this multipart API endpoint. Direct R2 presigned PUT support will be exposed "
            "through the same resource contract."
        ),
    }


@router.post("/datasets/{dataset_id}/upload")
def upload_dataset(
    dataset_id: str,
    request: Request,
    tenant_id: Tenant,
    file: Annotated[UploadFile, File()],
) -> dict[str, Any]:
    owned(request, "datasets", dataset_id, tenant_id)
    if not file.filename:
        raise HTTPException(status_code=422, detail="uploaded file must have a filename")
    try:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".csv", ".parquet"}:
            raise ValueError("only CSV and Parquet uploads are supported")
        object_ref, _ = runtime(request).objects.put_stream(
            runtime(request).objects.key(
                "tenants", tenant_id, "datasets", dataset_id, "source" + suffix
            ),
            file.file,
            file.content_type or "application/octet-stream",
            request.app.state.settings.max_upload_bytes,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    runtime(request).store.update(
        "datasets", dataset_id, upload_object=object_ref.as_dict(), state="uploaded"
    )
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "sha256": object_ref.sha256,
        "state": "uploaded",
    }


@router.post("/datasets/{dataset_id}/profile", status_code=status.HTTP_201_CREATED)
def profile_dataset(dataset_id: str, request: Request, tenant_id: Tenant) -> dict[str, Any]:
    dataset = owned(request, "datasets", dataset_id, tenant_id)
    if not dataset.get("upload_object"):
        raise HTTPException(status_code=409, detail="upload data before profiling")
    try:
        source = runtime(request).objects.materialize(ObjectRef.from_dict(dataset["upload_object"]))
        profile = profile_file(source)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not profile dataset: {exc}") from exc
    runtime(request).store.update("datasets", dataset_id, profile=profile, state="profiled")
    return profile


@router.get("/datasets/{dataset_id}/profile")
def get_profile(dataset_id: str, request: Request, tenant_id: Tenant) -> dict[str, Any]:
    dataset = owned(request, "datasets", dataset_id, tenant_id)
    if "profile" not in dataset:
        raise HTTPException(status_code=409, detail="dataset has not been profiled")
    return dataset["profile"]


@router.post("/datasets/{dataset_id}/finalize", status_code=status.HTTP_201_CREATED)
def finalize_dataset(
    dataset_id: str, body: DatasetManifest, request: Request, tenant_id: Tenant
) -> dict[str, Any]:
    dataset = owned(request, "datasets", dataset_id, tenant_id)
    if not dataset.get("upload_object"):
        raise HTTPException(status_code=409, detail="upload data before finalizing")
    try:
        source = runtime(request).objects.materialize(ObjectRef.from_dict(dataset["upload_object"]))
        frame = canonicalize(source, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    frequency, season = infer_frequency(frame)
    versions = dataset.get("versions", {})
    version = max((int(value) for value in versions), default=0) + 1
    with runtime(request).objects.temporary_path(".parquet") as normalized:
        frame.write_parquet(normalized, compression="zstd", statistics=True)
        data_ref = runtime(request).objects.put_file(
            runtime(request).objects.key(
                "tenants",
                tenant_id,
                "datasets",
                dataset_id,
                "versions",
                str(version),
                "canonical.parquet",
            ),
            normalized,
            "application/vnd.apache.parquet",
        )
    versions[str(version)] = {
        "version": version,
        "manifest": body.model_dump(mode="json"),
        "data_object": data_ref.as_dict(),
        "rows": frame.height,
        "items": frame.get_column("item_id").n_unique(),
        "inferred_frequency": frequency,
        "suggested_seasonal_period": season,
        "content_sha256": dataset["upload_object"]["sha256"],
    }
    runtime(request).store.update("datasets", dataset_id, versions=versions, state="ready")
    return versions[str(version)]


@router.get("/datasets/{dataset_id}/versions/{version}")
def get_dataset_version(
    dataset_id: str, version: int, request: Request, tenant_id: Tenant
) -> dict[str, Any]:
    dataset = owned(request, "datasets", dataset_id, tenant_id)
    try:
        return dataset["versions"][str(version)]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="dataset version not found") from exc


@router.post("/experiments", response_model=IdResponse, status_code=status.HTTP_202_ACCEPTED)
def create_experiment(
    body: ExperimentCreate,
    request: Request,
    tenant_id: Tenant,
    idempotency_key: IdempotencyKey = None,
) -> IdResponse:
    store = runtime(request).store
    dataset = owned(request, "datasets", body.dataset_id, tenant_id)
    if str(body.dataset_version) not in dataset.get("versions", {}):
        raise HTTPException(status_code=409, detail="finalize the requested dataset version first")
    experiment_id = store.new_id("exp")
    job_id = store.new_id("job")
    experiment, created = create_idempotent(
        request,
        [
            (
                "experiments",
                {
                    "id": experiment_id,
                    "tenant_id": tenant_id,
                    "dataset_id": body.dataset_id,
                    "dataset_version": body.dataset_version,
                    "state": "queued",
                    "job_id": job_id,
                    "config": body.model_dump(mode="json"),
                },
            ),
            (
                "jobs",
                {
                    "id": job_id,
                    "tenant_id": tenant_id,
                    "resource_id": experiment_id,
                    "state": "queued",
                    "stage": "queued",
                    "progress": 0,
                },
            ),
        ],
        tenant_id,
        "create_experiment",
        idempotency_key,
        body,
    )
    if dispatch_required(request, "experiments", experiment, created):
        runtime(request).submit_experiment(experiment["id"], experiment["job_id"])
    experiment = store.get("experiments", experiment["id"])
    return IdResponse(id=experiment["id"], state=experiment["state"])


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str, request: Request, tenant_id: Tenant) -> dict[str, Any]:
    return owned(request, "experiments", experiment_id, tenant_id)


@router.get("/experiments/{experiment_id}/leaderboard")
def experiment_leaderboard(
    experiment_id: str, request: Request, tenant_id: Tenant
) -> dict[str, Any]:
    experiment = owned(request, "experiments", experiment_id, tenant_id)
    return {
        "metrics": experiment.get("metrics", {}),
        "folds": experiment.get("folds", []),
        "models": experiment.get("leaderboard", []),
    }


@router.get("/experiments/{experiment_id}/report")
def experiment_report(experiment_id: str, request: Request, tenant_id: Tenant) -> dict[str, Any]:
    return owned(request, "experiments", experiment_id, tenant_id)


@router.post("/experiments/{experiment_id}/cancel")
def cancel_experiment(experiment_id: str, request: Request, tenant_id: Tenant) -> dict[str, str]:
    experiment = owned(request, "experiments", experiment_id, tenant_id)
    runtime(request).cancel(experiment["job_id"])
    runtime(request).store.update("experiments", experiment_id, state="cancelled")
    return {"id": experiment_id, "state": "cancelled"}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request, tenant_id: Tenant) -> dict[str, Any]:
    return owned(request, "jobs", job_id, tenant_id)


@router.get("/models")
def list_models(request: Request, tenant_id: Tenant) -> list[dict[str, Any]]:
    return runtime(request).store.list("models", tenant_id)


@router.get("/models/{model_id}")
def get_model(model_id: str, request: Request, tenant_id: Tenant) -> dict[str, Any]:
    return owned(request, "models", model_id, tenant_id)


@router.post("/models/{model_id}/promote")
def promote_model(
    model_id: str, body: PromotionRequest, request: Request, tenant_id: Tenant
) -> dict[str, Any]:
    model = owned(request, "models", model_id, tenant_id)
    if not model.get("baseline_beaten") and not body.force:
        raise HTTPException(
            status_code=409,
            detail="model did not beat its baseline; force with justification to promote",
        )
    champions = runtime(request).store.list(
        "models", tenant_id, dataset_id=model["dataset_id"], stage="champion"
    )
    for other in champions:
        runtime(request).store.update("models", other["id"], stage="candidate")
    return runtime(request).store.update(
        "models", model_id, stage="champion", promotion_justification=body.justification
    )


@router.post("/models/{model_id}/retire")
def retire_model(model_id: str, request: Request, tenant_id: Tenant) -> dict[str, Any]:
    owned(request, "models", model_id, tenant_id)
    return runtime(request).store.update("models", model_id, state="retired", stage="retired")


@router.post("/forecasts", response_model=IdResponse, status_code=status.HTTP_202_ACCEPTED)
def create_forecast(
    body: ForecastCreate,
    request: Request,
    tenant_id: Tenant,
    idempotency_key: IdempotencyKey = None,
) -> IdResponse:
    store = runtime(request).store
    model = owned(request, "models", body.model_id, tenant_id)
    if model["state"] != "ready":
        raise HTTPException(status_code=409, detail="model is not available for prediction")
    forecast_id = store.new_id("fc")
    job_id = store.new_id("job")
    forecast, created = create_idempotent(
        request,
        [
            (
                "forecasts",
                {
                    "id": forecast_id,
                    "tenant_id": tenant_id,
                    "model_id": body.model_id,
                    "state": "queued",
                    "job_id": job_id,
                    "request": body.model_dump(mode="json"),
                },
            ),
            (
                "jobs",
                {
                    "id": job_id,
                    "tenant_id": tenant_id,
                    "resource_id": forecast_id,
                    "state": "queued",
                    "stage": "queued",
                    "progress": 0,
                },
            ),
        ],
        tenant_id,
        "create_forecast",
        idempotency_key,
        body,
    )
    if dispatch_required(request, "forecasts", forecast, created):
        runtime(request).submit_forecast(forecast["id"], forecast["job_id"])
    forecast = store.get("forecasts", forecast["id"])
    return IdResponse(id=forecast["id"], state=forecast["state"])


@router.get("/forecasts/{forecast_id}")
def get_forecast(forecast_id: str, request: Request, tenant_id: Tenant) -> dict[str, Any]:
    return owned(request, "forecasts", forecast_id, tenant_id)


@router.get("/forecasts/{forecast_id}/download")
def download_forecast(forecast_id: str, request: Request, tenant_id: Tenant):
    forecast = owned(request, "forecasts", forecast_id, tenant_id)
    if forecast.get("state") != "succeeded":
        raise HTTPException(status_code=409, detail="forecast is not complete")
    ref = ObjectRef.from_dict(forecast["prediction_object"])
    filename = f"{forecast_id}.json"
    if url := runtime(request).objects.download_url(ref, filename):
        return RedirectResponse(url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)
    return FileResponse(
        runtime(request).objects.materialize(ref),
        media_type=ref.content_type,
        filename=filename,
    )


@router.post("/actuals", response_model=IdResponse, status_code=status.HTTP_201_CREATED)
def create_actuals(body: ActualsCreate, request: Request, tenant_id: Tenant) -> IdResponse:
    owned(request, "models", body.model_id, tenant_id)
    store = runtime(request).store
    actuals_id = store.new_id("act")
    store.create(
        "actuals",
        {
            "id": actuals_id,
            "tenant_id": tenant_id,
            "model_id": body.model_id,
            "state": "ready",
            "points": [point.model_dump() for point in body.points],
        },
    )
    return IdResponse(id=actuals_id, state="ready")


@router.get("/models/{model_id}/monitoring")
def model_monitoring(model_id: str, request: Request, tenant_id: Tenant) -> dict[str, Any]:
    return runtime(request).monitoring(tenant_id, model_id)
