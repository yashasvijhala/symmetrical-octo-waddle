import hashlib
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from forecasting_service.data import canonicalize, infer_frequency, profile_file
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
from forecasting_service.store import NotFoundError

router = APIRouter()
Tenant = Annotated[str, Header(alias="X-Tenant-ID", min_length=1, max_length=100)]


def runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def owned(request: Request, collection: str, resource_id: str, tenant_id: str) -> dict[str, Any]:
    try:
        return runtime(request).store.owned(collection, resource_id, tenant_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/datasets", response_model=IdResponse, status_code=status.HTTP_201_CREATED)
def create_dataset(body: DatasetCreate, request: Request, tenant_id: Tenant) -> IdResponse:
    store = runtime(request).store
    dataset_id = store.new_id("ds")
    store.create(
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
    return IdResponse(id=dataset_id, state="draft")


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
            "Local adapter. Replace with an S3 presigned PUT adapter in distributed deployments."
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
        path = runtime(request).store.save_upload(dataset_id, file.filename, file.file)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    digest_builder = hashlib.sha256()
    with path.open("rb") as uploaded:
        while chunk := uploaded.read(1024 * 1024):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    runtime(request).store.update(
        "datasets", dataset_id, upload_path=str(path), upload_sha256=digest, state="uploaded"
    )
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "sha256": digest,
        "state": "uploaded",
    }


@router.post("/datasets/{dataset_id}/profile", status_code=status.HTTP_201_CREATED)
def profile_dataset(dataset_id: str, request: Request, tenant_id: Tenant) -> dict[str, Any]:
    dataset = owned(request, "datasets", dataset_id, tenant_id)
    if not dataset.get("upload_path"):
        raise HTTPException(status_code=409, detail="upload data before profiling")
    try:
        profile = profile_file(Path(dataset["upload_path"]))
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
    if not dataset.get("upload_path"):
        raise HTTPException(status_code=409, detail="upload data before finalizing")
    try:
        frame = canonicalize(Path(dataset["upload_path"]), body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    frequency, season = infer_frequency(frame)
    versions = dataset.get("versions", {})
    version = max((int(value) for value in versions), default=0) + 1
    normalized = runtime(request).store.path("artifacts", f"{dataset_id}/v{version}/data.parquet")
    frame.write_parquet(normalized, compression="zstd", statistics=True)
    versions[str(version)] = {
        "version": version,
        "manifest": body.model_dump(mode="json"),
        "normalized_path": str(normalized),
        "rows": frame.height,
        "items": frame.get_column("item_id").n_unique(),
        "inferred_frequency": frequency,
        "suggested_seasonal_period": season,
        "content_sha256": dataset["upload_sha256"],
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
def create_experiment(body: ExperimentCreate, request: Request, tenant_id: Tenant) -> IdResponse:
    store = runtime(request).store
    dataset = owned(request, "datasets", body.dataset_id, tenant_id)
    if str(body.dataset_version) not in dataset.get("versions", {}):
        raise HTTPException(status_code=409, detail="finalize the requested dataset version first")
    experiment_id = store.new_id("exp")
    job_id = store.new_id("job")
    store.create(
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
    )
    store.create(
        "jobs",
        {
            "id": job_id,
            "tenant_id": tenant_id,
            "resource_id": experiment_id,
            "state": "queued",
            "stage": "queued",
            "progress": 0,
        },
    )
    runtime(request).submit_experiment(experiment_id, job_id)
    return IdResponse(id=experiment_id, state="queued")


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
    for other in runtime(request).store.list("models", tenant_id):
        if other.get("dataset_id") == model["dataset_id"] and other.get("stage") == "champion":
            runtime(request).store.update("models", other["id"], stage="candidate")
    return runtime(request).store.update(
        "models", model_id, stage="champion", promotion_justification=body.justification
    )


@router.post("/models/{model_id}/retire")
def retire_model(model_id: str, request: Request, tenant_id: Tenant) -> dict[str, Any]:
    owned(request, "models", model_id, tenant_id)
    return runtime(request).store.update("models", model_id, state="retired", stage="retired")


@router.post("/forecasts", response_model=IdResponse, status_code=status.HTTP_202_ACCEPTED)
def create_forecast(body: ForecastCreate, request: Request, tenant_id: Tenant) -> IdResponse:
    store = runtime(request).store
    model = owned(request, "models", body.model_id, tenant_id)
    if model["state"] != "ready":
        raise HTTPException(status_code=409, detail="model is not available for prediction")
    forecast_id = store.new_id("fc")
    job_id = store.new_id("job")
    store.create(
        "forecasts",
        {
            "id": forecast_id,
            "tenant_id": tenant_id,
            "model_id": body.model_id,
            "state": "queued",
            "job_id": job_id,
            "request": body.model_dump(mode="json"),
        },
    )
    store.create(
        "jobs",
        {
            "id": job_id,
            "tenant_id": tenant_id,
            "resource_id": forecast_id,
            "state": "queued",
            "stage": "queued",
            "progress": 0,
        },
    )
    runtime(request).submit_forecast(forecast_id, job_id)
    return IdResponse(id=forecast_id, state="queued")


@router.get("/forecasts/{forecast_id}")
def get_forecast(forecast_id: str, request: Request, tenant_id: Tenant) -> dict[str, Any]:
    return owned(request, "forecasts", forecast_id, tenant_id)


@router.get("/forecasts/{forecast_id}/download")
def download_forecast(forecast_id: str, request: Request, tenant_id: Tenant) -> FileResponse:
    forecast = owned(request, "forecasts", forecast_id, tenant_id)
    if forecast.get("state") != "succeeded":
        raise HTTPException(status_code=409, detail="forecast is not complete")
    return FileResponse(
        forecast["output_path"], media_type="application/json", filename=f"{forecast_id}.json"
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
