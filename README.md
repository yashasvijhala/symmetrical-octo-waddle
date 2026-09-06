# Forecasting Service

A runnable API-first platform for profiling arbitrary panel time-series data, compiling a safe
feature specification, rolling-origin backtesting, global LightGBM training, optional AutoGluon
training, model registration, cold-start forecasting, promotion, and accuracy monitoring.

## What is implemented

- Tenant-isolated CSV/Parquet upload, profiling, schema confirmation, immutable versioning, hashes,
  and compressed canonical Parquet.
- Conservative column roles: `static`, `known_future`, `past_only`, `hierarchy`, and `weight`.
- Calendar, lag, shifted rolling, scale, age, intermittency, metadata, future-covariate, and
  past-covariate features stored in the model's `FeatureSpec`.
- Direct multi-horizon global LightGBM with deterministic CPU settings.
- Leakage-safe expanding-window backtests against seasonal naive, WAPE/MAE/RMSE/bias metrics, and
  empirical residual quantiles.
- Optional AutoGluon `TimeSeriesPredictor` adapter and model-zoo leaderboard.
- Durable Hatchet experiment/forecast jobs with retries, tenant fairness, progress, cancellation,
  and separate CPU/GPU worker queues.
- Pooled PostgreSQL metadata, atomic job creation, worker-independent state, and payload-safe
  idempotency for dataset, experiment, and forecast creation.
- Model registry, baseline promotion gate, retirement, immutable lineage, actuals, and monitoring.
- Metadata/level-prior cold starts and coherent bottom-up hierarchy aggregates.
- Clean storage, orchestration, and model boundaries that preserve the HTTP contracts when local
  artifact storage is replaced by an object store.

The research and production evolution decisions are in
[the architecture plan](docs/forecasting-system-plan.md).

## Run locally

Requirements: Python 3.12+, `uv`, and OpenMP for LightGBM. On macOS:

```bash
brew install libomp
uv sync --dev
./scripts/db sync
uv run forecast-worker --kind cpu  # separate terminal
uv run fastapi dev src/forecasting_service/main.py
```

For the full AutoGluon/Chronos model zoo:

```bash
uv sync --dev --extra autogluon
```

Open `http://127.0.0.1:8000/docs`. In local/test mode, send `X-Tenant-ID` to exercise isolation;
otherwise the tenant defaults to `local`. Copy `.env.example` to `.env` to customize the service.
Set `HATCHET_CLIENT_TOKEN` for both API and worker processes. Run a GPU worker with
`uv run forecast-worker --kind gpu` when enabling AutoGluon; CPU-only deployments never load its
optional dependency.
In production, configure `FORECAST_API_KEYS` as a JSON tenant-to-secret map and send both
`X-Tenant-ID` and `X-API-Key`. Production starts fail closed when keys are absent.

## Database schema

Metadata uses PostgreSQL. The local setup targets
`postgresql://localhost:5432/symmetrical-octo-waddle`; tests create and remove isolated PostgreSQL
schemas so they exercise the same storage implementation without touching application data.

```bash
./scripts/db push     # create missing tables and indexes without deleting data
./scripts/db sync     # push and verify table compatibility
./scripts/db status   # verify the current database schema
```

Equivalent Make targets are `db-push`, `db-sync`, and `db-status`. Run `db push` as a deployment
release step before starting the API. Application workers never mutate the schema.

## Workflow

1. `POST /v1/datasets`, then upload CSV/Parquet to `/upload`.
2. `POST /profile`, review the proposed schema, and `POST /finalize` with confirmed column roles.
3. `POST /v1/experiments` and poll the experiment or its `job_id`.
4. Review `/leaderboard`, then promote the registered model if it beats the baseline.
5. `POST /v1/forecasts`, poll, and download tidy JSON predictions.
6. `POST /v1/actuals` and inspect `/v1/models/{id}/monitoring`.

`known_future` is a promise: forecasting fails when any required item/timestamp/column is missing.
This prevents accidental validation/production leakage. New items may provide `level_prior` and
metadata; otherwise the response uses a clearly labeled peer prior.

## Quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest --cov=forecasting_service
```

Runtime state, uploads, normalized data, models, and predictions live under `.forecast-state/` and
are ignored by Git.

## Deployment boundary

The API never executes ML work. Hatchet provides durable scheduling, retry, fairness, and worker
recovery; PostgreSQL remains the application-facing job-state projection. Scale CPU and GPU workers
independently. API and workers must share the artifact volume in this local/Compose profile. Replace
that volume with object storage before distributing workers across hosts. Accuracy and throughput
must be benchmarked against representative customer data before an SLO can be claimed.
