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
- Asynchronous persistent experiment/forecast jobs with progress, failure details, and cancellation.
- Transactional SQLite metadata in WAL mode, atomic job creation, restart recovery, and payload-safe
  idempotency for dataset, experiment, and forecast creation.
- Model registry, baseline promotion gate, retirement, immutable lineage, actuals, and monitoring.
- Metadata/level-prior cold starts and coherent bottom-up hierarchy aggregates.
- Local durable adapters requiring no cloud account. Storage/model/queue boundaries can be replaced
  by PostgreSQL, S3, and distributed workers without changing HTTP or modeling contracts.

The research and production evolution decisions are in
[the architecture plan](docs/forecasting-system-plan.md).

## Run locally

Requirements: Python 3.12+, `uv`, and OpenMP for LightGBM. On macOS:

```bash
brew install libomp
uv sync --dev
uv run fastapi dev src/forecasting_service/main.py
```

For the full AutoGluon/Chronos model zoo:

```bash
uv sync --dev --extra autogluon
```

Open `http://127.0.0.1:8000/docs`. In local/test mode, send `X-Tenant-ID` to exercise isolation;
otherwise the tenant defaults to `local`. Copy `.env.example` to `.env` to customize the service.
In production, configure `FORECAST_API_KEYS` as a JSON tenant-to-secret map and send both
`X-Tenant-ID` and `X-API-Key`. Production starts fail closed when keys are absent.

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

The included runtime is a production-hardened **single-node** profile: persistent SQLite/WAL,
bounded streaming uploads, atomic idempotency, restart recovery, API-key isolation, immutable
artifacts, non-root container execution, and health probes. Run one API process per state volume.

Horizontal scaling requires the distributed adapters described in the architecture plan:
PostgreSQL metadata, S3-compatible artifacts, and a leased external job queue. Those services need
real deployment credentials and infrastructure and are intentionally not simulated in this repo.
Accuracy and throughput must also be benchmarked against representative customer data before an
SLO can be claimed.
