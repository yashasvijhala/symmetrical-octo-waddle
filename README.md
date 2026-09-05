# Forecasting Service

An API-first Python service for ingesting panel time-series data, generating leakage-safe
features, backtesting candidate models, registering model versions, and serving probabilistic
forecasts to a separate Next.js application.

The repository currently contains the production-oriented FastAPI foundation and the reviewed
system design. Model and persistence work is intentionally split into the implementation phases
in [the forecasting system plan](docs/forecasting-system-plan.md).

## Local development

Requirements: Python 3.12+ and `uv`.

```bash
uv sync --dev
uv run fastapi dev src/forecasting_service/main.py
```

Open `http://127.0.0.1:8000/docs`. Run the checks with:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest --cov=forecasting_service
```

Copy `.env.example` to `.env` for local overrides. Local data, model artifacts, checkpoints,
and secrets are excluded by `.gitignore`.

## Current endpoints

- `GET /v1/health/live`
- `GET /v1/health/ready`

The complete proposed API, data contract, feature strategy, cold-start policy, architecture, and
delivery sequence are documented in [the implementation plan](docs/forecasting-system-plan.md).
