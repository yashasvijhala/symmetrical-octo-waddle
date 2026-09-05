FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FORECAST_STATE_DIR=/var/lib/forecasting

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"
WORKDIR /app
ARG FORECAST_EXTRAS=""
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN if [ "$FORECAST_EXTRAS" = "autogluon" ]; then \
      uv sync --frozen --no-dev --extra autogluon; \
    else \
      uv sync --frozen --no-dev; \
    fi

EXPOSE 8000
VOLUME ["/var/lib/forecasting"]
CMD ["uv", "run", "uvicorn", "forecasting_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
