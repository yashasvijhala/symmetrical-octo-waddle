FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STATE_DIR=/var/lib/forecasting

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && apt-get purge -y --auto-remove curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system forecasting \
    && useradd --system --gid forecasting --home-dir /app forecasting \
    && mkdir -p /var/lib/forecasting \
    && chown forecasting:forecasting /var/lib/forecasting

ENV PATH="/root/.local/bin:$PATH"
WORKDIR /app
ARG INSTALL_EXTRAS=""
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN if [ "$INSTALL_EXTRAS" = "autogluon" ]; then \
      uv sync --frozen --no-dev --extra autogluon; \
    else \
      uv sync --frozen --no-dev; \
    fi \
    && chown -R forecasting:forecasting /app

EXPOSE 8000
VOLUME ["/var/lib/forecasting"]
USER forecasting
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health/ready', timeout=2)"]
CMD ["/app/.venv/bin/uvicorn", "forecasting_service.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
