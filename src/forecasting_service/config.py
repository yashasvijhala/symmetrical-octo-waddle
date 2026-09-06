from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

OBJECT_CACHE_MAX_BYTES = 10_737_418_240
OBJECT_PRESIGN_SECONDS = 900
R2_KEY_PREFIX = "forecasting"
R2_MAX_CONNECTIONS = 32
R2_TRANSFER_CONCURRENCY = 4
R2_MULTIPART_THRESHOLD_BYTES = 67_108_864
R2_MULTIPART_CHUNK_BYTES = 16_777_216
JOB_RETRIES = 2
JOB_EXECUTION_TIMEOUT_SECONDS = 86_400
JOB_SCHEDULE_TIMEOUT_SECONDS = 86_400
TENANT_MAX_CONCURRENT_JOBS = 2


class Settings(BaseSettings):
    """Credentials and deployment settings. Transfer tunables live as code constants."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        extra="ignore",
        env_ignore_empty=True,
    )

    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"
    api_prefix: str = "/v1"
    state_dir: Path = Path(".forecast-state")
    object_store_backend: Literal["local", "r2"] = "local"
    object_cache_dir: Path | None = None
    r2_endpoint: str = ""
    r2_bucket: str = ""
    r2_access_key_id: str = Field(default="", repr=False)
    r2_secret_access_key: str = Field(default="", repr=False)
    database_url: str = "postgresql://localhost:5432/symmetrical-octo-waddle"
    database_pool_min_size: int = Field(default=1, ge=1, le=32)
    database_pool_max_size: int = Field(default=10, ge=1, le=128)
    test_max_workers: int = Field(default=2, ge=1, le=64)
    cpu_worker_slots: int = Field(default=2, ge=1, le=128)
    gpu_worker_slots: int = Field(default=1, ge=1, le=32)
    max_upload_bytes: int = Field(default=2_147_483_648, ge=1_048_576)
    api_keys: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_object_store(self) -> "Settings":
        if self.object_cache_dir is None:
            self.object_cache_dir = self.state_dir / "cache"
        if self.object_store_backend != "r2":
            return self
        missing = [
            name
            for name in (
                "r2_endpoint",
                "r2_bucket",
                "r2_access_key_id",
                "r2_secret_access_key",
            )
            if not getattr(self, name)
        ]
        if missing:
            raise ValueError(f"R2 object storage requires: {', '.join(missing)}")
        if not self.r2_endpoint.startswith("https://"):
            raise ValueError("R2 endpoint must use HTTPS")
        return self

    def cache_dir(self) -> Path:
        if self.object_cache_dir is None:
            return self.state_dir / "cache"
        return self.object_cache_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()
