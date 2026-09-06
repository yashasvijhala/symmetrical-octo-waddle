from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from FORECAST_* environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="FORECAST_",
        extra="ignore",
    )

    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"
    api_prefix: str = "/v1"
    state_dir: Path = Path(".forecast-state")
    database_url: str = "postgresql://localhost:5432/symmetrical-octo-waddle"
    database_pool_min_size: int = Field(default=1, ge=1, le=32)
    database_pool_max_size: int = Field(default=10, ge=1, le=128)
    max_workers: int = Field(default=2, ge=1, le=64)
    max_upload_bytes: int = Field(default=2_147_483_648, ge=1_048_576)
    api_keys: dict[str, str] = Field(default_factory=dict)


@lru_cache
def get_settings() -> Settings:
    return Settings()
