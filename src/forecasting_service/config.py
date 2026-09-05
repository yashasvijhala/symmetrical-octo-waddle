from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    max_workers: int = 2
    max_upload_bytes: int = 2_147_483_648


@lru_cache
def get_settings() -> Settings:
    return Settings()
