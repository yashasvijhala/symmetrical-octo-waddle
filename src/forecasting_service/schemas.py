from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ResourceState(StrEnum):
    DRAFT = "draft"
    READY = "ready"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETIRED = "retired"


class ColumnRole(StrEnum):
    TARGET = "target"
    TIMESTAMP = "timestamp"
    ITEM_ID = "item_id"
    STATIC = "static"
    KNOWN_FUTURE = "known_future"
    PAST_ONLY = "past_only"
    WEIGHT = "weight"
    HIERARCHY = "hierarchy"


class DatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)


class DatasetManifest(BaseModel):
    timestamp_column: str
    target_column: str
    item_id_column: str | None = None
    frequency: str = "auto"
    timezone: str = "UTC"
    horizon: int = Field(default=1, ge=1, le=1000)
    column_roles: dict[str, ColumnRole] = Field(default_factory=dict)
    duplicate_policy: Literal["reject", "sum", "mean", "last"] = "reject"
    non_negative: bool = False
    seasonal_periods: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def required_roles_are_distinct(self) -> "DatasetManifest":
        required = [self.timestamp_column, self.target_column]
        if self.item_id_column:
            required.append(self.item_id_column)
        if len(required) != len(set(required)):
            raise ValueError("timestamp, target, and item ID columns must be distinct")
        return self


class ExperimentCreate(BaseModel):
    dataset_id: str
    dataset_version: int = Field(default=1, ge=1)
    horizon: int | None = Field(default=None, ge=1, le=1000)
    validation_windows: int = Field(default=3, ge=1, le=10)
    model_policy: Literal["fast", "balanced", "high_accuracy", "autogluon"] = "balanced"
    primary_metric: Literal["wape", "mae", "rmse"] = "wape"
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    time_limit_seconds: int = Field(default=300, ge=10, le=86400)
    seed: int = 42
    num_threads: int = Field(default=2, ge=1, le=64)

    @model_validator(mode="after")
    def valid_quantiles(self) -> "ExperimentCreate":
        if not self.quantiles or any(not 0 < value < 1 for value in self.quantiles):
            raise ValueError("quantiles must contain values strictly between zero and one")
        self.quantiles = sorted(set(self.quantiles))
        return self


class NewItem(BaseModel):
    item_id: str
    level_prior: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ForecastCreate(BaseModel):
    model_id: str
    dataset_id: str | None = None
    dataset_version: int = Field(default=1, ge=1)
    horizon: int | None = Field(default=None, ge=1, le=1000)
    new_items: list[NewItem] = Field(default_factory=list)
    future_covariates: list[dict[str, Any]] = Field(default_factory=list)


class ActualPoint(BaseModel):
    item_id: str
    timestamp: str
    value: float


class ActualsCreate(BaseModel):
    model_id: str
    points: list[ActualPoint]


class PromotionRequest(BaseModel):
    force: bool = False
    justification: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def forced_requires_reason(self) -> "PromotionRequest":
        if self.force and not self.justification:
            raise ValueError("forcing promotion requires a justification")
        return self


class IdResponse(BaseModel):
    id: str
    state: str


class ProblemDetail(BaseModel):
    type: str = "about:blank"
    title: str
    status: int
    detail: str
    code: str
