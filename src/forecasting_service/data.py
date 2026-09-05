from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import polars as pl

from forecasting_service.schemas import ColumnRole, DatasetManifest


def scan(path: Path) -> pl.LazyFrame:
    if path.suffix.lower() == ".parquet":
        return pl.scan_parquet(path)
    return pl.scan_csv(path, try_parse_dates=True, infer_schema_length=10_000)


def profile_file(path: Path) -> dict[str, Any]:
    lazy = scan(path)
    schema = lazy.collect_schema()
    sample = lazy.head(10_000).collect()
    names = schema.names()
    lowered = {name: name.lower() for name in names}

    def candidate(words: tuple[str, ...], temporal: bool = False) -> str | None:
        exact = next((name for name in names if lowered[name] in words), None)
        if exact:
            return exact
        if temporal:
            return next(
                (name for name, dtype in schema.items() if dtype in (pl.Date, pl.Datetime)),
                None,
            )
        return next((name for name in names if any(word in lowered[name] for word in words)), None)

    timestamp = candidate(("timestamp", "date", "datetime", "ds", "time"), temporal=True)
    item = candidate(("item_id", "unique_id", "series_id", "sku", "id"))
    target = candidate(("target", "value", "y", "demand", "sales", "quantity"))
    if target is None:
        target = next(
            (name for name, dtype in schema.items() if dtype.is_numeric() and name != item),
            None,
        )

    column_stats = []
    for name, dtype in schema.items():
        series = sample.get_column(name)
        column_stats.append(
            {
                "name": name,
                "dtype": str(dtype),
                "null_count_sample": series.null_count(),
                "unique_count_sample": series.n_unique(),
                "suggested_role": _suggest_role(name, timestamp, target, item, sample.height),
            }
        )

    warnings: list[dict[str, str]] = []
    if not timestamp:
        warnings.append(
            {"code": "timestamp_not_detected", "message": "Confirm a timestamp column."}
        )
    if not target:
        warnings.append(
            {"code": "target_not_detected", "message": "Confirm a numeric target column."}
        )
    return {
        "format": path.suffix.lower().lstrip("."),
        "sample_rows": sample.height,
        "columns": column_stats,
        "proposal": {
            "timestamp_column": timestamp,
            "target_column": target,
            "item_id_column": item,
            "frequency": "auto",
            "column_roles": {stat["name"]: stat["suggested_role"] for stat in column_stats},
        },
        "warnings": warnings,
    }


def canonicalize(path: Path, manifest: DatasetManifest) -> pl.DataFrame:
    frame = scan(path).collect()
    required = {manifest.timestamp_column, manifest.target_column}
    if manifest.item_id_column:
        required.add(manifest.item_id_column)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    rename = {
        manifest.timestamp_column: "timestamp",
        manifest.target_column: "target",
    }
    if manifest.item_id_column:
        rename[manifest.item_id_column] = "item_id"
    frame = frame.rename(rename)
    if not manifest.item_id_column:
        frame = frame.with_columns(pl.lit("__single__").alias("item_id"))
    declared_optional = {
        name
        for name, role in manifest.column_roles.items()
        if role not in {ColumnRole.TIMESTAMP, ColumnRole.TARGET, ColumnRole.ITEM_ID}
    }
    missing_declared = declared_optional - set(frame.columns)
    if missing_declared:
        raise ValueError(f"declared feature columns are missing: {sorted(missing_declared)}")
    frame = frame.with_columns(
        pl.col("item_id").cast(pl.String),
        pl.col("target").cast(pl.Float64, strict=False),
    )
    if frame.schema["timestamp"] == pl.String:
        frame = frame.with_columns(pl.col("timestamp").str.to_datetime(strict=False))
    elif frame.schema["timestamp"] == pl.Date:
        frame = frame.with_columns(pl.col("timestamp").cast(pl.Datetime))
    elif frame.schema["timestamp"] != pl.Datetime:
        frame = frame.with_columns(pl.col("timestamp").cast(pl.Datetime, strict=False))
    if frame.get_column("timestamp").null_count() or frame.get_column("target").null_count():
        raise ValueError("timestamp and target values must be parseable and non-null")
    for name, role in manifest.column_roles.items():
        if role in {ColumnRole.STATIC, ColumnRole.HIERARCHY} and name in frame.columns:
            max_values = cast(
                int,
                frame.group_by("item_id").agg(pl.col(name).n_unique()).get_column(name).max(),
            )
            if max_values and max_values > 1:
                raise ValueError(f"{name!r} is declared {role.value} but changes within an item")

    keys = ["item_id", "timestamp"]
    duplicate_count = frame.select(keys).is_duplicated().sum()
    if duplicate_count:
        if manifest.duplicate_policy == "reject":
            raise ValueError(f"found {duplicate_count} duplicate item/timestamp rows")
        target_expr = getattr(pl.col("target"), manifest.duplicate_policy)()
        other = [name for name in frame.columns if name not in {*keys, "target"}]
        frame = frame.group_by(keys).agg(
            target_expr.alias("target"), *[pl.col(x).last() for x in other]
        )
    target_min = cast(float, frame.get_column("target").min())
    if manifest.non_negative and target_min < 0:
        raise ValueError("negative targets violate the manifest's non_negative constraint")
    return frame.sort(keys)


def infer_frequency(frame: pl.DataFrame) -> tuple[str, int]:
    groups = frame.partition_by("item_id", maintain_order=True)
    deltas: list[int] = []
    for group in groups[:100]:
        times = group.get_column("timestamp").to_list()
        deltas.extend(int((b - a).total_seconds()) for a, b in pairwise(times) if b > a)
    if not deltas:
        return "unknown", 1
    seconds = sorted(deltas)[len(deltas) // 2]
    choices = [(3600, "1h", 24), (86400, "1d", 7), (604800, "1w", 52), (2592000, "1mo", 12)]
    closest = min(choices, key=lambda value: abs(value[0] - seconds))
    return closest[1], closest[2]


def _suggest_role(
    name: str,
    timestamp: str | None,
    target: str | None,
    item: str | None,
    sample_rows: int,
) -> str:
    if name == timestamp:
        return ColumnRole.TIMESTAMP
    if name == target:
        return ColumnRole.TARGET
    if name == item:
        return ColumnRole.ITEM_ID
    lowered = name.lower()
    if any(word in lowered for word in ("holiday", "promo", "planned", "calendar")):
        return ColumnRole.KNOWN_FUTURE
    # Static is only a suggestion; finalize still requires caller confirmation.
    if any(word in lowered for word in ("category", "brand", "region", "type")):
        return ColumnRole.STATIC
    return ColumnRole.PAST_ONLY
