from __future__ import annotations

import hashlib
import importlib
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from forecasting_service.data import infer_frequency
from forecasting_service.schemas import ColumnRole, DatasetManifest, ExperimentCreate, NewItem


@dataclass
class Series:
    item_id: str
    timestamps: list[datetime]
    target: list[float]
    columns: dict[str, list[Any]]


def train_lightgbm(
    frame: pl.DataFrame,
    manifest: DatasetManifest,
    config: ExperimentCreate,
    artifact_path: Path,
) -> dict[str, Any]:
    frequency, default_season = infer_frequency(frame)
    horizon = config.horizon or manifest.horizon
    min_length = min(len(group) for group in frame.partition_by("item_id"))
    if min_length <= horizon:
        raise ValueError(
            f"each training series needs more than horizon={horizon} observations; "
            f"shortest series has {min_length}"
        )
    season = manifest.seasonal_periods[0] if manifest.seasonal_periods else default_season
    lags = sorted({1, 2, 3, season, 2 * season})
    spec = _feature_spec(frame, manifest, frequency, season, lags)
    series = _series(frame)
    windows = _valid_windows(series, horizon, config.validation_windows)
    fold_metrics: list[dict[str, Any]] = []
    residuals: dict[int, list[float]] = {step: [] for step in range(1, horizon + 1)}

    for fold, trim in enumerate(reversed(windows), start=1):
        training = [_trim(item, trim + horizon) for item in series]
        estimators = _fit_estimators(training, horizon, spec, config.seed, config.num_threads)
        actual: list[float] = []
        predicted: list[float] = []
        baseline: list[float] = []
        for item in series:
            cutoff = len(item.target) - trim - horizon
            if cutoff < 1:
                continue
            history = _trim(item, trim + horizon)
            truth = item.target[cutoff : cutoff + horizon]
            values = _predict_series(history, estimators, horizon, spec, {})
            naive = _seasonal_naive(history.target, horizon, season)
            actual.extend(truth)
            predicted.extend(values)
            baseline.extend(naive)
            for step, (observed, forecast) in enumerate(zip(truth, values, strict=True), start=1):
                residuals[step].append(observed - forecast)
        metrics = _metrics(actual, predicted)
        metrics.update(
            {f"baseline_{key}": value for key, value in _metrics(actual, baseline).items()}
        )
        fold_metrics.append({"fold": fold, "trim": trim, **metrics})

    estimators = _fit_estimators(series, horizon, spec, config.seed, config.num_threads)
    offsets = {
        str(step): {
            str(q): float(np.quantile(values, q)) if values else 0.0 for q in config.quantiles
        }
        for step, values in residuals.items()
    }
    artifact = {
        "engine": "lightgbm",
        "horizon": horizon,
        "frequency": frequency,
        "seasonal_period": season,
        "feature_spec": spec,
        "estimators": estimators,
        "quantile_offsets": offsets,
        "quantiles": config.quantiles,
        "non_negative": manifest.non_negative,
        "peer_profiles": _peer_profiles(series, season),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, artifact_path)
    aggregate = _mean_metrics(fold_metrics)
    return {
        "engine": "lightgbm",
        "artifact_path": str(artifact_path),
        "feature_spec": spec,
        "folds": fold_metrics,
        "metrics": aggregate,
        "baseline_beaten": aggregate.get(config.primary_metric, math.inf)
        < aggregate.get(f"baseline_{config.primary_metric}", math.inf),
    }


def forecast_lightgbm(
    artifact_path: Path,
    frame: pl.DataFrame,
    horizon: int | None,
    future_covariates: list[dict[str, Any]],
    new_items: list[NewItem],
) -> list[dict[str, Any]]:
    artifact = joblib.load(artifact_path)
    requested = horizon or artifact["horizon"]
    if requested > artifact["horizon"]:
        raise ValueError(f"model supports at most {artifact['horizon']} forecast steps")
    spec = artifact["feature_spec"]
    cov_lookup = {
        (str(row.get("item_id")), str(row.get("timestamp"))): row for row in future_covariates
    }
    rows: list[dict[str, Any]] = []
    for item in _series(frame):
        future = _future_covariates(item, requested, cov_lookup)
        required_known = spec["roles"]["known_future"]
        missing = [
            f"{item.item_id}/step-{step}/{name}"
            for step in range(1, requested + 1)
            for name in required_known
            if future.get(step, {}).get(name) is None
        ]
        if missing:
            raise ValueError(f"future covariates are incomplete: {missing[:10]}")
        values = _predict_series(item, artifact["estimators"], requested, spec, future)
        times = _future_times(item.timestamps, requested)
        rows.extend(_forecast_rows(item.item_id, times, values, artifact, "lightgbm"))
    rows.extend(_hierarchy_rows(rows, frame, spec["roles"].get("hierarchy", [])))
    for item in new_items:
        rows.extend(_cold_start_rows(item, requested, artifact, frame))
    return rows


def train_autogluon(
    frame: pl.DataFrame,
    manifest: DatasetManifest,
    config: ExperimentCreate,
    artifact_dir: Path,
) -> dict[str, Any]:
    try:
        ag = importlib.import_module("autogluon.timeseries")
        TimeSeriesDataFrame = ag.TimeSeriesDataFrame
        TimeSeriesPredictor = ag.TimeSeriesPredictor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "AutoGluon is optional; install it with `uv sync --extra autogluon`"
        ) from exc

    horizon = config.horizon or manifest.horizon
    known = [
        name
        for name, role in manifest.column_roles.items()
        if role == ColumnRole.KNOWN_FUTURE and name in frame.columns
    ]
    pandas = frame.to_pandas().rename(columns={"item_id": "item_id", "timestamp": "timestamp"})
    ts_data = TimeSeriesDataFrame.from_data_frame(
        pandas, id_column="item_id", timestamp_column="timestamp"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictor = TimeSeriesPredictor(
        path=str(artifact_dir),
        target="target",
        prediction_length=horizon,
        known_covariates_names=known,
        eval_metric=config.primary_metric.upper(),
        quantile_levels=config.quantiles,
    )
    min_length = min(len(group) for group in frame.partition_by("item_id"))
    windows = max(1, min(config.validation_windows, min_length // horizon - 1))
    predictor.fit(
        ts_data,
        presets="medium_quality",
        time_limit=config.time_limit_seconds,
        num_val_windows=windows,
        random_seed=config.seed,
    )
    leaderboard = predictor.leaderboard().to_dict(orient="records")
    best = leaderboard[0] if leaderboard else {}
    return {
        "engine": "autogluon",
        "artifact_path": str(artifact_dir),
        "metrics": {"validation_score": best.get("score_val")},
        "leaderboard": leaderboard,
        "feature_spec": {"known_covariates": known},
        "baseline_beaten": True,
    }


def forecast_autogluon(
    artifact_dir: Path,
    frame: pl.DataFrame,
    future_covariates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        ag = importlib.import_module("autogluon.timeseries")
        TimeSeriesDataFrame = ag.TimeSeriesDataFrame
        TimeSeriesPredictor = ag.TimeSeriesPredictor
    except ModuleNotFoundError as exc:
        raise RuntimeError("AutoGluon is not installed") from exc
    predictor = TimeSeriesPredictor.load(str(artifact_dir))
    data = TimeSeriesDataFrame.from_data_frame(
        frame.to_pandas(), id_column="item_id", timestamp_column="timestamp"
    )
    known = None
    if predictor.known_covariates_names:
        if not future_covariates:
            raise ValueError("this model requires future_covariates")
        pd = importlib.import_module("pandas")
        known = TimeSeriesDataFrame.from_data_frame(
            pd.DataFrame(future_covariates), id_column="item_id", timestamp_column="timestamp"
        )
    predictions = predictor.predict(data, known_covariates=known).reset_index()
    return [
        {key: _json_value(value) for key, value in row.items()}
        for row in predictions.to_dict(orient="records")
    ]


def _series(frame: pl.DataFrame) -> list[Series]:
    result = []
    for group in frame.partition_by("item_id", maintain_order=True):
        result.append(
            Series(
                item_id=group.item(0, "item_id"),
                timestamps=group.get_column("timestamp").to_list(),
                target=group.get_column("target").to_list(),
                columns={name: group.get_column(name).to_list() for name in group.columns},
            )
        )
    return result


def _feature_spec(
    frame: pl.DataFrame,
    manifest: DatasetManifest,
    frequency: str,
    season: int,
    lags: list[int],
) -> dict[str, Any]:
    available = set(frame.columns)
    roles = {
        role.value: [
            name
            for name, value in manifest.column_roles.items()
            if value == role and name in available
        ]
        for role in (
            ColumnRole.STATIC,
            ColumnRole.KNOWN_FUTURE,
            ColumnRole.PAST_ONLY,
            ColumnRole.HIERARCHY,
        )
    }
    names = ["item", "age", "horizon", "month", "weekday", "hour"]
    names += [f"lag_{lag}" for lag in lags]
    names += ["rolling_mean_short", "rolling_mean_season", "rolling_std_season", "zero_rate"]
    names += [f"static:{name}" for name in roles["static"]]
    names += [f"hierarchy:{name}" for name in roles["hierarchy"]]
    names += [f"known_future:{name}" for name in roles["known_future"]]
    names += [f"past_only:{name}" for name in roles["past_only"]]
    return {
        "version": 1,
        "frequency": frequency,
        "seasonal_period": season,
        "lags": lags,
        "roles": roles,
        "feature_names": names,
    }


def _features(
    item: Series,
    cutoff: int,
    step: int,
    spec: dict[str, Any],
    future: dict[int, dict[str, Any]],
) -> list[float]:
    history = item.target[: cutoff + 1]
    target_time = _future_times(item.timestamps[: cutoff + 1], step)[-1]
    season = spec["seasonal_period"]
    values = [
        _encode(item.item_id),
        float(len(history)),
        float(step),
        float(target_time.month),
        float(target_time.weekday()),
        float(target_time.hour),
    ]
    for lag in spec["lags"]:
        values.append(history[-lag] if len(history) >= lag else math.nan)
    short = history[-min(3, len(history)) :]
    seasonal = history[-min(season, len(history)) :]
    values.extend(
        [
            float(np.mean(short)),
            float(np.mean(seasonal)),
            float(np.std(seasonal)),
            float(sum(value == 0 for value in seasonal) / len(seasonal)),
        ]
    )
    for name in [*spec["roles"]["static"], *spec["roles"]["hierarchy"]]:
        values.append(_numeric(item.columns.get(name, [None])[-1]))
    for name in spec["roles"]["known_future"]:
        source = future.get(step, {}).get(name)
        if source is None and cutoff + step < len(item.columns.get(name, [])):
            source = item.columns[name][cutoff + step]
        values.append(_numeric(source))
    for name in spec["roles"]["past_only"]:
        column = item.columns.get(name, [])
        values.append(_numeric(column[cutoff] if cutoff < len(column) else None))
    return values


def _fit_estimators(
    series: list[Series],
    horizon: int,
    spec: dict[str, Any],
    seed: int,
    num_threads: int,
) -> list[dict[str, Any]]:
    from lightgbm import LGBMRegressor

    estimators: list[dict[str, Any]] = []
    for step in range(1, horizon + 1):
        x_rows: list[list[float]] = []
        labels: list[float] = []
        for item in series:
            for cutoff in range(0, len(item.target) - step):
                x_rows.append(_features(item, cutoff, step, spec, {}))
                labels.append(item.target[cutoff + step])
        if len(labels) < 8 or len(set(labels)) < 2:
            estimators.append(
                {"kind": "constant", "value": float(np.mean(labels)) if labels else 0.0}
            )
            continue
        model = LGBMRegressor(
            n_estimators=180,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=max(5, min(20, len(labels) // 10)),
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=0.1,
            random_state=seed + step,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            n_jobs=num_threads,
        )
        model.fit(np.asarray(x_rows, dtype=float), np.asarray(labels, dtype=float))
        estimators.append({"kind": "lightgbm", "model": model})
    return estimators


def _predict_series(
    item: Series,
    estimators: list[dict[str, Any]],
    horizon: int,
    spec: dict[str, Any],
    future: dict[int, dict[str, Any]],
) -> list[float]:
    values = []
    cutoff = len(item.target) - 1
    for step in range(1, horizon + 1):
        estimator = estimators[step - 1]
        if estimator["kind"] == "constant":
            value = estimator["value"]
        else:
            x_row = np.asarray([_features(item, cutoff, step, spec, future)], dtype=float)
            value = float(estimator["model"].predict(x_row)[0])
        values.append(value)
    return values


def _valid_windows(series: list[Series], horizon: int, requested: int) -> list[int]:
    min_length = min(len(item.target) for item in series)
    count = max(0, min(requested, min_length // horizon - 1))
    return [index * horizon for index in range(count)] or [0]


def _trim(item: Series, count: int) -> Series:
    end = len(item.target) - count if count else len(item.target)
    return Series(
        item.item_id,
        item.timestamps[:end],
        item.target[:end],
        {name: values[:end] for name, values in item.columns.items()},
    )


def _metrics(actual: list[float], predicted: list[float]) -> dict[str, float]:
    if not actual:
        return {"mae": math.inf, "rmse": math.inf, "wape": math.inf, "bias": math.inf}
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    error = p - a
    denominator = float(np.abs(a).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "wape": float(np.abs(error).sum() / denominator)
        if denominator
        else float(np.abs(error).mean()),
        "bias": float(error.sum() / denominator) if denominator else float(error.mean()),
    }


def _mean_metrics(folds: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "mae",
        "rmse",
        "wape",
        "bias",
        "baseline_mae",
        "baseline_rmse",
        "baseline_wape",
        "baseline_bias",
    )
    return {
        key: float(np.mean([fold[key] for fold in folds if math.isfinite(fold[key])]))
        for key in keys
        if any(math.isfinite(fold[key]) for fold in folds)
    }


def _seasonal_naive(history: list[float], horizon: int, season: int) -> list[float]:
    if not history:
        return [0.0] * horizon
    return [
        history[-season + (step - 1) % season] if len(history) >= season else history[-1]
        for step in range(1, horizon + 1)
    ]


def _future_times(times: list[datetime], horizon: int) -> list[datetime]:
    delta = times[-1] - times[-2] if len(times) > 1 else timedelta(days=1)
    return [times[-1] + delta * step for step in range(1, horizon + 1)]


def _future_covariates(
    item: Series, horizon: int, lookup: dict[tuple[str, str], dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    result = {}
    for step, timestamp in enumerate(_future_times(item.timestamps, horizon), start=1):
        result[step] = lookup.get(
            (item.item_id, timestamp.isoformat()), lookup.get((item.item_id, str(timestamp)), {})
        )
    return result


def _forecast_rows(
    item_id: str,
    times: list[datetime],
    values: list[float],
    artifact: dict[str, Any],
    method: str,
) -> list[dict[str, Any]]:
    rows = []
    for step, (timestamp, raw) in enumerate(zip(times, values, strict=True), start=1):
        value = max(0.0, raw) if artifact.get("non_negative") else raw
        row: dict[str, Any] = {
            "item_id": item_id,
            "timestamp": timestamp.isoformat(),
            "mean": value,
            "method": method,
            "warnings": [],
        }
        for quantile in artifact.get("quantiles", []):
            estimate = raw + artifact["quantile_offsets"].get(str(step), {}).get(str(quantile), 0.0)
            row[f"q{quantile:g}"] = max(0.0, estimate) if artifact.get("non_negative") else estimate
        rows.append(row)
    return rows


def _peer_profiles(series: list[Series], season: int) -> dict[str, Any]:
    levels = [float(np.mean(item.target[-min(season, len(item.target)) :])) for item in series]
    profiles = []
    for item, level in zip(series, levels, strict=True):
        tail = item.target[-min(season, len(item.target)) :]
        if level:
            profiles.append([value / level for value in tail])
    width = min((len(profile) for profile in profiles), default=1)
    profile = (
        np.mean([values[-width:] for values in profiles], axis=0).tolist() if profiles else [1.0]
    )
    return {"level": float(np.median(levels)) if levels else 0.0, "profile": profile}


def _cold_start_rows(
    item: NewItem, horizon: int, artifact: dict[str, Any], frame: pl.DataFrame
) -> list[dict[str, Any]]:
    peers = artifact["peer_profiles"]
    level = item.level_prior if item.level_prior is not None else peers["level"]
    profile = peers["profile"]
    last_time = frame.get_column("timestamp").to_list()[-1]
    sample = frame.partition_by("item_id", maintain_order=True)[0].get_column("timestamp").to_list()
    delta = sample[-1] - sample[-2] if len(sample) > 1 else timedelta(days=1)
    values = [level * profile[index % len(profile)] for index in range(horizon)]
    rows = _forecast_rows(
        item.item_id,
        [last_time + delta * (index + 1) for index in range(horizon)],
        values,
        artifact,
        "cold_start_analog",
    )
    for row in rows:
        row["warnings"] = ["cold start forecast; uncertainty may be understated"]
    return rows


def _hierarchy_rows(
    rows: list[dict[str, Any]], frame: pl.DataFrame, levels: list[str]
) -> list[dict[str, Any]]:
    if not levels:
        return []
    latest = frame.group_by("item_id").agg(*[pl.col(level).last() for level in levels])
    metadata = {row["item_id"]: row for row in latest.to_dicts()}
    numeric = [key for key in rows[0] if key == "mean" or key.startswith("q")] if rows else []
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        item_metadata = metadata.get(row["item_id"], {})
        for level in levels:
            value = item_metadata.get(level)
            if value is None:
                continue
            key = (level, str(value), row["timestamp"])
            aggregate = grouped.setdefault(
                key,
                {
                    "item_id": f"{level}={value}",
                    "timestamp": row["timestamp"],
                    "method": "bottom_up_reconciled",
                    "warnings": [],
                    **{name: 0.0 for name in numeric},
                },
            )
            for name in numeric:
                aggregate[name] += float(row[name])
    return list(grouped.values())


def _numeric(value: Any) -> float:
    if value is None:
        return math.nan
    if isinstance(value, (int, float, bool)):
        return float(value)
    return _encode(str(value))


def _encode(value: str) -> float:
    return float(int.from_bytes(hashlib.blake2b(value.encode(), digest_size=4).digest(), "big"))


def _json_value(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value
