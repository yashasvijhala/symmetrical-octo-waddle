# Forecasting Platform: Research-backed Architecture and Implementation Plan

**Audience:** engineering and data-science team  
**Decision date:** 5 September 2026  
**Scope:** a general-purpose, API-first service for panel time-series forecasting from a separate
Next.js product

## Implementation status

The repository now contains a complete runnable implementation of the data lifecycle,
profiling, immutable manifests, feature compilation, rolling backtests, global LightGBM,
AutoGluon adapter, model registry, durable Hatchet workers, forecasting, cold starts, bottom-up
hierarchy aggregation, actuals, and monitoring described below. An S3-compatible artifact adapter,
MinT reconciliation, foundation-model weight downloads, and infrastructure-specific authentication
remain deployment choices; their interfaces and API contracts are separated from execution.

## Executive decision

Build a hybrid forecasting platform with two complementary modeling lanes:

1. A **fast, controllable global LightGBM lane** using a versioned feature specification and
   leakage-safe lag, rolling, calendar, metadata, age, and availability features.
2. An **accuracy AutoGluon lane** that evaluates statistical baselines, tabular global models,
   suitable deep global models, Chronos-2, and a validation-trained weighted ensemble.

Do not hard-code one algorithm as the answer for every upload. Select the champion using rolling
origin backtests that match the requested horizon, latency, and business metric. Always compare it
with seasonal-naive and simple statistical baselines. AutoGluon itself follows this heterogeneous
model-and-ensemble strategy: its current time-series stack includes statistical models, LightGBM-
backed tabular models, deep global models, pretrained Chronos models, and a weighted ensemble
([AutoGluon overview](https://auto.gluon.ai/stable/tutorials/timeseries/index.html),
[in-depth guide](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-indepth.html)).

Use **Polars lazy scans and Parquet** for ingestion, profiling, validation, and materializing feature
tables. Convert to pandas only at the adapter boundary of libraries that require it. Polars' lazy
engine supports streaming and schema checks and applies optimizations such as predicate and
projection pushdown
([Polars lazy usage](https://docs.pola.rs/user-guide/lazy/using/),
[optimizer](https://docs.pola.rs/user-guide/lazy/optimizations/)).

## The answer to “how can we create features for unknown data?”

We can automate **mechanical features**, but not the business meaning or future availability of an
unknown column. The safe workflow is:

1. Ingest a sample and infer candidate timestamp, target, item identifier, frequency, dtypes,
   missingness, duplicate keys, series lengths, intermittency, and likely static columns.
2. Return a proposed schema and profile to the UI.
3. Require confirmation of the timestamp, target, item ID, timezone, aggregation rule, and the role
   of every optional column.
4. Treat undeclared optional columns as `past_only`. A column becomes `known_future` only when the
   caller explicitly promises values for the full forecast horizon.
5. Compile the confirmed schema into an immutable `FeatureSpec`. Use that exact artifact for every
   training fold and every future prediction.

The permitted column roles are:

| Role | Example | Training use | Forecast-time requirement |
|---|---|---|---|
| `target` | demand | labels and historical lags | history only |
| `static` | category, brand, region | repeated per item/global model | value for every item, including new items |
| `known_future` | planned price, promotion, holiday | past and future | every item × every horizon timestamp |
| `past_only` | observed weather, web traffic | historical context/lags only | never read beyond the forecast cutoff |
| `weight` | revenue importance | fitting/evaluation weight | optional |
| `hierarchy` | country/store/category | aggregation and reconciliation | value for every item |

This is not merely defensive engineering: using information unavailable at prediction time is data
leakage and produces optimistic validation results
([scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)).
AutoGluon likewise requires future values for declared known covariates across the complete forecast
horizon ([current predictor contract](https://auto.gluon.ai/stable/_modules/autogluon/timeseries/predictor.html)).

## Canonical data contract

Internally, normalize every upload into long-form Parquet:

```text
tenant_id | dataset_id | item_id | timestamp | target | [dynamic columns...]
```

Keep static metadata in a separate table keyed by `item_id`. The confirmed dataset manifest stores:

- frequency or `auto`, timezone, horizon, and optional seasonal periods;
- exact source-to-canonical column mappings and dtypes;
- per-column availability role and missing-value policy;
- target constraints (`non_negative`, integer/count, allowed zeros, censoring/stockout semantics);
- duplicate aggregation policy (`sum`, `mean`, `last`, or reject);
- hierarchy levels and optional evaluation weights;
- expected future-covariate coverage and late-data policy.

Ingestion must reject ambiguous duplicate `(item_id, timestamp)` keys unless an aggregation rule was
confirmed. Convert timestamps to UTC while retaining the source timezone in the manifest. Detect
mixed frequencies and either reject them or create separate frequency partitions; never silently
resample unrelated cadences. Preserve missing targets as missingness, not automatic zeros. A zero can
mean genuine zero demand, no inventory, a closed store, or missing reporting, and the service cannot
guess which.

Every finalized dataset version is immutable and content-addressed. Store the raw upload, normalized
Parquet, profile JSON, schema/feature spec, and content hash. Retraining creates new experiment and
model versions rather than mutating prior artifacts.

## Feature system

### FeatureSpec, not ad-hoc notebooks

The feature engine consumes `(DatasetManifest, FeatureSpec, cutoff)` and produces the same columns in
training and inference. Each feature definition records its name, source role, transform, parameters,
minimum history, null behavior, dtype, version, and lineage. Fitted state is scoped to a training
fold. Nothing may calculate across the fold cutoff.

Start with this feature library:

- **Calendar:** hour, day of week/month/year, week, month, quarter, weekend, month/quarter/year end,
  elapsed time, local holiday/event flags, and cyclical sine/cosine encodings where useful.
- **History:** lags 1/2/3 plus cadence candidates (hourly 24/168, daily 7/28/364, weekly 13/52,
  monthly 3/6/12), included only when enough history exists.
- **Shifted rolling:** mean, median, standard deviation, min/max, quantiles, non-zero rate, sum, and
  exponentially weighted statistics over short/seasonal windows. All target windows start at lag 1.
- **Trend and scale:** expanding mean/std, recent-vs-long mean ratios, differences, percent changes
  with safe denominators, local scale, and age since launch.
- **Intermittency:** time since last non-zero, non-zero count/rate, average inter-demand interval,
  recent zero run, and demand-size statistics.
- **Cross-series:** aggregate/category lags, peer-group statistics, item share of parent, and aggregate
  trend. These must also be computed as-of each cutoff.
- **Metadata and covariates:** native categorical metadata, numeric static attributes, known-future
  values, and historical lags of past-only values.
- **Quality indicators:** imputed/missing flag, newly launched flag, outlier flag, and observed-history
  length. Do not remove real spikes by default.

Nixtla's MLForecast provides a useful reference implementation for global lag/rolling pipelines,
target transforms, and time-series cross-validation
([API](https://nixtlaverse.nixtla.io/mlforecast/forecast.html),
[lag transforms](https://nixtlaverse.nixtla.io/mlforecast/lag_transforms.html),
[cross-validation](https://nixtlaverse.nixtla.io/mlforecast/docs/how-to-guides/cross_validation.html)).
We may use its primitives in the LightGBM adapter, but our own FeatureSpec remains the source of truth
so the platform can reproduce features across model adapters and versions.

### Automatic selection

The profiler proposes candidate seasonal periods from cadence, then checks their support using only
the training portion of each fold. It prunes features whose minimum history is unavailable, features
that are nearly constant, and duplicate/highly redundant transforms. LightGBM regularization and
feature subsampling are tuned inside the backtest, followed by validation permutation importance for
diagnostics—not by inspecting the final holdout.

For reproducibility on CPU, set explicit seeds, record the exact library/build versions, and use
LightGBM's deterministic mode with one of the forced histogram layouts. LightGBM documents that
determinism is system/version dependent and recommends `force_col_wise` or `force_row_wise` with
`deterministic=true`
([LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html#deterministic)).

## Model portfolio and routing

### Required baselines

Every experiment includes last-value naive, seasonal naive where feasible, drift, and a simple
seasonal/statistical candidate. Intermittent demand adds Croston/SBA-style candidates. A trained model
that cannot beat an appropriate naive baseline on the primary backtest should not be promoted.

### Custom LightGBM lane

Train a global model across related series so information can transfer to young items. Begin with a
direct multi-horizon design: add `horizon_step` and train either one pooled model or horizon buckets;
compare it against recursive one-step inference for cost. Direct forecasting avoids error propagation
and permits horizon-specific patterns; recursive forecasting is cheaper. Backtesting decides.

Use item/category metadata as native categorical features, per-series scaling or transformed targets
where backtests justify it, observation weights, early stopping, controlled thread counts, and
quantile objectives for requested quantiles. Enforce target constraints only as an explicit
post-processing policy. Quantile crossing is repaired monotonically and then interval coverage is
measured.

### AutoGluon lane

Use `TimeSeriesPredictor` as a challenger/ensemble layer, not as the data contract. Candidate presets
should be budgeted by dataset regime and infrastructure:

- local statistical models for mature regular series;
- `RecursiveTabular` and `DirectTabular` for fast global tree baselines;
- DeepAR/TFT/PatchTST only when panel size, history, GPU budget, and latency justify them;
- Chronos-2 as a strong zero-shot/global candidate for short or unfamiliar series;
- the validation-trained weighted ensemble.

AutoGluon supports multiple validation windows, but each series needs at least
`(num_val_windows + 1) × horizon` observations
([AutoGluon validation guide](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-indepth.html)).
The API must reduce fold count or route short series rather than silently dropping them.

Chronos-2 supports variable history lengths, past and future covariates, multivariate groups, and
optional cross-learning; its own documentation warns that cross-learning does not always improve
accuracy and must be tested
([Chronos-2 pipeline](https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py)).
Its project is Apache-2.0 licensed
([Chronos repository](https://github.com/amazon-science/chronos-forecasting)).

Google TimesFM is valuable as an optional research challenger, but **do not enable TimesFM 3.0 weights
in a commercial production path**: Google currently labels those weights non-commercial/non-production.
The repository says TimesFM 2.5 weights remain Apache-2.0, while 3.0 adds native multivariate and
past/future covariate support under the restricted weight license
([official TimesFM repository](https://github.com/google-research/timesfm)). This licensing distinction
must be represented in the model registry.

## Young-series and cold-start strategy

“Young” is a routing dimension, not one arbitrary length threshold:

| Regime | Evidence available | Default candidates |
|---|---|---|
| Cold, zero history | metadata/hierarchy only | metadata-neighbor analog, parent profile allocation, explicit prior |
| Very young | fewer than one useful season | global LightGBM, Chronos-2 zero-shot, analog blend, naive |
| Young | roughly 1–2 seasonal cycles | global LightGBM, Chronos-2, shrinkage toward peers, limited local models |
| Mature | enough rolling windows and seasons | full AutoGluon portfolio plus custom LightGBM and ensemble |
| Intermittent | long zero runs regardless of age | Croston/SBA, occurrence-size features, global models, WQL/WAPE metrics |

Amazon's documented cold-start approach uses static item metadata to find similar existing series and
derive a forecast from those neighbors
([item metadata](https://docs.aws.amazon.com/forecast/latest/dg/item-metadata-datasets.html),
[cold-start behavior](https://docs.aws.amazon.com/forecast/latest/dg/howitworks-forecast.html)). Adopt
that principle without pretending a zero-history item has target-derived features:

1. Encode confirmed metadata and find neighbors within the same tenant/dataset using mixed-type
   distance or learned embeddings.
2. Build each neighbor's normalized seasonal profile and level/trend priors.
3. Estimate the new item's level from a user-provided prior (planned volume, capacity, price tier),
   parent/category allocation, or peer median.
4. Produce a wide uncertainty interval and label the method `cold_start_analog`.
5. Blend toward the item's own global-model forecast as observations arrive.

If a zero-history item has neither metadata, hierarchy, nor a level prior, the service must return an
`insufficient_cold_start_context` warning or a clearly labeled dataset-level prior. There is no
statistically defensible item-specific forecast to infer from nothing.

Ordinary rolling validation does not test this behavior. Add two special evaluations:

- **launch simulation:** truncate mature items at ages 0, 1, 2, … periods and score their early-life
  forecasts;
- **cohort holdout:** remove entire later-launched items from training, then score metadata-only and
  first-observation forecasts.

Only metadata values that would have existed on the simulated launch date may be used.

## Backtesting, metrics, and uncertainty

Use expanding-window rolling origins with horizon exactly equal to production horizon. Fit all
preprocessing inside each window. Time-series cross-validation requires every test observation to
have only earlier observations in its training set
([Forecasting: Principles and Practice](https://otexts.com/fpp3/tscv.html)). Reserve a final untouched
holdout for the promotion decision when data volume allows.

Default evaluation bundle:

- **Primary probabilistic:** weighted quantile loss (WQL) when high-volume series matter, or scaled
  quantile loss when series should count equally.
- **Point:** WAPE plus MASE/RMSSE; report MAE/RMSE in original units.
- **Operations:** signed bias, interval coverage and width, inference latency, training time, peak
  memory, and failure rate.
- **Segments:** by horizon step, series age, volume, intermittency, category, and hierarchy level.

AutoGluon's metric guidance recommends quantile losses for probabilistic forecasts, scale-dependent
metrics for prioritizing larger series, and scaled metrics when series should count equally; it also
warns that percentage/scaled metrics can be poor for sparse demand
([metric guide](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-metrics.html)). Scaled
errors compare heterogeneous series against a naive training error
([accuracy reference](https://otexts.com/fpp3/accuracy.html)).

Generate quantile forecasts, then calibrate intervals from out-of-fold residuals per horizon and,
where sample size permits, per regime. Persist raw and calibrated quantiles and track empirical
coverage after actuals arrive. Never describe model quantiles as calibrated until coverage has been
measured.

If the user declares a hierarchy, return coherent totals. Start with bottom-up as the transparent
baseline and add MinT when residual covariance is estimable. MinT is designed to minimize total
reconciled forecast variance while satisfying aggregation constraints
([MinT paper](https://robjhyndman.com/publications/mint/),
[reconciliation overview](https://otexts.com/fpp3/reconciliation.html)). Evaluate reconciliation; do
not assume it always improves every node.

## Service architecture

```text
Next.js ──JWT/service token──> FastAPI control plane ──> PostgreSQL metadata
   │                               │                         ▲
   │                               └──jobs──> Hatchet ───────┤
   └──presigned upload────────> S3/MinIO <──artifacts──── CPU/GPU workers
                                   │                         │
                                   └──Parquet datasets───────┘
```

- **FastAPI control plane:** authentication context, validation, idempotency, resource ownership,
  signed upload/download URLs, OpenAPI contracts, and job submission. It never performs long training
  in the request process.
- **Worker plane:** distinct Hatchet tasks route CPU/GPU training and prediction to independently
  scalable worker pools. Jobs are idempotent, retryable, tenant-fair, resource-limited, cancellable
  between stages, and heartbeat-backed.
- **PostgreSQL:** tenants, datasets/versions, schemas, experiments, jobs, model registry, forecast
  requests, metrics, lineage, and audit events.
- **S3-compatible object store:** uploads, normalized Parquet, fold manifests, feature specs, model
  artifacts, predictions, and reports. Do not put large model blobs in PostgreSQL.
- **Hatchet:** PostgreSQL-backed durable scheduling, retries, timeouts, worker recovery, priorities,
  tenant concurrency, and cancellation. It is orchestration state, while the service database is
  the client-facing job projection.

Start as a modular monolith with worker processes. Keep ports/adapters around storage, queue, and model
engines so they can be separated later without creating premature microservices.

## API contract for the Next.js client

Resource-creation mutations accept `Idempotency-Key`; retries must use an identical payload. All
resources are tenant-scoped from the authenticated claim, never from a trusted request-body
`tenant_id`. Long operations return `202 Accepted` and a job URL.

### Dataset lifecycle

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/datasets` | Create draft dataset and upload intent |
| `POST` | `/v1/datasets/{id}/upload-url` | Create short-lived presigned upload URL |
| `POST` | `/v1/datasets/{id}/profile` | Start schema inference and data-quality profiling |
| `GET` | `/v1/datasets/{id}/profile` | Read proposed schema, warnings, and statistics |
| `POST` | `/v1/datasets/{id}/finalize` | Confirm manifest and create immutable version |
| `GET` | `/v1/datasets/{id}/versions/{version}` | Read version, lineage, and readiness |

`profile` is advisory; `finalize` requires explicit mappings. Production uploads go directly to
object storage. A size-limited multipart endpoint may exist only for local development.

### Experiments and models

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/experiments` | Submit feature generation + backtest + training |
| `GET` | `/v1/experiments/{id}` | State, stage, progress, configuration, timestamps |
| `GET` | `/v1/experiments/{id}/leaderboard` | Fold/segment metrics, costs, baseline comparisons |
| `GET` | `/v1/experiments/{id}/report` | Signed evaluation report URL |
| `POST` | `/v1/experiments/{id}/cancel` | Request cooperative cancellation |
| `POST` | `/v1/models/{id}/promote` | Atomically promote a validated model version |
| `GET` | `/v1/models/{id}` | Card, lineage, feature spec, metrics, constraints |
| `POST` | `/v1/models/{id}/retire` | Prevent new forecast requests without deleting lineage |

The experiment request contains dataset version, horizon, quantiles, primary metric, business weights,
compute budget, model policy (`fast`, `balanced`, `high_accuracy`, or explicit allow-list), validation
policy, hierarchy/reconciliation choice, and reproducibility seed.

### Forecasts and actuals

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/forecasts` | Submit batch forecast with context/future covariates |
| `GET` | `/v1/forecasts/{id}` | Status and summary |
| `GET` | `/v1/forecasts/{id}/download` | Signed Parquet/CSV result URL |
| `POST` | `/v1/actuals` | Ingest realized targets for monitoring |
| `GET` | `/v1/models/{id}/monitoring` | Accuracy, bias, coverage, drift, and staleness |

Forecast output is tidy and explicit:

```text
item_id | timestamp | mean | q0.1 | q0.5 | q0.9 | model_version | method | warnings
```

Validate that future covariates cover exactly the requested horizon and items. For new items, require
static metadata and route through the cold-start policy. Return per-item warnings instead of failing
an otherwise valid batch when partial results are explicitly allowed.

### Job state machine and errors

Use `queued → validating → profiling → featurizing → backtesting → training → calibrating → packaging
→ succeeded`, plus terminal `failed` and `cancelled`. Store monotonic stage progress and heartbeat.
Errors follow RFC 9457-style problem details with a stable machine code, message, field/path,
retryability, and correlation ID. Webhooks should be signed and retried; polling remains supported.

## Artifact and lineage contract

Every registered model version stores:

- dataset and normalized-data hashes;
- manifest and FeatureSpec versions;
- source commit, environment lock/container digest, model library versions, seed, and hardware class;
- every backtest cutoff, metric definition, segment result, baseline, and out-of-fold prediction URI;
- fitted model URI/checksum, required input schema, supported horizon/frequency/quantiles, and target
  constraints;
- model card, training logs, feature importance diagnostics, license/use restrictions, and promotion
  decision.

Promote through an atomic alias (`champion`) rather than copying artifacts. Predictions record the
resolved immutable model version.

## Implementation sequence

### Phase 0 — repository foundation (implemented)

- `src/` package layout, app factory, environment settings, health/readiness routes;
- lint/type/test configuration, example environment, comprehensive `.gitignore`;
- research-backed architecture and API plan.

**Exit check:** clean install and quality commands pass on Python 3.12.

### Phase 1 — contracts and persistence (implemented with local durable adapters)

- Pydantic request/response models and generated OpenAPI examples;
- PostgreSQL schema management, repository interfaces, tenant ownership, idempotency records;
- S3/MinIO adapter and presigned upload flow;
- dataset/profile/manifest state machines and job abstraction.

**Exit check:** a Next.js integration test uploads a file, confirms a manifest, polls a job, and can
only access its own tenant's immutable dataset version.

### Phase 2 — ingestion, profiler, and feature engine (implemented)

- CSV/Parquet streaming ingestion with Polars, content hashes, canonical Parquet output;
- duplicate/gap/frequency/timezone checks and data-quality report;
- schema proposal/confirmation UI contract;
- FeatureSpec compiler with cutoff-aware calendar, lag, rolling, metadata, age, intermittent, and
  aggregate features;
- golden tests proving batch/inference parity and synthetic leakage tests.

**Exit check:** inserting arbitrary future target values does not change any feature row at or before
the cutoff; large-file profiling stays within its memory budget.

### Phase 3 — backtesting and custom LightGBM (implemented)

- expanding-window splitter, baselines, metric/segment engine, launch/cohort simulations;
- global direct/recursive LightGBM adapters, quantile models, tuning budget, interval calibration;
- artifact packaging, model card, registry, batch prediction, actuals ingestion.

**Exit check:** reproducible end-to-end run on representative regular, intermittent, multi-series,
and young-series fixtures; champion cannot promote without beating/justifying baseline performance.

### Phase 4 — AutoGluon accuracy lane (adapter implemented; optional dependency)

- isolated CPU/GPU worker image and AutoGluon adapter;
- budget-aware model policies, Chronos-2 option, weighted ensembles, consistent metric conversion;
- adapter-level conversion from canonical Polars/Parquet to `TimeSeriesDataFrame`;
- resource cancellation, timeouts, OOM classification, and graceful candidate failure.

**Exit check:** AutoGluon and LightGBM predictions share one output schema and are compared on identical
cutoffs; failure of one candidate does not lose the experiment.

### Phase 5 — cold start and hierarchy (partially implemented; distributed hardening remains)

- metadata-neighbor/peer profile and level-prior API; age-dependent blending remains;
- bottom-up reconciliation is implemented; independently modeled aggregates and MinT remain;
- signed webhooks, quotas, rate limits, audit log, observability, drift/coverage dashboards;
- canary promotion, rollback alias, retention/deletion policy, load and chaos tests.

**Exit check:** zero-history and truncated-launch tests meet separately declared accuracy/coverage
targets; hierarchy outputs sum exactly; canary rollback does not change immutable lineage.

## Decisions deliberately deferred

- **One universal default metric:** must reflect whether large series, equal series, asymmetric
  under/over-forecast cost, or service levels matter.
- **Automatic resampling and imputation:** unsafe without domain semantics.
- **Fine-tuning foundation models:** zero-shot candidates come first; fine-tuning is justified only by
  repeated backtest gains after accounting for GPU cost and operational complexity.
- **Online per-request feature calculation:** initial production path is batch prediction; introduce
  an online feature store only if latency requirements demonstrate a need.
- **TimesFM 3.0 production use:** blocked by its current pretrained-weight license.

## Research limitations

No representative user dataset, business loss function, forecast cadence/horizon, scale target,
latency SLO, infrastructure provider, or commercial-license policy was supplied. Therefore this plan
defines an adaptive system and measurable decision gates rather than claiming that LightGBM,
AutoGluon, Chronos-2, or any particular feature set will always be most accurate. Final model policy,
fold count, feature windows, and compute presets must be benchmarked on representative tenant data.
