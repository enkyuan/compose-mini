# Universe Scaling Benchmark Implementation Plan

> **For implementation:** follow this plan task by task. Keep generated market
> data, reports, models, credentials, and caches untracked.

**Goal:** Determine whether increasing the frozen common-stock universe from
11 to 55 improves the horizon-13 `raw-17` forecast, first as a breadth audit of
the existing local models and then as a true shared-model scaling experiment.

**Architecture:** Fetch the largest frozen cohort once. Reuse one 55-stock
local-model calibration for exact 11/22/33/55 prefix summaries because every
current local fit is series-independent and the sweep has one candidate.
Retrain shared models separately at every cohort size because pooled training
changes with the cohort. Compare pooled models on one invariant core-11
evaluation set, and run a second curve that reserves ranks 45 through 55 as
stocks never used for gradients or model selection. Separate fixed-update
scientific comparisons from fixed-epoch practical comparisons. Keep the
reserved historical test and every `$100` policy replay closed until the
forecasting and cost-aware gates pass.

**Tech stack:** C11 runtime, Python standard library, PyTorch, Massive REST
API, Make, GitButler.

---

## Fixed experiment contract

### Inputs

- Selection package:
  `reports/universe-selection-20260724-06`
- Master manifest:
  `reports/universe-selection-20260724-06/manifests/liquid-common-55.json`
- Core cohorts: exact master prefixes `11`, `22`, `33`, and `55`
- Unseen-stock training cohorts: exact prefixes `11`, `22`, `33`, and `44`
- Unseen-stock evaluation: fixed ranks `45..55`, excluded from every gradient
  and model-selection decision
- Bars: adjusted, regular-session, 30-minute OHLCV
- Dates: `2024-11-01` through `2026-07-21`
- Target: `executable-return-v1`

\[
y_{i,t} =
\log\left(\frac{\operatorname{close}_{i,t+13}}
{\operatorname{open}_{i,t+1}}\right)
\]

- History: 17 completed bars
- Embargo: 12 bars
- Development protocol: two expanding walk-forward folds, five fixed seeds
- Candidate: the existing `raw-17` configuration only
- Evaluation unit: one stock, regardless of row count

Do not change the universe, dates, target, feature set, horizon, folds, seeds,
model dimensions, optimizer, or thresholds after observing results.

### Observed-bar gap contract

Massive omits an aggregate interval when no qualifying trade occurs. A
30-minute aggregate is therefore an observed bar with a 30-minute bin width,
not a guarantee that every wall-clock bin exists.

For this benchmark:

- retain strictly ordered, aligned regular-session bars that Massive observed;
- never synthesize, forward-fill, or interpolate OHLCV;
- record every missing internal interval and affected session in the fetch
  report;
- define `t+1` and `t+13` over the next observed bars, so the entry price
  remains executable;
- report the elapsed-time distribution from `as_of` to `target_time` by stock;
- support unequal row counts and timestamp grids in shared training;
- align paired comparisons by exact target timestamp.

The first live attempt proved this contract is necessary: CHE had missing
30-minute aggregates in 38 sessions. The strict fetch stopped after writing
only MSTR, ETR, and DTE, and published no report. Treat
`data/liquid-common-55-20260724-01` as an invalid partial attempt and never
reuse it.

### Two distinct questions

1. **Local breadth:** how consistently do independently trained local models
   work across a larger stock cross-section?
2. **Global scaling:** does one shared model improve when its training set grows
   from 11 to 22, 33, then 55 stocks?
3. **Unseen-stock transfer:** does a model trained on 11, 22, 33, then 44
   stocks improve on the same 11 stocks that never entered training?

The first question permits slicing one 55-stock local run. The second requires
four separate shared-model fits. The third requires four more unconditioned
shared-model fits; a series-ID embedding has no learned representation for an
unseen stock and is not a valid zero-shot model.

### Loss weighting

Report the stock-macro loss:

\[
L_{\text{macro}}(m) =
\frac{1}{N}\sum_{i=1}^{N}
\frac{1}{T_i}\sum_{t=1}^{T_i}
\ell\left(y_{i,t},\hat y_{m,i,t}\right)
\]

The shared-model training objective must estimate the same quantity. A plain
concatenation is valid only when all `T_i` are equal. Otherwise sample a stock
uniformly and then sample one of its timestamps uniformly, or apply exact
weights `1 / (N T_i)`.

For the primary scaling curve, also hold optimizer updates fixed:

\[
U \approx E\frac{\sum_i T_i}{B}
\]

With fixed epochs `E`, larger cohorts receive more optimizer updates and
therefore more compute. Report two labeled curves:

- **fixed-update:** same update budget at every cohort, primary causal
  comparison of data breadth;
- **fixed-epoch:** same pass count at every cohort, secondary practical
  comparison of data plus compute.

For each split, derive the fixed-update budget from the 11-stock control before
training:

\[
Q_{\text{split}} =
\left\lceil\frac{\sum_{i=1}^{11}T_{i,\text{train}}}{128}\right\rceil,\qquad
U_{\text{split}} = 100Q_{\text{split}}
\]

Evaluate validation every `Q_split` updates, execute all `U_split` updates, and
restore the best validation checkpoint afterward. This holds compute fixed
without allowing early stopping to give cohorts different update counts. The
fixed-epoch curve retains the existing patience-10 behavior.

### Paired comparison

For each baseline `b`, retain the aligned loss differential:

\[
d_{b,i,t} =
\left|y_{i,t}-\hat y_{b,i,t}\right|
-
\left|y_{i,t}-\hat y_{\text{transformer},i,t}\right|
\]

\[
\Delta_b =
\frac{1}{N}\sum_{i=1}^{N}
\frac{1}{T_i}\sum_{t=1}^{T_i}d_{b,i,t}
\]

Positive `Delta_b` favors the Transformer. Resample the same trading-date
blocks across all stocks and both models so the comparison remains paired and
preserves market-wide dependence.

For a shared model trained on cohort `K`, compare its seed-ensemble prediction
with the 11-stock training control on the same evaluation rows:

\[
\bar p_{i,t}^{K} = \frac{1}{S}\sum_{s=1}^{S}p_{i,t,s}^{K}
\]

\[
g_{i,t}^{K} =
\left|y_{i,t}-\bar p_{i,t}^{11}\right|
-
\left|y_{i,t}-\bar p_{i,t}^{K}\right|
\]

Positive `g` means the additional training stocks helped. Use only dates shared
by every compared cohort for both the point estimate and its interval; do not
report a full-sample point estimate beside a common-date interval.

### Cross-sectional effective count

Aggregate `d[b,i,t]` to aligned daily stock columns and estimate covariance
`Sigma_b`. Report:

\[
N_{\text{eff},b} =
\frac{N\,\operatorname{tr}(\Sigma_b)}
{\mathbf{1}^{\mathsf T}\Sigma_b\mathbf{1}}
\]

This is forecast-comparison information. Do not substitute strategy returns.
Treat it as descriptive breadth; do not multiply it by a temporal effective
sample size or use it to narrow the block-bootstrap interval.
Under equal variance and pairwise correlation `rho`, it reduces to:

\[
N_{\text{eff}} = \frac{N}{1 + (N-1)\rho}
\]

The formula explains why adding correlated stocks can have sharply diminishing
returns.

### Invariant core evaluation

For every shared-model cohort `N`, report:

- `core-11`: predictions for the same first 11 stocks and common target dates;
- `added`: predictions for stocks `12..N`;
- `all-N`: equal-stock macro metrics across the whole cohort.

Only the core-11 curve isolates the effect of added training stocks. The
all-N curve changes both the training data and the evaluation population.

Separately, evaluate each unconditioned model trained on `11/22/33/44` against
the same ranks `45..55` and common target dates. Report the majority of
held-out stocks improved, not only their aggregate mean. Keep selection and
threshold tuning entirely outside this held-out-stock set.

### Forecast gate

The shared Transformer may advance only if, on development calibration:

1. its held-out-stock macro return MAE improves by at least `1%` relative to
   the 11-stock shared-training control;
2. the paired expansion-gain interval versus that control has a strictly
   positive lower endpoint at 5-, 10-, and 20-trading-day blocks;
3. a majority of the 11 held-out stocks improve;
4. its core-11 macro return MAE degrades by no more than `1%`;
5. on core and held-out views, it remains below the zero-return, global linear,
   global MLP, and corresponding local Transformer baselines;
6. its direction accuracy exceeds the stock-macro majority-sign reference;
7. its mean reconstructed-close MAE improves on the zero-return reference;
8. the 33-to-44 unseen-stock marginal result is not negative.

These are development gates, not independent test evidence.

### Trading lock

Do not select a policy, open the reserved test, or replay `$100` while any
forecast gate fails. If all gates pass, freeze one cost-aware policy whose
entry condition is:

\[
\widehat{\mathbb E}[r\mid X]
>
\text{spread}+\text{slippage}+\text{fees}
+\lambda\widehat{\sigma}[r\mid X]
\]

Then freeze one policy before opening the reserved test. Its backtest must use
one shared `$100` account across all stocks, total gross exposure no greater
than one, fractional shares, and no unintended overlapping positions. Existing
per-stock `$100` replays are diagnostics, not one deployable portfolio.

---

## Task 1: Support audited observed-bar gaps

**Files:**

- Modify: `tools/fetch_massive.py`
- Modify: `tools/fetch_universe.py`
- Modify: `tests/python/test_massive.py`

### Step 1: Add failing gap-audit tests

Require:

- the existing single-ticker strict mode still rejects an internal gap;
- universe mode retains only observed regular-session bars;
- no synthetic timestamp or OHLCV row is created;
- each gap records its session, left timestamp, right timestamp, and number of
  absent bins;
- gap counts and affected-session counts are exact and deterministic;
- misaligned, duplicated, reversed, pre-market, or after-hours bars still
  fail or filter exactly as before;
- the fetch report binds the explicit `retain-observed-bars` policy and exact
  gap audit;
- a gap-audit mutation invalidates analysis provenance.

### Step 2: Implement one shared session scanner

Scan and validate regular-session bars once. Keep strict rejection as the
default for `tools/fetch_massive.py`. Let `tools/fetch_universe.py` request the
audited observed-bar result explicitly. Do not duplicate timezone, session, or
gap arithmetic between the two callers.

### Step 3: Run focused and aggregate tests

```zsh
PRIMARY=/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
"$PRIMARY" tests/python/test_massive.py
make -B PYTHON="$PRIMARY" check
```

Expected: all downloader, provenance, C, and Python suites pass.

---

## Task 2: Fetch and validate the 55-stock panel once

**Generated files:**

- Create: `data/liquid-common-55-20260724-02/*.csv`
- Create: `reports/liquid-common-55-20260724-02-fetch.json`

### Step 1: Confirm fresh ignored targets

Verify `.env` is nonempty without printing it. Confirm both output paths are
absent and that `.gitignore` excludes `.env`, `data/`, `reports/`, `models/`,
and Python caches.

### Step 2: Fetch through the bound system trust store

Run:

```zsh
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with truststore \
  python -c 'import runpy, truststore; truststore.inject_into_ssl(); runpy.run_path("tools/fetch_universe.py", run_name="__main__")' \
  reports/universe-selection-20260724-06/manifests/liquid-common-55.json \
  data/liquid-common-55-20260724-02 \
  reports/liquid-common-55-20260724-02-fetch.json \
  --requests-per-minute 5
```

Do not disable TLS. Do not retry into a path containing a partial attempt.

### Step 3: Validate the result

Require:

- 55 CSVs in exact manifest order;
- one report record per ticker;
- every recorded CSV hash and row count matches;
- all timestamps are strictly chronological;
- no timestamp duplication;
- every internal gap appears in the bound audit;
- elapsed horizon durations are finite and ordered;
- train, calibration, and reserved-test blocks remain nonempty after embargo;
- no API key byte sequence appears in the output directory or report.

Do not commit generated artifacts.

---

## Task 3: Add a pure scaling-metrics module

**Files:**

- Create: `tools/universe_scaling.py`
- Create: `tests/python/test_universe_scaling.py`

### Step 1: Write failing tests

Cover:

- exact nested prefixes `11/22/33/55`;
- rejection of reordered, missing, duplicated, or non-prefix members;
- stock-macro versus row-micro loss on unequal series lengths;
- paired absolute-error deltas with positive-is-better orientation;
- seeded common-date circular block bootstrap;
- robustness across 5-, 10-, and 20-trading-day blocks;
- forecast-loss `N_eff` from a hand-computed covariance matrix;
- constant, nonfinite, unaligned, or underidentified covariance inputs;
- core, added, and all-stock partitions;
- explicit train-stock and unseen-evaluation-stock roles;
- deterministic output ordering.

Run:

```zsh
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tests/python/test_universe_scaling.py
```

Expected: fail because `tools/universe_scaling.py` does not exist.

### Step 2: Implement only the tested pure functions

Keep parsing, integrity validation, training, and filesystem writes out of this
module. Reuse one macro reducer for MAE, direction, and paired deltas. Reuse one
aligned daily matrix for bootstrap and `N_eff`; do not create parallel metric
implementations.

### Step 3: Re-run the focused test

Expected: pass.

---

## Task 4: Add a frozen local-breadth analyzer

**Files:**

- Create: `tools/analyze_universe_scaling.py`
- Modify: `tests/python/test_universe_scaling.py`

### Step 1: Add synthetic red tests

Build a small generated fixture representing a one-candidate 55-stock local
calibration. Require the analyzer to reject:

- a selection report not bound to the current 55 manifest;
- a fetch report with wrong order, path, hash, row count, or request contract;
- a config other than the exact `raw-17` local protocol;
- an experiment whose series, source hashes, run counts, split boundaries,
  model grid, seed grid, or calibration-only status differs;
- a ledger with missing, extra, duplicated, reordered, or mismatched records;
- any test prediction or policy/backtest artifact;
- source or file-identity mutation during analysis;
- an existing or nested output path.

Require a valid fixture to emit one canonical report containing:

- direct hashes for selection, manifest, fetch, config, experiment, and ledger;
- per-stock and stock-macro metrics for `11/22/33/55`;
- paired loss deltas, win/tie/loss counts, block intervals, and forecast
  `N_eff`;
- marginal `11->22`, `22->33`, and `33->55` changes;
- the explicit label `local-breadth-not-shared-training`.

### Step 2: Implement the CLI

Use:

```zsh
python tools/analyze_universe_scaling.py \
  SELECTION MANIFEST FETCH CONFIG EXPERIMENT LEDGER OUTPUT
```

Freeze every input before parsing. Recheck byte hashes, file identities, and
exact directory membership immediately before exclusive output publication.
Reuse `tools.files`, `tools.fetch_universe.UniverseManifest`, and the pure
functions from Task 3.

### Step 3: Run focused and aggregate gates

```zsh
PRIMARY=/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
"$PRIMARY" tests/python/test_universe_scaling.py
make -B PYTHON="$PRIMARY" check
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch \
  python tests/python/test_training.py bin/transformer
```

Expected: all standard suites pass and C/PyTorch parity stays within the
existing tolerance.

---

## Task 5: Run one local 55-stock calibration

**Generated files:**

- Create: `reports/h13-universe-scaling-local-20260724-01/experiment.json`
- Create: `reports/h13-universe-scaling-local-20260724-01/calibration.jsonl`
- Create: `reports/h13-universe-scaling-local-20260724-01/analysis.json`

### Step 1: Build series arguments from verified data

Read manifest order, verify every path/hash against the fetch report, then
construct:

```text
TICKER=data/liquid-common-55-20260724-02/<ticker-lower>-30m.csv
```

Do not hand-maintain the 55 arguments.

### Step 2: Run the exact local sweep once

```zsh
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch python \
  tools/experiment.py \
  experiments/executable-h13-universe.example.json \
  reports/h13-universe-scaling-local-20260724-01/experiment.json \
  "${SERIES_ARGS[@]}" \
  --device cpu \
  --calibration-only \
  --calibration-predictions \
    reports/h13-universe-scaling-local-20260724-01/calibration.jsonl \
  --max-runs 2145
```

Expected local run counts:

| Stocks | Validation | Calibration | Total |
|---:|---:|---:|---:|
| 11 | 286 | 143 | 429 |
| 22 | 572 | 286 | 858 |
| 33 | 858 | 429 | 1,287 |
| 55 | 1,430 | 715 | 2,145 |

One 55-stock run saves 2,574 of the 4,719 fits required by four independent
local runs. This reuse is valid only because there is one candidate and every
local model is fitted per series.

### Step 3: Analyze exactly once

Invoke Task 4's analyzer into a fresh output. Treat any integrity error as an
invalid attempt. Do not derive a shared-training conclusion from this report.

---

## Task 6: Add stock-macro shared-model training

**Files:**

- Modify: `tools/experiment.py`
- Modify: `tests/python/test_experiment.py`
- Create: `experiments/executable-h13-universe-panel.example.json`

### Step 1: Add red unequal-length tests

Construct two series with unequal training and validation lengths. Require:

- equal total objective weight per stock;
- exact stock-macro validation loss for early stopping;
- unequal timestamp grids with exact per-stock split boundaries;
- deterministic batch order under each fixed seed;
- exact fixed-update stopping independent of cohort row count;
- a separately labeled fixed-epoch path;
- unchanged behavior when lengths are equal;
- unchanged local-model loaders and metrics;
- no reserved-test authorization for panel models.

### Step 2: Implement the smallest shared loader

Prefer a stock-uniform sampler over duplicating training loops. Preserve the
existing `DataSplits`, `fit_model`, and `fit_epochs` APIs for local callers.
Add only the minimum optional panel behavior required to make training and
validation estimate `L_macro`. Add an explicit update budget; do not infer it
from a cohort-dependent epoch count.

### Step 3: Add exact pooled controls

The shared comparison must contain:

- zero return;
- global ridge linear model;
- global MLP;
- unconditioned panel Transformer;
- series-conditioned panel Transformer;
- corresponding local Transformer.

Do not claim an architectural gain if the Transformer only beats local models
but loses to a simpler pooled control.

### Step 4: Run focused and aggregate gates

Run experiment, panel-analysis, panel-driver, aggregate C/Python, and optional
PyTorch parity suites.

---

## Task 7: Add and run the four-cohort shared benchmark

**Files:**

- Create: `tools/run_universe_scaling.py`
- Modify: `tests/python/test_universe_scaling.py`

### Step 1: Test the frozen driver contract

Require the driver to:

- bind the verified selection package, manifest hashes, fetch report, config,
  source tree, Python, PyTorch, commands, and fresh output directories;
- execute cohorts in fixed `11/22/33/55` order;
- execute unseen-stock training cohorts in fixed `11/22/33/44` order;
- fit shared models separately for every cohort and compute budget;
- evaluate each fit on core-11, added, and all-stock views;
- evaluate only unconditioned models on fixed unseen ranks `45..55`;
- emit distinct fixed-update and fixed-epoch results;
- finalize interruption and every catchable failure without reusing outputs;
- publish one canonical summary only after all inputs and child artifacts are
  reverified.

### Step 2: Run a non-evidentiary throughput probe

Measure one fixed epoch at 11 and 55 stocks without recording forecast metrics.
Use the result only to estimate wall-clock time and disk space. Do not tune
batch size, epochs, or architecture from the probe.

### Step 3: Run the frozen benchmark once

Store all generated outputs under one fresh ignored report directory. Do not
open the reserved historical test.

### Step 4: Decide whether to expand beyond 55

Massive can support larger frozen common-stock manifests. At the Basic
five-request-per-minute limit, a no-pagination fetch needs roughly two
requests per ticker:

| Stocks | Minimum requests | Minimum time |
|---:|---:|---:|
| 55 | 110 | 22 minutes |
| 66 | 132 | 27 minutes |
| 110 | 220 | 44 minutes |
| 220 | 440 | 88 minutes |
| 1,207 | 2,414 | 8.0 hours |

Create new point-in-time manifests for `66`, `110`, or `220` only if the fixed
55-stock benchmark passes and the 33-to-44 unseen-stock marginal gain remains
positive. Sixty-six stocks are the minimum needed to train on 55 while
preserving 11 unseen evaluation stocks. Never append symbols after observing
their model results.

---

## Task 8: Checkpoint verified tracked changes

Run `but diff`. Keep each coherent implementation with its tests. Preserve the
unrelated uncommitted `Makefile` and `docs/training.md` changes.

Create signed local GitButler checkpoints authored and committed by:

```text
enkyuan <yuan.enkng@gmail.com>
```

Verify every checkpoint signature against:

```text
SHA256:mnKfNNJJUV5ELrMXZCwdu6v/PusXBeAi8tWUnRXmWMQ
```

Do not push or land.
