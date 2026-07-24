# Shared Panel Transformer Implementation Plan

> **For Codex:** Execute this plan with `subagent-driven-development`. Follow
> `karpathy-guidelines`, `ponytail`, repository `AGENTS.md`, and the local-only
> GitButler checkpoint boundaries below. Do not push or land.

**Goal:** Determine whether sharing one unchanged Transformer across aligned
stocks improves horizon-13 executable-return calibration without opening the
reserved historical test.

**Architecture:** Continue fitting feature and target scalers independently
from each stock's retained training rows. Concatenate only the resulting
scaled window datasets, fit one unconditioned `ForecastTransformer`, and
inverse-scale every stock independently for metrics and prediction ledgers.
Initially require byte-identical timestamp grids so pooled training is
equal-stock weighted and cannot cross market-time boundaries.

**Tech:** Python 3.12+, PyTorch, existing experiment/report pipeline, procedural
Python tests, Make, GitButler. The armed attempt binds the full local source
closure and exact Python/uv/Torch runtime identity, not only entrypoint files.

**Delivery boundary:** Four signed local branches: reviewed plan,
implementation, armed attempt, and calibration evidence. Generated reports
remain ignored. Do not fetch data, access credentials, inspect reserved test
labels, authorize a policy, change Artifact V1, push, pull, land, or open a PR.

## Why This Experiment Comes Next

The admissible AAPL/MSFT/SPY horizon-13 reports are calibration-only and have
empty `test` arrays. They show:

| Metric | Transformer | MLP | Linear | Zero return |
| --- | ---: | ---: | ---: | ---: |
| Validation return MAE | 0.00960146 | 0.00957891 | 0.01011848 | 0.00962274 |
| Calibration return MAE | 0.01193267 | 0.01166609 | 0.01189875 | 0.01149941 |
| Calibration direction | 46.62% | 49.66% | 46.87% | flat prediction |

These macro means come from the bound calibration-only report. The local
Transformer has not established transferable signal; raw-17 also beat the
existing stationary candidate, so a broad feature rewrite is not first.

An inventory query touched historical-test metadata; no test-derived value is
used. Later confirmation requires labels unavailable after this plan.

A read-only pooled-ridge diagnostic used the exact calibration split
`(5157, 633, 645)` and each series' own training-only scalers:

```text
separate ridge macro return MAE = 0.011898754928663192
pooled ridge macro return MAE   = 0.011573988742235775
relative reduction              = 2.729451...%

separate ridge direction        = 0.46866771985255395
pooled ridge direction          = 0.47393364928909953
```

Pooling helped the linear model but still lost to zero return. This is
diagnostic evidence for shared statistical strength, not promotion evidence;
the macro is the arithmetic mean of the three per-series metrics.

DeepAR supports one model for related series (<https://arxiv.org/abs/1704.04110>);
asset-pricing evidence stresses nonlinear momentum/liquidity/volatility
interactions (<https://www.nber.org/papers/w25398>).

## Locked Mathematical Contract

For stock \(s\), feature \(j\), and forecast as-of/window-end index \(t\),
retain independent training-only scalers. The target index is \(t + H\):

```text
z[s,t,j] = (feature[s,t,j] - feature_mean[s,j]) / feature_scale[s,j]
u[s,t]   = (return[s,t] - target_mean[s]) / target_scale[s]

return[s,t] = log(close[s,t + H] / open[s,t + 1])
H = 13 completed 30-minute bars
```

The panel model has no instrument or sector input. It minimizes:

```text
L(theta) =
  (1 / N) * sum_s (
    (1 / T_s) * sum_t (
      transformer(theta, z[s,t-L+1:t]) - u[s,t]
    )^2
  )
```

The first implementation requires identical timestamps and therefore identical
split sizes for every stock. Ordinary concatenation then implements the
equal-stock objective exactly. If unequal calendars are supported later, use
common target-time cutoffs and explicit weight `1 / (N * T_s)`; do not silently
weight longer histories more heavily.

For each stock, reconstruct predictions with its own scaler:

```text
predicted_return[s,t] =
  predicted_scaled_return[s,t] * target_scale[s] + target_mean[s]

predicted_close[s,t] =
  open[s,t + 1] * exp(predicted_return[s,t])
```

Every input window remains inside one stock. Concatenation occurs between
already-formed datasets, never inside a tensor window.

For every walk-forward and final calibration split, index the common timestamp
grid and require:

```text
last_train_target_index <= first_validation_as_of_index
first_validation_as_of_index =
  first_validation_target_index - target_horizon_bars
```

With horizon 13 and embargo 12, equality is expected: the last training label
is available at the first validation decision time. Test this relationship
directly; timestamp equality alone is insufficient evidence of label
availability.

## Locked Scope

- Add only `panel_transformer`; reuse `ForecastTransformer` without changing
  its parameters, forward pass, or initialization.
- Use raw-17, horizon 13, `executable-return-v1`, folds 2, fraction 0.1,
  epochs 100, patience 10, batch size 128, and seeds 7, 19, 31, 43, 61.
- Compare local Transformer, panel Transformer, MLP, linear, rolling mean, and
  zero return in the same calibration-only run.
- Keep each series' feature and target scalers independent and training-only.
- Require the input series in explicit CLI order and require identical
  timestamp tuples before the first fit.
- Keep total batch size 128. Do not multiply it by the number of stocks.
- Count panel work in series-equivalent runs for the existing compute cap.
- Do not add symbol embeddings, sector embeddings, per-symbol heads, a new
  artifact schema, C runtime behavior, policy support, or test authorization.
- Do not use the blocked Massive recovery directory or create another fetch
  directory.

## Predeclared Input Identities

Task 4 adds `experiments/executable-h13-panel-inputs.json`. It binds this
experiment to exactly:

| Series | Rows | CSV SHA-256 |
| --- | ---: | --- |
| AAPL | 6488 | `a821339ae61f1a7169e2e95f5e221a2c9fdbe8d931aa15b9384c0760d394984c` |
| MSFT | 6488 | `715b8e27a73417271054985d8a5366a603b42068afe70bbe809fed83c5f59709` |
| SPY | 6488 | `486f199189e53134e1385497606b869ee77b4a4632efc94574d8016f116e562b` |

All three start at `2024-07-22T13:30:00Z`, end at
`2026-07-21T19:30:00Z`, and have the same ordered timestamp-stream SHA-256:

```text
fdd3c0e647c5312bac7eca3d2837a83d7d223a4e5931251b6af8c310178588e8
```

The timestamp digest is over each canonical timestamp followed by `\n`, without
the CSV header. Both the experiment preflight and analyzer must validate this
manifest against frozen regular-file inputs. A changed, missing, aliased, or
reordered input stops before fitting.

The same manifest also binds the admissible local comparator:

```text
reports/executable-h13-calibration.json
  SHA-256 0689539de9e7bc0400403b6f7dfe44ead880c11a8fb265d20def48f3e80cfd81
reports/executable-h13-calibration.jsonl
  SHA-256 8e8f1c9e53e1acaec71cc0abcf73fc402c735973037abd6f5b56bff4afeae2c5
```

The live report must reproduce the frozen baseline's local-model validation
records, calibration records, fingerprints, and calibration ledger rows
exactly before panel gates are computed. Compare semantic filtered arrays, not
whole reports: the new sweep and protocol legitimately add the panel model,
input manifest, attempt, and run accounting.

The attempt binds an ordered path/SHA-256 map and tree digest for
`experiment.py`, `panel_contract.py`, `train.py`, `artifact_v1.py`,
`data_v1.py`, `backtest.py`, `files.py`, `float32.py`, `analyze_panel.py`,
`run_panel_attempt.py`, and `finalize_panel_attempt.py`. Unrelated
universe/fetch tooling is not in the transitive panel closure. Hash records
sorted by relative path as
`path + NUL + sha256 + LF`.

Bind resolved primary-Python and uv executable SHA/version, exact Torch argv,
and a Torch probe: resolved Python SHA/version, Torch version/git/CUDA/config,
plus a sorted path/SHA map and tree digest of every nonsymlink regular file
beneath the resolved Torch package. One standard-library helper computes the
identity. Bind `PYTHONDONTWRITEBYTECODE=1` and a declared, absent
`PYTHONPYCACHEPREFIX` under the run directory so no local cache is executable.

## Fixed AAPL/MSFT/SPY Calibration

Add `experiments/executable-h13-panel.example.json` with one exact raw-17
candidate and models:

```text
transformer, linear, mlp, rolling_mean, last_close, panel_transformer
```

Existing three-series work is `117` series-equivalent runs. The panel model
adds:

```text
validation = 3 series * 2 folds * 5 seeds = 30 equivalents
calibration = 3 series * 5 seeds          = 15 equivalents
total                                        45 equivalents
```

The fixed report contract is therefore `162` series-equivalent runs even
though panel weights are physically fit only 15 times.

Use the exact runtime definitions embedded in the Task 7 driver throughout.

The armed attempt fixes `RUN_DIR=reports/h13-panel-20260723-01`. The directory
must be absent when armed and immediately before the process starts. The
calibration command will be:

```zsh
RUN_DIR=reports/h13-panel-20260723-01

"${TORCH[@]}" tools/experiment.py \
  experiments/executable-h13-panel.example.json \
  "$RUN_DIR/experiment.json" \
  AAPL=data/aapl-30m.csv \
  MSFT=data/msft-30m.csv \
  SPY=data/spy-30m.csv \
  --attempt-manifest experiments/executable-h13-panel-attempt.json \
  --input-manifest experiments/executable-h13-panel-inputs.json \
  --baseline-report reports/executable-h13-calibration.json \
  --baseline-ledger reports/executable-h13-calibration.jsonl \
  --device cpu \
  --calibration-only \
  --calibration-predictions "$RUN_DIR/calibration.jsonl" \
  --max-runs 162
```

Run this process exactly once after implementation and independent review.
Do not pass `--predictions`, `--policy`, or any test authorization.

## Predeclared Development Gates

Seed-ensemble calibration predictions are arithmetic means over the exact five
seeds within `(model, stock, as_of, target_time)`.

The implementation is technically valid regardless of model outcome. Promote
panel modeling to a later policy experiment only if all gates pass:

1. Panel macro validation return MAE is strictly below local Transformer, MLP,
   linear, rolling mean, and zero return.
2. The mean paired panel-minus-local Transformer validation MAE delta is
   negative, panel wins at least 20 of 30 `(stock, fold, seed)` pairs, and the
   mean delta is nonpositive for every stock.
3. Panel seed-ensemble calibration macro return MAE is strictly below local
   Transformer, MLP, linear, rolling mean, and zero return.
4. Panel calibration return MAE is strictly below zero return for AAPL, MSFT,
   and SPY separately.
5. Panel calibration macro direction is strictly above the per-stock
   majority-sign macro reference and is at least 50% for every stock.
6. Mean per-stock panel relative close-MAE improvement over zero return is
   strictly positive:

```text
relative_improvement[s] =
  (zero_close_MAE[s] - panel_close_MAE[s]) / zero_close_MAE[s]
```

No threshold changes, alternate seed subsets, policy tuning, or rerun may occur
after observing the output. A gate failure is a valid negative result.

For each stock, define:

```text
p_up   = count(actual > 0) / count(actual)
p_down = count(actual < 0) / count(actual)
p_flat = count(actual = 0) / count(actual)
majority_direction = max(p_up, p_down, p_flat)
direction_correct = sign(prediction) == sign(actual)
```

Validation deltas equal to zero are ties, never wins.

## Task 1: Review and Checkpoint This Plan

**Files:**

- Add: `docs/superpowers/plans/2026-07-23-panel-transformer.md`

Require two independent read-only reviews:

- engineering review: orchestration reuse, timestamp leakage, scaler isolation,
  run accounting, fingerprints, and Artifact V1 boundary;
- experiment review: admissible evidence, paired units, ensemble construction,
  gates, no policy/test access, and no outcome-conditioned rerun.

Resolve every Important finding in this file and re-review. Commit only this
plan on `enkyuan/panel-transformer-plan`, stacked above
`enkyuan/h13-feature-config`:

```text
docs(training): plan shared panel transformer
```

Verify author, committer, and ED25519 signature for
`enkyuan <yuan.enkng@gmail.com>`. Do not push.

## Task 2: Generalize the Existing Training Split Container

**Files:**

- Modify: `tools/train.py`
- Modify: `tests/python/test_training.py`

### Step 1: Add failing tests

Add a small test proving one object containing `train`, `validation`, and
`test` datasets works with:

- `data_loaders`;
- checkpoint-restoring `fit_model`;
- fixed-epoch `fit_epochs`;
- the existing guarantee that fixed-epoch fitting never iterates validation or
  test.
- existing positional `TrainingData` construction and `dataclasses.replace`.

Expected: fail because the functions require a full `TrainingData` type.

### Step 2: Extract the minimum container

Introduce one frozen three-field split container:

```python
@dataclass(frozen=True)
class DataSplits:
    train: Dataset
    validation: Dataset
    test: Dataset
```

Make `TrainingData` inherit those fields and retain all current scaler and
target metadata. Change only the type boundary of `data_loaders`, `fit_model`,
and `fit_epochs` to accept `DataSplits`.

Do not change batches, shuffling, generators, loss, optimizer, early stopping,
checkpoint restoration, scalers, target construction, or artifact training.

### Step 3: Verify

```zsh
"${TORCH[@]}" tests/python/test_training.py
make -B PYTHON="$PRIMARY_PYTHON" check
make PYTHON="${TORCH[*]}" check-training
```

Commit Task 2 with its tests on `enkyuan/panel-transformer`, stacked directly
above the plan:

```text
refactor(training): share chronological split loaders
```

## Task 3: Add Calibration-Only Panel Orchestration

**Files:**

- Add: `tools/panel_contract.py`
- Modify: `tools/experiment.py`
- Modify: `tests/python/test_experiment.py`

### Step 1: Add failing experiment tests

Use small synthetic CSVs and mocks where needed. Require:

- `panel_transformer` is a supported seeded experiment model;
- a panel run requires `--input-manifest`, `--attempt-manifest`,
  `--baseline-report`, and `--baseline-ledger`;
- a CSV/config/baseline mutation after standalone preflight still fails inside
  `experiment.py` before `_fit_neural`;
- a local source, Torch binary, runtime, or cache binding change after arming
  fails before fitting;
- full/test mode rejects a sweep containing `panel_transformer` before any fit;
- two input series with different timestamp tuples fail before any fit;
- every fold and final split proves
  `last_train_target_time <= first_validation_as_of`;
- equal timestamps produce one physical panel fit per candidate/fold/seed, not
  one fit per series;
- prepared panel members retain different per-series scaler tensors;
- concatenated length is the sum of member lengths, and the last stock-A sample
  plus first stock-B sample remain two intact single-stock windows;
- a later-row mutation cannot change training scalers or panel training inputs;
- every panel validation record has the same shared checkpoint epoch for its
  `(candidate, fold, seed)` while retaining its own series metrics;
- final panel calibration fits once per seed at the fold-selected epoch;
- missing, duplicated, or cross-series-divergent shared fold epochs fail before
  final fitting;
- prediction records retain original CLI series order and per-series hashes;
- shared weights produce one fingerprint per series, while changing one
  scaler changes only that series' fingerprint;
- physical fit call count and model-object identity prove sharing; combined
  per-series fingerprints are not treated as proof of shared weights;
- repeated CPU runs are deterministic;
- `expected_runs` returns `162` for the exact three-series config.

### Step 2: Implement strict shared manifests

In `tools/panel_contract.py`, add exact, canonical parsers for:

- `PanelInputs`: declared series order/path/hash/rows/time bounds/timestamp
  digest plus frozen baseline report and ledger identities;
- `PanelAttempt`: armed status, run ID/path, implementation and code hashes,
  source-tree/runtime identities, input/config/baseline hashes, exact argv
  arrays, output paths, and expected equivalent/physical fit counts.

Both expose validation against frozen direct inputs. `PanelAttempt` also
validates the calling stage's exact `sys.argv` and applicable path state.
Keep these contracts independent of PyTorch so the analyzer and finalizer can
reuse them under the primary Python runtime.

Add `--input-manifest`, `--attempt-manifest`, `--baseline-report`, and
`--baseline-ledger` to `experiment.py`. For a panel sweep, require all four.
Freeze attempt, input manifest, sweep, baseline report/ledger, and CSVs
together with the bound source closure. Validate every binding, runtime, and
timestamp identity inside the experiment process before `_candidate_data`,
`_fit_neural`, or optimizer construction, then verify the frozen set before
writing output. A mutation after standalone preflight must still fail.

Record immutable attempt and input-manifest path/SHA-256 provenance in the
panel report. Existing non-panel report fields remain byte-for-byte unchanged.

### Step 3: Implement one reusable panel dataset

Import `ConcatDataset` and add one helper that:

1. accepts prepared `TrainingData` values in CLI order;
2. returns `DataSplits` containing concatenated train, validation, and test
   datasets;
3. rejects empty input;
4. performs no scaling, copying, padding, oversampling, or timestamp logic.

Validate identical timestamp tuples once in `_run_experiment` when a panel
model is requested, before model fitting.

### Step 4: Reuse the local model

Add:

```text
TRANSFORMERS = {"transformer", "panel_transformer"}
PANEL_MODELS = {"panel_transformer"}
NEURAL = TRANSFORMERS | {"mlp"}
```

Generalize `_fit_neural` to construct `ForecastTransformer` for any
`TRANSFORMERS` member and `FlatMLP` only for `mlp`. Reuse `_fit_neural`,
`fit_model`, `_validation_record`, `evaluate`, `mean_loss`,
`_prediction_records`, and `_model_fingerprint`; do not add parallel versions
of their fit, metric, record, or fingerprint logic.

Keep local model loops unchanged except that they skip panel models. Add one
panel validation orchestrator that prepares each series independently, pools
its splits, calls the generalized `_fit_neural` once, and emits the existing
record shape once per series.

Add one panel calibration loop that:

- selects the existing candidate through current validation aggregation;
- uses a small `_panel_selected_epochs` helper that groups the duplicated
  validation records by fold, requires all series copies to name one identical
  shared `best_epoch`, and takes `median_low` over the two unique fold epochs;
- fits one shared model per seed on pooled training splits;
- evaluates each stock with its own loader and scaler;
- emits existing calibration records, fingerprints, and prediction records.

Do not retain panel models for policy-authorized test execution. In `main`,
reject a non-calibration panel sweep immediately after `Sweep.read`, before
policy decoding. Defensively repeat the rejection at the start of
`_run_experiment`, before `read_bars`, `_candidate_data`, `test_authorizer`, or
any fit. Tests must prove each downstream operation remains uncalled.

Lock output ordering:

- existing local validation, calibration, and ledger order remains unchanged;
- `series` and `test_contract` retain exact CLI series order;
- panel validation records append in candidate, CLI series, fold, seed order;
- panel calibration records append in CLI series, then seed order;
- panel ledger records append in CLI series, seed, target-time order;
- all fingerprints retain the existing lexical `(model, series, seed)` sort,
  with `None` ordered before integer seeds.

Collect the small panel records in memory and sort only that appended panel
section. Do not reorder existing local-model output.

### Step 5: Verify

```zsh
"${TORCH[@]}" tests/python/test_experiment.py
"$PRIMARY_PYTHON" tests/python/test_universe_analysis.py
make -B PYTHON="$PRIMARY_PYTHON" check
make PYTHON="${TORCH[*]}" check-training
```

Amend Task 3 into the same unpublished implementation checkpoint if it remains
one coherent panel feature. Otherwise use:

```text
feat(training): share transformer across aligned series
```

No tiny fixup commit.

## Task 4: Freeze the Panel Benchmark and Analyzer

**Files:**

- Add: `experiments/executable-h13-panel.example.json`
- Add: `experiments/executable-h13-panel-inputs.json`
- Add: `tools/analyze_panel.py`
- Add: `tools/run_panel_attempt.py`
- Add: `tools/finalize_panel_attempt.py`
- Add: `tests/python/test_panel_analysis.py`
- Add: `tests/python/test_panel_driver.py`
- Modify: `Makefile`
- Modify: `docs/training.md`

### Step 1: Lock the config exactly

Add exact decoded-JSON tests for every config and input-manifest field. Require
models in the declared order, the three predeclared input identities above,
and `expected_runs(sweep, 3) == 162`.

### Step 2: Add a strict calibration analyzer

`tools/analyze_panel.py` has three exact modes:

```text
validate-attempt ATTEMPT INPUTS CONFIG BASELINE_REPORT BASELINE_LEDGER \
  AAPL=CSV MSFT=CSV SPY=CSV
preflight ATTEMPT INPUTS CONFIG BASELINE_REPORT BASELINE_LEDGER \
  AAPL=CSV MSFT=CSV SPY=CSV
analyze ATTEMPT INPUTS CONFIG BASELINE_REPORT BASELINE_LEDGER \
  REPORT LEDGER OUTPUT AAPL=CSV MSFT=CSV SPY=CSV
```

`validate-attempt` statically validates every armed field/hash/argv and exact
real input without counting as an execution-stage preflight. Preflight repeats
those checks at execution time and additionally enforces all declared output
paths are absent.

Preflight freezes the armed attempt, input manifest, config, baseline report,
baseline ledger, three CSVs, and local source closure. It validates exact
attempt status/run ID/source tree/runtime/argv/output absence; exact input
identities and config; and verifies the frozen set before returning. It emits
no file.

Analyze freezes every direct input, requires fresh disjoint output, validates
canonical JSON/JSONL and every direct SHA-256, and rejects:

- any report schema, protocol, sweep, series, record, fingerprint, ledger, or
  ordering mismatch;
- nonempty test results, a test prediction ledger, policy metadata, or test
  authorization;
- any seed other than 7, 19, 31, 43, 61;
- any model/candidate/run grid other than the exact 162-equivalent protocol;
- any mismatch between filtered live local-model records/predictions and the
  frozen baseline report/ledger;
- a changed input during analysis.

Compute only the predeclared gates:

- report validation MAE paired by stock/fold/seed;
- seed-ensemble calibration return MAE and direction;
- per-stock majority-sign direction;
- reconstructed close MAE and zero-return relative improvement.

Output canonical JSON with:

```text
schema, status, inputs, protocol, validation, calibration, gates
```

`inputs` includes direct path/SHA-256 values for the immutable armed attempt,
input manifest, config, frozen baseline report/ledger, live report/ledger, and
CSVs, plus `run_id`.

Exit `0` for valid/pass and `3` for valid/gate-failure. Other nonzero exits are
integrity failures. Do not include policy or projected-return fields.

Import and reuse canonical JSON/JSONL, output-alias, and regular-file identity
helpers from `tools/panel_contract.py`; the panel must not depend on unrelated
universe/fetch branches. Panel-specific exact report validation and metrics
remain local to `analyze_panel.py`.

### Step 3: Test semantics, not only shape

Synthetic tests must cover:

- local versus panel paired signs and exact win count;
- seed averaging before calibration metrics;
- majority reference from unique actual targets, never seed-duplicated rows;
- executable reference open and target close reconstruction;
- strict inequalities at equality boundaries;
- one per-stock failure causing gate failure;
- valid exit 0 versus 3;
- duplicate fields, NaN/infinity, symlinks, aliases, mutation, reordered series,
  wrong hashes, forbidden test/policy fields, and output collisions.
- preflight rejecting any input-manifest mismatch before `_fit_neural` can run.
- `panel_transformer` remains absent from `SEEDED_MODELS`, `POLICY_MODELS`,
  policy parsing, selection, replay, and backtest authorization.
- exact equality between filtered live local-model records/ledger rows and the
  frozen local comparator before any gate is computed.

Register the standard-library analyzer test in `make check`.

### Step 4: Add one unconditional finalizer

`tools/finalize_panel_attempt.py` accepts the immutable attempt manifest, a
fresh tracked outcome path, start/end times, final stage, exit code, and
status. It validates exactly one transition:

```text
armed -> preflight-failure
armed -> setup-failure
armed -> experiment-failure
armed -> analysis-integrity-failure
armed -> gate-failure
armed -> pass
```

It trusts only its bound minimal closure (`finalize_panel_attempt.py`,
`panel_contract.py`, `files.py`) and primary-Python identity. It resolves the
attempt, declared outputs, and outcome as disjoint nonsymlink paths; freezes
present files and absent states together; hashes snapshots; then rechecks both
presence and absence immediately before no-clobber publication of canonical
`experiments/executable-h13-panel-outcome.json` with attempt path/hash/run ID,
timing, terminal stage/exit/status, and present/absent SHA-256 for every
declared output. Publish a completed same-directory temporary file with an
exclusive hard link; never use overwrite-capable `os.replace`.

Record current broader source/Torch mismatches on failure paths rather than
rejecting them. For `pass` or `gate-failure`, require report/analysis
provenance that preflight, experiment, and analyzer validated the full bound
closure. A changed trusted finalizer closure remains a hard failure.

Synthetic tests cover every transition, reject inconsistent
stage/exit/status combinations, aliases, symlinks, file mutation,
absent-to-present races, and outcome collisions.

### Step 5: Document only current behavior

Update `docs/training.md` to explain:

- local versus panel objectives;
- independent training-only scalers;
- identical-grid restriction;
- series-equivalent compute accounting;
- experiment-only and calibration-only status;
- no embeddings or Artifact V1 changes;
- predeclared gates and exact command.

Do not record a result before the one run.

### Step 6: Verify and checkpoint

```zsh
"$PRIMARY_PYTHON" tests/python/test_panel_analysis.py
"${TORCH[@]}" tests/python/test_experiment.py
make -B PYTHON="$PRIMARY_PYTHON" check
make PYTHON="${TORCH[*]}" check-training
```

Commit only Task 4 implementation/config/docs on
`enkyuan/panel-transformer`, keeping tests with behavior. Verify the complete
implementation diff and signature. Do not push.

## Task 5: Independent Implementation Review

Require two read-only reviews of the exact implementation checkpoint:

- code review: minimal reuse, no model duplication, split isolation, timestamp
  boundary, determinism, report/fingerprint compatibility, compute scaling;
- experiment/security review: no test/policy path, exact config and gates,
  canonical/frozen inputs, output freshness, no generated or credential files.

Fix Important findings by amending the owning unpublished checkpoint, rerun all
gates, and re-review.

## Task 6: Arm One Durable Attempt

**Files:**

- Add: `experiments/executable-h13-panel-attempt.json`

Only after implementation review, create a canonical attempt manifest on
`enkyuan/panel-transformer-attempt`, stacked directly above the implementation.
It binds:

```text
schema = 1
run_id = h13-panel-20260723-01
status = armed
run_dir = reports/h13-panel-20260723-01
implementation commit SHA
input-manifest path and SHA-256
config path and SHA-256
baseline report and ledger paths and SHA-256
ordered local source path/SHA-256 map and deterministic tree SHA-256
minimal finalizer source tree SHA-256
primary-Python and uv path/SHA/version identities
exact Torch argv and full-package Python/Torch probe identity
bound no-bytecode/cache-prefix environment
exact argument-vector arrays for validation, preflight, experiment, and
analyzer; exact finalizer template with typed timing/stage/exit/status slots
expected series-equivalent runs = 162
expected physical panel fits = 15
expected output relative paths
fresh tracked outcome path = experiments/executable-h13-panel-outcome.json
```

Require the run directory and every output to be absent before committing.
The synthetic `test_panel_analysis.py` contract tests from Task 4 cover every
attempt field and invalid binding. Run `analyze_panel.py validate-attempt` with
the real manifest and the same arguments as Task 7 preflight before committing;
it validates every bound file/hash and argument vector without consuming the
one execution-stage preflight. Commit:

```zsh
"$PRIMARY_PYTHON" tools/analyze_panel.py validate-attempt \
  experiments/executable-h13-panel-attempt.json \
  experiments/executable-h13-panel-inputs.json \
  experiments/executable-h13-panel.example.json \
  reports/executable-h13-calibration.json \
  reports/executable-h13-calibration.jsonl \
  AAPL=data/aapl-30m.csv \
  MSFT=data/msft-30m.csv \
  SPY=data/spy-30m.csv
```

```text
chore(training): arm shared panel calibration
```

After this checkpoint, do not change implementation, config, input manifest,
attempt manifest, CSVs, commands, seeds, thresholds, or output paths before
the process. If any bound value changes, the execution preflight records
`preflight-failure`; do not rewrite or re-arm it.

## Task 7: Run One Calibration and Record Evidence

Run the tracked standard-library driver once:

```zsh
/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3 \
  tools/run_panel_attempt.py
```

Preflight, experiment, and analyzer each run at most once. The finalizer runs
exactly once for normal exits, exceptions, `SIGHUP`, `SIGINT`, and `SIGTERM`.
It independently requires analyzer exit/status agreement for `0/pass` and
`3/gate-failure`. An unexpected analyzer exit is recorded as
`analysis-integrity-failure`; it never bypasses finalization. `SIGKILL` and
host loss are not catchable; if either leaves no outcome, do not rerun any
stage—preserve all files and request user direction. Apply the same rule if
the finalizer itself fails.

Append concise evidence to `docs/training.md`:

- date, run-directory basename, sanitized command, direct input/output hashes;
- exact 162-equivalent protocol and physical panel fit count;
- panel/local/baseline validation and calibration metrics;
- paired wins and per-stock results;
- each gate and final status;
- explicit calibration-only/no-test/no-policy language.

On a preflight, setup, or experiment failure, omit nonexistent model metrics
and state that no analyzer or policy ran. The immutable committed armed
manifest plus tracked outcome must preserve every failed attempt; do not
delete, replace, modify, or reuse its run directory.

Do not select or replay a `$100` policy in this checkpoint. A later reviewed
policy experiment is allowed only if every forecast gate passes. Existing
`$100` evidence remains calibration resubstitution and is not projected.

Run:

```zsh
"$PRIMARY_PYTHON" tests/python/test_panel_analysis.py
"${TORCH[@]}" tests/python/test_experiment.py
make -B PYTHON="$PRIMARY_PYTHON" check
make PYTHON="${TORCH[*]}" check-training
```

Commit only the new outcome manifest and `docs/training.md` on
`enkyuan/panel-transformer-evidence`, stacked directly above
`enkyuan/panel-transformer-attempt`:

```text
docs(training): record shared panel calibration
```

Verify enkyuan author/committer and the expected ED25519 signature. Do not
commit the report, ledger, analysis, CSVs, models, credentials, or bytecode.
Do not push or land.

## Final Audit

- [ ] Plan, implementation, armed-attempt, and evidence branches are signed
      and local.
- [ ] The Massive blocked and failed directories are unchanged and ignored.
- [ ] A panel input mismatch fails before fitting.
- [ ] Every scaler uses only its stock's retained training data.
- [ ] No window crosses stocks.
- [ ] One shared Transformer is fit per panel fold/seed.
- [ ] Panel compute is capped as series-equivalent work.
- [ ] Existing local models and Artifact V1 are unchanged.
- [ ] Full/test and policy paths reject panel models.
- [ ] The calibration process and analyzer each run at most once.
- [ ] The committed armed attempt binds inputs, code, command, and output path.
- [ ] Every success or failure has one committed immutable outcome manifest.
- [ ] No reserved test prediction, policy, or authorization exists.
- [ ] Gates are exact, predeclared, and unchanged after output.
- [ ] A gate failure is documented as a valid negative result.
- [ ] Generated artifacts and credentials remain untracked.
- [ ] Push, pull, PR, and landing remain absent.
