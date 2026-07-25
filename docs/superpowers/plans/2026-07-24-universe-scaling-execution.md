# Universe Scaling Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` (recommended) or inline execution to implement
> this plan task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement one immutable, development-only benchmark that measures
whether expanding the shared training universe improves horizon-13 forecasts,
then execute only after preflight proves complete declared gate coverage.

**Known preflight outcome:** The current frozen inputs are coverage-valid:
fold 0 evaluates 53 stocks, fold 1 evaluates 52, and calibration evaluates 51,
including all 11 unseen-transfer stocks.

**Architecture:** Bind the frozen selection package, four nested manifests,
schema-4 fetch report, exchange calendar, raw-17 configuration, 55 CSVs,
runtime, source tree, commands, and fresh outputs in one armed attempt. A
one-shot driver constructs continuity-safe common-calendar rows, fits every
declared cohort with stock-macro weighting, and publishes exclusive fit and
prediction ledgers. A separately bound finalizer verifies the complete closure
before publishing a development summary and forecast-gate decision.

**Tech Stack:** Python standard library, PyTorch, existing C11 parity runtime,
Massive frozen aggregates, Make, GitButler.

## Global Constraints

- Keep generated data, reports, models, credentials, and caches untracked.
- Use exact training cohorts `11/22/33/55` and transfer cohorts `11/22/33/44`.
- For `unseen-transfer` only, keep ranks `45..55` outside every pooled-model
  gradient and checkpoint-selection decision. The `cohort-scaling` 55-stock
  fit intentionally trains on all ranks `1..55`.
- Use seeds `7/19/31/43/61`, raw OHLCV history `17`, horizon `13`, embargo
  `12`, batch size `128`, two folds, and fold fraction `0.1`.
- Train with equal total weight per stock.
- Use core-11-derived update budgets unchanged for every larger cohort.
- Materialize only train and development-validation rows; test length is
  always zero.
- Do not add policy, test-authorization, replay, or backtest arguments or
  artifacts.
- Local-breadth analysis and expansion beyond 55 stocks are out of scope.
  Either requires a separately armed contract with its own frozen inputs.
- A development pass is not confirmatory evidence and does not authorize
  trading.
- Preserve unrelated `Makefile` and `docs/training.md` working-tree changes.
- Create signed local GitButler checkpoints only; do not push or land.

---

### Task 1: Bind the development attempt and preflight

**Files:**

- Create: `tools/universe_scaling_contract.py`
- Create: `tools/universe_scaling_inputs.py`
- Create: `tools/arm_universe_scaling.py`
- Modify: `tools/experiment.py`
- Modify: `tests/python/test_experiment.py`
- Create: `tests/python/test_universe_scaling_arm.py`
- Create: `tests/python/test_universe_scaling_driver.py`

**Interfaces:**

- Consumes: `FileBinding`, `ExecutableBinding`, `SourceTree`,
  `TorchIdentity`, `freeze_inputs()`, `UniverseManifest.read()`,
  `validate_fetch()`, `universe_roles()`, `common_calendar()`,
  `session_samples()`, `pack_rows()`, and `_prepare_packed()`.
- Produces:

```python
@dataclass(frozen=True)
class TreeBinding:
    root: str
    files: int
    sha256: str


@dataclass(frozen=True)
class ScalingAttempt:
    attempt_path: str
    run_id: str
    run_dir: str
    implementation_commit: str
    selection_tree: TreeBinding
    manifests: tuple[ManifestBinding, ...]
    fetch_report: FileBinding
    session_calendar: FileBinding
    config: FileBinding
    protocol: Mapping[str, object]
    budgets: tuple[tuple[str, UpdateBudget], ...]
    source_tree: SourceTree
    finalizer_tree: SourceTree
    primary_python: ExecutableBinding
    torch_argv: tuple[str, ...]
    torch_probe: TorchIdentity
    environment: Mapping[str, str]
    commands: Mapping[str, tuple[str, ...]]
    outputs: Mapping[str, str]
```

- `TreeBinding` fixes the literal relative selection root, 77-file count, and
  known tree digest. Ordered series, paths, hashes, and row counts are derived
  only from the exact hash-bound schema-4 fetch report; the attempt does not
  duplicate them.
- Parser:
  `ScalingAttempt.read(snapshot: Path, logical_path: Path,
  repository_root: Path) -> ScalingAttempt`
- Command order is exactly `validate`, `preflight`, `calibrate`, `analyze`,
  then the trusted `finalizer_prefix`.
- Outputs are exactly `fits`, `predictions`, `summary`, and `outcome`.

- [ ] **Step 1: Write synthetic red contract tests**

Construct one exact attempt fixture and require rejection of:

```python
mutations = (
    "selection byte, root, count, digest, or package member",
    "manifest hash, order, or prefix",
    "fetch schema, policy, calendar binding, or coordinated series mutation",
    "raw-17 configuration",
    "cohort, transfer, unseen, seed, phase, or budget order",
    "runtime, source, finalizer, command, environment, or parent identity",
    "attempt/output path, input collision, or output topology",
    "symlink, hardlink alias, existing output, extra output, or forbidden token",
)
```

Add boundary fixtures immediately before, on, and after each half-open block
edge. Assert exact `as_of`, entry, and target rows, the 12-opportunity embargo,
zero prepared test rows, and rejection when a lookback, entry, or target
crosses a missing expected-session bin.

Keep the trust boundary explicit: parser tests own canonical schema, literal
bindings, commands, and path topology; armer/executor tests re-observe live
source trees, runtimes, file identities, directory membership, and aliases.
Do not hardcode a machine runtime merely to make the parser reject it.

The valid fixture must prove:

```python
assert attempt.training_cohorts == (11, 22, 33, 55)
assert attempt.transfer_cohorts == (11, 22, 33, 44)
assert attempt.unseen_ranks == tuple(range(45, 56))
assert dict(attempt.budgets) == {
    "fold-0": UpdateBudget(34_992, 128, 100, 274, 27_400),
    "fold-1": UpdateBudget(41_042, 128, 100, 321, 32_100),
    "calibration": UpdateBudget(47_092, 128, 100, 368, 36_800),
}
assert all("test" not in command for command in attempt.commands.values())
```

- [ ] **Step 2: Run the contract test and verify it is red**

Run:

```zsh
PRIMARY=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3
"$PRIMARY" tests/python/test_universe_scaling_driver.py
```

Expected: the new arming, topology, and empty-added-validation assertions fail
before their implementation.

- [ ] **Step 3: Implement the smallest exact parser**

Reuse the hardened immutable-file types from `tools.panel_contract`; do not
parameterize panel profiles or move security-sensitive filesystem code.
`ScalingAttempt.read()` must accept one canonical schema and no optional
fields. Fix the selection tree and every direct file binding in code; do not
let the attempt self-declare series or selection digests. Validate all ordered
arrays as tuples and compare the entire protocol to fresh immutable constants:

```python
TRAINING_COHORTS = (11, 22, 33, 55)
TRANSFER_COHORTS = (11, 22, 33, 44)
UNSEEN_RANKS = tuple(range(45, 56))
SEEDS = (7, 19, 31, 43, 61)
MODES = ("fixed-update", "fixed-epoch")
MODELS = (
    "zero",
    "global_ridge",
    "global_mlp",
    "panel_transformer",
    "conditioned_panel_transformer",
    "local_transformer",
)
PHASES = ("fold-0", "fold-1", "calibration")
```

The environment is exactly:

```python
{
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPYCACHEPREFIX": f"{run_dir}/.pycache",
}
```

No inherited `MASSIVE_API_KEY` is part of the child environment.
The runner authenticates its adjacent `tools` package and exact CPython
launch before repository imports. Primary stages use `-I -S -B`; calibration
also uses `-I -S -B`, then exposes only the attested Torch package parent
after the executable, package tree, and source closure validate, replacing the
repository import root before Torch loads.

- [ ] **Step 4: Implement immutable arming and preflight**

`arm()` freezes every direct binding, discovers and freezes every source,
runtime, selection-package member, and CSV named by the bound fetch report,
validates the exact selection, manifest, fetch, calendar, configuration, and
sample counts, then publishes the canonical attempt with
`write_json_exclusive()`.
Resolve input and output paths without following aliases; reject input/output
overlap, existing outputs, changed directory membership, and any attempt path
other than `experiments/<run-id>-attempt.json`.

Preflight derives opportunities from the expected exchange grid, never from a
stock's sparse observed rows:

```python
calendar = common_calendar(5_505, 2, 0.1, 12)
assert calendar.folds == (
    (IndexRange(0, 3_293), IndexRange(3_305, 3_843)),
    (IndexRange(0, 3_843), IndexRange(3_855, 4_393)),
)
assert calendar.holdout[:2] == (
    IndexRange(0, 4_393), IndexRange(4_405, 4_943),
)
assert calendar.holdout[2] == IndexRange(4_955, 5_505)
```

First extend `_prepare_packed()` narrowly: require nonempty training, permit
empty development validation, keep test empty, and check the train/validation
embargo only when validation exists. Add focused regression tests for both
counts. Then derive every stock/phase member:

```python
packed = pack_rows(samples, blocks, 17, 13, 13)
coverage[(series, phase)] = packed.counts
data = _prepare_packed(rows, candidate, packed, 17, sweep)
assert len(data.train) > 0
assert len(data.validation) == packed.counts[1]
assert len(data.test) == 0
```

Every declared-prefix member remains in shared and ridge gradients. Every
core-11 member must also be phase-evaluable. Sparse added members with zero
validation rows remain explicit coverage misses: exclude them only from
checkpoint loss, predictions, and local-comparator fits. Never invent rows or
silently shrink the training cohort. The reserved block is recorded as
metadata but is never passed to `pack_rows()`.

For the current frozen inputs, preflight must record exact validation coverage
of 53 stocks for fold 0, 52 for fold 1, and 51 for calibration, including the
ordered misses:

```python
{
    "fold-0": ("ALTR", "ZI"),
    "fold-1": ("ALTR", "ZI", "INFA"),
    "calibration": ("ALTR", "ZI", "FYBR", "INFA"),
}
```

Calibration covers all 11 mandatory unseen stocks. Preserve the explicit
added-stock misses without shrinking a denominator or synthesizing rows.

- [ ] **Step 5: Run focused and aggregate checks**

```zsh
"$PRIMARY" tests/python/test_universe_scaling_driver.py
make -B PYTHON="$PRIMARY" check
TORCH=/Users/Enkang.Yuan1/.cache/uv/archive-v0/sVD6zcEJ9T4Luj6l/bin/python
make -B PYTHON="$TORCH" check-training
```

Expected: all tests pass, every prepared test split is empty, and no fit helper
is called by mutation/preflight tests.

- [ ] **Step 6: Create one signed local checkpoint**

Use GitButler to commit only the plan, helper change, contract, armer, and
focused tests with:

```text
feat(training): bind universe scaling preflight
```

---

### Task 2: Execute every pooled development fit

This task begins only after Task 1 produces a coverage-valid armed attempt.
The current frozen inputs satisfy that precondition.

**Files:**

- Create: `tools/run_universe_scaling.py`
- Modify: `tests/python/test_universe_scaling_driver.py`

**Interfaces:**

- Consumes: one `ScalingAttempt`, `_fit_shared_updates()`,
  `_fit_shared_epochs()`, `_fit_neural()`, and
  `stock_macro_linear_model()`.
- Produces exclusive `fits.jsonl` and `predictions.jsonl`.

- [ ] **Step 1: Add red schedule and failure tests**

Require one fit for each unique training identity:

```python
for mode in ("fixed-update", "fixed-epoch"):
    for cohort in (11, 22, 33, 44, 55):
        for phase in ("fold-0", "fold-1", "calibration"):
            for model in ("global_mlp", "panel_transformer"):
                for seed in (7, 19, 31, 43, 61):
                    yield mode, cohort, phase, model, seed
            if cohort in (11, 22, 33, 55):
                for seed in (7, 19, 31, 43, 61):
                    yield (
                        mode, cohort, phase,
                        "conditioned_panel_transformer", seed,
                    )
```

This would yield 420 neural pooled fits. Add 15 seedless `global_ridge` fits, once
per `(cohort, phase)`, and reuse them across modes and questions. Synthesize
analytic zero during analysis; it has no fit or prediction-ledger rows. Fit
local Transformers only for
phase-evaluable stock/phase pairs: current frozen coverage requires exactly
`5 * (53 + 52 + 51) = 780` seeded local fits. Reuse each local fit across
questions, cohorts, modes, and views with the same grid. The complete physical
schedule is therefore 1,215 fits for the current coverage; zero is analytic.

Map unique fits into question views afterward: `cohort-scaling` uses
`11/22/33/55` and all six comparators; `unseen-transfer` uses `11/22/33/44`
and excludes only the conditioned Transformer. Thus the overlapping
11/22/33 fits are references to the same model identity, not retraining. A
conditioned model must never appear in unseen transfer.

Mutation tests must fail before fitting when any phase loses a core-11
validation stock, hides or changes an added-stock coverage miss, changes a
budget with cohort size, repeats or drops a seed, changes a reused control
identity, or changes a paired prediction grid. Assert exact cohort, phase,
model, seed, coverage, and reuse-key closure rather than only an aggregate job
count.

- [ ] **Step 2: Implement one deterministic scheduler**

Use the existing helpers with their actual signatures; map the report label
`global_mlp` to runtime model name `mlp`:

```python
validation_indices = tuple(
    index for index, member in enumerate(members)
    if len(member.validation)
)
model, fit = _fit_shared_updates(
    runtime_model, candidate, members, sweep, seed,
    budget.updates_per_checkpoint, device,
    validation_indices=validation_indices,
)
assert fit.updates_trained == budget.total_updates

model, fit, loaders = _fit_shared_epochs(
    runtime_model, candidate, members, sweep, seed, device,
    validation_indices=validation_indices,
)

ridge = stock_macro_linear_model(members, candidate.ridge)

local, local_fit, local_loaders = _fit_neural(
    "transformer", candidate, data, sweep, seed, device,
)
```

Here `members` is the entire declared training prefix; every member has train
rows and enters shared/ridge gradients. `validation_indices` is the ordered
phase-evaluable subset. Checkpoint loss is stock-macro over every such
validation member, not core-11 only and not row-micro. In unseen transfer,
`members` can contain only the declared `11/22/33/44` training prefix, so
ranks `45..55` cannot influence a pooled fit or its checkpoint.

Use `_fit_neural("transformer", ...)` once per series, phase, and seed on the
same common-calendar data, then reuse it across views. On ranks `45..55` this
is explicitly a local-supervised comparator outside shared-transfer weights
and checkpoint selection; it is not labeled an unseen model. Never create a
second pooled loader, optimizer, or loss reducer.

Controls with an identical fit key are references to one canonical fit record,
not silently repeated training. Never use conditioned series embeddings for
unseen transfer.

Canonical identities omit evaluation-only axes:

```python
pooled_id = (mode, cohort, phase, model, seed, member_ids)
ridge_id = (cohort, phase, member_ids)
local_id = (phase, series, seed)
```

Question and view never enter a fit identity; mode enters only neural pooled
fits. The fixed-epoch sampler makes `sum(len(member.train))` stock-uniform
draws per epoch and retains patience stopping: equal-length members use the
shuffled without-replacement fast path; unequal lengths use weighted
replacement. Label it as a secondary cohort-sized-draw data-plus-compute
curve, not a literal pass over every row.

- [ ] **Step 3: Emit canonical ledgers**

Each physical fit record contains its canonical fit ID, applicable training
axes, ordered member IDs, exact budget, train/validation coverage, selected
checkpoint or epoch, optimizer updates, model fingerprint, and declared
question uses. Reused ridge and local controls retain one canonical provenance
ID; they cannot emit a second conflicting fit. Zero is synthesized by analysis
and has neither a fit nor a prediction-ledger row.

Each prediction record contains:

```python
{
    "schema": 2,
    "provenance_id": provenance_id,
    "model_fingerprint": model_fingerprint,
    "phase": phase,
    "series": series,
    "grid_sha256": grid_sha256,
    "predictions": {
        "encoding": "f32le-base64",
        "count": count,
        "base64": canonical_base64,
    },
}
```

Emit one record per physical fit and required destination in fit-schedule and
manifest order. Inverse each stock's target scaler before encoding finite
binary32 returns. Zero is synthesized by analysis and emits no ledger rows.

- [ ] **Step 4: Implement one-shot process control**

Mirror the tested process-group termination and signal handling from
`run_panel_attempt.py`. Execute bound stages with only the attempt's exact
child environment; finalize once on success and every catchable failure.
Signals received during finalization are deferred until that one finalizer
returns, then reflected in the controller exit. Never resume or reuse an
interrupted run directory.

- [ ] **Step 5: Run checks and checkpoint**

Run the focused driver test, aggregate suite, PyTorch suite, and an independent
read-only review. Commit only runner code and tests as:

```text
feat(training): run universe scaling benchmark
```

---

### Task 3: Finalize, analyze, and enforce forecast gates

This task is conditional on a completed, coverage-valid Task 2.

**Files:**

- Create: `tools/finalize_universe_scaling.py`
- Modify: `tests/python/test_universe_scaling_driver.py`

**Interfaces:**

- Consumes: armed attempt, fit ledger, prediction ledger, terminal stage/code.
- Produces: exclusive `summary.json` and `outcome.json`.

- [ ] **Step 1: Add red finalization tests**

Reject missing, extra, duplicate, reordered, aliased, or mutated artifacts;
nonfinite values; incomplete jobs; mismatched grids; unexpected run-directory
members; changed fit-reuse identity; and any policy/test/backtest field or
file. Prove that only fixed-update calibration rows can affect a gate. Simulate
every stage failure and SIGHUP/SIGINT/SIGTERM window.

- [ ] **Step 2: Build the development summary**

Use only `stock_macro_metrics()`, `cohort_views()`, `unseen_view()`, and
`paired_comparison()`. Average the five neural predictions in return space
before metrics. Stream the prediction ledger once, retain per-series metrics
for every phase and prediction arrays only for calibration, and synthesize
zero on demand. Emit core, added, all, and unseen views with 5-, 10-, and
20-trading-day paired intervals.

```python
views = cohort_views(master, cohort)
unseen = unseen_view(master)
metrics = stock_macro_metrics(points_by_stock)
comparison = paired_comparison(
    candidate_points, reference_points, block_days=(5, 10, 20),
)
```

The locks are exact:

```python
"locks": {
    "reserved_test_materialized_samples": 0,
    "policy_selected": False,
    "backtest_run": False,
    "trading_authorized": False,
}
```

- [ ] **Step 3: Evaluate the frozen forecast gate**

Evaluate all eight gates from fixed-update calibration predictions only.
Fold predictions exist solely to select/audit checkpoints; fixed-epoch results
are a labeled descriptive compute curve and cannot change a gate. Require the
gates from
`docs/superpowers/plans/2026-07-24-universe-scaling-benchmark.md`, including
positive lower paired-interval endpoints at all block lengths, majority unseen
stock improvement, no more than 1% core degradation, all pooled/local control
wins, direction above the stock-macro majority reference, close-MAE
improvement, and a nonnegative 33-to-44 marginal result.

The gated candidate is the unconditioned `panel_transformer`; the conditioned
curve is an architecture comparator and is never projected onto unseen stock
IDs. Missing coverage cannot be dropped from a denominator: failure to produce
the declared core-11 or all 11 unseen comparisons is an automatic gate
failure. `pass` means only that a separately authorized future task may
propose and freeze a cost-aware policy. `gate-failure` keeps every later phase
closed.

- [ ] **Step 4: Run checks and checkpoint**

Run focused failure-path tests, aggregate tests, optional PyTorch tests, and an
independent integrity review. Commit finalizer code and tests as:

```text
feat(training): finalize universe scaling evidence
```

---

### Task 4: Run the frozen development benchmark once

This task is conditional on a coverage-valid armed attempt; the current frozen
inputs are eligible.

**Generated files:**

- Create: one fresh ignored attempt manifest
- Create: one fresh ignored run directory containing `fits.jsonl`,
  `predictions.jsonl`, and `summary.json`
- Create: one fresh ignored terminal `outcome.json`

- [ ] **Step 1: Run a no-metric throughput probe**

Measure one cohort-sized draw epoch of the unconditioned Transformer at cohorts
11 and 55 without retaining predictions or changing any frozen hyperparameter.
Give the probe its own non-evidentiary run ID; none of its state may enter the
armed attempt.

- [ ] **Step 2: Arm one canonical attempt**

Use the verified implementation commit and exact frozen hashes. Confirm `.env`
is not inherited and every output is absent.

- [ ] **Step 3: Execute once**

Run the one-shot driver under the bound primary Python. Preserve every terminal
artifact even when the forecast gate fails.

If process or integrity failure requires another execution, arm a fresh
attempt with a new run ID and label it exploratory. Never overwrite, resume,
merge, or silently substitute it for the first terminal attempt.

- [ ] **Step 4: Audit the result**

Recompute record counts, artifact hashes, job coverage, paired grids, metrics,
intervals, and gate booleans from the frozen ledgers. Treat any disagreement as
an invalid attempt, not a model result.

- [ ] **Step 5: Record the development conclusion**

If a gate fails, record the observed control failure without opening the
reserved test. If every gate passes, record only
`eligible_for_separate_policy_authorization: true`; do not select or freeze a
policy in this plan.

---

## Later authorization boundary

This implementation ends after the development conclusion. Opening the
reserved test, selecting or freezing a policy, replaying `$100`, expanding the
universe beyond 55, or running the separate local-breadth question each
requires explicit later authorization and a newly hash-bound plan/attempt.
