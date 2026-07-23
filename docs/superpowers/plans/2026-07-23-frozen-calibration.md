# Frozen Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select trading policy parameters on a dedicated calibration block and
evaluate the final holdout only after deterministically reproducing the exact
calibrated model states.

**Architecture:** Reserve two chronological blocks after walk-forward model
selection: calibration and test. Choose final neural training epochs from the
earlier selection folds, fit once on all pre-calibration rows without consulting
calibration loss, and hash the fitted state plus its training-only scalers.
Calibration-only and test runs independently reconstruct the same states; a
schema-2 policy authorizes test only when every frozen hash matches.

**Tech Stack:** Python 3.12+, PyTorch, standard-library JSON/SHA-256/struct,
existing C11 runtime, GitButler.

## Global Constraints

- Keep the executable target exactly
  `log(close[t + H] / open[t + 1])`.
- Fit feature and target scalers from retained training rows only.
- Use the existing `H - 1` target-time embargo at every boundary.
- Keep each series as an independent `$100`, long-or-cash, unlevered account.
- Add no dependency and no new model architecture.
- Keep generated CSVs, ledgers, reports, policies, and model files untracked.
- Do not evaluate test labels until a policy containing exact model hashes
  exists.

---

### Task 1: Reserve calibration data and fit fixed-epoch final models

**Files:**
- Modify: `tools/experiment.py`
- Modify: `tools/train.py`
- Test: `tests/python/test_training.py`
- Test: `tests/python/test_experiment.py`

**Interfaces:**
- Consumes: existing `Sweep`, `TrainingData`, `train_epoch()`, and
  `data_loaders()`.
- Produces:
  `walk_forward_splits(samples: int, folds: int, fraction: float,
  reserved_blocks: int = 1) -> tuple[tuple[int, int], ...]`,
  `fit_epochs(model: nn.Module, data: TrainingData, batch_size: int,
  epochs: int, learning_rate: float, weight_decay: float, seed: int,
  device: torch.device) -> tuple[DataLoader, ...]`, and
  `_selected_epochs(records, model, candidate, series, seed) -> int`.

- [ ] **Step 1: Write split and fixed-fit tests**

Add these assertions beside the existing split and checkpoint tests:

```python
assert walk_forward_splits(100, 2, 0.1) == ((70, 10), (80, 10))
assert walk_forward_splits(100, 2, 0.1, 2) == ((60, 10), (70, 10))

model = ForecastTransformer(config)
loaders = fit_epochs(
    model, data, 8, 2, 3e-4, 1e-4, 7, torch.device("cpu"),
)
assert len(loaders) == 3
assert all(torch.isfinite(value).all() for value in model.state_dict().values())
```

Add a selection-record fixture with `best_epoch` values `3` and `7`, then:

```python
assert _selected_epochs(
    records, "transformer", "raw", "TEST", 7,
) == 3
```

`median_low` deliberately chooses the smaller of two middle epochs so the final
fit never trains longer because of an arbitrary rounding rule.

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```sh
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch \
  python tests/python/test_training.py bin/transformer
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch \
  python tests/python/test_experiment.py
```

Expected: imports or calls for `fit_epochs`, the fourth
`walk_forward_splits()` argument, and `_selected_epochs` fail.

- [ ] **Step 3: Extract the minimal fixed-epoch fitter**

Add this next to `fit_model()` in `tools/train.py`:

```python
def fit_epochs(model: nn.Module, data: TrainingData, batch_size: int,
               epochs: int, learning_rate: float, weight_decay: float,
               seed: int, device: torch.device
               ) -> tuple[DataLoader, ...]:
    """Fit a preselected epoch count without reading later split losses."""
    loaders = data_loaders(data, batch_size, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    for _ in range(epochs):
        if not math.isfinite(train_epoch(model, loaders[0], optimizer, device)):
            raise FloatingPointError("training produced a non-finite loss")
    return loaders
```

Change `walk_forward_splits()` in `tools/experiment.py` to:

```python
def walk_forward_splits(samples: int, folds: int, fraction: float,
                        reserved_blocks: int = 1
                        ) -> tuple[tuple[int, int], ...]:
    block = int(samples * fraction)
    initial = samples - (folds + reserved_blocks) * block
    if min(block, initial) <= 0:
        raise ValueError("series is too short for the requested walk-forward folds")
    return tuple((initial + fold * block, block) for fold in range(folds))
```

Import `median_low` and add:

```python
def _selected_epochs(records: Sequence[Mapping[str, object]], model: str,
                     candidate: str, series: str, seed: int) -> int:
    values = [
        int(record["best_epoch"]) for record in records
        if record["model"] == model and record["candidate"] == candidate and
        record["series"] == series and record["seed"] == seed
    ]
    if not values:
        raise ValueError("neural model has no selected epoch evidence")
    return median_low(values)
```

Use `reserved_blocks=2` when constructing the new calibration contract. The
existing default preserves legacy helper tests and makes the new boundary
explicit at its only call site.

- [ ] **Step 4: Run the focused tests**

Run the two commands from Step 2.

Expected: both scripts print their existing `passed` summaries.

- [ ] **Step 5: Create a local checkpoint**

Inspect with `but diff`, then commit only these four files:

```sh
but commit enkyuan/frozen-calibration -c \
  -m "feat(training): reserve calibration holdout" \
  --changes "$CHANGE_IDS"
```

Set `CHANGE_IDS` to the comma-separated file IDs printed by the immediately
preceding `but diff`; GitButler IDs are workspace-generated and must never be
invented.

---

### Task 2: Fingerprint reconstructed model states

**Files:**
- Modify: `tools/experiment.py`
- Test: `tests/python/test_experiment.py`

**Interfaces:**
- Consumes: a fitted PyTorch module, the originating `TrainingData`, and the
  selected candidate.
- Produces:
  `_model_fingerprint(model: nn.Module, data: TrainingData,
  candidate: Candidate) -> str`.

- [ ] **Step 1: Write fingerprint tests**

Add a tiny fitted model fixture and assert:

```python
first = _model_fingerprint(model, data, candidate)
assert first == _model_fingerprint(model, data, candidate)

with torch.no_grad():
    next(model.parameters()).view(-1)[0].add_(1)
assert _model_fingerprint(model, data, candidate) != first

restored = ForecastTransformer(candidate.config())
restored.load_state_dict(state)
changed_scale = replace(data, target_scale=data.target_scale * 2)
assert _model_fingerprint(restored, changed_scale, candidate) != first
```

Also fit the same seed twice with `fit_epochs()` and require identical hashes.

- [ ] **Step 2: Run the experiment test and confirm failure**

Run:

```sh
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch \
  python tests/python/test_experiment.py
```

Expected: `_model_fingerprint` cannot be imported.

- [ ] **Step 3: Add one canonical hash function**

Import `hashlib` and `struct`. Add:

```python
def _model_fingerprint(model: nn.Module, data: TrainingData,
                       candidate: Candidate) -> str:
    digest = hashlib.sha256(json.dumps(
        asdict(candidate), allow_nan=False, separators=(",", ":"),
        sort_keys=True,
    ).encode())
    tensors = (
        ("feature_mean", data.feature_mean),
        ("feature_scale", data.feature_scale),
        ("target_mean", data.target_mean),
        ("target_scale", data.target_scale),
        *sorted(model.state_dict().items()),
    )
    for name, tensor in tensors:
        values = tensor.detach().cpu().float().reshape(-1)
        digest.update(name.encode("ascii") + b"\0")
        for chunk in values.split(16_384):
            items = chunk.tolist()
            digest.update(struct.pack(f"<{len(items)}f", *items))
    return digest.hexdigest()
```

This reuses the artifact writer's bounded 16,384-float chunk size, hashes only
canonical binary32 values, bounds Python-float materialization to one chunk,
and avoids `torch.save` metadata or a new checkpoint format.

- [ ] **Step 4: Run the focused experiment test**

Run the command from Step 2.

Expected: the script prints `experiment tests passed`.

- [ ] **Step 5: Amend the local checkpoint**

Use `but diff`, then amend the new test and implementation hunks into the
unpublished `feat(training): reserve calibration holdout` commit:

```sh
but amend "$COMMIT_ID" --changes "$CHANGE_IDS"
```

Set `COMMIT_ID` from the latest GitButler workspace output and `CHANGE_IDS` from
the immediately preceding `but diff`.

---

### Task 3: Emit calibration ledgers and freeze schema-2 policies

**Files:**
- Modify: `tools/backtest.py`
- Modify: `tools/experiment.py`
- Modify: `tools/select_policy.py`
- Modify: `tests/python/test_backtest.py`
- Modify: `tests/python/test_experiment.py`
- Modify: `tests/python/test_policy.py`

**Interfaces:**
- Consumes: model-selection records, fixed-epoch final models, per-series
  calibration windows, and schema-3 calibration forecast records.
- Produces:
  experiment report schema `6`, forecast ledger schema `3`, policy schema `2`,
  `report["calibration"]`, and `policy["model_fingerprints"]`.

- [ ] **Step 1: Write contract-first tests**

Extend `Forecast.parse()` tests with one valid record:

```python
record = prediction_record | {
    "schema": 3, "split": "calibration", "fold": None,
    "target_kind": EXECUTABLE_RETURN_TARGET,
}
assert Forecast.parse(record).split == "calibration"
```

Reject a calibration record with a non-null fold and a validation record with a
null fold.

Update the policy fixture to schema 2 and require:

```python
"model_fingerprints": [
    {
        "model": "transformer", "series": "TEST",
        "seed": 3, "epochs": 4, "sha256": "3" * 64,
    },
    {
        "model": "transformer", "series": "TEST",
        "seed": 7, "epochs": 6, "sha256": "4" * 64,
    },
],
"calibration_report": {
    "path": "calibration.json", "sha256": "0" * 64,
},
"calibration_prediction_ledger": {
    "path": "calibration.jsonl", "sha256": "1" * 64,
    "source_records": 2, "selected_records": 2,
},
```

Require rejection for duplicate/missing series-seed fingerprints, unsorted
fingerprints, invalid digests, calibration reports with phase other than
`selection-and-calibration`, and ledgers that omit any selected model record.
Use `seed: null, epochs: null` for deterministic models.

Add boundary assertions for the final split:

```python
def as_of(dataset: Windows, offset: int) -> int:
    return feature_lookback(candidate.feature_set) + dataset.start + \
        candidate.seq_len - 1 + offset

assert as_of(data.train, len(data.train) - 1) + data.horizon_bars <= \
    as_of(data.validation, 0)
assert as_of(data.validation, len(data.validation) - 1) + \
    data.horizon_bars <= as_of(data.test, 0)
```

This is the same index arithmetic used by `_prediction_records()` and proves
each earlier block's last target is mature by the next block's first completed
bar.

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```sh
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tests/python/test_backtest.py
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tests/python/test_policy.py
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch \
  python tests/python/test_experiment.py
```

Expected: schema, calibration-phase, boundary, and run-count assertions fail.

- [ ] **Step 3: Extend the ledger without duplicating its fields**

Keep `LEDGER_V2_FIELDS` unchanged and define:

```python
LEDGER_V3_FIELDS = LEDGER_V2_FIELDS
```

Select fields with:

```python
fields = {
    1: LEDGER_V1_FIELDS,
    2: LEDGER_V2_FIELDS,
    3: LEDGER_V3_FIELDS,
}.get(schema)
```

Accept splits with:

```python
if split not in ("calibration", "validation", "test") or \
   split == "validation" and (type(fold) is not int or fold < 0) or \
   split != "validation" and fold is not None:
    raise ValueError("forecast split, fold, or target kind is invalid")
```

Apply the same split/fold rule in `_prediction_records()`, and emit schema 3
for every newly written record. Rename the in-memory argument
`validation_prediction_records` to `calibration_prediction_records`; fill it
from the final model's calibration loader rather than the selection-fold
loaders.

Allow `run_backtests()` to describe a calibration split as a hypothetical
calibration backtest; keep execution, cost, and `$100` arithmetic unchanged.

- [ ] **Step 4: Build final models without calibration feedback**

In `run_experiment()`:

1. Build selection folds with `reserved_blocks=2`.
2. Select candidates from those folds.
3. Build the existing three-way final split as
   `(pre_calibration, calibration, test)`.
4. For each selected neural model, series, and seed, choose epochs through
   `_selected_epochs()`, call `torch.manual_seed(seed)` immediately before
   constructing the model, call `fit_epochs()` exactly once, hash the state,
   evaluate `data.validation`, and retain that model plus its loaders for a
   possible authorized test.
5. For deterministic models, fit once on `data.train`, hash the state, evaluate
   the same calibration loader, and retain the same object.
6. Store each fingerprint as `(model, series, seed, epochs, sha256)`, using
   `epochs=None` for deterministic models. Sort with the explicit key
   `(model, series, -1 if seed is None else seed)`.
7. Build the complete contract and call the test authorizer. Only after it
   succeeds, evaluate authorized models with the retained `loaders[2]`.
   Recompute `_model_fingerprint()` immediately before each test evaluation and
   require equality with both the retained calibration hash and the policy.

The retained models are the exact objects used for both calibration and test;
there is no second fit. This keeps run count fixed and prevents a separately
trained state from inheriting another state's authorization.

Collapse `expected_runs()` to
`expected_runs(sweep: Sweep, series_count: int) -> int`; both phases perform the
same fits, so mode and authorized-model arguments would be dead distinctions.
Count one final selected fit per model, series, and seed:

```python
selection = _model_runs(sweep, sweep.models) * sweep.folds * \
    len(sweep.candidates)
calibration = _model_runs(sweep, sweep.models)
return series_count * (selection + calibration)
```

`evaluate_test` changes evaluation only, not fitting. Replace the two focused
assertions with `assert expected_runs(sweep, 1) == 25`; the documented
three-series horizon sweep remains exactly 117 fits in both phases.

Retain the existing protocol fields and replace only `phase` and
`test_policy` with:

```python
"phase": (
    "selection-calibration-and-test" if evaluate_test
    else "selection-and-calibration"
),
"calibration_policy": (
    "evaluate test only after exact model and policy reproduction"
    if evaluate_test else "deferred until policy selection"
),
```

Delete the old `test_policy` entry and return schema 6 with:

```python
"validation": validation,
"calibration": calibration,
"model_fingerprints": fingerprints,
"test": test,
```

The calibration fingerprint payload becomes:

```python
payload = {field: report[field] for field in (
    "series", "sweep", "selection", "validation", "calibration",
    "model_fingerprints", "test_contract",
)}
```

Rename the policy field `validation_fingerprint` to
`calibration_fingerprint`; it hashes the complete selection, calibration,
model-state, and test-grid contract above.

- [ ] **Step 5: Freeze schema-2 policy provenance**

In `tools/select_policy.py`, require report schema 6, phase
`selection-and-calibration`, and an empty test list. Select records from
`report["calibration"]`; validate a complete `(series, seed)` grid with
`fold is None` and calibration target boundaries.

In `tools/backtest.py`, define schema-2 policy fields by replacing:

```text
validation_report
validation_prediction_ledger
```

with:

```text
calibration_report
calibration_prediction_ledger
model_fingerprints
```

Require sorted, unique, complete `(series, seed)` entries and lowercase SHA-256
digests. The action, safety grid, costs, candidate, target, series, and test-grid
checks remain unchanged.

Rename the experiment CLI flags to `--calibration-only` and
`--calibration-predictions`. Reject the legacy validation-only flag combination
rather than silently changing its meaning.

In `main()`, pass the calibration list into `write_predictions()` and record:

```python
report["calibration_prediction_ledger"] = {
    "schema": 3, "path": str(args.calibration_predictions),
    "records": len(calibration_predictions),
    "sha256": _sha256(args.calibration_predictions),
}
```

Write schema 3 for the test prediction ledger as well. Include both output
paths in `require_disjoint()` and report them in the final CLI JSON.

- [ ] **Step 6: Run the focused tests**

Run all three commands from Step 2.

Expected: all scripts print their existing `passed` summaries.

- [ ] **Step 7: Create the policy-contract checkpoint**

Inspect with `but diff`, then commit these six files on a branch stacked above
`enkyuan/frozen-calibration`:

```sh
but commit enkyuan/calibration-policy -c \
  -m "feat(training): freeze calibration policy" \
  --changes "$CHANGE_IDS"
```

Set `CHANGE_IDS` to the comma-separated file IDs from the immediately preceding
`but diff`.

---

### Task 4: Require exact reconstruction before test evaluation

**Files:**
- Modify: `tools/experiment.py`
- Modify: `tools/backtest.py`
- Modify: `tests/python/test_experiment.py`
- Modify: `tests/python/test_backtest.py`

**Interfaces:**
- Consumes: schema-2 policies and the reconstructed
  `report["model_fingerprints"]`.
- Produces: test authorization that binds policy hashes, experiment
  fingerprint, selected models, and every series-seed model hash.

- [ ] **Step 1: Write negative authorization tests**

Add policy mutations that change one fingerprint digest, remove one seed, add
an unrequested model, and reorder entries. Each must make `_authorize_test()` or
`validate_test_experiment()` raise `ValueError`.

Add a positive test where two separately reconstructed reports contain the same
fingerprints and the policy authorizes the selected model.

After authorization, mutate one retained parameter and pass that model to the
new `_verify_test_state()` helper. It must raise `ValueError` before
`evaluate()` is called or any test batch is read.

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```sh
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch \
  python tests/python/test_experiment.py
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tests/python/test_backtest.py
```

Expected: altered model fingerprints are still accepted.

- [ ] **Step 3: Bind policy authorization to model states**

In `_authorize_test()`, derive the expected fingerprints for each policy model
and require exact equality:

```python
fingerprints = [
    item for item in contract["model_fingerprints"]
    if item["model"] == model
]
if policy["model_fingerprints"] != fingerprints:
    raise ValueError("test policy does not match reconstructed model states")
```

Return the authorized policies keyed by model, not merely their names. Extract
the pre-inference check so it can be tested independently:

```python
def _verify_test_state(model: nn.Module, data: TrainingData,
                       candidate: Candidate,
                       fingerprint: Mapping[str, object],
                       policy: Mapping[str, object], series: str,
                       seed: int | None) -> None:
    expected = next(
        item["sha256"] for item in policy["model_fingerprints"]
        if item["series"] == series and item["seed"] == seed
    )
    actual = _model_fingerprint(model, data, candidate)
    if actual != fingerprint["sha256"] or actual != expected:
        raise ValueError("test model state does not match its frozen policy")
```

In the test loop, call `_verify_test_state()` on the retained object immediately
before `evaluate()`. Do not call `_fit_neural()`, `fit_epochs()`, or
`_deterministic()` inside that loop. Those calls belong only to the single
final-model construction pass before authorization.

In `validate_test_experiment()`, require schema 6, phase
`selection-calibration-and-test`, and the same selection/calibration
fingerprint used by the policy.

Do not expose a CLI override for epochs, hashes, costs, seeds, safety, or model
selection in policy-bound test mode.

- [ ] **Step 4: Run the focused tests**

Run the two commands from Step 2.

Expected: both scripts print their `passed` summaries.

- [ ] **Step 5: Amend the policy checkpoint**

Use `but diff`, then amend these exact-reconstruction changes into the
unpublished `feat(training): freeze calibration policy` commit:

```sh
but amend "$COMMIT_ID" --changes "$CHANGE_IDS"
```

Use the current commit ID and change IDs printed by GitButler; do not reuse IDs
from an earlier history mutation.

---

### Task 5: Document and benchmark the frozen workflow

**Files:**
- Modify: `docs/training.md`
- Test: `tests/python/test_experiment.py`

**Interfaces:**
- Consumes: schema-6 experiment commands and schema-2 policy files.
- Produces: one reproducible selection/calibration/test procedure and an
  exploratory `$100` comparison.

- [ ] **Step 1: Replace the old two-phase commands**

Document these exact phases:

```zsh
series=(
  AAPL=data/aapl-30m.csv
  MSFT=data/msft-30m.csv
  SPY=data/spy-30m.csv
)

python tools/experiment.py experiments/horizons.example.json \
  reports/executable-h13-calibration.json \
  "${series[@]}" \
  --horizon-bars 13 --target-kind executable-return-v1 --max-runs 117 \
  --calibration-predictions reports/executable-h13-calibration.jsonl \
  --calibration-only

for model in transformer mlp linear; do
  python tools/select_policy.py reports/executable-h13-calibration.json \
    reports/executable-h13-calibration.jsonl \
    "reports/executable-h13-${model}-policy-v2.json" \
    "${series[@]}" \
    --model "$model" --safety-bps 0 3 6 10 \
    --initial-cash 100 --spread-bps 1 --slippage-bps 1 --fee-bps 0
done

python tools/experiment.py experiments/horizons.example.json \
  reports/executable-h13-test-v2.json \
  "${series[@]}" \
  --horizon-bars 13 --target-kind executable-return-v1 --max-runs 117 \
  --predictions reports/executable-h13-test-v2.jsonl \
  --policy reports/executable-h13-transformer-policy-v2.json \
  --policy reports/executable-h13-mlp-policy-v2.json \
  --policy reports/executable-h13-linear-policy-v2.json

for model in transformer mlp linear; do
  python tools/backtest.py reports/executable-h13-test-v2.jsonl \
    "reports/executable-h13-${model}-final-v2.json" \
    "${series[@]}" \
    --policy "reports/executable-h13-${model}-policy-v2.json" \
    --experiment-report reports/executable-h13-test-v2.json
done
```

The three policies must exist and pass validation before the combined test
command runs. The test experiment accepts all three policies together and opens
the holdout once; no model gets an earlier look at test labels.

State explicitly that the present historical test interval is exploratory
because earlier work already inspected it; confirmation requires later data
whose labels were unavailable when the policy hash was registered.

- [ ] **Step 2: Run complete verification**

Run:

```sh
make -B \
  PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  check
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch \
  python tests/python/test_training.py bin/transformer
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch \
  python tests/python/test_experiment.py
```

Expected: all nine C suites, all standard Python suites, artifact parity, and
experiment/policy tests pass.

- [ ] **Step 3: Run calibration, freeze policies, and backtest**

Run the four phases from Step 1: calibration, all three policy freezes, one
combined test, and all three backtests. If any policy freeze fails, stop before
the test command.

Record, without committing generated files:

- calibration winner and safety margin;
- calibration objective, mean equity, turnover, and trade count;
- test return MAE and direction accuracy;
- per-series final equity from `$100`, maximum drawdown, and trades;
- macro mean terminal log growth versus cash, buy-and-hold, always-up, linear,
  and MLP baselines.

Reject any model change that wins only on the already-opened test interval or
loses to the strongest calibration baseline.

- [ ] **Step 4: Commit documentation**

Inspect with `but diff`, then commit the documentation and any command-contract
test on a branch stacked above `enkyuan/calibration-policy`:

```sh
but commit enkyuan/calibration-docs -c \
  -m "docs(training): explain frozen calibration" \
  --changes "$CHANGE_IDS"
```

Set `CHANGE_IDS` from the immediately preceding `but diff`.

- [ ] **Step 5: Verify authorship and signatures**

For every new checkpoint returned by GitButler, run:

```sh
git verify-commit "$COMMIT_SHA"
git show -s --format='%H%n%an <%ae>%n%cn <%ce>%n%G? %GF' "$COMMIT_SHA"
```

Expected: author and committer are
`enkyuan <yuan.enkng@gmail.com>`, signature status is `G`, and fingerprint is
`SHA256:mnKfNNJJUV5ELrMXZCwdu6v/PusXBeAi8tWUnRXmWMQ`.

Do not push or land these checkpoints.
