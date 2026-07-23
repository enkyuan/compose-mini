# Horizon-13 Feature Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine, without reopening historical test labels, whether
`stationary-v1` fixes the raw-price scaling drift in the horizon-13
executable-return Transformer.

**Architecture:** Run one paired, Transformer-only calibration experiment over
AAPL, MSFT, and SPY. Keep the raw 17-bar control and the stationary 16-token
candidate on the same 17-bar source footprint, then require both validation and
cost-aware calibration-policy gates before promotion.

**Tech Stack:** Python 3.12+, PyTorch, strict JSON, existing experiment,
policy-selection, and backtest tools, GitButler.

## Evidence and Constraints

- `reports/features.json` is legacy schema 2: it omits the target contract,
  used the default horizon-1 close return, and selected raw-17. It cannot answer
  the horizon-13 executable-return question.
- The frozen raw-17 exploratory test produced macro terminal log growth
  `-0.0032079332`, mean equity `$99.680745`, and 24 trades, all on MSFT.
  AAPL and SPY had zero execution coverage.
- Raw-price z-scores for AAPL and SPY moved beyond their training ranges.
  This motivates the representation ablation; it is not evidence that
  `stationary-v1` wins.
- Data through `2026-07-21` is development-only. Do not run a historical test,
  pass a frozen policy to `tools/experiment.py`, or inspect test labels.
- Keep the target exactly `log(close[t + 13] / open[t + 1])`.
- Fit all scalers on retained training rows only and keep the existing
  horizon-13 embargo and frozen-calibration boundary.
- Add no dependency, model architecture, candidate, seed, or policy threshold.
- Generated reports, ledgers, backtests, and policies remain ignored.
- Future confirmation requires externally pre-registering the complete policy
  and boundary against labels strictly after `2026-07-21`.

---

### Task 1: Lock the calibration-only feature contract

**Files:**
- Create: `experiments/executable-h13-features.example.json`
- Modify: `tests/python/test_experiment.py`
- Modify: `docs/training.md`

**Interfaces:**
- Consumes: `Candidate.parse()`, `Sweep.read()`, `expected_runs()`, and the
  existing frozen-calibration CLI.
- Produces: one strict two-candidate sweep requiring exactly 75 fits over three
  series, plus a documented calibration-only command.

- [ ] **Step 1: Add the strict config test**

In `tests/python/test_experiment.py`, add this beside the existing horizon
sweep assertion:

```python
feature_sweep = Sweep.read(
    ROOT / "experiments/executable-h13-features.example.json"
)
assert feature_sweep == Sweep(
    (
        Candidate(
            "raw-17", 17, 16, 2, 32, 1, 3e-4, 1e-4,
            32, 8, 1e-3, "ohlcv",
        ),
        Candidate(
            "stationary-16", 16, 16, 2, 32, 1, 3e-4, 1e-4,
            32, 8, 1e-3, "stationary-v1",
        ),
    ),
    ("transformer",), (7, 19, 31, 43, 61),
    2, 0.1, 100, 10, 128, 13, 13, EXECUTABLE_RETURN_TARGET,
)
assert expected_runs(feature_sweep, 3) == 75
```

The dataclass equality makes every accepted field explicit; `Sweep.read()`
already rejects unknown fields.

- [ ] **Step 2: Run the focused test and confirm failure**

Run:

```sh
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch \
  python tests/python/test_experiment.py
```

Expected: failure because
`experiments/executable-h13-features.example.json` does not exist.

- [ ] **Step 3: Add the exact two-candidate sweep**

Create `experiments/executable-h13-features.example.json` with:

```json
{
  "batch_size": 128,
  "candidates": [
    {
      "feature_set": "ohlcv",
      "ff_dim": 32,
      "heads": 2,
      "layers": 1,
      "learning_rate": 0.0003,
      "mlp_dim": 32,
      "model_dim": 16,
      "name": "raw-17",
      "ridge": 0.001,
      "rolling_window": 8,
      "seq_len": 17,
      "weight_decay": 0.0001
    },
    {
      "feature_set": "stationary-v1",
      "ff_dim": 32,
      "heads": 2,
      "layers": 1,
      "learning_rate": 0.0003,
      "mlp_dim": 32,
      "model_dim": 16,
      "name": "stationary-16",
      "ridge": 0.001,
      "rolling_window": 8,
      "seq_len": 16,
      "weight_decay": 0.0001
    }
  ],
  "epochs": 100,
  "fold_fraction": 0.1,
  "folds": 2,
  "models": ["transformer"],
  "patience": 10,
  "seeds": [7, 19, 31, 43, 61],
  "target_horizon_bars": 13,
  "alignment_horizon_bars": 13,
  "target_kind": "executable-return-v1"
}
```

Selection uses `5 seeds * 2 folds * 2 candidates = 20` fits per series.
Frozen calibration adds 5 selected-candidate fits per series, so the three
series require exactly `(20 + 5) * 3 = 75` fits.

- [ ] **Step 4: Document only the calibration command**

Add a short subsection after the existing feature sweep in
`docs/training.md`:

```zsh
series=(
  AAPL=data/aapl-30m.csv
  MSFT=data/msft-30m.csv
  SPY=data/spy-30m.csv
)

python tools/experiment.py \
  experiments/executable-h13-features.example.json \
  reports/executable-h13-feature-calibration.json \
  "${series[@]}" \
  --max-runs 75 \
  --calibration-predictions \
    reports/executable-h13-feature-calibration.jsonl \
  --calibration-only
```

State directly below it:

```text
This ablation is calibration-only. Do not run a policy-authorized historical
test or inspect labels through 2026-07-21. Confirmatory evaluation requires a
pre-registered policy against later, previously unavailable labels.
```

- [ ] **Step 5: Run focused and full gates**

Run:

```sh
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch \
  python tests/python/test_experiment.py
PRIMARY_PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3
make -B PYTHON="$PRIMARY_PYTHON" check
make \
  PYTHON="/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch python" \
  check-training
```

Expected: the experiment test passes, all C and standard Python checks pass,
and both PyTorch integration suites pass.

- [ ] **Step 6: Create the config/docs checkpoint**

Inspect with `but diff`, then use the selected-change fast path for only the
config, test, and training doc:

```sh
but commit enkyuan/h13-feature-config -c \
  -m "feat(training): define h13 feature calibration" \
  --changes "$CHANGE_IDS"
```

Set `CHANGE_IDS` from the immediately preceding `but diff`. Read the commit's
returned workspace state. If it does not show the new branch directly above
`enkyuan/h13-feature-plan`, stack the existing branches with:

```sh
but move enkyuan/h13-feature-config enkyuan/h13-feature-plan
```

Do not use `but branch new -a`, create another branch, or include generated
calibration artifacts. The successful `but commit` or `but move` output is the
stack verification; do not rerun status unless that output lacks it.

---

### Task 2: Run the paired calibration and apply the promotion gate

**Files:**
- Create, ignored: `reports/executable-h13-feature-calibration.json`
- Create, ignored: `reports/executable-h13-feature-calibration.jsonl`
- Create, ignored: `reports/executable-h13-feature-transformer-policy.json`
- Create, ignored:
  `reports/executable-h13-feature-calibration-backtest.json`
- Modify only if persistent evidence is requested: `docs/training.md`

**Interfaces:**
- Consumes: the Task 1 sweep, three frozen CSV inputs, schema-6 calibration
  report, schema-3 ledger, and schema-2 selector.
- Produces: paired return-MAE evidence, six series-by-fold comparisons, a
  frozen Transformer policy, and a single pass/fail promotion decision.

- [ ] **Step 1: Run exactly the ignored calibration outputs**

Run the command documented in Task 1. Require its final JSON to name both
requested outputs and require:

```sh
jq -e '
  .schema == 6 and
  .protocol.phase == "selection-and-calibration" and
  .protocol.run_count == 75 and
  .protocol.target_horizon_bars == 13 and
  .protocol.target_kind == "executable-return-v1" and
  (.test | length) == 0
' reports/executable-h13-feature-calibration.json
```

Expected: `true`. Stop if any condition fails.

- [ ] **Step 2: Freeze the selected Transformer policy**

Run:

```zsh
python tools/select_policy.py \
  reports/executable-h13-feature-calibration.json \
  reports/executable-h13-feature-calibration.jsonl \
  reports/executable-h13-feature-transformer-policy.json \
  "${series[@]}" \
  --model transformer --safety-bps 0 3 6 10 \
  --initial-cash 100 --spread-bps 1 --slippage-bps 1 --fee-bps 0
```

This freezes whichever Transformer candidate won model selection. Do not use
the policy to authorize a test run.

- [ ] **Step 3: Materialize per-series calibration evidence**

If the policy action is `cash`, skip only the calibration-backtest command and
continue to the common Step 4 gate. Step 4 records zero per-series coverage,
prints the complete evidence JSON, and exits nonzero. Otherwise run the selected
safety threshold diagnostically on the calibration ledger:

```zsh
safety_bps=$(
  jq -er 'select(.action == "long_above") | .safety_bps' \
    reports/executable-h13-feature-transformer-policy.json
)

python tools/backtest.py \
  reports/executable-h13-feature-calibration.jsonl \
  reports/executable-h13-feature-calibration-backtest.json \
  "${series[@]}" \
  --model transformer --ensemble-seeds --safety-bps "$safety_bps" \
  --initial-cash 100 --spread-bps 1 --slippage-bps 1 --fee-bps 0
```

This is a calibration backtest only. Its per-series results supply the coverage
gate that the policy's macro threshold trial does not expose.

- [ ] **Step 4: Compute all promotion conditions once**

Run:

```sh
PRIMARY_PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3
"$PRIMARY_PYTHON" - <<'PY'
from collections import defaultdict
from pathlib import Path
from statistics import fmean
import json

root = Path("reports")
report = json.loads(
    (root / "executable-h13-feature-calibration.json").read_text()
)
policy = json.loads(
    (root / "executable-h13-feature-transformer-policy.json").read_text()
)

values = defaultdict(dict)
for record in report["validation"]:
    if record["model"] != "transformer":
        continue
    key = (record["series"], record["fold"], record["seed"])
    values[record["candidate"]][key] = record["metrics"]["return_mae"]

raw = values["raw-17"]
stationary = values["stationary-16"]
assert set(raw) == set(stationary)
assert len(raw) == 3 * 2 * 5

reduction = (fmean(raw.values()) - fmean(stationary.values())) / fmean(
    raw.values()
)
buckets = sorted({(series, fold) for series, fold, _ in raw})
wins = sum(
    fmean(stationary[key] for key in stationary if key[:2] == bucket)
    < fmean(raw[key] for key in raw if key[:2] == bucket)
    for bucket in buckets
)
assert len(buckets) == 6

trial = next(
    item for item in policy["threshold_trials"]
    if item["action"] == policy["action"]
    and item["safety_bps"] == policy["safety_bps"]
)
coverage = {name: 0.0 for name in ("AAPL", "MSFT", "SPY")}
backtest = root / "executable-h13-feature-calibration-backtest.json"
if policy["action"] == "long_above":
    evidence = json.loads(backtest.read_text())
    coverage = {
        item["series"]:
        item["strategies"]["forecast_long_cash"]["execution_coverage"]
        for item in evidence["results"]
    }

checks = {
    "stationary_selected": policy["candidate"] == "stationary-16",
    "return_mae_reduction_at_least_5pct": reduction >= 0.05,
    "series_fold_wins_at_least_5_of_6": wins >= 5,
    "policy_objective_positive": trial["objective"] > 0.0,
    "policy_trades_at_least_30": trial["trade_count"] >= 30,
    "every_series_execution_coverage_positive":
        set(coverage) == {"AAPL", "MSFT", "SPY"}
        and all(value > 0.0 for value in coverage.values()),
}
result = {
    "raw_macro_return_mae": fmean(raw.values()),
    "stationary_macro_return_mae": fmean(stationary.values()),
    "relative_reduction": reduction,
    "series_fold_wins": wins,
    "selected_trial": trial,
    "execution_coverage": coverage,
    "checks": checks,
    "promote_stationary_v1": all(checks.values()),
}
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["promote_stationary_v1"] else 1)
PY
```

Promotion requires every check to pass. A tie is not a bucket win. If any check
fails, retain raw-17 and execute Task 3; do not weaken a threshold after seeing
the result.

- [ ] **Step 5: Preserve evidence separately only if requested**

Generated reports, ledgers, policies, and backtests stay ignored. If durable
evidence is wanted, add only the Step 4 JSON summary and calibration boundary
to `docs/training.md`. Inspect with `but diff`, then checkpoint only that
documentation:

```sh
but commit enkyuan/h13-feature-benchmark -c \
  -m "docs(training): record h13 feature calibration" \
  --changes "$CHANGE_IDS"
```

Set `CHANGE_IDS` from that diff. If the returned state does not show the
benchmark directly above the config branch, run:

```sh
but move enkyuan/h13-feature-benchmark enkyuan/h13-feature-config
```

Do not amend it into the config/docs checkpoint or describe the result as
confirmatory.

---

### Task 3: Plan per-timestamp ensemble abstention only if promotion fails

**Files:**
- Create only on gate failure:
  `docs/superpowers/plans/2026-07-23-h13-seed-disagreement.md`

**Interfaces:**
- Consumes: the failed Step 4 gate, calibration prediction ledger, and existing
  seed-ensemble execution path.
- Produces: a separate implementation plan; this task changes no experiment,
  selector, model, or report code.

- [ ] **Step 1: Write the separate later plan**

The new plan must keep candidate validation selection unchanged and apply seed
disagreement only to each timestamp's calibration trading signal:

```text
decision_signal[t] =
  mean(seed predicted returns at t)
  - lambda * pstdev(seed predicted returns at t)
```

Specify `lambda in {0, 0.5, 1}` selected jointly with
`safety_bps in {0, 3, 6, 10}` by the existing macro calibration-P&L objective.
Require `lambda = 0` to reproduce the present seed-mean signal exactly. Bump the
policy schema to freeze the selected lambda, keep the reader backward-compatible
by interpreting the prior schema as `lambda = 0`, and replay the frozen lambda
identically during authorized evaluation.

The later plan must name focused tests for exact lambda-zero parity, timestamp
alignment across seeds, population-standard-deviation arithmetic, joint-grid
tie-breaking, old-policy compatibility, new-policy validation, and identical
policy replay. Keep the same candidates, seeds, horizon-13 executable target,
costs, data boundary, and no-test rule.

- [ ] **Step 2: Keep this plan non-implementing**

Do not add the penalty, rerun the ablation, or open test labels while executing
this plan. Inspect with `but diff`, then checkpoint only the later plan:

```sh
but commit enkyuan/h13-seed-disagreement-plan -c \
  -m "docs(training): plan seed-aware abstention" \
  --changes "$CHANGE_IDS"
```

Set `CHANGE_IDS` from that diff. If the optional benchmark branch exists, stack
the returned branch directly above it:

```sh
but move \
  enkyuan/h13-seed-disagreement-plan \
  enkyuan/h13-feature-benchmark
```

Otherwise stack it directly above the config branch:

```sh
but move \
  enkyuan/h13-seed-disagreement-plan \
  enkyuan/h13-feature-config
```

Run only the one applicable `but move`, and only when the commit output does not
already show the required dependency. Implementation requires a new session and
review.

---

## Checkpoint Boundaries

1. This plan only: `enkyuan/h13-feature-plan`.
2. Config, strict test, and command docs: `enkyuan/h13-feature-config`.
3. Optional durable benchmark evidence only:
   `enkyuan/h13-feature-benchmark` above the config branch.
4. On failure, the seed-disagreement plan only:
   `enkyuan/h13-seed-disagreement-plan` above the latest session-owned
   checkpoint.

Do not push, pull, land, or open a pull request for any checkpoint.
