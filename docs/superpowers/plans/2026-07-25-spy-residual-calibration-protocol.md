# SPY-Residual Calibration Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` or execute the task inline with its stated
> checks. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze one Torch-free, development-only protocol for evaluating
stock-minus-SPY residual forecasts without authorizing an absolute forecast or
trade.

**Architecture:** Add one exact JSON profile and one pure validator. Reuse the
existing context phase budgets, bootstrap constants, public model names, and
SPY-residual target string. Move the target string into the protocol module so
the PyTorch adapter and the later armer share one definition.

**Tech Stack:** Python 3.12+, existing exact-JSON helpers, GitButler.

## Global Constraints

- Use history `17`, horizon `13`, phases `fold-1` and `calibration`, and seeds
  `7/19/31/43/61`.
- Transfer the authenticated context phase budgets: `302` updates per
  checkpoint for `fold-1` and `349` for `calibration`.
- Compare `global_ridge`, `global_mlp`, and `panel_transformer`; zero residual
  is a baseline, not a fitted model.
- Give only `panel_transformer` the completed SPY context. Keep both controls
  stock-only.
- Define residualization as stock return minus SPY return with fixed beta `1`;
  beta-adjusted and multifactor targets remain separate future experiments.
- Fit stock features, aligned SPY features, and residual targets from each
  stock's training prefix only.
- For residual truth \(z_{i,t}=r_{i,t}^{stock}-r_t^{SPY}\), define pooled raw
  residual R-squared against zero as
  \(1-\sum_{i,t}(z_{i,t}-\hat z_{i,t})^2/\sum_{i,t}z_{i,t}^2\).
- Define paired gain as
  \(|z-\hat z_{reference}|-|z-\hat z_{candidate}|\), so positive values favor
  the candidate. Average observations within each stock, then average stocks;
  use shared circular date blocks and equal-tailed 95% intervals for this
  paired metric only.
- Make those two measures primary. Centered cross-sectional R-squared and
  mean valid-timestamp RankIC are secondary because subtracting one common SPY
  return leaves within-timestamp ordering unchanged.
- Average the five neural predictions before computing primary metrics. At
  each observation, report the population standard deviation across seed
  predictions, then summarize it by averaging each stock's common-grid values
  and averaging stocks.
- Freeze the 30-minute SPY interval from `2024-11-01` through `2026-07-21`;
  require one final completed SPY row aligned to each horizon-13 input.
- Keep absolute forecasting, backtesting, forward-clean claims, trading, and
  universe expansion disabled.
- Do not read market data, truth labels, prior predictions, or model state in
  this checkpoint.
- Keep generated data, reports, attempts, models, credentials, and caches
  untracked.

---

### Task 1: Exact Residual-Calibration Profile

**Files:**

- Create: `experiments/executable-h13-spy-residual.example.json`
- Create: `tools/relative_context_contract.py`
- Modify: `tools/relative_context.py`
- Create: `tests/python/test_relative_context_contract.py`

**Interfaces:**

- Produces:
  `expected_residual_protocol() -> dict[str, object]` and
  `validate_residual_protocol(value: object) -> Mapping[str, object]`.
- Moves `SPY_RESIDUAL_TARGET` from `tools.relative_context` to
  `tools.relative_context_contract`; the former re-exports the imported name
  for compatibility.
- The first name in each paired comparison is the candidate and the second is
  the baseline.

- [x] **Step 1: Write the failing contract test**

```python
#!/usr/bin/env python3
"""Verify the exact Torch-free SPY-residual calibration protocol."""

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.panel_contract import read_canonical_json
from tools.relative_context_contract import (
    HISTORY_BARS, HORIZON_BARS, MODELS, PHASE_BUDGETS,
    SPY_RESIDUAL_TARGET, expected_residual_protocol,
    validate_residual_protocol,
)

CONFIG = ROOT / "experiments/executable-h13-spy-residual.example.json"


def rejects(value: object) -> None:
    try:
        validate_residual_protocol(value)
    except ValueError:
        return
    raise AssertionError("invalid residual calibration protocol was accepted")


def verify_exact_protocol() -> None:
    value = read_canonical_json(CONFIG)
    assert validate_residual_protocol(value) == expected_residual_protocol()
    assert (HISTORY_BARS, HORIZON_BARS) == (17, 13)
    assert MODELS == (
        "global_ridge", "global_mlp", "panel_transformer",
    )
    assert PHASE_BUDGETS == (("fold-1", 302), ("calibration", 349))
    assert value["target_kind"] == SPY_RESIDUAL_TARGET
    assert set(value["locks"].values()) == {False}
    assert value["metrics"]["primary"][0] == \
        "pooled-raw-residual-r2-vs-zero"
    assert value["paired_absolute_error_convention"] == \
        "reference-mae-minus-candidate-mae-positive"
    assert value["seed_aggregation"]["primary"] == \
        "arithmetic-mean-predictions-before-metrics"
    assert "torch" not in sys.modules

    value["models"].pop()
    assert len(expected_residual_protocol()["models"]) == len(MODELS)


def verify_rejections() -> None:
    value = expected_residual_protocol()
    for mutate in (
        lambda item: item["models"].reverse(),
        lambda item: item["models"].append("conditioned_panel_transformer"),
        lambda item: item["seeds"].reverse(),
        lambda item: item["phases"].reverse(),
        lambda item: item["paired_absolute_error_comparisons"].reverse(),
        lambda item: item["locks"].update({"backtest_run": True}),
        lambda item: item.update({"history_bars": 34}),
        lambda item: item.update({"target_horizon_bars": 1}),
        lambda item: item.update({"target_kind": "executable-return-v1"}),
        lambda item: item.update({"extra": True}),
    ):
        invalid = deepcopy(value)
        mutate(invalid)
        rejects(invalid)

    invalid = deepcopy(value)
    invalid["phases"][0]["updates_per_checkpoint"] = True
    rejects(invalid)


def main() -> None:
    verify_exact_protocol()
    verify_rejections()
    print("relative-context contract tests passed")


if __name__ == "__main__":
    main()
```

- [x] **Step 2: Run the test and verify the red state**

Run:

```sh
$PYTHON tests/python/test_relative_context_contract.py
```

Expected: failure because `tools.relative_context_contract` does not exist.

- [x] **Step 3: Add the exact profile**

```json
{
  "alignment_horizon_bars": 13,
  "baselines": ["zero"],
  "batch_size": 128,
  "bootstrap": {
    "applies_to": "stock-macro-paired-absolute-error",
    "block_days": [5, 10, 20],
    "interval": "equal-tailed-95-percentile",
    "method": "shared-circular-date-block",
    "replicates": 10000,
    "seed": 20260725,
    "weighting": "stock-macro"
  },
  "data_interval": {
    "end": "2026-07-21",
    "minutes": 30,
    "start": "2024-11-01"
  },
  "evidence_role": "development-calibration-not-forward-clean",
  "feature_set": "ohlcv",
  "history_bars": 17,
  "locks": {
    "absolute_forecast_authorized": false,
    "backtest_run": false,
    "forward_clean": false,
    "trading_authorized": false,
    "universe_expansion_authorized": false
  },
  "metrics": {
    "primary": [
      "pooled-raw-residual-r2-vs-zero",
      "stock-macro-paired-absolute-error"
    ],
    "secondary": [
      "pooled-timestamp-centered-cross-sectional-r2",
      "mean-valid-timestamp-spearman-rank-ic"
    ]
  },
  "model_inputs": {
    "global_mlp": "stock-only",
    "global_ridge": "stock-only",
    "panel_transformer": "stock-plus-final-completed-spy-row"
  },
  "models": ["global_ridge", "global_mlp", "panel_transformer"],
  "neural_checkpoint_policy": "reuse-context-prior-selections",
  "output_role": "residual-only-not-executable-return",
  "paired_absolute_error_comparisons": [
    ["global_ridge", "zero"],
    ["global_mlp", "zero"],
    ["global_mlp", "global_ridge"],
    ["panel_transformer", "zero"],
    ["panel_transformer", "global_ridge"],
    ["panel_transformer", "global_mlp"]
  ],
  "paired_absolute_error_convention":
    "reference-mae-minus-candidate-mae-positive",
  "phases": [
    {"name": "fold-1", "updates_per_checkpoint": 302},
    {"name": "calibration", "updates_per_checkpoint": 349}
  ],
  "residualization": "stock-minus-spy-fixed-beta-1",
  "sampling_policy": "stock-balanced",
  "scalers": {
    "residual_target": "per-stock-training-prefix",
    "spy_features": "per-stock-aligned-training-prefix",
    "stock_features": "per-stock-training-prefix"
  },
  "seed_aggregation": {
    "primary": "arithmetic-mean-predictions-before-metrics",
    "report": {
      "per_observation": "population-standard-deviation-across-seeds",
      "summary": "stock-macro-mean-over-common-grid"
    }
  },
  "seeds": [7, 19, 31, 43, 61],
  "target_horizon_bars": 13,
  "target_kind": "spy-residual-executable-return-v1",
  "timestamp_policy": "exact-triples-and-common-cross-section"
}
```

- [x] **Step 4: Implement the pure validator**

```python
"""Freeze one development-only SPY-residual calibration protocol."""

from collections.abc import Mapping

from tools.panel_contract import _exact_json
from tools.universe_cross_section import CROSS_SECTION_SEED
from tools.universe_scaling import (
    BOOTSTRAP_BLOCK_DAYS, BOOTSTRAP_REPLICATES,
)

SPY_RESIDUAL_TARGET = "spy-residual-executable-return-v1"
EVIDENCE_ROLE = "development-calibration-not-forward-clean"
HISTORY_BARS = 17
HORIZON_BARS = 13
INTERVAL_MINUTES = 30
SPY_START = "2024-11-01"
SPY_END = "2026-07-21"
MODELS = ("global_ridge", "global_mlp", "panel_transformer")
SEEDS = (7, 19, 31, 43, 61)
PHASE_BUDGETS = (("fold-1", 302), ("calibration", 349))
PAIRED_COMPARISONS = (
    ("global_ridge", "zero"),
    ("global_mlp", "zero"),
    ("global_mlp", "global_ridge"),
    ("panel_transformer", "zero"),
    ("panel_transformer", "global_ridge"),
    ("panel_transformer", "global_mlp"),
)


def expected_residual_protocol() -> dict[str, object]:
    """Return a fresh copy of the frozen residual-calibration choices."""
    return {
        "alignment_horizon_bars": HORIZON_BARS,
        "baselines": ["zero"],
        "batch_size": 128,
        "bootstrap": {
            "applies_to": "stock-macro-paired-absolute-error",
            "block_days": list(BOOTSTRAP_BLOCK_DAYS),
            "interval": "equal-tailed-95-percentile",
            "method": "shared-circular-date-block",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": CROSS_SECTION_SEED,
            "weighting": "stock-macro",
        },
        "data_interval": {
            "end": SPY_END,
            "minutes": INTERVAL_MINUTES,
            "start": SPY_START,
        },
        "evidence_role": EVIDENCE_ROLE,
        "feature_set": "ohlcv",
        "history_bars": HISTORY_BARS,
        "locks": {
            "absolute_forecast_authorized": False,
            "backtest_run": False,
            "forward_clean": False,
            "trading_authorized": False,
            "universe_expansion_authorized": False,
        },
        "metrics": {
            "primary": [
                "pooled-raw-residual-r2-vs-zero",
                "stock-macro-paired-absolute-error",
            ],
            "secondary": [
                "pooled-timestamp-centered-cross-sectional-r2",
                "mean-valid-timestamp-spearman-rank-ic",
            ],
        },
        "model_inputs": {
            "global_mlp": "stock-only",
            "global_ridge": "stock-only",
            "panel_transformer":
                "stock-plus-final-completed-spy-row",
        },
        "models": list(MODELS),
        "neural_checkpoint_policy": "reuse-context-prior-selections",
        "output_role": "residual-only-not-executable-return",
        "paired_absolute_error_comparisons": [
            list(pair) for pair in PAIRED_COMPARISONS
        ],
        "paired_absolute_error_convention":
            "reference-mae-minus-candidate-mae-positive",
        "phases": [
            {"name": name, "updates_per_checkpoint": updates}
            for name, updates in PHASE_BUDGETS
        ],
        "residualization": "stock-minus-spy-fixed-beta-1",
        "sampling_policy": "stock-balanced",
        "scalers": {
            "residual_target": "per-stock-training-prefix",
            "spy_features": "per-stock-aligned-training-prefix",
            "stock_features": "per-stock-training-prefix",
        },
        "seed_aggregation": {
            "primary": "arithmetic-mean-predictions-before-metrics",
            "report": {
                "per_observation":
                    "population-standard-deviation-across-seeds",
                "summary": "stock-macro-mean-over-common-grid",
            },
        },
        "seeds": list(SEEDS),
        "target_horizon_bars": HORIZON_BARS,
        "target_kind": SPY_RESIDUAL_TARGET,
        "timestamp_policy": "exact-triples-and-common-cross-section",
    }


def validate_residual_protocol(
    value: object,
) -> Mapping[str, object]:
    """Reject any choice outside the predeclared residual calibration."""
    expected = expected_residual_protocol()
    if not _exact_json(value, expected):
        raise ValueError("config does not match the residual calibration")
    return expected
```

Replace the local target declaration in `tools/relative_context.py` with:

```python
from tools.relative_context_contract import SPY_RESIDUAL_TARGET
```

### Task 2: Source-Derived Preflight Records

**Files:**

- Modify: `tools/relative_context_contract.py`
- Modify: `tests/python/test_relative_context_contract.py`

**Interfaces:**

- Produces immutable `ResidualScalerInput` and `ResidualPhaseInput` records.
- Produces
  `residual_scaler_inputs_sha256(master, values) -> str`,
  `parse_residual_phases(value, source) -> tuple[ResidualPhaseInput, ...]`,
  `expected_residual_fits(master, phase) -> tuple[ContextFit, ...]`, and
  `validate_spy_session_audit(value) -> Mapping[str, object]`.
- Reuses the authenticated `ContextPhase` as the owner of fit schedules,
  update budgets, and stock grids.

- [x] **Step 1: Add failing source-binding tests**

Add exact tests that:

```python
from dataclasses import FrozenInstanceError, replace

sys.path.insert(0, str(Path(__file__).parent))
from test_context_diagnostic_finalizer import MASTER, phase_for
from tools.context_diagnostic_contract import (
    ContextPhase, _phase_value, context_phase_sha256,
)
from tools.relative_context_contract import (
    ResidualPhaseInput, ResidualScalerInput, expected_residual_fits,
    parse_residual_phases, residual_scaler_inputs_sha256,
    validate_spy_session_audit,
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def scaler_inputs(
    phase: str,
) -> tuple[ResidualScalerInput, ...]:
    return tuple(
        ResidualScalerInput(
            series,
            digest(f"{phase}-{series}-stock-prefix"),
            digest(f"{phase}-{series}-spy-prefix"),
            index + 1,
            digest(f"{phase}-{series}-training-grid"),
        )
        for index, series in enumerate(MASTER)
    )


def source_phase(name: str) -> ContextPhase:
    value = _phase_value(phase_for(name))
    count, updates = {
        "fold-1": (3_513, 302),
        "calibration": (4_060, 349),
    }[name]
    for row in value["training_rows"]:
        row["count"] = count
    value["updates_per_checkpoint"] = updates
    return ContextPhase.parse(value, MASTER)


def phase_value(
    phase: ContextPhase, scaler_sha256: str,
) -> dict[str, str]:
    return {
        "aligned_evaluation_grid_sha256":
            phase.evaluation_grid_sha256,
        "aligned_training_grid_sha256": phase.training_grid_sha256,
        "phase": phase.phase,
        "scaler_inputs_sha256": scaler_sha256,
        "source_phase_sha256": context_phase_sha256(phase),
    }
```

Verify one deterministic scaler digest, sensitivity to every field, exact
55-series order, positive non-Boolean training counts, valid hashes, frozen
records, two ordered source phases, exact source/grid hashes, and an 11-fit
history-17 closure per phase. The phase parser requires distinct well-formed
scaler digests; the later armed-attempt binding owns their exact values.

Verify the only accepted SPY audit is:

```python
{
    "scope": "all-expected-session-bins",
    "expected_sessions": 428,
    "affected_sessions": 0,
    "missing_sessions": [],
    "expected_bins": 5_534,
    "missing_bins": 0,
    "ranges": [],
}
```

Mutating any field or adding an extra field must fail.

- [x] **Step 2: Run the expanded test and verify the red state**

Run:

```sh
$PYTHON tests/python/test_relative_context_contract.py
```

Expected: import failure for `ResidualScalerInput`.

- [x] **Step 3: Implement the source-derived records**

Add:

```python
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tools.context_diagnostic_contract import (
    ContextFit, ContextPhase, _json_sha256, context_phase_sha256,
    expected_context_fits,
)
from tools.panel_contract import (
    _exact_json, _integer, _object, _sha256, _string,
)
from tools.universe_contract import universe_roles


@dataclass(frozen=True, slots=True)
class ResidualScalerInput:
    series: str
    stock_training_prefix_sha256: str
    spy_training_prefix_sha256: str
    training_rows: int
    training_grid_sha256: str


def residual_scaler_inputs_sha256(
    master: Sequence[str],
    values: Sequence[ResidualScalerInput],
) -> str:
    """Bind each stock's aligned training-only scaler inputs."""
    names = tuple(master)
    universe_roles(names)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("residual scaler inputs are invalid")
    inputs = tuple(values)
    if len(inputs) != len(names):
        raise ValueError("residual scaler inputs must contain 55 stocks")
    records = []
    for index, (value, series) in enumerate(zip(
        inputs, names, strict=True,
    )):
        if not isinstance(value, ResidualScalerInput) or \
           _string(value.series, f"scaler input {index} series") != series:
            raise ValueError("residual scaler input order changed")
        records.append({
            "series": series,
            "spy_training_prefix_sha256": _sha256(
                value.spy_training_prefix_sha256,
                f"{series} SPY training prefix",
            ),
            "stock_training_prefix_sha256": _sha256(
                value.stock_training_prefix_sha256,
                f"{series} stock training prefix",
            ),
            "training_grid_sha256": _sha256(
                value.training_grid_sha256,
                f"{series} training grid",
            ),
            "training_rows": _integer(
                value.training_rows, f"{series} training rows",
            ),
        })
    return _json_sha256({
        "history_bars": HISTORY_BARS,
        "inputs": records,
        "role": "residual-scaler-inputs",
        "schema": 1,
    })


@dataclass(frozen=True, slots=True)
class ResidualPhaseInput:
    phase: str
    source_phase_sha256: str
    aligned_training_grid_sha256: str
    aligned_evaluation_grid_sha256: str
    scaler_inputs_sha256: str

    @classmethod
    def parse(
        cls, value: object, source: ContextPhase,
    ) -> "ResidualPhaseInput":
        if not isinstance(source, ContextPhase):
            raise ValueError("residual source phase is invalid")
        expected_updates = dict(PHASE_BUDGETS).get(source.phase)
        if source.updates_per_checkpoint != expected_updates:
            raise ValueError("residual source phase budget changed")
        item = _object(value, {
            "aligned_evaluation_grid_sha256",
            "aligned_training_grid_sha256",
            "phase",
            "scaler_inputs_sha256",
            "source_phase_sha256",
        }, "residual phase")
        parsed = cls(
            _string(item["phase"], "residual phase name"),
            _sha256(item["source_phase_sha256"], "source phase"),
            _sha256(item["aligned_training_grid_sha256"], "training grid"),
            _sha256(
                item["aligned_evaluation_grid_sha256"], "evaluation grid",
            ),
            _sha256(item["scaler_inputs_sha256"], "scaler inputs"),
        )
        if (
            parsed.phase != source.phase
            or parsed.source_phase_sha256 != context_phase_sha256(source)
            or parsed.aligned_training_grid_sha256
            != source.training_grid_sha256
            or parsed.aligned_evaluation_grid_sha256
            != source.evaluation_grid_sha256
        ):
            raise ValueError("residual phase differs from its source")
        return parsed


def parse_residual_phases(
    value: object,
    source: Sequence[ContextPhase],
) -> tuple[ResidualPhaseInput, ...]:
    """Bind the two residual phases to their authenticated source phases."""
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)):
        raise ValueError("residual phases changed")
    phases = tuple(source)
    if not isinstance(value, list) or len(value) != len(PHASE_BUDGETS) or \
       len(phases) != len(PHASE_BUDGETS) or any(
           not isinstance(phase, ContextPhase) for phase in phases
       ) or tuple(
           phase.phase for phase in phases
       ) != tuple(name for name, _ in PHASE_BUDGETS):
        raise ValueError("residual phases changed")
    parsed = tuple(
        ResidualPhaseInput.parse(item, phase)
        for item, phase in zip(value, phases, strict=True)
    )
    if len({phase.scaler_inputs_sha256 for phase in parsed}) != len(parsed):
        raise ValueError("residual phase scaler inputs were reused")
    return parsed


def expected_residual_fits(
    master: Sequence[str], phase: ContextPhase,
) -> tuple[ContextFit, ...]:
    """Reuse only the selected history-17 context fit schedule."""
    if not isinstance(phase, ContextPhase) or \
       phase.updates_per_checkpoint != dict(PHASE_BUDGETS).get(phase.phase):
        raise ValueError("residual source phase budget changed")
    fits = tuple(
        fit for fit in expected_context_fits(master, phase)
        if fit.history == HISTORY_BARS
    )
    expected = (
        (phase.phase, "global_ridge", HISTORY_BARS, None),
        *(
            (phase.phase, model, HISTORY_BARS, seed)
            for model in MODELS[1:] for seed in SEEDS
        ),
    )
    if tuple(
        (fit.phase, fit.model, fit.history, fit.seed) for fit in fits
    ) != expected:
        raise ValueError("residual fit closure changed")
    return fits


def validate_spy_session_audit(
    value: object,
) -> Mapping[str, object]:
    """Validate the declared clean SPY grid; the armer binds source bytes."""
    expected = {
        "scope": "all-expected-session-bins",
        "expected_sessions": 428,
        "affected_sessions": 0,
        "missing_sessions": [],
        "expected_bins": 5_534,
        "missing_bins": 0,
        "ranges": [],
    }
    if not _exact_json(value, expected):
        raise ValueError("SPY session grid is incomplete")
    return expected
```

- [x] **Step 4: Run the expanded contract test**

Run:

```sh
$PYTHON tests/python/test_relative_context_contract.py
```

Expected: pass with 55 scaler records, two phases, 11 fits per phase, and one
exact clean SPY audit.

- [x] **Step 5: Run focused checks**

Run:

```sh
$PYTHON tests/python/test_relative_context_contract.py
$TORCH_PYTHON tests/python/test_relative_context.py
$PYTHON tests/python/test_relative_context_inputs.py
```

Expected: all pass.

- [x] **Step 6: Run repository checks**

Run:

```sh
make -B PYTHON=$PYTHON check
```

Expected: all mandatory C and Python suites pass. The PyTorch residual test is
run separately because PyTorch remains optional in the aggregate gate.

- [x] **Step 7: Create signed local checkpoints**

Commit the plan, profile, validator, adapter import, and test together as one
reviewable checkpoint:

```text
feat(training): freeze residual calibration protocol
```

Stack it above `enkyuan/spy-grid-plan`. Verify author, committer, and ED25519
signature. Do not push.

## Next Checkpoint Boundary

Write a separate preflight plan that:

1. authenticates the successful
   `h13-context-diagnostic-20260725-03` attempt and terminal outcome;
2. re-fetches SPY through `scan_regular_bars()` with the bound exchange
   calendar and requires a zero-missing-bin `session_grid_audit()`;
3. freezes the coverage overlay, calendar, SPY CSV, every stock CSV, aligned
   timestamp grids, training-only scaler inputs, source tree, runtimes, and
   Torch package;
4. creates an exclusive attempt before reading residual calibration truth; and
5. preserves every lock in this profile.

Do not create a recovery attestation for the older failed attempt: `-03` is the
current successful source. Do not run the residual calibration or `$100`
backtest until that additive preflight is green.
