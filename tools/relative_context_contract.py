"""Freeze one development-only SPY-residual calibration protocol."""

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
            "panel_transformer": "stock-plus-final-completed-spy-row",
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
