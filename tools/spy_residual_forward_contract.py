"""Freeze one SPY-direction-gated residual forward test."""

from collections.abc import Mapping

from tools.panel_contract import _exact_json
from tools.relative_context_contract import (
    HISTORY_BARS, HORIZON_BARS, INTERVAL_MINUTES, SEEDS,
    SPY_RESIDUAL_TARGET,
)
from tools.spy_residual_gate import SPY_DIRECTION_SCALE
from tools.universe_cross_section import CROSS_SECTION_SEED
from tools.universe_scaling import (
    BOOTSTRAP_BLOCK_DAYS, BOOTSTRAP_REPLICATES,
)

FORWARD_SOURCES = (
    (
        "residual_attempt",
        "experiments/h13-spy-residual-20260725-01-attempt.json",
        "0fb90623c90b418dfff93d35dde1bb49024c25d3b2b27b799ce752b8deed9ea3",
    ),
    (
        "residual_outcome",
        "experiments/h13-spy-residual-20260725-01-outcome.json",
        "132c17cd7dde7abcdf625581d6b399d7e9b0011f4a9c8bc6d4d0f4065d3a0488",
    ),
    (
        "alignment_report",
        "reports/h13-spy-residual-20260725-01-alignment/alignment.json",
        "8aee02761357927097a74d13c3d62271fd57f629a4eeb6ac1bebc84a77fd3fa1",
    ),
    (
        "calibration_fits",
        "reports/h13-spy-residual-20260725-01/calibration-fits.jsonl",
        "93237f9962a64665950252094e9119d1e1a806d7288af0d5c331cd724954203e",
    ),
)
FORWARD_CALENDAR = (
    "forward_calendar",
    "universes/us-equities-core-forward-2026-05-19_2026-08-18.json",
    "997d751a5a2ae8b2c51f4b500bd27ec94155359e211d1f9cfef198d5f156c362",
)
STATE_FINGERPRINTS = (
    "fe12e7e77d81eb6761defa27f423739e634d55c608beb30d9125b2631fc1049b",
    "3e2ccdc591baaf9d3b94efdeeddd86b6fed913bf7ea7fce29b6ec38f0c0fc2d2",
    "3f09fcd8c5af5e17019ed4f907c4ade5d2a62c77a958718341efff40bdedb770",
    "723554c420e19286431ce476d9f5ebfe4ed4941c3d5cbbd49c4bdae0b99f11e6",
    "3a5309a9aa318eb2b4093a95a7185610f6cce5a1f15d3ba2856b8a8bdae7132e",
)
FORWARD_UNIVERSE = (
    "KRYS", "TGT", "STM", "SSNC", "NWL", "AAON",
    "GEV", "SWKS", "BMRN", "ACI", "HUN",
)
EXECUTION_LOCKS = (
    "absolute_forecast_authorized",
    "price_reconstruction_authorized",
    "policy_selected",
    "backtest_run",
    "trading_authorized",
    "forward_data_refit_authorized",
    "candidate_search_authorized",
    "truth_access_before_verified_prediction_ledger_authorized",
)


def expected_forward_protocol() -> dict[str, object]:
    """Return a fresh copy of the sole allowed forward-test profile."""
    return {
        "schema": 1,
        "evidence_role": "preregistered-forward-test-not-yet-executed",
        "candidate": {
            "name": "spy-direction-gated-five-seed-mean",
            "model": "panel_transformer",
            "target_kind": SPY_RESIDUAL_TARGET,
            "history_bars": HISTORY_BARS,
            "target_horizon_bars": HORIZON_BARS,
            "seed_aggregation": "arithmetic-mean-before-gate",
            "gate": {
                "feature": "log(spy.close[as_of]/spy.close[as_of-16])",
                "threshold": 0.0,
                "comparison": "greater-than-or-equal",
                "scale": SPY_DIRECTION_SCALE,
                "active_output": "scale-times-mean-prediction",
                "inactive_output": 0.0,
            },
        },
        "sources": {
            name: {"path": path, "sha256": sha256}
            for name, path, sha256 in (*FORWARD_SOURCES, FORWARD_CALENDAR)
        },
        "transformer_states": [
            {"seed": seed, "state_fingerprint": fingerprint}
            for seed, fingerprint in zip(
                SEEDS, STATE_FINGERPRINTS, strict=True,
            )
        ],
        "universe": list(FORWARD_UNIVERSE),
        "forward_window": {
            "interval_minutes": INTERVAL_MINUTES,
            "session": "regular",
            "sampling": "session_samples-identical-canonical-triples",
            "entry_boundary":
                "strictly-after-final-inspected-calibration-target",
            "first_target_session": "earliest-full-target-session",
            "target_session_count": 60,
            "target_session_sequence": "first-plus-next-calendar-sessions",
            "optional_stopping": False,
            "coverage": "complete-universe-and-spy-grids-or-fail",
            "missing_coverage": "fail-without-changing-window",
            "sample_selection":
                "all-canonical-triples-with-target-in-fixed-session-set",
            "window_derivation": "bound-calendar-before-market-data",
        },
        "metrics": {
            "paired_gain":
                "reference-squared-error-minus-candidate-squared-error",
            "references": ["zero", "unchanged-five-seed-mean"],
            "bootstrap": {
                "method": "shared-circular-target-session-block",
                "block_sessions": list(BOOTSTRAP_BLOCK_DAYS),
                "decision_block_sessions": 20,
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": CROSS_SECTION_SEED,
                "interval": "equal-tailed-95-percentile",
                "weighting": "stock-macro",
            },
            "descriptive": [
                "mean-absolute-error",
                "direction-accuracy",
                "market-regime-cells",
                "seed-dispersion",
                "nondecision-block-intervals",
            ],
            "descriptive_role": "report-only-cannot-rescue-failed-gate",
            "raw_residual_r2_convention":
                "one-minus-sse-over-uncentered-zero-baseline-truth-energy",
            "leave_one_stock_out_convention":
                "scoring-omission-only-without-refit",
            "stock_win_convention":
                "strictly-positive-per-stock-mean-paired-squared-error-gain",
        },
        "gates": {
            "pooled_raw_residual_r2_vs_zero_min_exclusive": 0.0,
            "decision_paired_mse_gain_lower_bound_vs_zero_min_exclusive": 0.0,
            (
                "decision_paired_mse_gain_lower_bound_vs_"
                "unchanged_five_seed_mean_min_exclusive"
            ): 0.0,
            "minimum_leave_one_stock_out_raw_residual_r2_vs_zero_exclusive":
                0.0,
            "minimum_stocks_with_positive_mse_gain_vs_zero": 6,
            "policy": "all-required",
        },
        "locks": {name: False for name in EXECUTION_LOCKS},
    }


def validate_forward_protocol(
    value: object,
) -> Mapping[str, object]:
    """Reject any choice outside the preregistered forward test."""
    expected = expected_forward_protocol()
    if not _exact_json(value, expected):
        raise ValueError("config does not match the SPY residual forward test")
    return expected
