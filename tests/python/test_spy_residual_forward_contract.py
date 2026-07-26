#!/usr/bin/env python3
"""Verify the exact, Torch-free SPY-residual forward protocol."""

from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.files import file_sha256
from tools.panel_contract import _exact_json, read_canonical_json
from tools.spy_residual_forward_contract import (
    FORWARD_CALENDAR, expected_forward_protocol, validate_forward_protocol,
)

CONFIG = ROOT / \
    "experiments/executable-h13-spy-direction-forward.example.json"
STATES = (
    (7, "fe12e7e77d81eb6761defa27f423739e634d55c608beb30d9125b2631fc1049b"),
    (19, "3e2ccdc591baaf9d3b94efdeeddd86b6fed913bf7ea7fce29b6ec38f0c0fc2d2"),
    (31, "3f09fcd8c5af5e17019ed4f907c4ade5d2a62c77a958718341efff40bdedb770"),
    (43, "723554c420e19286431ce476d9f5ebfe4ed4941c3d5cbbd49c4bdae0b99f11e6"),
    (61, "3a5309a9aa318eb2b4093a95a7185610f6cce5a1f15d3ba2856b8a8bdae7132e"),
)
UNIVERSE = (
    "KRYS", "TGT", "STM", "SSNC", "NWL", "AAON",
    "GEV", "SWKS", "BMRN", "ACI", "HUN",
)
SOURCES = {
    "residual_attempt": {
        "path": "experiments/h13-spy-residual-20260725-01-attempt.json",
        "sha256":
            "0fb90623c90b418dfff93d35dde1bb49024c25d3b2b27b799ce752b8deed9ea3",
    },
    "residual_outcome": {
        "path": "experiments/h13-spy-residual-20260725-01-outcome.json",
        "sha256":
            "132c17cd7dde7abcdf625581d6b399d7e9b0011f4a9c8bc6d4d0f4065d3a0488",
    },
    "alignment_report": {
        "path":
            "reports/h13-spy-residual-20260725-01-alignment/alignment.json",
        "sha256":
            "8aee02761357927097a74d13c3d62271fd57f629a4eeb6ac1bebc84a77fd3fa1",
    },
    "calibration_fits": {
        "path":
            "reports/h13-spy-residual-20260725-01/calibration-fits.jsonl",
        "sha256":
            "93237f9962a64665950252094e9119d1e1a806d7288af0d5c331cd724954203e",
    },
    "forward_calendar": {
        "path":
            "universes/us-equities-core-forward-2026-05-19_2026-08-18.json",
        "sha256":
            "997d751a5a2ae8b2c51f4b500bd27ec94155359e211d1f9cfef198d5f156c362",
    },
}
LOCKS = {
    "absolute_forecast_authorized": False,
    "price_reconstruction_authorized": False,
    "policy_selected": False,
    "backtest_run": False,
    "trading_authorized": False,
    "forward_data_refit_authorized": False,
    "candidate_search_authorized": False,
    "truth_access_before_verified_prediction_ledger_authorized": False,
}
GATES = {
    "decision_paired_mse_gain_lower_bound_vs_zero_min_exclusive": 0.0,
    (
        "decision_paired_mse_gain_lower_bound_vs_"
        "unchanged_five_seed_mean_min_exclusive"
    ): 0.0,
    "minimum_leave_one_stock_out_raw_residual_r2_vs_zero_exclusive": 0.0,
    "minimum_stocks_with_positive_mse_gain_vs_zero": 6,
    "policy": "all-required",
    "pooled_raw_residual_r2_vs_zero_min_exclusive": 0.0,
}


def rejects(mutate: object) -> None:
    value = expected_forward_protocol()
    mutate(value)  # type: ignore[operator]
    try:
        validate_forward_protocol(value)
    except ValueError:
        return
    raise AssertionError("changed forward protocol was accepted")


def swap_state_field(value: dict[str, object], field: str) -> None:
    states = value["transformer_states"]  # type: ignore[assignment]
    states[0][field], states[-1][field] = \
        states[-1][field], states[0][field]


def verify_exact_profile_and_literals() -> None:
    value = read_canonical_json(CONFIG)
    expected = expected_forward_protocol()
    assert _exact_json(value, expected)
    assert validate_forward_protocol(value) == expected
    assert value["schema"] == 1
    assert value["evidence_role"] == \
        "preregistered-forward-test-not-yet-executed"
    candidate = value["candidate"]
    assert candidate["name"] == "spy-direction-gated-five-seed-mean"
    assert candidate["model"] == "panel_transformer"
    assert candidate["target_kind"] == \
        "spy-residual-executable-return-v1"
    assert candidate["seed_aggregation"] == "arithmetic-mean-before-gate"
    assert (candidate["history_bars"], candidate["target_horizon_bars"]) == \
        (17, 13)
    assert tuple(
        (state["seed"], state["state_fingerprint"])
        for state in value["transformer_states"]
    ) == STATES
    assert candidate["gate"] == {
        "feature": "log(spy.close[as_of]/spy.close[as_of-16])",
        "threshold": 0.0,
        "comparison": "greater-than-or-equal",
        "scale": 0.4029492434939931,
        "active_output": "scale-times-mean-prediction",
        "inactive_output": 0.0,
    }
    assert value["sources"] == SOURCES
    name, path, sha256 = FORWARD_CALENDAR
    assert value["sources"][name] == {
        "path": path, "sha256": sha256,
    }
    assert file_sha256(ROOT / path) == sha256
    assert tuple(value["universe"]) == UNIVERSE
    assert value["forward_window"] == {
        "interval_minutes": 30,
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
    }
    assert value["metrics"] == {
        "paired_gain":
            "reference-squared-error-minus-candidate-squared-error",
        "references": ["zero", "unchanged-five-seed-mean"],
        "bootstrap": {
            "method": "shared-circular-target-session-block",
            "block_sessions": [5, 10, 20],
            "decision_block_sessions": 20,
            "replicates": 10_000,
            "seed": 20_260_725,
            "interval": "equal-tailed-95-percentile",
            "weighting": "stock-macro",
        },
        "descriptive": [
            "mean-absolute-error", "direction-accuracy",
            "market-regime-cells", "seed-dispersion",
            "nondecision-block-intervals",
        ],
        "descriptive_role": "report-only-cannot-rescue-failed-gate",
        "raw_residual_r2_convention":
            "one-minus-sse-over-uncentered-zero-baseline-truth-energy",
        "leave_one_stock_out_convention":
            "scoring-omission-only-without-refit",
        "stock_win_convention":
            "strictly-positive-per-stock-mean-paired-squared-error-gain",
    }
    assert value["gates"] == GATES
    assert value["locks"] == LOCKS


def verify_fresh_copy() -> None:
    first, second = expected_forward_protocol(), expected_forward_protocol()
    assert first is not second
    assert first["candidate"] is not second["candidate"]
    assert first["candidate"]["gate"] is not second["candidate"]["gate"]
    assert first["transformer_states"] is not second["transformer_states"]
    assert first["transformer_states"][0] is not \
        second["transformer_states"][0]
    first["candidate"]["gate"]["scale"] = 0.0
    first["transformer_states"][0]["seed"] = 0
    first["transformer_states"].pop()
    assert _exact_json(second, read_canonical_json(CONFIG))
    validated = validate_forward_protocol(second)
    assert validated is not second
    validated["universe"].pop()
    assert _exact_json(
        validate_forward_protocol(second), expected_forward_protocol(),
    )


def verify_required_rejections() -> None:
    mutations = (
        lambda item: item["transformer_states"].reverse(),
        lambda item: swap_state_field(item, "seed"),
        lambda item: swap_state_field(item, "state_fingerprint"),
        lambda item: item["universe"].reverse(),
        lambda item: item["transformer_states"][0].update(
            {"state_fingerprint": "0" * 64},
        ),
        lambda item: item["candidate"]["gate"].update(
            {"scale": 0.4},
        ),
        lambda item: item["candidate"]["gate"].update({"threshold": 0.1}),
        lambda item: item["candidate"].update({"name": "alternative"}),
        lambda item: item["candidate"].update({"model": "global_mlp"}),
        lambda item: item["candidate"].update(
            {"target_kind": "executable-return-v1"},
        ),
        lambda item: item["forward_window"].update(
            {"target_session_count": 59},
        ),
        lambda item: item["forward_window"].update(
            {"target_session_count": 61},
        ),
        lambda item: item["forward_window"].update(
            {"optional_stopping": True},
        ),
        lambda item: item["forward_window"].update(
            {"missing_coverage": "extend-until-complete"},
        ),
        lambda item: item["forward_window"].update(
            {"window_derivation": "after-market-data"},
        ),
        lambda item: item["forward_window"].update(
            {"sample_selection": "first-canonical-triple-per-session"},
        ),
        lambda item: item["sources"]["residual_attempt"].update(
            {"path": "experiments/other.json"},
        ),
        lambda item: item["sources"]["residual_outcome"].update(
            {"sha256": "0" * 64},
        ),
        lambda item: item["sources"]["forward_calendar"].update(
            {"sha256": "0" * 64},
        ),
        lambda item: item["metrics"]["references"].reverse(),
        lambda item: item["metrics"].update(
            {"raw_residual_r2_convention": "centered-r2"},
        ),
        lambda item: item["metrics"].update(
            {"leave_one_stock_out_convention": "refit-after-omission"},
        ),
        lambda item: item["metrics"].update(
            {"stock_win_convention": "positive-point-count"},
        ),
        lambda item: item["gates"].update(
            {"minimum_stocks_with_positive_mse_gain_vs_zero": 5},
        ),
        lambda item: item["metrics"]["bootstrap"].update(
            {"decision_block_sessions": 10},
        ),
        lambda item: item.update({"extra": True}),
        lambda item: item.pop("metrics"),
        lambda item: item.update({"schema": True}),
    )
    for mutate in mutations:
        rejects(mutate)
    for name in LOCKS:
        rejects(lambda item, name=name: item["locks"].update({name: True}))


def verify_isolated_import() -> None:
    code = f"""
import os
from pathlib import Path
import sys

sys.path.insert(0, {str(ROOT)!r})
root = Path({str(ROOT)!r})
data_roots = {{"data", "experiments", "models", "reports", "universes"}}
data_suffixes = {{
    ".bin", ".csv", ".json", ".jsonl", ".npy", ".npz", ".pt", ".pth",
    ".safetensors",
}}

def reads_data(value):
    if isinstance(value, bytes):
        value = os.fsdecode(value)
    if not isinstance(value, (str, os.PathLike)):
        return False
    path = Path(value)
    if path.suffix in data_suffixes:
        return True
    if not path.is_absolute():
        return bool(path.parts) and path.parts[0] in data_roots
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    return relative.parts[0] in data_roots or \
        path.suffix not in (".py", ".pyc")

def audit(event, args):
    if event == "open" and args and reads_data(args[0]):
        raise AssertionError("forward contract read data")

class NoTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise AssertionError("forward contract imported PyTorch")

sys.addaudithook(audit)
sys.meta_path.insert(0, NoTorch())

import tools.spy_residual_forward_contract as contract
expected = contract.expected_forward_protocol()
assert contract.validate_forward_protocol(expected) == expected
assert "torch" not in sys.modules
"""
    with tempfile.TemporaryDirectory() as directory:
        subprocess.run(
            (sys.executable, "-I", "-B", "-c", code),
            cwd=directory, check=True,
        )


def main() -> None:
    verify_exact_profile_and_literals()
    verify_fresh_copy()
    verify_required_rejections()
    verify_isolated_import()
    print("SPY residual forward contract tests passed")


if __name__ == "__main__":
    main()
