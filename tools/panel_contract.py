"""Validate immutable inputs for one calibration-only panel experiment."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from types import MappingProxyType
import hashlib
import json
import math
import os
import re
import stat
import subprocess

from tools.files import FrozenInput, file_sha256, freeze_inputs, verify_frozen

SOURCE_PATHS = (
    "tools/experiment.py",
    "tools/panel_contract.py",
    "tools/train.py",
    "tools/artifact_v1.py",
    "tools/data_v1.py",
    "tools/backtest.py",
    "tools/files.py",
    "tools/float32.py",
    "tools/analyze_panel.py",
    "tools/run_panel_attempt.py",
    "tools/finalize_panel_attempt.py",
)
FINALIZER_SOURCE_PATHS = (
    "tools/finalize_panel_attempt.py",
    "tools/panel_contract.py",
    "tools/files.py",
)
COMMANDS = ("validate_attempt", "preflight", "experiment", "analyze")
OUTPUTS = (
    "experiment_report", "calibration_ledger", "analysis_report", "outcome",
)
NAME = re.compile(r"[A-Za-z0-9._-]{1,64}")
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,63}")
HEX = re.compile(r"[0-9a-f]{64}")
LOCAL_MODELS = (
    "transformer", "linear", "mlp", "rolling_mean", "last_close",
)
SEEDS = (7, 19, 31, 43, 61)
SERIES = ("AAPL", "MSFT", "SPY")
TARGET_KIND = "executable-return-v1"
MINIMUM_RELATIVE_IMPROVEMENT = 0.01
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_724
BOOTSTRAP_BLOCKS = (13, 29, 39)


@dataclass(frozen=True)
class PanelProfile:
    models: tuple[str, ...]
    panel_models: tuple[str, ...]
    candidate: str
    reference: str
    expected_runs: int
    expected_panel_fits: int
    analysis_schema: int


LEGACY_PROFILE = PanelProfile(
    (*LOCAL_MODELS, "panel_transformer"),
    ("panel_transformer",),
    "panel_transformer", "transformer", 162, 15, 1,
)
COMPARISON_PROFILE = PanelProfile(
    (*LOCAL_MODELS, "panel_transformer", "conditioned_panel_transformer"),
    ("panel_transformer", "conditioned_panel_transformer"),
    "conditioned_panel_transformer", "panel_transformer", 207, 30, 2,
)
PROFILES = (LEGACY_PROFILE, COMPARISON_PROFILE)


def _exact_json(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(
            _exact_json(value[key], item) for key, item in expected.items()
        )
    if isinstance(expected, list):
        return len(value) == len(expected) and all(
            _exact_json(item, other)
            for item, other in zip(value, expected, strict=True)
        )
    return value == expected


def expected_panel_sweep(profile: PanelProfile) -> dict[str, object]:
    if profile not in PROFILES:
        raise ValueError("unknown panel profile")
    return {
        "alignment_horizon_bars": 13,
        "batch_size": 128,
        "candidates": [{
            "feature_set": "ohlcv", "ff_dim": 32, "heads": 2, "layers": 1,
            "learning_rate": 0.0003, "mlp_dim": 32, "model_dim": 16,
            "name": "raw-17", "ridge": 0.001, "rolling_window": 8,
            "seq_len": 17, "weight_decay": 0.0001,
        }],
        "epochs": 100,
        "fold_fraction": 0.1,
        "folds": 2,
        "models": list(profile.models),
        "patience": 10,
        "seeds": list(SEEDS),
        "target_horizon_bars": 13,
        "target_kind": TARGET_KIND,
    }


def panel_profile(config: Mapping[str, object]) -> PanelProfile:
    matches = tuple(
        profile for profile in PROFILES
        if _exact_json(config, expected_panel_sweep(profile))
    )
    if len(matches) != 1:
        raise ValueError("config does not match an exact panel profile")
    return matches[0]


def _object(value: object, fields: set[str], label: str
            ) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is invalid")
    return value


def _integer(value: object, label: str, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} is invalid")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError(f"{label} is invalid")
    return value


def _relative(value: object, label: str) -> str:
    path = Path(_string(value, label))
    if path.is_absolute() or not path.parts or \
       any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{label} must be a normalized relative path")
    return path.as_posix()


def _analysis_float(
    value: object, label: str, minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value) or \
       minimum is not None and value < minimum or \
       maximum is not None and value > maximum:
        raise ValueError(f"{label} is invalid")
    return value


def _analysis_binding(value: object, label: str) -> None:
    item = _object(value, {"path", "sha256"}, label)
    _string(item["path"], f"{label}.path")
    _sha256(item["sha256"], f"{label}.sha256")


def _analysis_inputs(value: object) -> None:
    item = _object(
        value,
        {
            "run_id", "attempt", "input_manifest", "config",
            "baseline_report", "baseline_ledger", "experiment_report",
            "calibration_ledger", "series",
        },
        "analysis.inputs",
    )
    _string(item["run_id"], "analysis.inputs.run_id")
    for name in (
        "attempt", "input_manifest", "config", "baseline_report",
        "baseline_ledger", "experiment_report", "calibration_ledger",
    ):
        _analysis_binding(item[name], f"analysis.inputs.{name}")
    raw_series = item["series"]
    if not isinstance(raw_series, list) or len(raw_series) != len(SERIES):
        raise ValueError("analysis.inputs.series is invalid")
    for index, (entry, name) in enumerate(zip(
        raw_series, SERIES, strict=True,
    )):
        record = _object(
            entry, {"name", "path", "sha256"},
            f"analysis.inputs.series[{index}]",
        )
        if record["name"] != name:
            raise ValueError("analysis input series order is invalid")
        _string(record["path"], "analysis series path")
        _sha256(record["sha256"], "analysis series sha256")


def panel_analysis_protocol(profile: PanelProfile) -> dict[str, object]:
    if profile not in PROFILES:
        raise ValueError("unknown panel analysis profile")
    protocol: dict[str, object] = {
        "candidate": "raw-17",
        "models": list(profile.models),
        "seeds": list(SEEDS),
        "series": list(SERIES),
        "folds": 2,
        "fold_fraction": 0.1,
        "target_horizon_bars": 13,
        "target_kind": TARGET_KIND,
        "series_equivalent_runs": profile.expected_runs,
        "physical_panel_fits": profile.expected_panel_fits,
        "validation_pair": "stock/fold/seed",
        "calibration_ensemble":
            "arithmetic mean by model/stock/target before metrics",
        "macro_unit": "stock",
        "majority_reference": "unique actual calibration targets",
    }
    if profile == COMPARISON_PROFILE:
        protocol.update({
            "candidate_model": profile.candidate,
            "reference_model": profile.reference,
            "minimum_relative_mae_improvement":
                MINIMUM_RELATIVE_IMPROVEMENT,
            "calibration_role": "reused-development-diagnostic",
            "uncertainty_interpretation":
                "conditional-descriptive-not-confirmatory",
            "bootstrap": {
                "kind": "paired-noncircular-moving-block",
                "block_rows": list(BOOTSTRAP_BLOCKS),
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
                "lower_percentile": 0.025,
                "gate_aggregation":
                    "minimum-lower-percentile-across-block-lengths",
            },
        })
    return protocol


def _model_metric_map(
    value: object, models: Sequence[str], label: str,
    *, direction: bool = False,
) -> Mapping[str, object]:
    item = _object(value, set(models), label)
    for model in models:
        _analysis_float(
            item[model], f"{label}.{model}", 0.0, 1.0 if direction else None,
        )
    return item


def _stock_models(value: object, profile: PanelProfile, label: str) -> None:
    models = _object(value, set(profile.models), label)
    for model in profile.models:
        metrics = _object(
            models[model],
            {"return_mae", "direction_accuracy", "close_mae"},
            f"{label}.{model}",
        )
        _analysis_float(
            metrics["return_mae"], f"{label}.{model}.return_mae", 0.0,
        )
        _analysis_float(
            metrics["direction_accuracy"],
            f"{label}.{model}.direction_accuracy", 0.0, 1.0,
        )
        _analysis_float(
            metrics["close_mae"], f"{label}.{model}.close_mae", 0.0,
        )


def _majority(value: object, label: str) -> None:
    item = _object(value, {"p_up", "p_down", "p_flat", "reference"}, label)
    for name in item:
        _analysis_float(item[name], f"{label}.{name}", 0.0, 1.0)
    if not math.isclose(
        item["p_up"] + item["p_down"] + item["p_flat"], 1.0,
        rel_tol=0.0, abs_tol=1e-12,
    ) or item["reference"] != max(
        item["p_up"], item["p_down"], item["p_flat"],
    ):
        raise ValueError(f"{label} is inconsistent")


def _legacy_comparators(values: Mapping[str, object]) -> dict[str, float]:
    return {
        "local_transformer": float(values["transformer"]),
        "mlp": float(values["mlp"]),
        "linear": float(values["linear"]),
        "rolling_mean": float(values["rolling_mean"]),
        "zero_return": float(values["last_close"]),
    }


def _legacy_gates(
    validation: Mapping[str, object], calibration: Mapping[str, object],
) -> dict[str, object]:
    validation_macro = validation["macro_return_mae"]
    validation_panel = validation_macro[LEGACY_PROFILE.candidate]
    validation_comparators = _legacy_comparators(validation_macro)
    validation_margin = min(validation_comparators.values()) - \
        validation_panel
    paired = validation["paired_panel_minus_local_transformer"]
    stock_delta = paired["per_stock_mean_delta"]
    paired_pass = paired["mean_delta"] < 0.0 and paired["wins"] >= 20 and all(
        value <= 0.0 for value in stock_delta.values()
    )

    calibration_macro = calibration["macro_return_mae"]
    calibration_panel = calibration_macro[LEGACY_PROFILE.candidate]
    calibration_comparators = _legacy_comparators(calibration_macro)
    calibration_margin = min(calibration_comparators.values()) - \
        calibration_panel
    per_stock = calibration["per_stock"]
    zero = {
        name: {
            "panel": stock["models"][
                LEGACY_PROFILE.candidate
            ]["return_mae"],
            "zero_return": stock["zero_return_return_mae"],
            "margin": (
                stock["zero_return_return_mae"] -
                stock["models"][LEGACY_PROFILE.candidate]["return_mae"]
            ),
        }
        for name, stock in per_stock.items()
    }
    for item in zero.values():
        item["pass"] = item["margin"] > 0.0

    direction = calibration[
        "macro_direction_accuracy"
    ][LEGACY_PROFILE.candidate]
    majority = calibration["macro_majority_direction"]
    stock_direction = {
        name: {
            "panel": stock["models"][
                LEGACY_PROFILE.candidate
            ]["direction_accuracy"],
            "minimum": 0.5,
            "margin": (
                stock["models"][LEGACY_PROFILE.candidate][
                    "direction_accuracy"
                ] - 0.5
            ),
        }
        for name, stock in per_stock.items()
    }
    for item in stock_direction.values():
        item["pass"] = item["margin"] >= 0.0
    close = calibration["mean_panel_close_relative_improvement"]

    gates = {
        "validation_macro_mae": {
            "pass": validation_margin > 0.0,
            "panel": validation_panel,
            "comparators": validation_comparators,
            "margin": validation_margin,
        },
        "validation_paired": {
            "pass": paired_pass,
            "mean_delta": paired["mean_delta"],
            "wins": paired["wins"],
            "required_wins": 20,
            "per_stock_mean_delta": stock_delta,
        },
        "calibration_macro_mae": {
            "pass": calibration_margin > 0.0,
            "panel": calibration_panel,
            "comparators": calibration_comparators,
            "margin": calibration_margin,
        },
        "calibration_per_stock_zero": {
            "pass": all(item["pass"] for item in zero.values()),
            "per_stock": zero,
        },
        "calibration_direction": {
            "pass": direction > majority and
                    all(item["pass"] for item in stock_direction.values()),
            "panel_macro": direction,
            "majority_macro": majority,
            "macro_margin": direction - majority,
            "per_stock": stock_direction,
        },
        "calibration_close_mae": {
            "pass": close > 0.0,
            "mean_relative_improvement": close,
            "margin": close,
        },
    }
    gates["all_pass"] = all(
        item["pass"] for item in gates.values()
        if isinstance(item, dict)
    )
    return gates


def _comparison_gates(
    validation: Mapping[str, object], calibration: Mapping[str, object],
    profile: PanelProfile,
) -> dict[str, object]:
    required = MINIMUM_RELATIVE_IMPROVEMENT
    validation_macro = validation["macro_return_mae"]
    validation_candidate = validation_macro[profile.candidate]
    validation_comparators = _comparison_comparators(validation_macro)
    validation_margin = min(validation_comparators.values()) - \
        validation_candidate
    paired = validation["paired_candidate_minus_reference"]
    paired_pass = paired["mean_delta"] < 0.0 and paired["wins"] >= 20 and \
        all(item["mean"] < 0.0 for item in paired["by_stock"].values()) and \
        all(item["mean"] < 0.0 for item in paired["by_fold"].values()) and \
        sum(
            item["mean"] < 0.0 for item in paired["by_seed"].values()
        ) >= 4

    calibration_macro = calibration["macro_return_mae"]
    calibration_candidate = calibration_macro[profile.candidate]
    calibration_comparators = _comparison_comparators(calibration_macro)
    calibration_margin = min(calibration_comparators.values()) - \
        calibration_candidate
    relative = calibration["relative_improvement_vs_reference"]
    leave_out = calibration["leave_one_seed_out"]
    bootstrap = calibration["bootstrap"]
    bootstrap_lower = bootstrap["mae_relative_improvement_lower_025"]
    calibration_macro_pass = relative >= required and \
        calibration_candidate < min(calibration_comparators.values()) and \
        bootstrap_lower >= required and \
        all(
            item["relative_improvement"] > 0.0
            for item in leave_out.values()
        )

    per_stock = calibration["per_stock"]
    zero = {
        name: {
            "candidate_return_mae":
                stock["models"][profile.candidate]["return_mae"],
            "reference_return_mae":
                stock["models"][profile.reference]["return_mae"],
            "zero_return_mae": stock["zero_return_return_mae"],
            "reference_margin": (
                stock["models"][profile.reference]["return_mae"] -
                stock["models"][profile.candidate]["return_mae"]
            ),
            "zero_margin": (
                stock["zero_return_return_mae"] -
                stock["models"][profile.candidate]["return_mae"]
            ),
        }
        for name, stock in per_stock.items()
    }
    for item in zero.values():
        item["reference_pass"] = item["reference_margin"] > 0.0
        item["zero_pass"] = item["zero_margin"] > 0.0

    candidate_direction = calibration[
        "macro_direction_accuracy"
    ][profile.candidate]
    reference_direction = calibration[
        "macro_direction_accuracy"
    ][profile.reference]
    majority = calibration["macro_majority_direction"]
    reference_margin = candidate_direction - reference_direction
    majority_margin = candidate_direction - majority
    stock_direction = {
        name: {
            "candidate":
                stock["models"][profile.candidate]["direction_accuracy"],
            "reference":
                stock["models"][profile.reference]["direction_accuracy"],
            "minimum": 0.5,
            "reference_margin": (
                stock["models"][profile.candidate]["direction_accuracy"] -
                stock["models"][profile.reference]["direction_accuracy"]
            ),
            "minimum_margin": (
                stock["models"][profile.candidate]["direction_accuracy"] - 0.5
            ),
        }
        for name, stock in per_stock.items()
    }
    for item in stock_direction.values():
        item["reference_pass"] = item["reference_margin"] > 0.0
        item["minimum_pass"] = item["minimum_margin"] >= 0.0
    reference_lower = bootstrap[
        "direction_candidate_minus_reference_lower_025"
    ]
    majority_lower = bootstrap[
        "direction_candidate_minus_majority_lower_025"
    ]
    close_zero = calibration[
        "mean_candidate_close_relative_improvement_over_zero"
    ]
    close_reference = calibration[
        "mean_candidate_close_relative_improvement_over_reference"
    ]
    gates = {
        "validation_macro_mae": {
            "pass": paired["relative_improvement"] >= required and
                    validation_candidate < min(
                        validation_comparators.values()
                    ),
            "candidate_model": profile.candidate,
            "reference_model": profile.reference,
            "candidate": validation_candidate,
            "comparators": validation_comparators,
            "margin": validation_margin,
            "relative_improvement": paired["relative_improvement"],
            "required_relative_improvement": required,
        },
        "validation_paired": {
            "pass": paired_pass,
            "candidate_model": profile.candidate,
            "reference_model": profile.reference,
            "mean_delta": paired["mean_delta"],
            "wins": paired["wins"],
            "ties": paired["ties"],
            "losses": paired["losses"],
            "required_wins": 20,
            "by_stock": paired["by_stock"],
            "by_fold": paired["by_fold"],
            "by_seed": paired["by_seed"],
            "required_improving_seeds": 4,
        },
        "calibration_macro_mae": {
            "pass": calibration_macro_pass,
            "candidate_model": profile.candidate,
            "reference_model": profile.reference,
            "candidate": calibration_candidate,
            "comparators": calibration_comparators,
            "margin": calibration_margin,
            "relative_improvement": relative,
            "required_relative_improvement": required,
            "bootstrap_lower_025": bootstrap_lower,
            "leave_one_seed_out": leave_out,
        },
        "calibration_per_stock_zero": {
            "pass": all(
                item["reference_pass"] and item["zero_pass"]
                for item in zero.values()
            ),
            "candidate_model": profile.candidate,
            "reference_model": profile.reference,
            "per_stock": zero,
        },
        "calibration_direction": {
            "pass": reference_margin > 0.0 and majority_margin > 0.0 and
                    reference_lower > 0.0 and majority_lower > 0.0 and
                    all(
                        item["reference_pass"] and item["minimum_pass"]
                        for item in stock_direction.values()
                    ),
            "candidate_model": profile.candidate,
            "reference_model": profile.reference,
            "candidate_macro": candidate_direction,
            "reference_macro": reference_direction,
            "majority_macro": majority,
            "reference_margin": reference_margin,
            "majority_margin": majority_margin,
            "reference_bootstrap_lower_025": reference_lower,
            "majority_bootstrap_lower_025": majority_lower,
            "per_stock": stock_direction,
        },
        "calibration_close_mae": {
            "pass": close_zero > 0.0 and close_reference > 0.0,
            "candidate_model": profile.candidate,
            "reference_model": profile.reference,
            "mean_relative_improvement_over_zero": close_zero,
            "mean_relative_improvement_over_reference": close_reference,
        },
    }
    gates["all_pass"] = all(
        item["pass"] for item in gates.values()
        if isinstance(item, dict)
    )
    return gates


def panel_gates(
    validation: Mapping[str, object], calibration: Mapping[str, object],
    profile: PanelProfile = LEGACY_PROFILE,
) -> dict[str, object]:
    if profile == LEGACY_PROFILE:
        return _legacy_gates(validation, calibration)
    if profile == COMPARISON_PROFILE:
        return _comparison_gates(validation, calibration, profile)
    raise ValueError("unknown panel analysis profile")


def _legacy_analysis(
    validation_value: object, calibration_value: object, gates_value: object,
) -> bool:
    validation = _object(
        validation_value,
        {"macro_return_mae", "paired_panel_minus_local_transformer"},
        "analysis.validation",
    )
    validation_macro = _model_metric_map(
        validation["macro_return_mae"], LEGACY_PROFILE.models,
        "analysis.validation.macro_return_mae",
    )
    paired = _object(
        validation["paired_panel_minus_local_transformer"],
        {"mean_delta", "wins", "ties", "losses", "per_stock_mean_delta"},
        "analysis.validation.paired",
    )
    _analysis_float(paired["mean_delta"], "paired mean delta")
    counts = tuple(
        _integer(paired[name], f"paired {name}", 0)
        for name in ("wins", "ties", "losses")
    )
    if sum(counts) != 30:
        raise ValueError("paired counts are invalid")
    stock_delta = _object(
        paired["per_stock_mean_delta"], set(SERIES), "paired stock deltas",
    )
    for name in SERIES:
        _analysis_float(stock_delta[name], f"paired stock delta {name}")

    calibration = _object(
        calibration_value,
        {
            "macro_return_mae", "macro_direction_accuracy",
            "macro_majority_direction", "mean_panel_close_relative_improvement",
            "per_stock",
        },
        "analysis.calibration",
    )
    calibration_macro = _model_metric_map(
        calibration["macro_return_mae"], LEGACY_PROFILE.models,
        "analysis.calibration.macro_return_mae",
    )
    calibration_direction = _model_metric_map(
        calibration["macro_direction_accuracy"], LEGACY_PROFILE.models,
        "analysis.calibration.macro_direction_accuracy", direction=True,
    )
    majority = _analysis_float(
        calibration["macro_majority_direction"],
        "analysis.calibration.macro_majority_direction", 0.0, 1.0,
    )
    close = _analysis_float(
        calibration["mean_panel_close_relative_improvement"],
        "analysis.calibration.mean_panel_close_relative_improvement",
    )
    per_stock = _object(
        calibration["per_stock"], set(SERIES), "analysis.calibration.per_stock",
    )
    for name in SERIES:
        stock = _object(
            per_stock[name],
            {
                "samples", "models", "majority_direction",
                "zero_return_return_mae", "zero_return_close_mae",
                "panel_close_relative_improvement",
            },
            f"analysis.calibration.per_stock.{name}",
        )
        _integer(stock["samples"], f"{name} samples")
        _stock_models(stock["models"], LEGACY_PROFILE, f"{name} models")
        _majority(stock["majority_direction"], f"{name} majority")
        _analysis_float(
            stock["zero_return_return_mae"], f"{name} zero return MAE", 0.0,
        )
        _analysis_float(
            stock["zero_return_close_mae"], f"{name} zero close MAE", 0.0,
        )
        _analysis_float(
            stock["panel_close_relative_improvement"],
            f"{name} close improvement",
        )

    expected = panel_gates(validation, calibration, LEGACY_PROFILE)
    if not _exact_json(gates_value, expected):
        raise ValueError("legacy analysis gates are inconsistent")
    return bool(expected["all_pass"])


def _axis(
    value: object, keys: Sequence[str], count: int, label: str,
) -> Mapping[str, object]:
    axis = _object(value, set(keys), label)
    for key in keys:
        stats = _object(
            axis[key], {"count", "mean", "stddev", "minimum", "maximum"},
            f"{label}.{key}",
        )
        if _integer(stats["count"], f"{label}.{key}.count") != count:
            raise ValueError(f"{label}.{key}.count is invalid")
        for name in ("mean", "stddev", "minimum", "maximum"):
            _analysis_float(
                stats[name], f"{label}.{key}.{name}",
                0.0 if name == "stddev" else None,
            )
        if not stats["minimum"] <= stats["mean"] <= stats["maximum"]:
            raise ValueError(f"{label}.{key} bounds are invalid")
    return axis


def _require_exact(
    actual: float, expected: float, label: str,
) -> None:
    if actual != expected:
        raise ValueError(f"{label} is inconsistent")


def _require_regrouped(
    actual: float, expected: float, label: str,
) -> None:
    relation = lambda value: (value > 0.0) - (value < 0.0)
    if relation(actual) != relation(expected) or abs(actual - expected) > \
            4 * max(math.ulp(actual), math.ulp(expected)):
        raise ValueError(f"{label} is inconsistent")


def _weighted_axis_mean(axis: Mapping[str, object]) -> float:
    count = sum(item["count"] for item in axis.values())
    return math.fsum(
        item["count"] * item["mean"] for item in axis.values()
    ) / count


def _comparison_comparators(
    values: Mapping[str, object],
) -> dict[str, float]:
    return {
        "unconditioned_panel": float(values["panel_transformer"]),
        "local_transformer": float(values["transformer"]),
        "mlp": float(values["mlp"]),
        "linear": float(values["linear"]),
        "rolling_mean": float(values["rolling_mean"]),
        "zero_return": float(values["last_close"]),
    }


def _comparison_analysis(
    validation_value: object, calibration_value: object, gates_value: object,
) -> bool:
    profile = COMPARISON_PROFILE
    validation = _object(
        validation_value,
        {"macro_return_mae", "paired_candidate_minus_reference"},
        "analysis.validation",
    )
    validation_macro = _model_metric_map(
        validation["macro_return_mae"], profile.models,
        "analysis.validation.macro_return_mae",
    )
    paired = _object(
        validation["paired_candidate_minus_reference"],
        {
            "candidate_model", "reference_model", "relative_improvement",
            "mean_delta", "wins", "ties", "losses",
            "by_stock", "by_fold", "by_seed",
        },
        "analysis.validation.paired",
    )
    if paired["candidate_model"] != profile.candidate or \
       paired["reference_model"] != profile.reference:
        raise ValueError("paired model identity is invalid")
    relative = _analysis_float(
        paired["relative_improvement"], "paired relative improvement",
    )
    _analysis_float(paired["mean_delta"], "paired mean delta")
    counts = tuple(
        _integer(paired[name], f"paired {name}", 0)
        for name in ("wins", "ties", "losses")
    )
    if sum(counts) != 30:
        raise ValueError("paired counts are invalid")
    by_stock = _axis(paired["by_stock"], SERIES, 10, "paired.by_stock")
    by_fold = _axis(paired["by_fold"], ("0", "1"), 15, "paired.by_fold")
    seed_keys = tuple(str(seed) for seed in SEEDS)
    by_seed = _axis(paired["by_seed"], seed_keys, 6, "paired.by_seed")
    reference_validation = float(validation_macro[profile.reference])
    if reference_validation <= 0.0 or relative != 1.0 - \
            float(validation_macro[profile.candidate]) / reference_validation:
        raise ValueError("validation relative improvement is inconsistent")
    mean_delta = float(paired["mean_delta"])
    _require_regrouped(
        mean_delta,
        float(validation_macro[profile.candidate]) - reference_validation,
        "paired mean delta",
    )
    for label, axis in (
        ("stock", by_stock), ("fold", by_fold), ("seed", by_seed),
    ):
        _require_regrouped(
            mean_delta, _weighted_axis_mean(axis),
            f"paired {label} mean delta",
        )

    calibration = _object(
        calibration_value,
        {
            "macro_return_mae", "macro_direction_accuracy",
            "macro_majority_direction", "relative_improvement_vs_reference",
            "leave_one_seed_out", "bootstrap",
            "mean_candidate_close_relative_improvement_over_zero",
            "mean_candidate_close_relative_improvement_over_reference",
            "per_stock",
        },
        "analysis.calibration",
    )
    calibration_macro = _model_metric_map(
        calibration["macro_return_mae"], profile.models,
        "analysis.calibration.macro_return_mae",
    )
    calibration_direction = _model_metric_map(
        calibration["macro_direction_accuracy"], profile.models,
        "analysis.calibration.macro_direction_accuracy", direction=True,
    )
    majority = _analysis_float(
        calibration["macro_majority_direction"],
        "analysis.calibration.macro_majority_direction", 0.0, 1.0,
    )
    calibration_relative = _analysis_float(
        calibration["relative_improvement_vs_reference"],
        "analysis.calibration.relative_improvement_vs_reference",
    )
    calibration_reference = float(calibration_macro[profile.reference])
    if calibration_reference <= 0.0 or calibration_relative != 1.0 - \
            float(calibration_macro[profile.candidate]) / calibration_reference:
        raise ValueError("calibration relative improvement is inconsistent")
    leave_out = _object(
        calibration["leave_one_seed_out"], set(seed_keys),
        "analysis.calibration.leave_one_seed_out",
    )
    for seed in seed_keys:
        item = _object(
            leave_out[seed], {"relative_improvement"},
            f"leave_one_seed_out.{seed}",
        )
        _analysis_float(
            item["relative_improvement"],
            f"leave_one_seed_out.{seed}.relative_improvement",
        )

    bootstrap = _object(
        calibration["bootstrap"],
        {
            "by_block_rows", "mae_relative_improvement_lower_025",
            "direction_candidate_minus_reference_lower_025",
            "direction_candidate_minus_majority_lower_025",
        },
        "analysis.calibration.bootstrap",
    )
    block_keys = tuple(str(block) for block in BOOTSTRAP_BLOCKS)
    by_block = _object(
        bootstrap["by_block_rows"], set(block_keys),
        "analysis.calibration.bootstrap.by_block_rows",
    )
    bootstrap_names = (
        "mae_relative_improvement_lower_025",
        "direction_candidate_minus_reference_lower_025",
        "direction_candidate_minus_majority_lower_025",
    )
    for block in block_keys:
        item = _object(
            by_block[block], set(bootstrap_names),
            f"analysis.calibration.bootstrap.by_block_rows.{block}",
        )
        for name in bootstrap_names:
            _analysis_float(item[name], f"bootstrap {block} {name}")
    for name in bootstrap_names:
        value = _analysis_float(bootstrap[name], f"bootstrap {name}")
        if value != min(by_block[block][name] for block in block_keys):
            raise ValueError("bootstrap aggregate is inconsistent")

    candidate_close_zero = _analysis_float(
        calibration[
            "mean_candidate_close_relative_improvement_over_zero"
        ],
        "mean candidate close improvement over zero",
    )
    candidate_close_reference = _analysis_float(
        calibration[
            "mean_candidate_close_relative_improvement_over_reference"
        ],
        "mean candidate close improvement over reference",
    )
    per_stock = _object(
        calibration["per_stock"], set(SERIES), "analysis.calibration.per_stock",
    )
    for name in SERIES:
        stock = _object(
            per_stock[name],
            {
                "samples", "models", "majority_direction",
                "zero_return_return_mae", "zero_return_close_mae",
                "candidate_close_relative_improvement_over_zero",
                "candidate_close_relative_improvement_over_reference",
            },
            f"analysis.calibration.per_stock.{name}",
        )
        _integer(stock["samples"], f"{name} samples")
        _stock_models(stock["models"], profile, f"{name} models")
        _majority(stock["majority_direction"], f"{name} majority")
        zero_return = _analysis_float(
            stock["zero_return_return_mae"], f"{name} zero return MAE", 0.0,
        )
        zero_close = _analysis_float(
            stock["zero_return_close_mae"], f"{name} zero close MAE", 0.0,
        )
        last_close = stock["models"]["last_close"]
        _require_exact(
            float(last_close["return_mae"]), zero_return,
            f"{name} last-close return MAE",
        )
        _require_exact(
            float(last_close["close_mae"]), zero_close,
            f"{name} last-close close MAE",
        )
        reference_close = stock["models"][profile.reference]["close_mae"]
        candidate_close = stock["models"][profile.candidate]["close_mae"]
        if zero_close <= 0.0 or reference_close <= 0.0:
            raise ValueError("close MAE denominator is invalid")
        expected_zero = 1.0 - candidate_close / zero_close
        expected_reference = 1.0 - candidate_close / reference_close
        reported_zero = _analysis_float(
            stock["candidate_close_relative_improvement_over_zero"],
            f"{name} close improvement over zero",
        )
        reported_reference = _analysis_float(
            stock["candidate_close_relative_improvement_over_reference"],
            f"{name} close improvement over reference",
        )
        if reported_zero != expected_zero or \
           reported_reference != expected_reference:
            raise ValueError("stock close improvement is inconsistent")

    for model in profile.models:
        _require_exact(
            float(calibration_macro[model]),
            fmean(
                float(per_stock[name]["models"][model]["return_mae"])
                for name in SERIES
            ),
            f"{model} macro return MAE",
        )
        _require_exact(
            float(calibration_direction[model]),
            fmean(
                float(
                    per_stock[name]["models"][model]["direction_accuracy"]
                )
                for name in SERIES
            ),
            f"{model} macro direction accuracy",
        )
    _require_exact(
        majority,
        fmean(
            float(per_stock[name]["majority_direction"]["reference"])
            for name in SERIES
        ),
        "macro majority direction",
    )
    _require_exact(
        candidate_close_zero,
        fmean(
            float(
                per_stock[name][
                    "candidate_close_relative_improvement_over_zero"
                ]
            )
            for name in SERIES
        ),
        "mean close improvement over zero",
    )
    _require_exact(
        candidate_close_reference,
        fmean(
            float(
                per_stock[name][
                    "candidate_close_relative_improvement_over_reference"
                ]
            )
            for name in SERIES
        ),
        "mean close improvement over reference",
    )

    expected = panel_gates(validation, calibration, profile)
    if not _exact_json(gates_value, expected):
        raise ValueError("comparison analysis gates are inconsistent")
    return bool(expected["all_pass"])


def validate_panel_analysis(
    value: object, profile: PanelProfile,
) -> None:
    if profile not in PROFILES:
        raise ValueError("unknown panel analysis profile")
    analysis = _object(
        value,
        {"schema", "status", "inputs", "protocol",
         "validation", "calibration", "gates"},
        "analysis",
    )
    if _integer(analysis["schema"], "analysis.schema") != \
            profile.analysis_schema or \
       analysis["status"] not in ("pass", "gate-failure") or \
       not _exact_json(
           analysis["protocol"], panel_analysis_protocol(profile),
       ):
        raise ValueError("analysis profile is invalid")
    _analysis_inputs(analysis["inputs"])
    passed = (
        _legacy_analysis(
            analysis["validation"], analysis["calibration"], analysis["gates"],
        )
        if profile == LEGACY_PROFILE else
        _comparison_analysis(
            analysis["validation"], analysis["calibration"], analysis["gates"],
        )
    )
    if analysis["status"] != ("pass" if passed else "gate-failure"):
        raise ValueError("analysis status and gates disagree")


def _finite_tree(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("manifest numbers must be finite")
    if isinstance(value, Mapping):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            _finite_tree(item)


def _pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise ValueError(f"duplicate manifest field: {key}")
        result[key] = value
    return result


def read_canonical_json(path: Path) -> Mapping[str, object]:
    """Decode one finite, duplicate-free canonical JSON object."""
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw, object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"invalid numeric constant: {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read canonical manifest: {error}") from error
    _finite_tree(value)
    if not isinstance(value, dict) or raw != (
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode():
        raise ValueError("manifest must be a canonical JSON object")
    return value


def read_canonical_json_lines(
    path: Path,
) -> tuple[Mapping[str, object], ...]:
    """Decode finite, duplicate-free, canonical JSON objects by line."""
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise ValueError(f"cannot read canonical ledger: {error}") from error
    if not lines:
        raise ValueError("canonical ledger must not be empty")
    values = []
    for line in lines:
        try:
            value = json.loads(
                line, object_pairs_hook=_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"invalid numeric constant: {token}")
                ),
            )
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read canonical ledger: {error}") from error
        _finite_tree(value)
        if not isinstance(value, dict) or line != (
            json.dumps(value, allow_nan=False, sort_keys=True) + "\n"
        ).encode():
            raise ValueError("ledger rows must be canonical JSON objects")
        values.append(value)
    return tuple(values)


@contextmanager
def _open_parent(path: Path) -> Iterator[tuple[int, str]]:
    """Open each lexical parent without following directory symlinks."""
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | \
        getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:-1]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor, absolute.name
    finally:
        os.close(descriptor)


def _regular_identity(path: Path) -> tuple[int, int]:
    try:
        with _open_parent(path) as (parent, name):
            metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"input is unavailable: {path}") from error
    if stat.S_IFMT(metadata.st_mode) != stat.S_IFREG:
        raise ValueError(f"input must be a nonsymlink regular file: {path}")
    return metadata.st_dev, metadata.st_ino


def _directory_identity(path: Path) -> tuple[int, int]:
    try:
        with _open_parent(path) as (parent, name):
            metadata = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"directory is unavailable: {path}") from error
    if stat.S_IFMT(metadata.st_mode) != stat.S_IFDIR:
        raise ValueError(f"directory must be nonsymlink: {path}")
    return metadata.st_dev, metadata.st_ino


def _open_directory(path: Path) -> tuple[int, tuple[int, int]]:
    try:
        with _open_parent(path) as (parent, name):
            descriptor = os.open(
                name, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
                getattr(os, "O_NOFOLLOW", 0), dir_fd=parent,
            )
    except OSError as error:
        raise ValueError(f"directory is unavailable: {path}") from error
    metadata = os.fstat(descriptor)
    if stat.S_IFMT(metadata.st_mode) != stat.S_IFDIR:
        os.close(descriptor)
        raise ValueError(f"directory must be nonsymlink: {path}")
    return descriptor, (metadata.st_dev, metadata.st_ino)


def mkdir_nofollow(path: Path) -> None:
    """Create one directory only through nonsymlink lexical parents."""
    try:
        with _open_parent(path) as (parent, name):
            os.mkdir(name, dir_fd=parent)
            metadata = os.stat(
                name, dir_fd=parent, follow_symlinks=False,
            )
    except OSError as error:
        raise ValueError(f"cannot create directory: {path}") from error
    identity = metadata.st_dev, metadata.st_ino
    if stat.S_IFMT(metadata.st_mode) != stat.S_IFDIR or \
       _directory_identity(path) != identity:
        raise ValueError(f"created directory changed: {path}")


def _regular_inputs(
    paths: Sequence[Path],
) -> tuple[tuple[Path, tuple[int, int]], ...]:
    identities = tuple(_regular_identity(path) for path in paths)
    if len(identities) != len(set(identities)):
        raise ValueError("panel inputs must not alias each other")
    return tuple(zip(paths, identities, strict=True))


def regular_file_identities(
    paths: Sequence[Path],
) -> tuple[tuple[int, int], ...]:
    return tuple(identity for _, identity in _regular_inputs(paths))


def _verify_identities(
    identities: Sequence[tuple[Path, tuple[int, int]]],
) -> None:
    if any(_regular_identity(path) != identity
           for path, identity in identities):
        raise ValueError("panel input identity changed during the command")


def _absent(path: Path, label: str) -> None:
    try:
        aliases = (
            path, Path(os.path.abspath(path)), path.resolve(strict=False),
        )
    except (OSError, RuntimeError) as error:
        raise ValueError(f"{label} path is invalid") from error
    if any(os.path.lexists(alias) for alias in aliases):
        raise ValueError(f"{label} must be absent")


def resolve_fresh_output(path: Path) -> Path:
    """Reject lexical aliases and return one absent resolved output path."""
    _absent(path, "output")
    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError("output path is invalid") from error


def _tree_digest(files: Sequence[FileBinding]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files, key=lambda file: file.path):
        digest.update(
            item.path.encode("utf-8") + b"\0" +
            item.sha256.encode("ascii") + b"\n"
        )
    return digest.hexdigest()


@dataclass(frozen=True)
class FileBinding:
    path: str
    sha256: str

    @classmethod
    def parse(cls, value: object, label: str, *, relative: bool = True
              ) -> FileBinding:
        item = _object(value, {"path", "sha256"}, label)
        path = (_relative(item["path"], f"{label}.path") if relative else
                _string(item["path"], f"{label}.path"))
        return cls(path, _sha256(item["sha256"], f"{label}.sha256"))

    def validate(self, frozen: FrozenInput, label: str) -> None:
        if str(frozen.source) != self.path or frozen.sha256 != self.sha256:
            raise ValueError(f"{label} does not match its frozen binding")


@dataclass(frozen=True)
class ExecutableBinding(FileBinding):
    version: str

    @classmethod
    def parse(cls, value: object, label: str) -> ExecutableBinding:
        item = _object(value, {"path", "sha256", "version"}, label)
        path = _string(item["path"], f"{label}.path")
        if not Path(path).is_absolute():
            raise ValueError(f"{label}.path must be absolute")
        return cls(
            path, _sha256(item["sha256"], f"{label}.sha256"),
            _string(item["version"], f"{label}.version"),
        )

    def validate_live(self, label: str) -> None:
        path = Path(self.path)
        _regular_identity(path)
        if file_sha256(path) != self.sha256 or \
           _command_version(path) != self.version:
            raise ValueError(f"{label} runtime identity changed")


@dataclass(frozen=True)
class SourceTree:
    root: str
    files: tuple[FileBinding, ...]
    sha256: str

    @classmethod
    def parse(cls, value: object, label: str,
              expected: Sequence[str] | None = None) -> SourceTree:
        item = _object(value, {"root", "files", "sha256"}, label)
        root = _string(item["root"], f"{label}.root")
        if not Path(root).is_absolute():
            raise ValueError(f"{label}.root must be absolute")
        raw = item["files"]
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"{label}.files must be nonempty")
        files = tuple(
            FileBinding.parse(entry, f"{label}.files[{index}]")
            for index, entry in enumerate(raw)
        )
        paths = tuple(file.path for file in files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)) or \
           expected is not None and set(paths) != set(expected):
            raise ValueError(f"{label}.files are invalid")
        digest = _sha256(item["sha256"], f"{label}.sha256")
        if digest != _tree_digest(files):
            raise ValueError(f"{label} tree digest is invalid")
        return cls(root, files, digest)

    def validate(self, frozen: Mapping[str, FrozenInput], label: str) -> None:
        if set(frozen) != {item.path for item in self.files}:
            raise ValueError(f"{label} source set changed")
        for item in self.files:
            if frozen[item.path].sha256 != item.sha256:
                raise ValueError(f"{label} source changed")


@dataclass(frozen=True)
class TorchIdentity:
    python: ExecutableBinding
    version: str
    git_version: str | None
    cuda_version: str | None
    config: str
    package_tree: SourceTree

    @classmethod
    def parse(cls, value: object) -> TorchIdentity:
        item = _object(
            value,
            {
                "python", "version", "git_version", "cuda_version", "config",
                "package_tree",
            },
            "runtime.torch_probe",
        )
        optional = (item["git_version"], item["cuda_version"])
        if any(value is not None and not isinstance(value, str)
               for value in optional):
            raise ValueError("Torch optional runtime fields are invalid")
        return cls(
            ExecutableBinding.parse(item["python"], "runtime.torch_python"),
            _string(item["version"], "runtime.torch_version"),
            item["git_version"], item["cuda_version"],
            _string(item["config"], "runtime.torch_config"),
            SourceTree.parse(item["package_tree"], "runtime.package_tree"),
        )


@dataclass(frozen=True)
class PanelSeries:
    name: str
    csv: FileBinding
    rows: int
    first_timestamp: str
    last_timestamp: str
    timestamp_sha256: str

    @classmethod
    def parse(cls, value: object, index: int) -> PanelSeries:
        label = f"series[{index}]"
        item = _object(
            value,
            {
                "name", "csv", "rows", "first_timestamp",
                "last_timestamp", "timestamp_sha256",
            },
            label,
        )
        name = _string(item["name"], f"{label}.name")
        if not NAME.fullmatch(name):
            raise ValueError(f"{label}.name is invalid")
        return cls(
            name,
            FileBinding.parse(item["csv"], f"{label}.csv"),
            _integer(item["rows"], f"{label}.rows"),
            _string(item["first_timestamp"], f"{label}.first_timestamp"),
            _string(item["last_timestamp"], f"{label}.last_timestamp"),
            _sha256(
                item["timestamp_sha256"], f"{label}.timestamp_sha256",
            ),
        )


@dataclass(frozen=True)
class PanelInputs:
    series: tuple[PanelSeries, ...]
    baseline_report: FileBinding
    baseline_ledger: FileBinding

    @classmethod
    def read(cls, path: Path) -> PanelInputs:
        value = _object(
            read_canonical_json(path),
            {"schema", "series", "baseline_report", "baseline_ledger"},
            "input manifest",
        )
        if _integer(value["schema"], "schema") != 1 or \
           not isinstance(value["series"], list) or \
           not value["series"]:
            raise ValueError("input manifest schema or series is invalid")
        series = tuple(
            PanelSeries.parse(item, index)
            for index, item in enumerate(value["series"])
        )
        if len({item.name for item in series}) != len(series):
            raise ValueError("input manifest series names must be unique")
        return cls(
            series,
            FileBinding.parse(value["baseline_report"], "baseline_report"),
            FileBinding.parse(value["baseline_ledger"], "baseline_ledger"),
        )

    def validate_direct(
        self, series: Sequence[tuple[str, FrozenInput]],
        baseline_report: FrozenInput, baseline_ledger: FrozenInput,
    ) -> None:
        if tuple(name for name, _ in series) != \
                tuple(item.name for item in self.series):
            raise ValueError("series order does not match the input manifest")
        for declared, (_, frozen) in zip(self.series, series, strict=True):
            declared.csv.validate(frozen, f"{declared.name} CSV")
        self.baseline_report.validate(baseline_report, "baseline report")
        self.baseline_ledger.validate(baseline_ledger, "baseline ledger")

    def validate_timestamps(
        self, series: Sequence[tuple[str, FrozenInput, Sequence[str]]],
    ) -> None:
        if tuple(name for name, _, _ in series) != \
                tuple(item.name for item in self.series):
            raise ValueError("series order does not match the input manifest")
        for declared, (_, frozen, timestamps) in zip(
            self.series, series, strict=True,
        ):
            declared.csv.validate(frozen, f"{declared.name} CSV")
            if not timestamps:
                raise ValueError(f"{declared.name} timestamp grid is empty")
            digest = hashlib.sha256(
                "".join(f"{timestamp}\n" for timestamp in timestamps).encode(
                    "ascii",
                )
            ).hexdigest()
            if len(timestamps) != declared.rows or \
               timestamps[0] != declared.first_timestamp or \
               timestamps[-1] != declared.last_timestamp or \
               digest != declared.timestamp_sha256:
                raise ValueError(
                    f"{declared.name} timestamps do not match the input manifest"
                )


def _argv(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or \
       any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{label} must be a nonempty argument array")
    return tuple(value)


@dataclass(frozen=True)
class PanelAttempt:
    run_id: str
    run_dir: str
    implementation_commit: str
    input_manifest: FileBinding
    config: FileBinding
    baseline_report: FileBinding
    baseline_ledger: FileBinding
    source_tree: SourceTree
    finalizer_tree: SourceTree
    primary_python: ExecutableBinding
    uv: ExecutableBinding
    torch_argv: tuple[str, ...]
    torch_probe: TorchIdentity
    environment: Mapping[str, str]
    commands: Mapping[str, tuple[str, ...]]
    expected_equivalent_runs: int
    expected_panel_fits: int
    outputs: Mapping[str, str]

    @classmethod
    def read(cls, path: Path) -> PanelAttempt:
        value = _object(
            read_canonical_json(path),
            {
                "schema", "run_id", "status", "run_dir",
                "implementation_commit", "input_manifest", "config",
                "baseline_report", "baseline_ledger", "source_tree",
                "finalizer_tree", "primary_python", "uv", "torch_argv",
                "torch_probe", "environment", "commands",
                "expected_equivalent_runs", "expected_panel_fits", "outputs",
            },
            "attempt manifest",
        )
        if _integer(value["schema"], "schema") != 1 or \
           value["status"] != "armed":
            raise ValueError("attempt must be schema 1 and armed")
        run_id = _string(value["run_id"], "run_id")
        if not RUN_ID.fullmatch(run_id):
            raise ValueError("run_id is invalid")
        commit = _string(value["implementation_commit"],
                         "implementation_commit")
        if len(commit) not in (40, 64) or \
           any(byte not in "0123456789abcdef" for byte in commit):
            raise ValueError("implementation_commit is invalid")
        environment = _object(
            value["environment"],
            {"PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX"},
            "environment",
        )
        if environment["PYTHONDONTWRITEBYTECODE"] != "1":
            raise ValueError("bytecode must be disabled")
        commands = _object(
            value["commands"], {*COMMANDS, "finalizer_prefix"}, "commands",
        )
        parsed_commands = {
            name: _argv(commands[name], f"commands.{name}")
            for name in (*COMMANDS, "finalizer_prefix")
        }
        outputs = _object(value["outputs"], set(OUTPUTS), "outputs")
        parsed_outputs = {
            name: _relative(outputs[name], f"outputs.{name}")
            for name in OUTPUTS
        }
        run_dir = _relative(value["run_dir"], "run_dir")
        if environment["PYTHONPYCACHEPREFIX"] != \
                f"{run_dir}/.pycache" or \
           parsed_outputs["experiment_report"] != \
                f"{run_dir}/experiment.json" or \
           parsed_outputs["calibration_ledger"] != \
                f"{run_dir}/calibration.jsonl" or \
           parsed_outputs["analysis_report"] != \
                f"{run_dir}/analysis.json" or \
           len({
               Path(path).resolve(strict=False)
               for path in parsed_outputs.values()
           }) != len(parsed_outputs):
            raise ValueError("attempt output or cache paths are invalid")
        source = SourceTree.parse(
            value["source_tree"], "source_tree", SOURCE_PATHS,
        )
        finalizer = SourceTree.parse(
            value["finalizer_tree"], "finalizer_tree",
            FINALIZER_SOURCE_PATHS,
        )
        if source.root != finalizer.root:
            raise ValueError("source roots do not match")
        uv = ExecutableBinding.parse(value["uv"], "uv")
        torch_argv = _argv(value["torch_argv"], "torch_argv")
        if torch_argv != (
            uv.path, "run", "--offline", "--with", "torch", "python",
        ):
            raise ValueError("torch_argv is invalid")
        return cls(
            run_id, run_dir, commit,
            FileBinding.parse(value["input_manifest"], "input_manifest"),
            FileBinding.parse(value["config"], "config"),
            FileBinding.parse(value["baseline_report"], "baseline_report"),
            FileBinding.parse(value["baseline_ledger"], "baseline_ledger"),
            source, finalizer,
            ExecutableBinding.parse(value["primary_python"], "primary_python"),
            uv, torch_argv, TorchIdentity.parse(value["torch_probe"]),
            MappingProxyType(dict(environment)),
            MappingProxyType(parsed_commands),
            _integer(
                value["expected_equivalent_runs"],
                "expected_equivalent_runs",
            ),
            _integer(value["expected_panel_fits"], "expected_panel_fits"),
            MappingProxyType(parsed_outputs),
        )

    def source_paths(self) -> tuple[Path, ...]:
        paths = dict.fromkeys(
            item.path for tree in (self.source_tree, self.finalizer_tree)
            for item in tree.files
        )
        return tuple(Path(self.source_tree.root) / path for path in paths)

    def validate_stage(
        self, stage: str, argv: Sequence[str], environment: Mapping[str, str],
        torch_probe: TorchIdentity,
    ) -> None:
        if stage not in COMMANDS or tuple(argv) != self.commands[stage]:
            raise ValueError("calling stage arguments do not match the attempt")
        if dict(self.environment) != {
            name: environment.get(name) for name in self.environment
        }:
            raise ValueError("panel environment does not match the attempt")
        self.primary_python.validate_live("primary Python")
        self.uv.validate_live("uv")
        if torch_probe != self.torch_probe:
            raise ValueError("Torch runtime identity changed")

    def validate_paths(self, stage: str) -> None:
        if stage == "validate_attempt":
            return
        run_dir = Path(self.run_dir)
        cache = Path(self.environment["PYTHONPYCACHEPREFIX"])
        if stage == "preflight":
            _absent(run_dir, "run directory")
        elif not run_dir.is_dir() or run_dir.is_symlink():
            raise ValueError("run directory must be a nonsymlink directory")
        _absent(cache, "Python cache prefix")
        _absent(Path(self.outputs["outcome"]), "attempt outcome")
        if stage == "experiment":
            for name in (
                "experiment_report", "calibration_ledger", "analysis_report",
            ):
                _absent(Path(self.outputs[name]), f"{name} output")
        elif stage == "analyze":
            for name in ("experiment_report", "calibration_ledger"):
                _regular_identity(Path(self.outputs[name]))
            _absent(Path(self.outputs["analysis_report"]), "analysis output")


@dataclass(frozen=True)
class PanelExecution:
    attempt_input: FrozenInput
    input_manifest_input: FrozenInput
    config_input: FrozenInput
    baseline_report_input: FrozenInput
    baseline_ledger_input: FrozenInput
    source_inputs: Mapping[str, FrozenInput]
    attempt: PanelAttempt
    inputs: PanelInputs
    series: tuple[tuple[str, FrozenInput], ...]
    torch_probe: TorchIdentity
    observe_torch: Callable[[], TorchIdentity]
    argv: tuple[str, ...]
    frozen: tuple[FrozenInput, ...]
    identities: tuple[tuple[Path, tuple[int, int]], ...]
    run_directory_fd: int
    run_directory_identity: tuple[int, int]

    def validate(self) -> None:
        self.verify()
        self.attempt.input_manifest.validate(
            self.input_manifest_input, "input manifest",
        )
        self.attempt.config.validate(self.config_input, "sweep config")
        self.attempt.baseline_report.validate(
            self.baseline_report_input, "baseline report",
        )
        self.attempt.baseline_ledger.validate(
            self.baseline_ledger_input, "baseline ledger",
        )
        self.attempt.source_tree.validate(
            {
                item.path: self.source_inputs[item.path]
                for item in self.attempt.source_tree.files
            },
            "implementation",
        )
        self.attempt.finalizer_tree.validate(
            {
                item.path: self.source_inputs[item.path]
                for item in self.attempt.finalizer_tree.files
            },
            "finalizer",
        )
        self.attempt.validate_stage(
            "experiment", self.argv, os.environ, self.torch_probe,
        )
        self.attempt.validate_paths("experiment")
        self.inputs.validate_direct(
            self.series, self.baseline_report_input,
            self.baseline_ledger_input,
        )

    def verify(self) -> None:
        verify_frozen(self.frozen)
        _verify_identities(self.identities)
        metadata = os.fstat(self.run_directory_fd)
        if (metadata.st_dev, metadata.st_ino) != \
                self.run_directory_identity or \
           _directory_identity(Path(self.attempt.run_dir)) != \
                self.run_directory_identity:
            raise ValueError("panel run directory changed")
        if self.observe_torch() != self.attempt.torch_probe:
            raise ValueError("Torch runtime identity changed")

    def validate_outputs(
        self, report: Path, ledger: Path | None,
    ) -> None:
        actual = (str(report), None if ledger is None else str(ledger))
        expected = (
            self.attempt.outputs["experiment_report"],
            self.attempt.outputs["calibration_ledger"],
        )
        if actual != expected:
            raise ValueError("panel outputs do not match the attempt")

    def prepare_output(self, name: str, path: Path) -> None:
        if str(path) != self.attempt.outputs[name]:
            raise ValueError("panel output does not match the attempt")
        self.verify()
        try:
            os.stat(
                path.name, dir_fd=self.run_directory_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        raise ValueError(f"{name} output must be absent")

    def provenance(self) -> dict[str, object]:
        return {
            "attempt_manifest": {
                "path": str(self.attempt_input.source),
                "sha256": self.attempt_input.sha256,
                "run_id": self.attempt.run_id,
            },
            "input_manifest": {
                "path": str(self.input_manifest_input.source),
                "sha256": self.input_manifest_input.sha256,
            },
        }


def _command_version(path: Path) -> str:
    try:
        result = subprocess.run(
            (str(path), "--version"), check=True, capture_output=True,
            text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"cannot identify runtime executable: {path}") from error
    return (result.stdout or result.stderr).strip()


def executable_binding(path: Path, version: str) -> ExecutableBinding:
    resolved = path.resolve(strict=True)
    _regular_identity(resolved)
    return ExecutableBinding(str(resolved), file_sha256(resolved), version)


def source_tree(root: Path) -> SourceTree:
    """Hash every nonsymlink regular file below one resolved package root."""
    resolved = root.resolve(strict=True)
    files = []
    for path in resolved.rglob("*"):
        metadata = path.stat(follow_symlinks=False)
        if stat.S_IFMT(metadata.st_mode) == stat.S_IFREG:
            files.append(FileBinding(
                path.relative_to(resolved).as_posix(), file_sha256(path),
            ))
    files.sort(key=lambda item: item.path)
    if not files:
        raise ValueError("runtime package contains no regular files")
    return SourceTree(str(resolved), tuple(files), _tree_digest(files))


def selected_source_tree(
    root: Path, paths: Sequence[str],
) -> SourceTree:
    """Hash exactly one declared set of nonsymlink source files."""
    resolved = root.resolve(strict=True)
    relative = tuple(_relative(path, "source path") for path in paths)
    if not relative or len(relative) != len(set(relative)):
        raise ValueError("selected source paths must be unique and nonempty")
    files = []
    for item in sorted(relative):
        path = resolved / item
        _regular_identity(path)
        files.append(FileBinding(item, file_sha256(path)))
    return SourceTree(str(resolved), tuple(files), _tree_digest(files))


def observe_torch(
    torch_argv: Sequence[str], root: Path,
) -> TorchIdentity:
    script = (
        "from dataclasses import asdict\n"
        "import json\n"
        "from tools.experiment import _torch_identity\n"
        "print(json.dumps(asdict(_torch_identity()),"
        "allow_nan=False,sort_keys=True))\n"
    )
    try:
        result = subprocess.run(
            (*torch_argv, "-c", script), cwd=root, check=True,
            capture_output=True, text=True, timeout=300,
        )
        value = json.loads(result.stdout)
    except (
        OSError, subprocess.SubprocessError, json.JSONDecodeError,
    ) as error:
        raise ValueError("cannot observe the bound Torch runtime") from error
    return TorchIdentity.parse(value)


def expected_panel_commands(
    attempt_path: Path,
    input_manifest_path: str,
    config_path: str,
    baseline_report_path: str,
    baseline_ledger_path: str,
    outputs: Mapping[str, str],
    inputs: PanelInputs,
    profile: PanelProfile,
) -> Mapping[str, tuple[str, ...]]:
    if profile not in PROFILES or set(outputs) != set(OUTPUTS):
        raise ValueError("panel command profile is invalid")
    common = (
        str(attempt_path), input_manifest_path, config_path,
        baseline_report_path, baseline_ledger_path,
    )
    series = tuple(f"{item.name}={item.csv.path}" for item in inputs.series)
    analyzer = "tools/analyze_panel.py"
    return {
        "validate_attempt": (
            analyzer, "validate-attempt", *common, *series,
        ),
        "preflight": (analyzer, "preflight", *common, *series),
        "experiment": (
            "tools/experiment.py", config_path,
            outputs["experiment_report"], *series,
            "--attempt-manifest", str(attempt_path),
            "--input-manifest", input_manifest_path,
            "--baseline-report", baseline_report_path,
            "--baseline-ledger", baseline_ledger_path,
            "--device", "cpu", "--calibration-only",
            "--calibration-predictions", outputs["calibration_ledger"],
            "--max-runs", str(profile.expected_runs),
        ),
        "analyze": (
            analyzer, "analyze", *common,
            outputs["experiment_report"], outputs["calibration_ledger"],
            outputs["analysis_report"], *series,
        ),
        "finalizer_prefix": (
            "tools/finalize_panel_attempt.py", str(attempt_path),
            outputs["outcome"],
        ),
    }


@contextmanager
def freeze_panel_execution(
    attempt_path: Path, input_manifest_path: Path, config_path: Path,
    baseline_report_path: Path, baseline_ledger_path: Path,
    series: Sequence[tuple[str, Path]], root: Path,
    observe_torch: Callable[[], TorchIdentity], argv: Sequence[str],
) -> Iterator[PanelExecution]:
    """Freeze the discovered panel closure and yield one validated execution."""
    direct = (
        attempt_path, input_manifest_path, config_path, baseline_report_path,
        baseline_ledger_path, *(path for _, path in series),
    )
    discovery_identity = _regular_inputs(direct)
    with freeze_inputs((attempt_path,)) as discovery:
        discovered = PanelAttempt.read(discovery[0].snapshot)
        if Path(discovered.source_tree.root) != root.resolve(strict=True):
            raise ValueError("attempt source root does not match the repository")
        source_paths = discovered.source_paths()
        unique_sources = tuple(dict.fromkeys(source_paths))
        executable_paths = tuple(dict.fromkeys(map(Path, (
            discovered.primary_python.path, discovered.uv.path,
            discovered.torch_probe.python.path,
        ))))
        paths = tuple(dict.fromkeys(
            (*direct, *unique_sources, *executable_paths),
        ))
        identities = _regular_inputs(paths)
        _verify_identities(discovery_identity)
        with freeze_inputs(paths) as frozen:
            verify_frozen(discovery)
            _verify_identities(identities)
            by_source = dict(zip(paths, frozen, strict=True))
            attempt = PanelAttempt.read(by_source[attempt_path].snapshot)
            if attempt != discovered:
                raise ValueError("attempt changed during source discovery")
            for binding, label in (
                (attempt.primary_python, "primary Python"),
                (attempt.uv, "uv"),
                (attempt.torch_probe.python, "Torch Python"),
            ):
                binding.validate(by_source[Path(binding.path)], label)
            source_inputs = MappingProxyType({
                item.path: by_source[Path(tree.root) / item.path]
                for tree in (attempt.source_tree, attempt.finalizer_tree)
                for item in tree.files
            })
            directory_fd, directory_identity = _open_directory(
                Path(attempt.run_dir),
            )
            try:
                execution = PanelExecution(
                    by_source[attempt_path], by_source[input_manifest_path],
                    by_source[config_path], by_source[baseline_report_path],
                    by_source[baseline_ledger_path], source_inputs, attempt,
                    PanelInputs.read(by_source[input_manifest_path].snapshot),
                    tuple(
                        (name, by_source[path]) for name, path in series
                    ),
                    observe_torch(), observe_torch, tuple(argv), frozen,
                    identities, directory_fd, directory_identity,
                )
                execution.validate()
                yield execution
                execution.verify()
            finally:
                os.close(directory_fd)
