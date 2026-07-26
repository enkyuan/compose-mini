#!/usr/bin/env python3
"""Verify the exact Torch-free SPY-residual calibration protocol."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from test_context_diagnostic_finalizer import MASTER, phase_for
from tools.context_diagnostic_contract import (
    ContextPhase, _phase_value, context_phase_sha256,
)
from tools.panel_contract import read_canonical_json
from tools.relative_context_contract import (
    HISTORY_BARS, HORIZON_BARS, MODELS, PHASE_BUDGETS,
    SPY_RESIDUAL_TARGET, ResidualPhaseInput, ResidualScalerInput,
    expected_residual_fits, expected_residual_protocol,
    parse_residual_phases, residual_scaler_inputs_sha256,
    validate_residual_protocol, validate_spy_session_audit,
)

CONFIG = ROOT / "experiments/executable-h13-spy-residual.example.json"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def rejects(value: object) -> None:
    try:
        validate_residual_protocol(value)
    except ValueError:
        return
    raise AssertionError("invalid residual calibration protocol was accepted")


def raises(function: object, *args: object) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except (FrozenInstanceError, TypeError, ValueError):
        return
    raise AssertionError("invalid relative-context contract was accepted")


def scaler_inputs(
    phase: str = "fold-1",
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


def source_phase(name: str = "fold-1") -> ContextPhase:
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


def spy_audit() -> dict[str, object]:
    return {
        "scope": "all-expected-session-bins",
        "expected_sessions": 428,
        "affected_sessions": 0,
        "missing_sessions": [],
        "expected_bins": 5_534,
        "missing_bins": 0,
        "ranges": [],
    }


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
    assert value["alignment_horizon_bars"] == HORIZON_BARS
    assert value["metrics"]["primary"][0] == \
        "pooled-raw-residual-r2-vs-zero"
    assert value["paired_absolute_error_convention"] == \
        "reference-mae-minus-candidate-mae-positive"
    assert value["seed_aggregation"]["primary"] == \
        "arithmetic-mean-predictions-before-metrics"
    assert value["seed_aggregation"]["report"] == {
        "per_observation":
            "population-standard-deviation-across-seeds",
        "summary": "stock-macro-mean-over-common-grid",
    }
    assert value["bootstrap"]["applies_to"] == \
        "stock-macro-paired-absolute-error"
    assert value["sampling_policy"] == "stock-balanced"
    assert value["output_role"] == \
        "residual-only-not-executable-return"
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


def verify_scaler_closure() -> None:
    inputs = scaler_inputs()
    original = residual_scaler_inputs_sha256(MASTER, inputs)
    assert original == residual_scaler_inputs_sha256(MASTER, list(inputs))

    for index in (0, 44, 54):
        for field in (
            "stock_training_prefix_sha256",
            "spy_training_prefix_sha256",
            "training_rows",
            "training_grid_sha256",
        ):
            changed = list(inputs)
            value = changed[index].training_rows + 1 \
                if field == "training_rows" else digest(f"{index}-{field}")
            changed[index] = replace(changed[index], **{field: value})
            assert residual_scaler_inputs_sha256(MASTER, changed) != original

    raises(setattr, inputs[0], "training_rows", 2)
    for invalid in (
        inputs[:-1],
        (*inputs, inputs[0]),
        (inputs[1], inputs[0], *inputs[2:]),
        (*inputs[:1], inputs[0], *inputs[2:]),
        object(),
        "invalid",
        (item for item in inputs),
    ):
        raises(residual_scaler_inputs_sha256, MASTER, invalid)
    for field, value in (
        ("stock_training_prefix_sha256", "invalid"),
        ("spy_training_prefix_sha256", "invalid"),
        ("training_rows", 0),
        ("training_rows", True),
        ("training_grid_sha256", "invalid"),
        ("series", MASTER[1]),
    ):
        changed = list(inputs)
        changed[0] = replace(changed[0], **{field: value})
        raises(residual_scaler_inputs_sha256, MASTER, changed)
    raises(residual_scaler_inputs_sha256, MASTER[:-1], inputs[:-1])


def verify_phase_closure() -> None:
    sources = (source_phase(), source_phase("calibration"))
    scalers = tuple(
        residual_scaler_inputs_sha256(
            MASTER, scaler_inputs(phase.phase),
        )
        for phase in sources
    )
    values = [
        phase_value(phase, scaler)
        for phase, scaler in zip(sources, scalers, strict=True)
    ]
    parsed = parse_residual_phases(values, sources)
    assert tuple(item.phase for item in parsed) == (
        "fold-1", "calibration",
    )
    raises(setattr, parsed[0], "phase", "calibration")

    for phase in sources:
        assert phase.updates_per_checkpoint == \
            dict(PHASE_BUDGETS)[phase.phase]
        fits = expected_residual_fits(MASTER, phase)
        assert len(fits) == 11
        assert tuple(
            (fit.phase, fit.model, fit.history, fit.seed)
            for fit in fits
        ) == (
            (phase.phase, "global_ridge", HISTORY_BARS, None),
            *(
                (phase.phase, model, HISTORY_BARS, seed)
                for model in MODELS[1:]
                for seed in (7, 19, 31, 43, 61)
            ),
        )
        assert all(
            fit.optimizer_updates ==
            fit.selected_checkpoint * phase.updates_per_checkpoint
            for fit in fits[1:]
        )

    for invalid in (
        values[:-1],
        list(reversed(values)),
        tuple(values),
        object(),
    ):
        raises(parse_residual_phases, invalid, sources)
    raises(parse_residual_phases, values, tuple(reversed(sources)))
    raises(parse_residual_phases, values, object())
    raises(parse_residual_phases, values, (phase for phase in sources))
    wrong_budget = replace(sources[0], updates_per_checkpoint=1)
    raises(
        parse_residual_phases,
        [phase_value(wrong_budget, scalers[0]), values[1]],
        (wrong_budget, sources[1]),
    )
    raises(expected_residual_fits, MASTER, wrong_budget)
    for field in (
        "source_phase_sha256",
        "aligned_training_grid_sha256",
        "aligned_evaluation_grid_sha256",
    ):
        invalid = deepcopy(values)
        invalid[0][field] = digest(f"wrong-{field}")
        raises(parse_residual_phases, invalid, sources)
    invalid = deepcopy(values)
    invalid[0]["scaler_inputs_sha256"] = "invalid"
    raises(parse_residual_phases, invalid, sources)
    invalid = deepcopy(values)
    invalid[1]["scaler_inputs_sha256"] = invalid[0][
        "scaler_inputs_sha256"
    ]
    raises(parse_residual_phases, invalid, sources)
    invalid = deepcopy(values)
    invalid[0]["extra"] = True
    raises(parse_residual_phases, invalid, sources)


def verify_spy_audit() -> None:
    expected = spy_audit()
    assert validate_spy_session_audit(expected) == expected
    for field, value in (
        ("scope", "regular-session-bins"),
        ("expected_sessions", 427),
        ("affected_sessions", 1),
        ("affected_sessions", False),
        ("expected_bins", 5_539),
        ("missing_bins", 1),
        ("missing_bins", False),
        ("missing_sessions", ["2024-11-29"]),
        ("ranges", [{"absent_bins": 1}]),
    ):
        invalid = deepcopy(expected)
        invalid[field] = value
        raises(validate_spy_session_audit, invalid)
    invalid = deepcopy(expected)
    invalid.pop("scope")
    raises(validate_spy_session_audit, invalid)
    raises(validate_spy_session_audit, expected | {"extra": True})


def main() -> None:
    verify_exact_protocol()
    verify_rejections()
    verify_scaler_closure()
    verify_phase_closure()
    verify_spy_audit()
    print("relative-context contract tests passed")


if __name__ == "__main__":
    main()
