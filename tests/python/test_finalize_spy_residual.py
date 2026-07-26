#!/usr/bin/env python3
"""Verify residual metrics, common-grid joins, and non-trading closure."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from math import isclose, sqrt
from pathlib import Path
from unittest.mock import patch
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/python"))

from test_context_diagnostic_finalizer import (
    MASTER, truth_rows,
)
from test_relative_context_contract import source_phase
from tools.context_diagnostic_contract import (
    ContextPhase, context_phase_sha256,
)
from tools.finalize_spy_residual import (
    BOOTSTRAP_SEED, TERMINAL_PHASE_SHAPES, TERMINAL_SERIES,
    _paired_metrics, _spearman, evaluate_residual_phase,
    finalize_residual_run,
)
from tools.panel_contract import FileBinding
from tools.relative_context_contract import (
    MODELS, PAIRED_COMPARISONS, PHASE_BUDGETS, SEEDS,
    ResidualPhaseInput, ResidualPredictionEvidence, ResidualTruthRow,
    expected_residual_predictions, residual_fit_provenance_id,
)
from tools.universe_scaling import (
    BOOTSTRAP_BLOCK_DAYS, BOOTSTRAP_REPLICATES,
)
from tools.universe_scaling_contract import timestamp_grid_sha256

OFFSETS = dict(zip(SEEDS, (-0.002, -0.001, 0.0, 0.001, 0.002)))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def raises(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except (TypeError, ValueError):
        return
    raise AssertionError("expected residual finalizer failure")


def residual_phase(source: ContextPhase) -> ResidualPhaseInput:
    return ResidualPhaseInput(
        source.phase, context_phase_sha256(source),
        source.training_grid_sha256, source.evaluation_grid_sha256,
        digest(f"{source.phase}-residual-scalers"),
    )


def truth_for(
    source: ContextPhase,
) -> dict[str, tuple[ResidualTruthRow, ...]]:
    base = truth_rows()
    return {
        series: tuple(
            ResidualTruthRow(
                row.as_of, row.entry_time, row.target_time,
                (stock - 5) / 1_000 + (index % 3 - 1) / 10_000,
            )
            for index, row in enumerate(base[:count])
        )
        for stock, (series, count, _) in enumerate(source.evaluation_rows)
    }


def uneven_fixture() -> tuple[
    ContextPhase, dict[str, tuple[ResidualTruthRow, ...]],
]:
    source = source_phase()
    truth = truth_for(source)
    series = source.evaluation_rows[-1][0]
    truth[series] = truth[series][2:]
    grid = tuple(
        (row.as_of, row.entry, row.target) for row in truth[series]
    )
    source = replace(
        source,
        evaluation_grid_sha256=digest("uneven-evaluation"),
        evaluation_rows=(
            *source.evaluation_rows[:-1],
            (series, len(grid), timestamp_grid_sha256(grid)),
        ),
    )
    return source, truth


def evidence_for(
    source: ContextPhase, phase: ResidualPhaseInput,
    truth: dict[str, tuple[ResidualTruthRow, ...]],
    *,
    ridge_factor: float = 0.5,
    outside_common_scale: float = 1.0,
) -> tuple[ResidualPredictionEvidence, ...]:
    missing = source.evaluation_rows[-1][0]
    records = []
    for prediction in expected_residual_predictions(MASTER, source):
        fit = prediction.fit
        factor = {
            "global_ridge": ridge_factor,
            "global_mlp": 0.75,
            "panel_transformer": 1.0,
        }[fit.model]
        offset = 0.0 if fit.seed is None else OFFSETS[fit.seed]
        values = tuple(
            factor * row.value + offset * (
                outside_common_scale
                if prediction.series != missing and index < 2 else 1.0
            )
            for index, row in enumerate(truth[prediction.series])
        )
        records.append(ResidualPredictionEvidence(
            prediction,
            residual_fit_provenance_id(fit, source, phase, MASTER),
            digest(f"{fit}-state"), values,
        ))
    return tuple(records)


def evaluate(
    source: ContextPhase | None = None,
    truth: dict[str, tuple[ResidualTruthRow, ...]] | None = None,
    **evidence_options: float,
) -> dict[str, object]:
    source = source or source_phase()
    truth = truth or truth_for(source)
    phase = residual_phase(source)
    with patch(
        "tools.finalize_spy_residual.circular_block_interval",
        return_value=(-0.01, 0.01),
    ):
        return evaluate_residual_phase(
            MASTER, source, phase,
            evidence_for(source, phase, truth, **evidence_options), truth,
        )


def test_ensembles_precede_primary_metrics() -> dict[str, object]:
    result = evaluate()
    r2 = result["primary"]["pooled_raw_residual_r2_vs_zero"]
    assert isclose(r2["global_ridge"], 0.75)
    assert isclose(r2["global_mlp"], 0.9375)
    assert isclose(r2["panel_transformer"], 1.0)
    assert result["observation_count"] == 220
    assert result["stock_count"] == 11
    assert result["timestamp_count"] == 20
    assert all(
        comparison["mean_gain"] > 0.0
        for comparison in result["primary"]["paired_absolute_error"]
    )
    expected = sqrt(2.0) / 1_000
    assert all(isclose(value, expected) for value in
               result["seed_dispersion"].values())
    assert result["locks"] == {
        "absolute_forecast_authorized": False,
        "backtest_run": False,
        "forward_clean": False,
        "trading_authorized": False,
        "universe_expansion_authorized": False,
    }
    return result


def test_uneven_grids_join_exact_common_timestamps() -> None:
    source, truth = uneven_fixture()
    result = evaluate(
        source, truth, outside_common_scale=1_000.0,
    )
    assert result["observation_count"] == 218
    assert result["timestamp_count"] == 18
    assert isclose(
        result["secondary"]["centered_cross_sectional_r2"]
        ["panel_transformer"], 1.0,
    )
    assert all(
        isclose(value, sqrt(2.0) / 1_000)
        for value in result["seed_dispersion"].values()
    )


def test_stock_macro_gain_uses_the_common_target_dates() -> None:
    def row(
        as_of: str, target: str, value: float,
    ) -> ResidualTruthRow:
        return ResidualTruthRow(as_of, target, target, value)

    truth = {
        "A": (
            row("2026-01-01T10:00:00Z", "2026-01-02T10:00:00Z", 1.0),
            row("2026-01-02T10:00:00Z", "2026-01-02T11:00:00Z", 3.0),
            row("2026-01-03T10:00:00Z", "2026-01-03T11:00:00Z", 1_000.0),
        ),
        "B": (
            row("2026-01-02T10:00:00Z", "2026-01-02T11:00:00Z", 10.0),
        ),
    }
    predictions = {
        "global_ridge": {
            series: tuple(row.value for row in rows)
            for series, rows in truth.items()
        },
    }
    captured = []

    def interval(
        values: object, width: int, **kwargs: object,
    ) -> tuple[float, float]:
        captured.append((values, width, kwargs))
        return 0.0, 1.0

    with patch(
        "tools.finalize_spy_residual.circular_block_interval",
        side_effect=interval,
    ):
        result = _paired_metrics(
            truth, predictions, "global_ridge", "zero",
        )
    assert result["date_count"] == 1
    assert result["per_stock_mean_gain"] == {"A": 2.0, "B": 10.0}
    assert result["mean_gain"] == 6.0
    assert result["mean_gain"] != (1.0 + 3.0 + 10.0) / 3.0
    assert all(
        tuple(values[series]) == ("2026-01-02", "2026-01-03")
        if series == "A" else tuple(values[series]) == ("2026-01-02",)
        for values, _, _ in captured for series in values
    )
    worse = {
        "global_ridge": {
            series: tuple(-row.value for row in rows)
            for series, rows in truth.items()
        },
    }
    with patch(
        "tools.finalize_spy_residual.circular_block_interval",
        return_value=(-7.0, -5.0),
    ):
        assert _paired_metrics(
            truth, worse, "global_ridge", "zero",
        )["mean_gain"] == -6.0


def test_rank_ties_and_constant_timestamp_exclusion() -> None:
    assert isclose(_spearman((1.0, 1.0, 2.0), (1.0, 2.0, 2.0)), 0.5)
    source = source_phase()
    truth = truth_for(source)
    phase = residual_phase(source)
    evidence = evidence_for(source, phase, truth)
    first = tuple(
        replace(
            record,
            values=(0.0, *record.values[1:]),
        ) if record.prediction.fit.model == "global_ridge" else record
        for record in evidence
    )
    with patch(
        "tools.finalize_spy_residual.circular_block_interval",
        return_value=(-0.01, 0.01),
    ):
        result = evaluate_residual_phase(
            MASTER, source, phase, first, truth,
        )
    rank = result["secondary"]["spearman_rank_ic"]["global_ridge"]
    assert rank == {
        "excluded_timestamp_count": 1,
        "mean": 1.0,
        "valid_timestamp_count": 19,
    }


def test_shared_circular_date_bootstrap() -> None:
    source = source_phase()
    truth = truth_for(source)
    phase, calls = residual_phase(source), []

    def interval(
        values: object, width: int, **kwargs: object,
    ) -> tuple[float, float]:
        calls.append((values, width, kwargs))
        return -0.01, 0.01

    with patch(
        "tools.finalize_spy_residual.circular_block_interval",
        side_effect=interval,
    ):
        evaluate_residual_phase(
            MASTER, source, phase,
            evidence_for(source, phase, truth), truth,
        )
    assert len(calls) == len(PAIRED_COMPARISONS) * len(
        BOOTSTRAP_BLOCK_DAYS,
    )
    assert all(kwargs == {
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    } for _, _, kwargs in calls)
    date_grids = {
        tuple(sorted(set.intersection(*(
            set(by_date) for by_date in values.values()
        ))))
        for values, _, _ in calls
    }
    assert len(date_grids) == 1


def test_denominator_and_evidence_rejections() -> None:
    source = source_phase()
    truth = truth_for(source)
    phase = residual_phase(source)

    zero = {
        series: tuple(replace(row, value=0.0) for row in rows)
        for series, rows in truth.items()
    }
    raises(
        evaluate_residual_phase, MASTER, source, phase,
        evidence_for(source, phase, zero), zero,
    )
    common = {
        series: tuple(
            replace(row, value=(index + 1) / 1_000)
            for index, row in enumerate(rows)
        )
        for series, rows in truth.items()
    }
    constant_predictions = evidence_for(
        source, phase, truth, ridge_factor=0.0,
    )
    with patch(
        "tools.finalize_spy_residual.circular_block_interval",
        return_value=(-0.01, 0.01),
    ):
        raises(
            evaluate_residual_phase, MASTER, source, phase,
            evidence_for(source, phase, common), common,
        )
        raises(
            evaluate_residual_phase, MASTER, source, phase,
            constant_predictions, truth,
        )

    evidence = evidence_for(source, phase, truth)
    forged = replace(evidence[0], fit_provenance_id=digest("forged"))
    nonfinite = replace(
        evidence[0], values=(float("nan"), *evidence[0].values[1:]),
    )
    model_permuted = (
        *evidence[11:22], *evidence[:11], *evidence[22:],
    )
    seed_permuted = (
        *evidence[:11], *evidence[22:33], *evidence[11:22],
        *evidence[33:],
    )
    for changed in (
        evidence[:-1],
        (evidence[1], evidence[0], *evidence[2:]),
        model_permuted,
        seed_permuted,
        (forged, *evidence[1:]),
        (nonfinite, *evidence[1:]),
    ):
        raises(
            evaluate_residual_phase, MASTER, source, phase, changed, truth,
        )
    reversed_truth = dict(reversed(tuple(truth.items())))
    raises(
        evaluate_residual_phase, MASTER, source, phase,
        evidence, reversed_truth,
    )
    series = next(iter(truth))
    changed_truth = truth | {
        series: (truth[series][1], truth[series][0], *truth[series][2:]),
    }
    raises(
        evaluate_residual_phase, MASTER, source, phase,
        evidence, changed_truth,
    )


def binding(path: str) -> dict[str, str]:
    return {"path": path, "sha256": digest(path)}


def terminal_inputs() -> tuple[
    FileBinding, tuple[ResidualPhaseInput, ...],
    dict[str, dict[str, object]], list[dict[str, object]],
]:
    sources = (source_phase(), source_phase("calibration"))
    phases = tuple(map(residual_phase, sources))
    evaluations = {}
    for source in sources:
        value = evaluate(source, truth_for(source))
        shape = TERMINAL_PHASE_SHAPES[source.phase]
        value["observation_count"] = shape["observations"]
        value["timestamp_count"] = shape["timestamps"]
        for comparison in value["primary"]["paired_absolute_error"]:
            comparison["date_count"] = shape["dates"]
            gains = comparison["per_stock_mean_gain"]
            comparison["per_stock_mean_gain"] = dict(zip(
                TERMINAL_SERIES, gains.values(), strict=True,
            ))
        for rank in value["secondary"]["spearman_rank_ic"].values():
            rank["excluded_timestamp_count"] = 0
            rank["valid_timestamp_count"] = shape["timestamps"]
        evaluations[source.phase] = value
    inputs = [
        {
            "access": binding(f"reports/run/{phase.phase}-access.json"),
            "evaluation":
                binding(f"reports/run/{phase.phase}-evaluation.json"),
            "fits": binding(f"reports/run/{phase.phase}-fits.jsonl"),
            "phase": phase.phase,
            "predictions":
                binding(f"reports/run/{phase.phase}-predictions.jsonl"),
            "receipt": binding(f"reports/run/{phase.phase}-receipt.json"),
        }
        for phase in phases
    ]
    return (
        FileBinding("experiments/residual-attempt.json", digest("attempt")),
        phases, evaluations, inputs,
    )


def test_terminal_closure_rejects_metric_tampering() -> None:
    attempt, phases, evaluations, inputs = terminal_inputs()
    output = finalize_residual_run(
        attempt, phases, evaluations, inputs, digest("source-tree"),
    )
    assert set(output) == {
        "schema", "evidence_role", "inputs", "phases", "decision", "locks",
        "integrity",
    }
    assert output["decision"] == {
        "output_role": "residual-only-not-executable-return",
    }

    for mutate in (
        lambda value: value["fold-1"]["primary"]
            ["paired_absolute_error"][0].__setitem__("wins", 0),
        lambda value: value["fold-1"]["primary"]
            ["paired_absolute_error"][1].__setitem__("date_count", 19),
        lambda value: value["fold-1"]["secondary"]
            ["spearman_rank_ic"]["global_ridge"].__setitem__("mean", 1.1),
        lambda value: value["fold-1"]["locks"].__setitem__(
            "backtest_run", 0,
        ),
        lambda value: value["fold-1"].__setitem__(
            "observation_count", value["fold-1"]["observation_count"] + 1,
        ),
    ):
        changed = deepcopy(evaluations)
        mutate(changed)
        raises(
            finalize_residual_run, attempt, phases, changed, inputs,
            digest("source-tree"),
        )
    inflated = deepcopy(evaluations)
    phase = inflated["fold-1"]
    phase["timestamp_count"] += 1
    phase["observation_count"] += phase["stock_count"]
    for rank in phase["secondary"]["spearman_rank_ic"].values():
        rank["valid_timestamp_count"] += 1
    raises(
        finalize_residual_run, attempt, phases, inflated, inputs,
        digest("source-tree"),
    )
    renamed = deepcopy(evaluations)
    for comparison in renamed["fold-1"]["primary"][
        "paired_absolute_error"
    ]:
        gains = comparison["per_stock_mean_gain"]
        gains["FAKE"] = gains.pop(TERMINAL_SERIES[0])
    raises(
        finalize_residual_run, attempt, phases, renamed, inputs,
        digest("source-tree"),
    )
    duplicate = deepcopy(inputs)
    duplicate[1]["fits"] = duplicate[0]["fits"]
    raises(
        finalize_residual_run, attempt, phases, evaluations, duplicate,
        digest("source-tree"),
    )
    raises(
        finalize_residual_run, attempt, tuple(reversed(phases)),
        evaluations, inputs, digest("source-tree"),
    )


def main() -> None:
    test_ensembles_precede_primary_metrics()
    test_uneven_grids_join_exact_common_timestamps()
    test_stock_macro_gain_uses_the_common_target_dates()
    test_rank_ties_and_constant_timestamp_exclusion()
    test_shared_circular_date_bootstrap()
    test_denominator_and_evidence_rejections()
    test_terminal_closure_rejects_metric_tampering()
    print("SPY residual finalizer tests passed")


if __name__ == "__main__":
    main()
