#!/usr/bin/env python3
"""Verify zero-anchored residual scaling and its input boundary."""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from math import isclose
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tools.analyze_spy_residual_shrinkage as analyzer
from tools.analyze_spy_residual_shrinkage import (
    _analyze_alignment, _analyze_phases, _final_report, _fit_report,
    _market_regimes, _phase_market_regimes, _publish, _validate_published,
    alignment_diagnostic, evaluate_frozen_scale, fit_diagnostic,
    fit_zero_anchored_scale, pooled_r2, scale_predictions,
    zero_anchored_scale,
)
from tools.arm_spy_residual import ResidualLease
from tools.data_v1 import FEATURE_COUNT
from tools.files import FrozenInput
from tools.panel_contract import FileBinding, _directory_identity
from tools.relative_context_contract import (
    MODELS, ResidualTruthRow, expected_residual_protocol,
)
from tools.session_samples import SampleRows
from tools.spy_residual_controller import _PhaseRows
from tools.universe_contract import PackedRows

REPORT_INPUTS = {
    "analysis_source": {
        "path": "tools/analyze_spy_residual_shrinkage.py",
        "sha256": "a" * 64,
    },
    "attempt": {"path": "experiments/attempt.json", "sha256": "b" * 64},
    "outcome": {"path": "experiments/outcome.json", "sha256": "c" * 64},
    "phases": [],
}


def rows(*values: float) -> tuple[ResidualTruthRow, ...]:
    return tuple(
        ResidualTruthRow(
            f"2026-01-{index + 1:02d}T14:00:00Z",
            f"2026-01-{index + 1:02d}T14:05:00Z",
            f"2026-01-{index + 1:02d}T15:05:00Z",
            value,
        )
        for index, value in enumerate(values)
    )


def panel_fixture() -> tuple[
    dict[str, tuple[ResidualTruthRow, ...]],
    dict[str, dict[str, tuple[float, ...]]],
    tuple[tuple[str, str, str], ...],
    dict[str, object],
]:
    names = tuple(f"S{index:02d}" for index in range(11))
    truth = {
        name: rows(*([float(index + 1)] * 20))
        for index, name in enumerate(names)
    }
    transformer = {
        name: tuple(2.0 * row.value for row in values)
        for name, values in truth.items()
    }
    reference = {
        name: tuple(0.1 * row.value for row in values)
        for name, values in truth.items()
    }
    return truth, {
        model: transformer if model == "panel_transformer" else reference
        for model in MODELS
    }, tuple(
        (row.as_of, row.entry, row.target)
        for row in next(iter(truth.values()))
    ), {
        "excluded_timestamp_count": 0,
        "mean": 1.0,
        "valid_timestamp_count": 20,
    }


def rejects(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except (OSError, TypeError, ValueError):
        return
    raise AssertionError("invalid shrinkage input was accepted")


def bars(*closes: float) -> tuple[float, ...]:
    return tuple(
        value
        for close in closes
        for value in (close, close, close, close, 1.0)
    )


def test_scale_is_the_clipped_mse_minimizer() -> None:
    truth = {"A": rows(1.0, -1.0)}
    fit = fit_zero_anchored_scale(truth, {"A": (2.0, -2.0)})
    assert (
        fit.scale, fit.unclipped_scale, fit.numerator,
        fit.denominator, fit.observation_count,
    ) == (0.5, 0.5, 4.0, 8.0, 2)
    assert zero_anchored_scale(truth, {"A": (2.0, -2.0)}) == 0.5
    assert zero_anchored_scale(truth, {"A": (-2.0, 2.0)}) == 0.0
    assert zero_anchored_scale(truth, {"A": (0.5, -0.5)}) == 1.0
    assert zero_anchored_scale(truth, {"A": (0.0, 0.0)}) == 0.0


def test_scale_is_global_and_leave_one_stock_out() -> None:
    truth = {"A": rows(1.0), "B": rows(0.0)}
    predictions = {"A": (2.0,), "B": (2.0,)}
    assert zero_anchored_scale(truth, predictions) == 0.25
    assert zero_anchored_scale(truth, predictions, ("B",)) == 0.5
    assert zero_anchored_scale(truth, predictions, ("A",)) == 0.0


def test_common_rescaling_and_duplication_are_invariant() -> None:
    truth = {"A": rows(1.0, -2.0), "B": rows(3.0)}
    predictions = {"A": (2.0, -1.0), "B": (4.0,)}
    expected = zero_anchored_scale(truth, predictions)
    doubled_truth = {
        name: tuple(replace(row, value=2.0 * row.value) for row in values)
        for name, values in truth.items()
    }
    doubled_predictions = {
        name: tuple(2.0 * value for value in values)
        for name, values in predictions.items()
    }
    duplicate_truth = {
        name: values + values for name, values in truth.items()
    }
    duplicate_predictions = {
        name: values + values for name, values in predictions.items()
    }
    assert isclose(
        zero_anchored_scale(doubled_truth, doubled_predictions), expected,
    )
    assert isclose(
        zero_anchored_scale(duplicate_truth, duplicate_predictions), expected,
    )


def test_scaling_preserves_shape_and_r2_uses_zero() -> None:
    truth = {"A": rows(1.0, -1.0), "B": rows(2.0)}
    predictions = {"A": (2.0, -2.0), "B": (4.0,)}
    scaled = scale_predictions(
        predictions, zero_anchored_scale(truth, predictions),
    )
    assert scaled == {"A": (1.0, -1.0), "B": (2.0,)}
    assert tuple(scaled) == tuple(predictions)
    assert pooled_r2(truth, scaled) == 1.0
    assert pooled_r2(truth, {
        name: (0.0,) * len(values) for name, values in predictions.items()
    }) == 0.0


def test_alignment_decomposition_reconciles_partitions() -> None:
    truth = {"A": rows(2.0, 2.0), "B": rows(-1.0, -1.0)}
    predictions = {"A": (1.0, 1.0), "B": (1.0, 1.0)}
    diagnostic = alignment_diagnostic(truth, predictions, {
        "A": ("nonnegative", "nonnegative"),
        "B": ("negative", "negative"),
    })
    global_ = diagnostic["global"]
    assert global_ == {
        "observation_count": 4,
        "numerator": 2.0,
        "prediction_square_sum": 4.0,
        "scale": 0.5,
        "selection_eligible": False,
        "truth_square_sum": 10.0,
        "unclipped_scale": 0.5,
    }

    partitions = (
        tuple(diagnostic["by_stock"].values()),
        tuple(diagnostic["by_market_regime"].values()),
        tuple(
            cell
            for stock in diagnostic["by_stock_and_market_regime"].values()
            for cell in stock.values()
        ),
    )
    for partition in partitions:
        assert sum(cell["observation_count"] for cell in partition) == 4
        for key in (
            "numerator", "prediction_square_sum", "truth_square_sum",
        ):
            assert sum(cell[key] for cell in partition) == global_[key]
    assert diagnostic["by_stock"]["A"]["numerator"] == 4.0
    assert diagnostic["by_stock"]["B"]["numerator"] == -2.0
    assert diagnostic["by_market_regime"]["negative"]["scale"] == 0.0
    empty = diagnostic["by_stock_and_market_regime"]["A"]["negative"]
    assert empty["observation_count"] == 0
    assert empty["unclipped_scale"] is None
    assert empty["scale"] is None


def test_market_regime_uses_only_completed_spy_window() -> None:
    increasing = bars(*(100.0 + index for index in range(17)))
    decreasing = bars(*(116.0 - index for index in range(17)))
    assert len(increasing) == 17 * FEATURE_COUNT
    assert _market_regimes(increasing, (16,)) == {16: "nonnegative"}
    assert _market_regimes(decreasing, (16,)) == {16: "negative"}
    rejects(_market_regimes, increasing + bars(117.0), (16,))
    rejects(_market_regimes, bars(*range(1, 17)), (15,))
    rejects(_market_regimes, bars(*([1.0] * 16), 0.0), (16,))
    rejects(
        _market_regimes,
        bars(*([1.0] * 16), float("nan")),
        (16,),
    )
    invalid_interior = [1.0] * 17
    invalid_interior[8] = float("nan")
    rejects(_market_regimes, bars(*invalid_interior), (16,))
    rejects(_market_regimes, increasing, (16, 16))


def test_phase_market_regime_reads_exact_fold_prefix() -> None:
    timestamps = tuple(
        f"2026-01-02T14:{index:02d}:00Z" for index in range(20)
    )
    source = SimpleNamespace(evaluation_rows=(
        ("A", 2, "a" * 64), ("B", 2, "b" * 64),
    ))
    packed = (
        (
            "A",
            PackedRows((
                SampleRows(0, 1, 2, 0),
                SampleRows(16, 17, 18, 16),
                SampleRows(17, 18, 19, 17),
            ), (1, 2)),
        ),
        (
            "B",
            PackedRows((
                SampleRows(0, 1, 2, 0),
                SampleRows(18, 19, 19, 18),
                SampleRows(19, 19, 19, 19),
            ), (1, 2)),
        ),
    )
    state = _PhaseRows(source, object(), (), timestamps, packed)
    closes = (*([100.0] * 16), 116.0, 50.0, 118.0, 90.0)
    events = []

    with tempfile.TemporaryDirectory(
        prefix="spy-residual-regime-", dir=ROOT,
    ) as directory:
        snapshot = Path(directory) / "spy.csv"
        snapshot.write_bytes(b"frozen")
        snapshot.chmod(0o400)
        metadata = snapshot.stat(follow_symlinks=False)
        frozen = FrozenInput(
            snapshot, snapshot, hashlib.sha256(b"frozen").hexdigest(),
            (metadata.st_dev, metadata.st_ino),
        )
        lease = ResidualLease(
            object(), (("spy_csv", frozen),),
            lambda: events.append("lease"),
        )

        def prefix(path: Path, grid: object, stop: str) -> object:
            events.append("prefix")
            assert path == snapshot
            assert tuple(grid) == timestamps
            assert stop == timestamps[-1]
            return bars(*closes)

        with patch.object(
            analyzer, "context_bar_prefix", side_effect=prefix,
        ):
            result = _phase_market_regimes(state, lease)

    assert result == {
        "A": ("nonnegative", "negative"),
        "B": ("nonnegative", "negative"),
    }
    assert events == ["lease", "prefix", "lease"]


def test_invalid_alignment_inputs_are_rejected() -> None:
    truth = {"A": rows(1.0), "B": rows(-1.0)}
    predictions = {"A": (1.0,), "B": (1.0,)}
    rejects(alignment_diagnostic, truth, predictions, {
        "B": ("negative",), "A": ("nonnegative",),
    })
    rejects(alignment_diagnostic, truth, predictions, {
        "A": ("unknown",), "B": ("negative",),
    })
    rejects(alignment_diagnostic, truth, predictions, {
        "A": (), "B": ("negative",),
    })


def test_diagnostics_use_one_frozen_global_scale() -> None:
    truth, predictions, common, rank = panel_fixture()
    fit = fit_diagnostic(truth, predictions["panel_transformer"])
    evaluation = evaluate_frozen_scale(
        fit["scale"], truth, common, predictions, 0.02, rank,
    )
    assert fit["scale"] == 0.5
    assert fit["scale_leave_one_stock_out_range"] == 0.0
    assert evaluation["pooled_raw_residual_r2_vs_zero"] == 1.0
    assert all(
        value == 1.0 for value in
        evaluation["pooled_raw_residual_r2_without_stock"].values()
    )
    assert evaluation[
        "later_residual_holdout_preregistration_warranted"
    ] is True
    assert evaluation["paired_squared_error"]["zero"]["wins"] == 11
    assert evaluation["scaled_seed_dispersion"] == 0.01
    assert evaluation["spearman_rank_ic"] == rank

    rejected = evaluate_frozen_scale(
        0.0, truth, common, predictions, 0.02, rank,
    )
    assert rejected[
        "later_residual_holdout_preregistration_warranted"
    ] is False
    assert rejected["spearman_rank_ic"] is None


def test_reports_remain_residual_only_and_non_executable() -> None:
    diagnostic = {
        "denominator": 2.0,
        "numerator": 1.0,
        "observation_count": 2,
        "pooled_raw_residual_r2_vs_zero": 0.5,
        "scale": 0.5,
        "scale_leave_one_stock_out": {"A": 0.5},
        "scale_leave_one_stock_out_delta": {"A": 0.0},
        "scale_leave_one_stock_out_range": 0.0,
        "unclipped_scale": 0.5,
    }
    fit = _fit_report(REPORT_INPUTS, diagnostic, "d" * 40)
    evaluation = {
        "later_residual_holdout_preregistration_warranted": False,
    }
    result = _final_report(
        REPORT_INPUTS,
        FileBinding("reports/run/shrinkage-fit.json", "e" * 64),
        fit, evaluation, "d" * 40,
    )
    assert fit["locks"] == result["locks"] == \
        expected_residual_protocol()["locks"]
    assert result["decision"] == {
        "later_residual_holdout_preregistration_warranted": False,
        "output_role": "adaptive-residual-only-not-executable-return",
    }
    serialized = json.dumps(result, sort_keys=True)
    assert "raw_truth" not in serialized
    assert "absolute_price" not in serialized
    assert result["evidence_role"] == \
        "development-post-hoc-not-forward-clean"


def test_publication_is_exclusive_and_mode_bound() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-residual-shrinkage-", dir=ROOT,
    ) as directory:
        output = Path(directory) / "shrinkage-fit.json"
        descriptor = os.open(directory, os.O_RDONLY)
        calls = 0

        def verify() -> None:
            nonlocal calls
            calls += 1

        try:
            _publish(output, {"schema": 1}, descriptor, verify)
            assert calls == 2
            _validate_published(output, {"schema": 1})
            rejects(_publish, output, {"schema": 2}, descriptor, verify)
        finally:
            os.close(descriptor)


def test_fit_is_durable_before_calibration_truth() -> None:
    truth, predictions, common, rank = panel_fixture()
    phases = tuple(SimpleNamespace(
        predictions=predictions,
        evaluation={
            "secondary": {"spearman_rank_ic": {
                "panel_transformer": rank,
            }},
            "seed_dispersion": {"panel_transformer": 0.02},
        },
    ) for _ in range(2))
    states, events = (object(), object()), []

    with tempfile.TemporaryDirectory(
        prefix="spy-residual-order-", dir=ROOT,
    ) as directory:
        output = Path(directory)
        fit_path = output / "shrinkage-fit.json"
        result_path = output / "shrinkage.json"
        descriptor = os.open(directory, os.O_RDONLY)
        real_publish = analyzer._publish
        real_validate = analyzer._validate_published

        def phase_truth(state: object, lease: object) -> object:
            label = "fold" if state is states[0] else "calibration"
            events.append(f"{label}-truth")
            if label == "calibration":
                assert fit_path.exists()
                real_validate(
                    fit_path,
                    json.loads(fit_path.read_text(encoding="utf-8")),
                )
            return truth, common

        def publish(*args: object) -> None:
            events.append(f"publish-{Path(args[0]).name}")
            real_publish(*args)

        def validate(*args: object) -> object:
            events.append(f"validate-{Path(args[0]).name}")
            return real_validate(*args)

        try:
            with patch.object(
                analyzer, "_phase_truth", side_effect=phase_truth,
            ), patch.object(
                analyzer, "_publish", side_effect=publish,
            ), patch.object(
                analyzer, "_validate_published", side_effect=validate,
            ):
                _analyze_phases(
                    states, phases, object(), {
                        "analysis_source": {},
                        "attempt": {},
                        "outcome": {},
                        "phases": [],
                    }, "d" * 40, fit_path, result_path, descriptor,
                    _directory_identity(output), lambda: events.append(
                        "verify",
                    ),
                )
        finally:
            os.close(descriptor)
    ordered = (
        events.index("fold-truth"),
        events.index("publish-shrinkage-fit.json"),
        events.index("validate-shrinkage-fit.json"),
        events.index("calibration-truth"),
        events.index("publish-shrinkage.json"),
    )
    assert ordered == tuple(sorted(ordered))

    with tempfile.TemporaryDirectory(
        prefix="spy-residual-order-fail-", dir=ROOT,
    ) as directory:
        output = Path(directory)
        descriptor = os.open(directory, os.O_RDONLY)
        opened = []

        def fail_publish(*args: object) -> None:
            raise OSError("fit publication failed")

        def record_truth(*args: object) -> object:
            opened.append(args[0])
            return truth, common

        try:
            with patch.object(
                analyzer, "_phase_truth", side_effect=record_truth,
            ), patch.object(
                analyzer, "_publish", side_effect=fail_publish,
            ):
                rejects(
                    _analyze_phases, states, phases, object(), {},
                    "d" * 40, output / "shrinkage-fit.json",
                    output / "shrinkage.json", descriptor,
                    _directory_identity(output), lambda: None,
                )
        finally:
            os.close(descriptor)
        assert opened == [states[0]]


def test_alignment_read_and_publication_order_is_fold_one_only() -> None:
    truth = {"A": rows(1.0), "B": rows(-1.0)}
    predictions = {"A": (1.0,), "B": (1.0,)}
    state = SimpleNamespace(source=SimpleNamespace(phase="fold-1"))
    phase = SimpleNamespace(
        source=state.source,
        predictions={"panel_transformer": predictions},
    )
    events = []

    with tempfile.TemporaryDirectory(
        prefix="spy-residual-alignment-", dir=ROOT,
    ) as directory:
        output = Path(directory)
        result_path = output / "alignment.json"
        descriptor = os.open(directory, os.O_RDONLY)
        real_publish = analyzer._publish

        def regimes(*args: object) -> object:
            events.append("regimes")
            return {"A": ("nonnegative",), "B": ("negative",)}

        def phase_truth(*args: object) -> object:
            events.append("truth")
            return truth, ()

        def publish(*args: object) -> None:
            events.append("publish")
            real_publish(*args)

        try:
            with patch.object(
                analyzer, "_phase_market_regimes", side_effect=regimes,
            ), patch.object(
                analyzer, "_phase_truth", side_effect=phase_truth,
            ), patch.object(
                analyzer, "_publish", side_effect=publish,
            ):
                result = _analyze_alignment(
                    state, phase, object(), REPORT_INPUTS, "d" * 40,
                    result_path, descriptor, _directory_identity(output),
                    lambda: events.append("verify"),
                )
        finally:
            os.close(descriptor)

    assert events.index("regimes") < events.index("truth") < \
        events.index("publish")
    assert result["diagnostic"]["phase"] == "fold-1"
    assert result["decision"] == {
        "model_change_authorized": False,
        "output_role": "descriptive-residual-alignment-only",
    }
    assert result["locks"] == expected_residual_protocol()["locks"]
    assert "candidate" not in result
    assert result["subject"]["model"] == "panel_transformer"
    serialized = json.dumps(result, sort_keys=True)
    assert "raw_truth" not in serialized
    assert "absolute_price" not in serialized
    assert result["evidence_role"] == \
        "development-post-hoc-not-forward-clean"


def test_cli_summaries_are_mode_bound() -> None:
    parsed = tuple(analyzer.parse_args((
        "attempt.json", "--implementation-commit", "d" * 40, *flag,
    )) for flag in ((), ("--alignment",)))
    assert tuple(value.alignment for value in parsed) == (False, True)
    arguments = SimpleNamespace(
        attempt=Path("attempt.json"),
        implementation_commit="d" * 40,
        alignment=False,
    )
    reports = (
        {
            "decision": {
                "later_residual_holdout_preregistration_warranted": False,
            },
            "fit": {"scale": 0.25},
        },
        {
            "diagnostic": {
                "global": {"unclipped_scale": -0.5},
            },
        },
    )
    expected = (
        {
            "later_residual_holdout_preregistration_warranted": False,
            "scale": 0.25,
            "status": "analyzed",
        },
        {
            "mode": "alignment",
            "status": "analyzed",
            "unclipped_scale": -0.5,
        },
    )
    for alignment, report, summary in zip(
        (False, True), reports, expected, strict=True,
    ):
        arguments.alignment = alignment
        output = StringIO()
        with patch.object(
            analyzer, "parse_args", return_value=arguments,
        ), patch.object(
            analyzer, "analyze_residual_shrinkage", return_value=report,
        ) as analyze, redirect_stdout(output):
            analyzer.main()
        assert json.loads(output.getvalue()) == summary
        analyze.assert_called_once_with(
            arguments.attempt, arguments.implementation_commit,
            alignment=alignment,
        )


def test_imported_entrypoint_cannot_analyze() -> None:
    code = (
        "import sys;"
        f"sys.path.append({str(ROOT)!r});"
        "from pathlib import Path;"
        "from tools.analyze_spy_residual_shrinkage import "
        "analyze_residual_shrinkage;"
        "analyze_residual_shrinkage(Path('missing.json'),'0'*40)"
    )
    result = subprocess.run(
        (sys.executable, "-I", "-S", "-B", "-c", code),
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "residual shrinkage requires" in result.stderr


def test_invalid_inputs_are_rejected() -> None:
    truth = {"A": rows(1.0), "B": rows(2.0)}
    predictions = {"A": (1.0,), "B": (2.0,)}
    invalid_row = rows(1.0)[0]
    object.__setattr__(invalid_row, "value", float("inf"))
    rejects(zero_anchored_scale, {}, {})
    rejects(zero_anchored_scale, truth, {"B": (2.0,), "A": (1.0,)})
    rejects(zero_anchored_scale, truth, {"A": (1.0,)})
    rejects(zero_anchored_scale, truth, {**predictions, "C": (3.0,)})
    rejects(zero_anchored_scale, truth, {"A": (), "B": (2.0,)})
    rejects(zero_anchored_scale, truth, {"A": (1.0, 2.0), "B": (2.0,)})
    rejects(zero_anchored_scale, {"A": ("bad",), "B": rows(2.0)}, predictions)
    rejects(zero_anchored_scale, truth, {"A": (float("nan"),), "B": (2.0,)})
    rejects(
        zero_anchored_scale,
        {"A": (invalid_row,), "B": rows(2.0)},
        predictions,
    )
    rejects(
        zero_anchored_scale,
        {"A": (invalid_row,), "B": rows(2.0)},
        predictions, ("A",),
    )
    rejects(zero_anchored_scale, truth, predictions, ("A", "A"))
    rejects(zero_anchored_scale, truth, predictions, ("C",))
    rejects(zero_anchored_scale, truth, predictions, ("A", "B"))
    rejects(scale_predictions, predictions, -0.1)
    rejects(scale_predictions, predictions, 1.1)
    rejects(scale_predictions, predictions, True)
    rejects(scale_predictions, {"A": ()}, 0.5)
    rejects(zero_anchored_scale, {"A": rows(1e308)}, {"A": (1e308,)})
    rejects(pooled_r2, {"A": rows(0.0)}, {"A": (0.0,)})


def main() -> None:
    test_scale_is_the_clipped_mse_minimizer()
    test_scale_is_global_and_leave_one_stock_out()
    test_common_rescaling_and_duplication_are_invariant()
    test_scaling_preserves_shape_and_r2_uses_zero()
    test_alignment_decomposition_reconciles_partitions()
    test_market_regime_uses_only_completed_spy_window()
    test_phase_market_regime_reads_exact_fold_prefix()
    test_invalid_alignment_inputs_are_rejected()
    test_diagnostics_use_one_frozen_global_scale()
    test_reports_remain_residual_only_and_non_executable()
    test_publication_is_exclusive_and_mode_bound()
    test_fit_is_durable_before_calibration_truth()
    test_alignment_read_and_publication_order_is_fold_one_only()
    test_cli_summaries_are_mode_bound()
    test_imported_entrypoint_cannot_analyze()
    test_invalid_inputs_are_rejected()
    print("SPY residual shrinkage tests passed")


if __name__ == "__main__":
    main()
