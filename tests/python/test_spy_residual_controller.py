#!/usr/bin/env python3
"""Verify causal derivation, preparation, and deferred residual truth."""

from __future__ import annotations

from array import array
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("residual controller tests require PyTorch") from error

from tools.arm_context_diagnostic import ContextLease, ContextSnapshots
from tools.context_diagnostic_contract import (
    PHASE_RANGES, ContextAttempt, ContextPhase, ContextScalerInput,
    context_scaler_inputs_sha256,
)
from tools.context_diagnostic_inputs import (
    context_grid_sha256, timestamp_rows,
)
from tools.files import FrozenInput
from tools.relative_context_contract import (
    PHASE_BUDGETS, RESIDUAL_BENCHMARK, ResidualPhaseInput,
)
from tools.session_samples import SampleRows, SessionSamples
from tools.spy_residual_controller import (
    _PhaseRows, derive_residual_phases, prepare_residual_phase,
)
from tools.universe_contract import PackedRows
from tools.universe_scaling_contract import timestamp_grid_sha256

MASTER = tuple(f"S{index:02d}" for index in range(55))
TIMESTAMPS = tuple(
    f"2026-01-02T14:{index:02d}:00Z" for index in range(50)
)
TRAIN = (
    SampleRows(16, 17, 29, 16),
    SampleRows(17, 18, 30, 17),
)
EVALUATE = (SampleRows(31, 32, 44, 31),)
PACKED = PackedRows(TRAIN + EVALUATE, (len(TRAIN), len(EVALUATE)))


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def rejects(function: object, *args: object) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except (TypeError, ValueError):
        return
    raise AssertionError("invalid residual controller input was accepted")


def prefix(path: Path, stop: str) -> str:
    return digest(f"{path}:{stop}")


def phase_for(name: str, packed: PackedRows = PACKED) -> ContextPhase:
    boundary = packed.counts[0]
    training = packed.rows[:boundary]
    evaluation = packed.rows[boundary:]
    training_grid = timestamp_rows(TIMESTAMPS, training)
    evaluation_grid = timestamp_rows(TIMESTAMPS, evaluation)
    stock_prefix = prefix(Path("stock.csv"), TIMESTAMPS[training[-1].target])
    return ContextPhase(
        name, PHASE_RANGES[name],
        tuple((series, boundary) for series in MASTER[:44]),
        tuple(
            (
                series, len(evaluation),
                timestamp_grid_sha256(evaluation_grid),
            )
            for series in MASTER[44:]
        ),
        context_grid_sha256(
            "training", MASTER[:44],
            {series: training_grid for series in MASTER[:44]},
        ),
        context_grid_sha256(
            "evaluation", MASTER[44:],
            {series: evaluation_grid for series in MASTER[44:]},
        ),
        context_scaler_inputs_sha256(MASTER, tuple(
            ContextScalerInput(
                series, stock_prefix, boundary,
                timestamp_grid_sha256(training_grid),
            )
            for series in MASTER
        )),
        dict(PHASE_BUDGETS)[name], (),
    )


def context_for(
    phases: tuple[ContextPhase, ...],
    stock: FrozenInput,
) -> tuple[ContextAttempt, ContextLease]:
    snapshots = ContextSnapshots(
        FrozenInput(Path("config"), Path("config"), digest("config")),
        FrozenInput(Path("manifest"), Path("manifest"), digest("manifest")),
        FrozenInput(Path("calendar"), Path("calendar"), digest("calendar")),
        tuple((series, stock) for series in MASTER),
    )
    return (
        ContextAttempt(
            "experiments/source.json", "source", "reports/source",
            "0" * 40, (), None, phases, None, None, (), None, {},
        ),
        ContextLease(snapshots, lambda: None),
    )


def verify_phase_derivation() -> None:
    fold_rows = PACKED
    calibration_rows = PackedRows(
        TRAIN + (SampleRows(18, 19, 31, 18),) + EVALUATE,
        (3, 1),
    )
    phases = (
        phase_for("fold-1", fold_rows),
        phase_for("calibration", calibration_rows),
    )
    stock = FrozenInput(Path("stock.csv"), Path("stock.csv"), digest("stock"))
    context, lease = context_for(phases, stock)
    spy = FrozenInput(
        Path("spy.csv"), Path("spy.csv"),
        RESIDUAL_BENCHMARK["spy_csv"].sha256,
    )
    all_rows = tuple(sorted(
        set((*fold_rows.rows, *calibration_rows.rows)),
        key=lambda row: row.as_of_ordinal,
    ))
    calls: list[tuple[Path, str]] = []

    def hashed(path: Path, stop: str) -> str:
        calls.append((path, stop))
        return prefix(path, stop)

    with patch(
        "tools.spy_residual_controller._validate_inputs",
        return_value=(
            {
                "target_horizon_bars": 13,
                "alignment_horizon_bars": 13,
            },
            SimpleNamespace(
                start=date(2024, 11, 1), end=date(2026, 7, 21),
                interval_minutes=30,
            ),
            object(),
        ),
    ), patch(
        "tools.spy_residual_controller.context_cutoff_timestamp",
        return_value=TIMESTAMPS[-1],
    ), patch(
        "tools.spy_residual_controller.read_timestamps_until",
        return_value=TIMESTAMPS,
    ), patch(
        "tools.spy_residual_controller.session_samples",
        return_value=SessionSamples(all_rows, len(all_rows)),
    ), patch(
        "tools.spy_residual_controller.context_all_phase_rows",
        return_value=(
            ("fold-1", fold_rows),
            ("calibration", calibration_rows),
        ),
    ), patch(
        "tools.spy_residual_controller.context_csv_prefix_sha256",
        side_effect=hashed,
    ), patch(
        "tools.spy_residual_controller.context_phase_sha256",
        side_effect=lambda value: digest(value.phase),
    ), patch(
        "tools.spy_residual_controller.verify_frozen",
    ):
        derived = derive_residual_phases(context, lease, spy)
        assert tuple(value.phase for value in derived) == (
            "fold-1", "calibration",
        )
        assert len({value.scaler_inputs_sha256 for value in derived}) == 2
        assert all(
            value.source_phase_sha256 == digest(value.phase)
            for value in derived
        )
        assert {stop for _, stop in calls} == {
            TIMESTAMPS[TRAIN[-1].target],
            TIMESTAMPS[calibration_rows.rows[2].target],
        }

        missing = SessionSamples(all_rows[:-1], len(all_rows))
        with patch(
            "tools.spy_residual_controller.session_samples",
            return_value=missing,
        ):
            rejects(derive_residual_phases, context, lease, spy)

        def changed(path: Path, stop: str) -> str:
            value = prefix(path, stop)
            return digest(value) if path == Path("stock.csv") else value

        with patch(
            "tools.spy_residual_controller.context_csv_prefix_sha256",
            side_effect=changed,
        ):
            rejects(derive_residual_phases, context, lease, spy)

        for changed_source in (
            replace(
                phases[0],
                training_rows=(
                    (MASTER[0], phases[0].training_rows[0][1] + 1),
                    *phases[0].training_rows[1:],
                ),
            ),
            replace(
                phases[0], training_grid_sha256=digest("changed-grid"),
            ),
            replace(
                phases[0],
                training_rows=(
                    phases[0].training_rows[1],
                    phases[0].training_rows[0],
                    *phases[0].training_rows[2:],
                ),
            ),
        ):
            changed_context, changed_lease = context_for(
                (changed_source, phases[1]), stock,
            )
            rejects(
                derive_residual_phases,
                changed_context, changed_lease, spy,
            )


def bars(multiplier: float) -> array:
    values = array("f")
    for index in range(len(TIMESTAMPS)):
        open_ = multiplier + index * (0.7 + multiplier / 1_000)
        close = open_ * (1 + ((index % 5) - 2) / (2_000 + multiplier))
        values.extend((
            open_, max(open_, close) + 1, min(open_, close) - 1,
            close, 1_000 + index,
        ))
    return values


def verify_preparation_and_truth() -> None:
    source = phase_for("fold-1")
    binding = ResidualPhaseInput(
        source.phase, digest("source"), source.training_grid_sha256,
        source.evaluation_grid_sha256, digest("residual-scalers"),
    )
    stock_input = FrozenInput(
        Path("stock.csv"), Path("stock.csv"), digest("stock"),
    )
    config = ROOT / "experiments/executable-h13-context.example.json"
    snapshots = ContextSnapshots(
        FrozenInput(config, config, digest("config")),
        FrozenInput(Path("manifest"), Path("manifest"), digest("manifest")),
        FrozenInput(Path("calendar"), Path("calendar"), digest("calendar")),
        tuple((series, stock_input) for series in MASTER),
    )
    context = ContextAttempt(
        "experiments/source.json", "source", "reports/source",
        "0" * 40, (), None, (source,), None, None, (), None, {},
    )
    lease = ContextLease(snapshots, lambda: None)
    spy_input = FrozenInput(
        Path("spy.csv"), Path("spy.csv"),
        RESIDUAL_BENCHMARK["spy_csv"].sha256,
    )
    state = _PhaseRows(
        source, binding,
        tuple((series, TIMESTAMPS, PACKED) for series in MASTER),
        TIMESTAMPS,
        tuple((series, PACKED) for series in MASTER),
    )
    stock_bars, spy_bars = bars(100.0), bars(250.0)
    stop_index = {timestamp: index for index, timestamp in enumerate(TIMESTAMPS)}
    reads: list[tuple[Path, str]] = []

    def read(path: Path, _timestamps: object, stop: str) -> array:
        reads.append((path, stop))
        source_rows = spy_bars if path == spy_input.snapshot else stock_bars
        return source_rows[:(stop_index[stop] + 1) * 5]

    with patch(
        "tools.spy_residual_controller._collect_inputs",
        return_value=(state,),
    ), patch(
        "tools.spy_residual_controller.context_bar_prefix",
        side_effect=read,
    ), patch(
        "tools.spy_residual_controller.verify_frozen",
    ):
        prepared, truth = prepare_residual_phase(
            context, source, binding, lease, spy_input,
        )
        assert tuple(series for series, _ in prepared.training) == MASTER
        assert tuple(item.series for item in prepared.forward) == MASTER[44:]
        assert all(len(item.stock) == len(item.market) == 1
                   for item in prepared.forward)
        assert TIMESTAMPS[EVALUATE[-1].target] not in {
            stop for _, stop in reads
        }
        observed = truth()

    assert tuple(observed) == MASTER[44:]
    assert all(len(rows) == 1 for rows in observed.values())
    assert TIMESTAMPS[EVALUATE[-1].target] in {stop for _, stop in reads}
    row = EVALUATE[0]
    expected = math.log(
        stock_bars[row.target * 5 + 3] /
        stock_bars[row.entry * 5]
    ) - math.log(
        spy_bars[row.target * 5 + 3] /
        spy_bars[row.entry * 5]
    )
    assert math.isclose(observed[MASTER[44]][0].value, expected)


def main() -> None:
    verify_phase_derivation()
    verify_preparation_and_truth()
    print("SPY residual controller tests passed")


if __name__ == "__main__":
    main()
