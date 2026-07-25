"""Prepare causal data and own one authenticated context phase."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from tools.arm_context_diagnostic import ContextLease
from tools.context_diagnostic_contract import (
    MAX_HISTORY, ContextAttempt, ContextPhase, ContextPredictionEvidence,
    ContextScalerInput, context_scaler_inputs_sha256,
    validate_context_sweep,
)
from tools.context_diagnostic_inputs import (
    context_bar_prefix, context_csv_prefix_sha256,
    context_cutoff_timestamp, context_grid_sha256,
    context_phase_rows, timestamp_rows,
)
from tools.context_diagnostic_runtime import ContextRuntime
from tools.data_v1 import (
    CLOSE_RETURN_TARGET, EXECUTABLE_RETURN_TARGET, FEATURE_COUNT,
    read_timestamps_until,
)
from tools.experiment import Sweep, _prepare_packed
from tools.fetch_universe import UniverseManifest
from tools.panel_contract import read_canonical_json
from tools.run_universe_scaling import _runtime_sha256
from tools.session_calendar import SessionCalendar
from tools.universe_contract import PackedRows
from tools.universe_forward_runner import ForwardFeatureWindows
from tools.universe_scaling_contract import timestamp_grid_sha256


def _phase_rows(
    attempt: ContextAttempt, phase: ContextPhase, lease: ContextLease,
    sweep: Sweep, manifest: UniverseManifest, calendar: SessionCalendar,
) -> tuple[
    Mapping[str, tuple[str, ...]],
    Mapping[str, PackedRows],
]:
    cutoff = context_cutoff_timestamp(
        calendar, manifest.start, manifest.end, manifest.interval_minutes,
        sweep.target_horizon_bars, sweep.alignment_horizon_bars,
        (phase.phase,),
    )
    timestamps, packed = {}, {}
    for series, frozen in lease.snapshots.csv:
        values = read_timestamps_until(frozen.snapshot, cutoff)
        timestamps[series] = values
        packed[series] = context_phase_rows(
            values, manifest.interval_minutes, calendar,
            manifest.start, manifest.end, phase.phase,
            sweep.target_horizon_bars, sweep.alignment_horizon_bars,
        )

    training = tuple(series for series, _ in phase.training_rows)
    evaluation = tuple(series for series, _, _ in phase.evaluation_rows)
    training_grids = {
        series: timestamp_rows(
            timestamps[series],
            packed[series].rows[:packed[series].counts[0]],
        )
        for series in training
    }
    evaluation_grids = {
        series: timestamp_rows(
            timestamps[series],
            packed[series].rows[packed[series].counts[0]:],
        )
        for series in evaluation
    }
    if tuple(
        (series, packed[series].counts[0]) for series in training
    ) != phase.training_rows or tuple(
        (
            series, packed[series].counts[1],
            timestamp_grid_sha256(evaluation_grids[series]),
        )
        for series in evaluation
    ) != phase.evaluation_rows or context_grid_sha256(
        "training", training, training_grids,
    ) != phase.training_grid_sha256 or context_grid_sha256(
        "evaluation", evaluation, evaluation_grids,
    ) != phase.evaluation_grid_sha256:
        raise ValueError("context phase grid changed")

    return timestamps, packed


def prepare_context_phase(
    attempt: ContextAttempt, phase: ContextPhase, lease: ContextLease,
) -> tuple[object, object, object]:
    """Build the sole fit, prediction, and receipt-gated truth callbacks."""
    if not isinstance(attempt, ContextAttempt) or \
       not isinstance(phase, ContextPhase) or \
       not isinstance(lease, ContextLease) or \
       phase not in attempt.phases or \
       tuple(series for series, _ in lease.snapshots.csv) != attempt.master:
        raise ValueError("context preparation inputs are invalid")
    lease()
    validate_context_sweep(read_canonical_json(
        lease.snapshots.config.snapshot,
    ))
    sweep = Sweep.read(lease.snapshots.config.snapshot)
    candidate = next(
        item for item in sweep.candidates if item.seq_len == MAX_HISTORY
    )
    manifest = UniverseManifest.read(lease.snapshots.manifest.snapshot)
    calendar = SessionCalendar.read(lease.snapshots.calendar.snapshot)
    timestamps, packed = _phase_rows(
        attempt, phase, lease, sweep, manifest, calendar,
    )

    data, forward, scaler_inputs = {}, {}, []
    evaluation = tuple(series for series, _, _ in phase.evaluation_rows)
    csv = dict(lease.snapshots.csv)
    for series in attempt.master:
        samples = packed[series]
        boundary = samples.counts[0]
        training = samples.rows[:boundary]
        if not training:
            raise ValueError("context training prefix is empty")
        rows = context_bar_prefix(
            csv[series].snapshot, timestamps[series],
            timestamps[series][training[-1].target],
        )
        scaler_inputs.append(ContextScalerInput(
            series, context_csv_prefix_sha256(
                csv[series].snapshot,
                timestamps[series][training[-1].target],
            ),
            boundary, timestamp_grid_sha256(timestamp_rows(
                timestamps[series], training,
            )),
        ))
        data[series] = _prepare_packed(
            rows, candidate, PackedRows(training, (boundary, 0)),
            MAX_HISTORY, sweep,
        )
        if series in evaluation:
            prediction = samples.rows[boundary:]
            if not prediction:
                raise ValueError("context prediction grid is empty")
            rows = context_bar_prefix(
                csv[series].snapshot, timestamps[series],
                timestamps[series][prediction[-1].as_of],
            )
            forward[series] = ForwardFeatureWindows(
                rows, prediction, MAX_HISTORY, candidate.feature_set,
                data[series].feature_mean, data[series].feature_scale,
            )
    if context_scaler_inputs_sha256(
        attempt.master, scaler_inputs,
    ) != phase.scaler_inputs_sha256:
        raise ValueError("context scaler inputs changed")
    torch.use_deterministic_algorithms(True)
    runtime = ContextRuntime(
        attempt.master, phase, data, forward, torch.device("cpu"),
        _runtime_sha256(attempt),
    )
    lease()

    def truth(
        evidence: Sequence[ContextPredictionEvidence],
    ) -> Mapping[str, object]:
        from tools.finalize_context_diagnostic import (
            ContextTruthRow, evaluate_context_phase,
        )

        lease()
        values = {}
        for series in evaluation:
            samples = packed[series]
            rows = samples.rows[samples.counts[0]:]
            bars = context_bar_prefix(
                csv[series].snapshot, timestamps[series],
                timestamps[series][rows[-1].target],
            )
            values[series] = tuple(
                ContextTruthRow(
                    timestamps[series][row.as_of],
                    timestamps[series][row.entry],
                    timestamps[series][row.target],
                    float(bars[
                        (
                            row.as_of * FEATURE_COUNT + 3
                            if sweep.target_kind == CLOSE_RETURN_TARGET else
                            row.entry * FEATURE_COUNT
                        )
                    ]),
                    float(bars[row.target * FEATURE_COUNT + 3]),
                )
                for row in rows
            )
        if sweep.target_kind not in (
            CLOSE_RETURN_TARGET, EXECUTABLE_RETURN_TARGET,
        ):
            raise ValueError("context target kind changed")
        result = evaluate_context_phase(
            attempt.master, phase, evidence, values,
        )
        lease()
        return result

    return runtime.fit_one, runtime.predict_one, truth
