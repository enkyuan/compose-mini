"""Derive causal stock-minus-SPY phase inputs and defer their truth."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from math import log
from typing import TYPE_CHECKING

from tools.context_diagnostic_contract import (
    MAX_HISTORY, ContextAttempt, ContextPhase, ContextScalerInput,
    context_phase_sha256,
    context_scaler_inputs_sha256, validate_context_sweep,
)
from tools.context_diagnostic_inputs import (
    context_all_phase_rows, context_bar_prefix,
    context_csv_prefix_sha256, context_cutoff_timestamp,
    context_grid_sha256, timestamp_rows,
)
from tools.data_v1 import FEATURE_COUNT, read_timestamps_until
from tools.files import FrozenInput, verify_frozen
from tools.panel_contract import read_canonical_json
from tools.relative_context_contract import (
    HISTORY_BARS, INTERVAL_MINUTES, PHASE_BUDGETS, RESIDUAL_BENCHMARK,
    SPY_END, SPY_RESIDUAL_TARGET, SPY_START, ResidualPhaseInput,
    ResidualScalerInput, ResidualTruthRow, residual_scaler_inputs_sha256,
)
from tools.relative_context_inputs import align_spy_rows
from tools.session_calendar import SessionCalendar
from tools.session_samples import SampleRows, session_samples
from tools.universe_contract import PackedRows
from tools.universe_scaling_contract import timestamp_grid_sha256

if TYPE_CHECKING:
    import torch

    from tools.arm_context_diagnostic import ContextLease
    from tools.fetch_universe import UniverseManifest
    from tools.relative_context import MarketContextForwardWindows
    from tools.train import TrainingData
    from tools.universe_forward_runner import ForwardFeatureWindows


@dataclass(frozen=True, slots=True)
class ResidualForwardSeries:
    """Keep one label-free stock/SPY input pair on its authenticated grid."""

    series: str
    stock: ForwardFeatureWindows
    market: MarketContextForwardWindows
    samples: tuple[SampleRows, ...]
    spy_feature_mean: torch.Tensor
    spy_feature_scale: torch.Tensor

    def __post_init__(self) -> None:
        import torch

        from tools.relative_context import MarketContextForwardWindows
        from tools.universe_forward_runner import ForwardFeatureWindows

        scalers = (self.spy_feature_mean, self.spy_feature_scale)
        if type(self.series) is not str or not self.series or \
           type(self.stock) is not ForwardFeatureWindows or \
           type(self.market) is not MarketContextForwardWindows or \
           self.market.stock is not self.stock or \
           type(self.samples) is not tuple or \
           len(self.samples) != len(self.stock) or \
           len(self.samples) != len(self.market) or any(
               type(row) is not SampleRows for row in self.samples
           ) or any(
               not isinstance(value, torch.Tensor) or
               value.shape != (FEATURE_COUNT,) or
               value.dtype != torch.float32 or
               value.device.type != "cpu" or
               not torch.isfinite(value).all()
               for value in scalers
           ) or not torch.all(self.spy_feature_scale > 0) or \
           not torch.equal(
               self.market.spy.feature_mean, self.spy_feature_mean,
           ) or not torch.equal(
               self.market.spy.feature_scale, self.spy_feature_scale,
           ):
            raise ValueError("residual forward series is invalid")


@dataclass(frozen=True, slots=True)
class ResidualPreparedPhase:
    """Own training-only residual data and label-free evaluation inputs."""

    source: ContextPhase
    binding: ResidualPhaseInput
    training: tuple[tuple[str, TrainingData], ...]
    forward: tuple[ResidualForwardSeries, ...]

    def __post_init__(self) -> None:
        from tools.train import TrainingData

        if type(self.source) is not ContextPhase or \
           type(self.binding) is not ResidualPhaseInput:
            raise ValueError("prepared residual phase is invalid")
        master = (
            *(series for series, _ in self.source.training_rows),
            *(series for series, _, _ in self.source.evaluation_rows),
        )
        evaluation = tuple(
            series for series, _, _ in self.source.evaluation_rows
        )
        if self.binding.phase != self.source.phase or \
           type(self.training) is not tuple or any(
               type(item) is not tuple or len(item) != 2 or
               type(item[0]) is not str or
               type(item[1]) is not TrainingData
               for item in self.training
           ) or \
           tuple(series for series, _ in self.training) != master or any(
               data.target_kind != SPY_RESIDUAL_TARGET
               for _, data in self.training
           ) or type(self.forward) is not tuple or any(
               type(item) is not ResidualForwardSeries
               for item in self.forward
           ) or \
           tuple(item.series for item in self.forward) != evaluation:
            raise ValueError("prepared residual phase is invalid")


@dataclass(frozen=True, slots=True)
class _PhaseRows:
    source: ContextPhase
    binding: ResidualPhaseInput
    stock: tuple[tuple[str, tuple[str, ...], PackedRows], ...]
    spy_timestamps: tuple[str, ...]
    spy: tuple[tuple[str, PackedRows], ...]


def _validate_inputs(
    context: ContextAttempt, lease: ContextLease, spy_csv: FrozenInput,
) -> tuple[Mapping[str, object], UniverseManifest, SessionCalendar]:
    from tools.arm_context_diagnostic import ContextLease
    from tools.fetch_universe import UniverseManifest

    if type(context) is not ContextAttempt or \
       type(lease) is not ContextLease or \
       type(spy_csv) is not FrozenInput or \
       spy_csv.sha256 != RESIDUAL_BENCHMARK["spy_csv"].sha256 or \
       any(type(phase) is not ContextPhase for phase in context.phases) or \
       tuple(phase.phase for phase in context.phases) != tuple(
           name for name, _ in PHASE_BUDGETS
       ) or \
       tuple(series for series, _ in lease.snapshots.csv) != context.master:
        raise ValueError("residual controller inputs are invalid")
    lease()
    verify_frozen((spy_csv,))
    config = validate_context_sweep(read_canonical_json(
        lease.snapshots.config.snapshot,
    ))
    manifest = UniverseManifest.read(lease.snapshots.manifest.snapshot)
    calendar = SessionCalendar.read(lease.snapshots.calendar.snapshot)
    if manifest.interval_minutes != INTERVAL_MINUTES or \
       manifest.start.isoformat() != SPY_START or \
       manifest.end.isoformat() != SPY_END:
        raise ValueError("residual benchmark interval differs from source")
    return config, manifest, calendar


def _phase_binding(
    context: ContextAttempt, source: ContextPhase,
    stock: tuple[tuple[str, tuple[str, ...], PackedRows], ...],
    spy_timestamps: tuple[str, ...], spy_rows: tuple[tuple[str, PackedRows], ...],
    csv: Mapping[str, FrozenInput], spy_csv: FrozenInput,
) -> ResidualPhaseInput:
    training = tuple(series for series, _ in source.training_rows)
    evaluation = tuple(series for series, _, _ in source.evaluation_rows)
    aligned = dict(spy_rows)
    training_grids, evaluation_grids = {}, {}
    context_scalers, residual_scalers = [], []
    spy_prefixes: dict[str, str] = {}

    for series, timestamps, packed in stock:
        boundary = packed.counts[0]
        train, evaluate = packed.rows[:boundary], packed.rows[boundary:]
        if not train:
            raise ValueError("residual training prefix is empty")
        train_grid = timestamp_rows(timestamps, train)
        grid_sha256 = timestamp_grid_sha256(train_grid)
        stock_stop = timestamps[train[-1].target]
        spy_train = aligned[series].rows[:boundary]
        spy_stop = spy_timestamps[spy_train[-1].target]
        stock_prefix = context_csv_prefix_sha256(
            csv[series].snapshot, stock_stop,
        )
        if spy_stop not in spy_prefixes:
            spy_prefixes[spy_stop] = context_csv_prefix_sha256(
                spy_csv.snapshot, spy_stop,
            )
        spy_prefix = spy_prefixes[spy_stop]
        context_scalers.append(ContextScalerInput(
            series, stock_prefix, boundary, grid_sha256,
        ))
        residual_scalers.append(ResidualScalerInput(
            series, stock_prefix, spy_prefix, boundary, grid_sha256,
        ))
        if series in training:
            training_grids[series] = train_grid
        if series in evaluation:
            evaluation_grids[series] = timestamp_rows(
                timestamps, evaluate,
            )

    expected_evaluation = {
        series: (count, digest)
        for series, count, digest in source.evaluation_rows
    }
    if tuple(
        (series, len(training_grids.get(series, ())))
        for series in training
    ) != source.training_rows or \
       tuple(training_grids) != training or \
       tuple(evaluation_grids) != evaluation or \
       tuple(
           (series, len(evaluation_grids[series]),
            timestamp_grid_sha256(evaluation_grids[series]))
           for series in evaluation
       ) != tuple(
           (series, *expected_evaluation[series])
           for series in evaluation
       ) or context_grid_sha256(
           "training", training, training_grids,
       ) != source.training_grid_sha256 or context_grid_sha256(
           "evaluation", evaluation, evaluation_grids,
       ) != source.evaluation_grid_sha256 or \
       context_scaler_inputs_sha256(
           context.master, context_scalers,
       ) != source.scaler_inputs_sha256:
        raise ValueError("residual source phase grid changed")

    return ResidualPhaseInput(
        source.phase, context_phase_sha256(source),
        source.training_grid_sha256, source.evaluation_grid_sha256,
        residual_scaler_inputs_sha256(context.master, residual_scalers),
    )


def _collect_snapshot_inputs(
    context: ContextAttempt,
    config: Mapping[str, object],
    start: date,
    end: date,
    interval_minutes: int,
    calendar: SessionCalendar,
    csv_inputs: tuple[tuple[str, FrozenInput], ...],
    spy_csv: FrozenInput,
    verify: Callable[[], None],
) -> tuple[_PhaseRows, ...]:
    """Derive phase rows from frozen market data without execution authority."""
    if type(context) is not ContextAttempt or \
       not isinstance(config, Mapping) or type(start) is not date or \
       type(end) is not date or type(interval_minutes) is not int or \
       type(calendar) is not SessionCalendar or \
       type(csv_inputs) is not tuple or \
       tuple(series for series, _ in csv_inputs) != context.master or \
       any(type(item) is not FrozenInput for _, item in csv_inputs) or \
       type(spy_csv) is not FrozenInput or \
       spy_csv.sha256 != RESIDUAL_BENCHMARK["spy_csv"].sha256 or \
       not callable(verify) or any(
           type(phase) is not ContextPhase for phase in context.phases
       ) or tuple(phase.phase for phase in context.phases) != tuple(
           name for name, _ in PHASE_BUDGETS
       ):
        raise ValueError("residual snapshot inputs are invalid")
    verify()
    verify_frozen((*tuple(item for _, item in csv_inputs), spy_csv))
    horizon = config["target_horizon_bars"]
    alignment = config["alignment_horizon_bars"]
    if type(horizon) is not int or type(alignment) is not int:
        raise ValueError("residual horizons are invalid")
    cutoff = context_cutoff_timestamp(
        calendar, start, end, interval_minutes,
        horizon, alignment,
    )
    spy_timestamps = read_timestamps_until(spy_csv.snapshot, cutoff)
    spy_samples = session_samples(
        spy_timestamps, interval_minutes, calendar,
        start, end, MAX_HISTORY, horizon, alignment,
    )
    by_phase = {phase.phase: [] for phase in context.phases}
    for series, frozen in csv_inputs:
        timestamps = read_timestamps_until(frozen.snapshot, cutoff)
        rows = dict(context_all_phase_rows(
            timestamps, interval_minutes, calendar,
            start, end, horizon, alignment,
        ))
        if tuple(rows) != tuple(by_phase):
            raise ValueError("residual source phase order changed")
        for phase in by_phase:
            by_phase[phase].append((series, timestamps, rows[phase]))

    csv = dict(csv_inputs)
    result = []
    for source in context.phases:
        stock = tuple(by_phase[source.phase])
        aligned = tuple(
            (
                series,
                align_spy_rows(
                    timestamps, rows, spy_timestamps, spy_samples,
                ),
            )
            for series, timestamps, rows in stock
        )
        result.append(_PhaseRows(
            source,
            _phase_binding(
                context, source, stock, spy_timestamps, aligned,
                csv, spy_csv,
            ),
            stock, spy_timestamps, aligned,
        ))
    verify()
    verify_frozen((*tuple(item for _, item in csv_inputs), spy_csv))
    return tuple(result)


def _collect_inputs(
    context: ContextAttempt, lease: ContextLease, spy_csv: FrozenInput,
) -> tuple[_PhaseRows, ...]:
    config, manifest, calendar = _validate_inputs(context, lease, spy_csv)
    return _collect_snapshot_inputs(
        context, config, manifest.start, manifest.end,
        manifest.interval_minutes, calendar, lease.snapshots.csv,
        spy_csv, lease,
    )


def derive_residual_phases(
    context: ContextAttempt, lease: ContextLease, spy_csv: FrozenInput,
) -> tuple[ResidualPhaseInput, ...]:
    """Bind both residual phases without decoding market values."""
    return tuple(
        phase.binding for phase in _collect_inputs(context, lease, spy_csv)
    )


def prepare_residual_phase(
    context: ContextAttempt, source_phase: ContextPhase,
    phase: ResidualPhaseInput, lease: ContextLease, spy_csv: FrozenInput,
) -> tuple[
    ResidualPreparedPhase,
    Callable[[], Mapping[str, tuple[ResidualTruthRow, ...]]],
]:
    """Build training/forward inputs and return one deferred truth reader."""
    states = _collect_inputs(context, lease, spy_csv)
    matches = tuple(
        value for value in states if value.source == source_phase
    )
    if len(matches) != 1 or type(phase) is not ResidualPhaseInput or \
       phase != matches[0].binding:
        raise ValueError("residual phase binding changed")
    state = matches[0]

    from tools.experiment import Sweep, _prepare_packed
    from tools.relative_context import (
        MarketContextForwardWindows, spy_residual_data,
    )
    from tools.universe_forward_runner import ForwardFeatureWindows

    sweep = Sweep.read(lease.snapshots.config.snapshot)
    candidate = next(
        item for item in sweep.candidates if item.seq_len == HISTORY_BARS
    )
    csv, aligned = dict(lease.snapshots.csv), dict(state.spy)
    data, forward = [], []
    evaluation = tuple(
        series for series, _, _ in source_phase.evaluation_rows
    )

    for series, timestamps, packed in state.stock:
        boundary = packed.counts[0]
        stock_train = packed.rows[:boundary]
        spy_train = aligned[series].rows[:boundary]
        stock = _prepare_packed(
            context_bar_prefix(
                csv[series].snapshot, timestamps,
                timestamps[stock_train[-1].target],
            ),
            candidate, PackedRows(stock_train, (boundary, 0)),
            MAX_HISTORY, sweep,
        )
        spy = _prepare_packed(
            context_bar_prefix(
                spy_csv.snapshot, state.spy_timestamps,
                state.spy_timestamps[spy_train[-1].target],
            ),
            candidate, PackedRows(spy_train, (boundary, 0)),
            MAX_HISTORY, sweep,
        )
        data.append((series, spy_residual_data(stock, spy)))

        if series in evaluation:
            samples = packed.rows[boundary:]
            spy_samples = aligned[series].rows[boundary:]
            stock_forward = ForwardFeatureWindows(
                context_bar_prefix(
                    csv[series].snapshot, timestamps,
                    timestamps[samples[-1].as_of],
                ),
                samples, HISTORY_BARS, candidate.feature_set,
                stock.feature_mean, stock.feature_scale,
            )
            spy_forward = ForwardFeatureWindows(
                context_bar_prefix(
                    spy_csv.snapshot, state.spy_timestamps,
                    state.spy_timestamps[spy_samples[-1].as_of],
                ),
                spy_samples, HISTORY_BARS, candidate.feature_set,
                spy.feature_mean, spy.feature_scale,
            )
            forward.append(ResidualForwardSeries(
                series=series,
                stock=stock_forward,
                market=MarketContextForwardWindows(
                    stock_forward, spy_forward,
                ),
                samples=samples,
                spy_feature_mean=spy.feature_mean,
                spy_feature_scale=spy.feature_scale,
            ))

    result = ResidualPreparedPhase(
        source_phase, phase, tuple(data), tuple(forward),
    )
    stock_rows = {
        series: (timestamps, packed)
        for series, timestamps, packed in state.stock
    }
    lease()
    verify_frozen((spy_csv,))

    def read_truth() -> Mapping[str, tuple[ResidualTruthRow, ...]]:
        lease()
        verify_frozen((spy_csv,))
        truth = {}
        for item in result.forward:
            series = item.series
            timestamps, packed = stock_rows[series]
            boundary = packed.counts[0]
            samples = packed.rows[boundary:]
            spy_samples = aligned[series].rows[boundary:]
            stock_bars = context_bar_prefix(
                csv[series].snapshot, timestamps,
                timestamps[samples[-1].target],
            )
            spy_bars = context_bar_prefix(
                spy_csv.snapshot, state.spy_timestamps,
                state.spy_timestamps[spy_samples[-1].target],
            )
            truth[series] = tuple(
                ResidualTruthRow(
                    timestamps[row.as_of], timestamps[row.entry],
                    timestamps[row.target],
                    log(
                        stock_bars[row.target * FEATURE_COUNT + 3] /
                        stock_bars[row.entry * FEATURE_COUNT]
                    ) - log(
                        spy_bars[spy_row.target * FEATURE_COUNT + 3] /
                        spy_bars[spy_row.entry * FEATURE_COUNT]
                    ),
                )
                for row, spy_row in zip(
                    samples, spy_samples, strict=True,
                )
            )
        lease()
        verify_frozen((spy_csv,))
        return truth

    return result, read_truth
