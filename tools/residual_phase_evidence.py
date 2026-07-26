"""Derive residual truth and causal regimes from frozen phase snapshots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import log

from tools.context_diagnostic_inputs import context_bar_prefix
from tools.data_v1 import FEATURE_COUNT
from tools.files import FrozenInput, verify_frozen
from tools.finalize_spy_residual import _truth
from tools.relative_context_contract import ResidualTruthRow
from tools.spy_residual_controller import _PhaseRows
from tools.spy_residual_gate import market_regimes

Verify = Callable[[], None]


def phase_truth(
    state: _PhaseRows,
    csv: Mapping[str, FrozenInput],
    spy_csv: FrozenInput,
    verify: Verify,
) -> tuple[
    Mapping[str, tuple[ResidualTruthRow, ...]],
    tuple[tuple[str, str, str], ...],
]:
    """Reconstruct one phase's residual truth from frozen market snapshots."""
    if not isinstance(state, _PhaseRows) or \
       not isinstance(csv, Mapping) or \
       tuple(csv) != tuple(series for series, _, _ in state.stock) or \
       any(type(value) is not FrozenInput for value in csv.values()) or \
       type(spy_csv) is not FrozenInput or not callable(verify):
        raise ValueError("residual truth inputs are invalid")
    verify()
    verify_frozen((*tuple(csv.values()), spy_csv))
    stock = {
        series: (timestamps, packed)
        for series, timestamps, packed in state.stock
    }
    aligned = dict(state.spy)
    truth = {}
    for series, _, _ in state.source.evaluation_rows:
        timestamps, packed = stock[series]
        boundary = packed.counts[0]
        samples = packed.rows[boundary:]
        spy_samples = aligned[series].rows[boundary:]
        if not samples or len(samples) != len(spy_samples):
            raise ValueError(f"{series} residual truth grid changed")
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
                timestamps[row.as_of],
                timestamps[row.entry],
                timestamps[row.target],
                log(
                    stock_bars[row.target * FEATURE_COUNT + 3] /
                    stock_bars[row.entry * FEATURE_COUNT]
                ) - log(
                    spy_bars[spy_row.target * FEATURE_COUNT + 3] /
                    spy_bars[spy_row.entry * FEATURE_COUNT]
                ),
            )
            for row, spy_row in zip(samples, spy_samples, strict=True)
        )
    verify()
    verify_frozen((*tuple(csv.values()), spy_csv))
    return _truth(state.source, state.binding, truth)


def phase_market_regimes(
    state: _PhaseRows,
    spy_csv: FrozenInput,
    verify: Verify,
) -> Mapping[str, tuple[str, ...]]:
    """Derive causal SPY regime labels from completed as-of bars."""
    if not isinstance(state, _PhaseRows) or \
       type(spy_csv) is not FrozenInput or not callable(verify):
        raise ValueError("residual regime inputs are invalid")
    verify()
    verify_frozen((spy_csv,))
    aligned, indices = dict(state.spy), {}
    for series, count, _ in state.source.evaluation_rows:
        try:
            packed = aligned[series]
        except KeyError as error:
            raise ValueError(f"{series} SPY rows are missing") from error
        rows = packed.rows[packed.counts[0]:]
        values = tuple(row.as_of for row in rows)
        if len(values) != count:
            raise ValueError(f"{series} SPY regime grid changed")
        indices[series] = values
    unique = tuple(sorted({
        index for values in indices.values() for index in values
    }))
    if not unique:
        raise ValueError("SPY regime grid is empty")
    bars = context_bar_prefix(
        spy_csv.snapshot, state.spy_timestamps,
        state.spy_timestamps[unique[-1]],
    )
    labels = market_regimes(bars, unique)
    result = {
        series: tuple(labels[index] for index in values)
        for series, values in indices.items()
    }
    verify()
    verify_frozen((spy_csv,))
    return result
