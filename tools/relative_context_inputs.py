"""Align canonical stock and SPY rows for market-relative training."""

from __future__ import annotations

from collections.abc import Sequence

from tools.context_diagnostic_inputs import timestamp_rows
from tools.session_samples import SessionSamples
from tools.universe_contract import PackedRows


def _timestamps(value: object) -> bool:
    return (
        type(value) is tuple
        and bool(value)
        and all(type(item) is str and item for item in value)
        and all(left < right for left, right in zip(value, value[1:]))
    )


def align_spy_rows(
    stock_timestamps: Sequence[str],
    stock: PackedRows,
    spy_timestamps: Sequence[str],
    spy: SessionSamples,
) -> PackedRows:
    """Return SPY rows whose causal timestamp triples equal the stock's."""
    valid = (
        _timestamps(stock_timestamps)
        and _timestamps(spy_timestamps)
        and isinstance(stock, PackedRows)
        and type(stock.rows) is tuple
        and type(stock.counts) is tuple
        and len(stock.counts) == 2
        and all(type(count) is int for count in stock.counts)
        and stock.counts[0] > 0
        and stock.counts[1] >= 0
        and sum(stock.counts) == len(stock.rows)
        and isinstance(spy, SessionSamples)
        and type(spy.rows) is tuple
        and type(spy.opportunities) is int
        and spy.opportunities >= len(spy.rows)
    )
    if not valid:
        raise ValueError("relative-context rows are invalid")

    stock_grid = timestamp_rows(stock_timestamps, stock.rows)
    timestamp_rows(spy_timestamps, spy.rows)
    by_ordinal = {row.as_of_ordinal: row for row in spy.rows}
    try:
        selected = tuple(
            by_ordinal[row.as_of_ordinal] for row in stock.rows
        )
    except KeyError as error:
        raise ValueError("SPY is missing a required stock row") from error
    if timestamp_rows(spy_timestamps, selected) != stock_grid:
        raise ValueError("stock and SPY timestamp grids differ")
    return PackedRows(selected, stock.counts)
