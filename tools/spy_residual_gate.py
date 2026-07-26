"""Apply the fixed causal SPY-direction residual gate."""

from collections.abc import Sequence
from math import isfinite

from tools.data_v1 import FEATURE_COUNT
from tools.relative_context_contract import HISTORY_BARS

MARKET_REGIMES = ("negative", "nonnegative")
SPY_DIRECTION_SCALE = 0.4029492434939931


def market_regimes(
    bars: Sequence[float],
    as_of: Sequence[int],
) -> dict[int, str]:
    """Classify exact completed SPY windows by close direction."""
    if isinstance(bars, (str, bytes)) or not isinstance(bars, Sequence) or \
       isinstance(as_of, (str, bytes)) or not isinstance(as_of, Sequence):
        raise TypeError("market regime inputs are invalid")
    values, indices = tuple(bars), tuple(as_of)
    rows, lookback = len(values) // FEATURE_COUNT, HISTORY_BARS - 1
    if not values or len(values) % FEATURE_COUNT or not indices or \
       any(type(index) is not int for index in indices) or \
       len(indices) != len(set(indices)) or \
       indices != tuple(sorted(indices)) or min(indices) < lookback or \
       max(indices) != rows - 1:
        raise ValueError("market regime window changed")
    needed = tuple(sorted({
        row
        for index in indices
        for row in range(index - lookback, index + 1)
    }))
    closes = {
        row: values[row * FEATURE_COUNT + 3]
        for row in needed
    }
    if any(
        type(close) not in (int, float) or not isfinite(close) or close <= 0.0
        for close in closes.values()
    ):
        raise ValueError("market regime closes must be positive and finite")
    return {
        index: (
            "negative"
            if closes[index] < closes[index - lookback]
            else "nonnegative"
        )
        for index in indices
    }


def gate_mean_predictions(
    mean_predictions: Sequence[float],
    regimes: Sequence[str],
) -> tuple[float, ...]:
    """Apply the frozen scale only in the nonnegative SPY regime."""
    if isinstance(mean_predictions, (str, bytes)) or \
       not isinstance(mean_predictions, Sequence) or \
       isinstance(regimes, (str, bytes)) or \
       not isinstance(regimes, Sequence):
        raise TypeError("SPY-direction gate inputs are invalid")
    values, labels = tuple(mean_predictions), tuple(regimes)
    if not values or len(values) != len(labels) or \
       any(label not in MARKET_REGIMES for label in labels) or \
       any(
           type(value) not in (int, float) or not isfinite(value)
           for value in values
       ):
        raise ValueError("SPY-direction gate inputs changed")
    return tuple(
        0.0 if label == "negative" else SPY_DIRECTION_SCALE * value
        for value, label in zip(values, labels, strict=True)
    )
