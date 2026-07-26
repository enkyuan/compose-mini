#!/usr/bin/env python3
"""Verify the fixed causal SPY-direction residual gate."""

from math import isclose
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.spy_residual_gate import (
    SPY_DIRECTION_SCALE, gate_mean_predictions, market_regimes,
)


def bars(*closes: float) -> tuple[float, ...]:
    return tuple(
        value
        for close in closes
        for value in (100.0, 101.0, 99.0, close, 1.0)
    )


def rejects(function: object, *args: object) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except (TypeError, ValueError):
        return
    raise AssertionError("invalid SPY-direction gate input was accepted")


def test_regimes_use_completed_window_endpoints() -> None:
    assert market_regimes(bars(*([2.0] * 16), 1.0), (16,)) == {
        16: "negative",
    }
    assert market_regimes(bars(*([2.0] * 17)), (16,)) == {
        16: "nonnegative",
    }
    closes = (1.0, *([99.0] * 15), 2.0, 0.5)
    assert market_regimes(bars(*closes), (16, 17)) == {
        16: "nonnegative", 17: "negative",
    }
    rejects(market_regimes, bars(*closes, 3.0), (16, 17))


def test_import_is_torch_free() -> None:
    assert "torch" not in sys.modules


def test_gate_is_fixed_and_shape_preserving() -> None:
    assert SPY_DIRECTION_SCALE == 0.4029492434939931
    result = gate_mean_predictions(
        (1.0, -2.0, 3.0),
        ("negative", "nonnegative", "nonnegative"),
    )
    assert len(result) == 3 and result[0] == 0.0
    assert isclose(result[1], -2.0 * SPY_DIRECTION_SCALE)
    assert isclose(result[2], 3.0 * SPY_DIRECTION_SCALE)


def test_invalid_inputs_are_rejected() -> None:
    for predictions, regimes in (
        ((), ()),
        ("invalid", ("negative",)),
        ((1.0,), "negative"),
        ((1.0,), ()),
        ((1.0,), ("unknown",)),
        ((float("nan"),), ("negative",)),
        ((float("inf"),), ("nonnegative",)),
    ):
        rejects(gate_mean_predictions, predictions, regimes)
    for closes, indices in (
        ((*([1.0] * 16), 0.0), (16,)),
        ((*([1.0] * 16), float("nan")), (16,)),
        ((1.0, *([float("inf")] * 15), 2.0), (16,)),
        ((*([1.0] * 17),), (15,)),
        ((*([1.0] * 17),), (16, 16)),
    ):
        rejects(market_regimes, bars(*closes), indices)


def main() -> None:
    test_regimes_use_completed_window_endpoints()
    test_import_is_torch_free()
    test_gate_is_fixed_and_shape_preserving()
    test_invalid_inputs_are_rejected()
    print("SPY residual gate tests passed")


if __name__ == "__main__":
    main()
