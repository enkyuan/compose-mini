#!/usr/bin/env python3
"""Verify label-free rolling-origin feature windows."""

from array import array
from pathlib import Path
import math
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.session_samples import SampleRows
from tools.universe_forward_runner import ForwardFeatureWindows


def raises(*args: object) -> None:
    try:
        ForwardFeatureWindows(*args)  # type: ignore[arg-type]
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def bars() -> array:
    return array("f", (
        10, 12, 9, 11, 100,
        11, 13, 10, 12, 110,
        12, 15, 11, 14, 120,
        14, 16, 13, 15, 130,
        15, 18, 14, 17, 140,
        17, 20, 16, 19, 150,
    ))


def scalers() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(5), torch.ones(5)


def test_ohlcv_windows() -> None:
    mean = torch.tensor((1, 2, 3, 4, 5), dtype=torch.float32)
    scale = torch.tensor((2, 2, 2, 2, 5), dtype=torch.float32)
    samples = (SampleRows(2, 3, 5, 2), SampleRows(5, 6, 6, 5))
    windows = ForwardFeatureWindows(
        bars(), samples, 2, "ohlcv", mean, scale,
    )
    assert len(windows) == 2
    torch.testing.assert_close(windows[0], torch.tensor((
        (5, 5.5, 3.5, 4, 21), (5.5, 6.5, 4, 5, 23),
    ), dtype=torch.float32))
    torch.testing.assert_close(windows[1], torch.tensor((
        (7, 8, 5.5, 6.5, 27), (8, 9, 6.5, 7.5, 29),
    ), dtype=torch.float32))
    assert not hasattr(windows, "targets")


def test_stationary_windows() -> None:
    mean, scale = scalers()
    window = ForwardFeatureWindows(
        bars()[:15], (SampleRows(2, 3, 5, 2),), 2, "stationary-v1",
        mean, scale,
    )[0]
    expected = torch.tensor(tuple(
        (
            math.log(open_ / prior_close),
            math.log(close / open_),
            math.log(high / max(open_, close)),
            math.log(min(open_, close) / low),
            math.log1p(volume) - math.log1p(prior_volume),
        )
        for prior_close, prior_volume, open_, high, low, close, volume in (
            (11, 100, 11, 13, 10, 12, 110),
            (12, 110, 12, 15, 11, 14, 120),
        )
    ), dtype=torch.float32)
    torch.testing.assert_close(window, expected)


def test_causal_slices() -> None:
    mean, scale = scalers()
    samples = (SampleRows(2, 3, 5, 2), SampleRows(5, 6, 6, 5))
    original = ForwardFeatureWindows(
        bars(), samples, 2, "ohlcv", mean, scale,
    )
    changed = bars()
    changed[25:30] = array("f", (170, 200, 160, 190, 1500))
    mutated = ForwardFeatureWindows(
        changed, samples[::-1], 2, "ohlcv", mean, scale,
    )
    torch.testing.assert_close(mutated[1], original[0])
    assert not torch.equal(mutated[0], original[1])


def test_rejections() -> None:
    mean, scale = scalers()
    samples = (SampleRows(5, 6, 6, 5),)
    malformed, nonfinite, trailing = bars(), bars(), bars()
    malformed.pop()
    nonfinite[3] = math.nan
    trailing.extend((20, 22, 18, 21, 160))
    for values in (
        (array("d", bars()), samples, 2, "ohlcv", mean, scale),
        (malformed, samples, 2, "ohlcv", mean, scale),
        (nonfinite, samples, 2, "ohlcv", mean, scale),
        (trailing, samples, 2, "ohlcv", mean, scale),
        (bars(), None, 2, "ohlcv", mean, scale),
        (bars(), (), 2, "ohlcv", mean, scale),
        (bars(), samples, 0, "ohlcv", mean, scale),
        (bars(), (SampleRows(1, 2, 3, 1),), 2, "stationary-v1",
         mean, scale),
        (bars(), (SampleRows("5", 6, 6, 5),), 2, "ohlcv", mean, scale),
        (bars(), (SampleRows(5, 5, 6, 5),), 2, "ohlcv", mean, scale),
        (bars(), (SampleRows(5, 6, 5, 5),), 2, "ohlcv", mean, scale),
        (bars(), (SampleRows(5, 6, 6, -1),), 2, "ohlcv", mean, scale),
        (bars(), (SampleRows(7, 8, 8, 7),), 2, "ohlcv", mean, scale),
        (bars(), samples, 2, "unknown", mean, scale),
        (bars(), samples, 2, "ohlcv", mean[:4], scale),
        (bars(), samples, 2, "ohlcv", mean, torch.zeros(5)),
        (bars(), samples, 2, "ohlcv", mean, torch.full(
            (5,), torch.finfo(torch.float32).tiny,
        )),
    ):
        raises(*values)


def main() -> None:
    test_ohlcv_windows()
    test_stationary_windows()
    test_causal_slices()
    test_rejections()


if __name__ == "__main__":
    main()
