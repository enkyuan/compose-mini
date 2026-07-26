"""Build stock-minus-SPY targets and causal market-context batches.

Callers must authenticate one common calendar and timestamp grid before
pairing series by session ordinal.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.utils.data import Dataset

from tools.artifact_v1 import Config
from tools.data_v1 import EXECUTABLE_RETURN_TARGET
from tools.relative_context_contract import SPY_RESIDUAL_TARGET
from tools.session_samples import SampleRows
from tools.train import ForecastTransformer, TrainingData, Windows

_SHARED_WINDOW_DATA = ("features", "targets", "references", "outcomes")


def _finite(value: object) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.is_floating_point()
        and bool(torch.isfinite(value).all())
    )


def _prepared(value: object) -> bool:
    return (
        _finite(value)
        and value.dtype == torch.float32
        and value.device.type == "cpu"
    )


def _source_windows(data: TrainingData) -> tuple[Windows, Windows, Windows]:
    if not isinstance(data, TrainingData) or \
       data.target_kind != EXECUTABLE_RETURN_TARGET:
        raise ValueError("relative context requires executable-return data")
    splits = (data.train, data.validation, data.test)
    if any(not isinstance(split, Windows) or not split.indexed
           for split in splits):
        raise ValueError("relative context requires indexed windows")
    source = splits[0]
    if not len(source) or any(
        split.seq_len != source.seq_len or
        split.feature_starts != source.feature_starts or
        split.sample_rows != source.sample_rows or any(
            getattr(split, name) is not getattr(source, name)
            for name in _SHARED_WINDOW_DATA
        )
        for split in splits
    ):
        raise ValueError("relative context requires one shared preparation")
    if not isinstance(source.features, torch.Tensor) or \
       not isinstance(source.targets, torch.Tensor):
        raise ValueError("relative context preparation is invalid")
    width = source.features.shape[1] if source.features.ndim == 2 else -1
    scalers = (
        data.feature_mean, data.feature_scale,
        data.target_mean, data.target_scale,
    )
    if source.targets.ndim != 1 or \
       any(not isinstance(row, SampleRows) for row in source.sample_rows) or \
       not _prepared(source.features) or not _prepared(source.targets) or \
       any(not _prepared(value) for value in scalers) or \
       data.feature_mean.shape != (width,) or \
       data.feature_scale.shape != (width,) or \
       data.target_mean.ndim != 0 or data.target_scale.ndim != 0 or \
       not bool(torch.all(data.feature_scale > 0)) or \
       not bool(data.target_scale > 0):
        raise ValueError("relative context preparation is invalid")
    return splits


def _aligned(stock: Windows, spy: Windows) -> bool:
    return (
        isinstance(stock, Windows)
        and isinstance(spy, Windows)
        and stock.indexed
        and spy.indexed
        and stock.seq_len == spy.seq_len
        and (stock.start, stock.count) == (spy.start, spy.count)
        and stock.features.ndim == spy.features.ndim == 2
        and stock.features.shape[1] == spy.features.shape[1]
        and stock.features.dtype == spy.features.dtype
        and stock.features.device == spy.features.device
        and tuple(row.as_of_ordinal for row in stock.sample_rows)
        == tuple(row.as_of_ordinal for row in spy.sample_rows)
    )


class MarketContextWindows(Dataset):
    """Yield model inputs and residual targets without absolute-price fields."""

    def __init__(self, stock: Windows, spy: Windows) -> None:
        if not _aligned(stock, spy):
            raise ValueError("stock and SPY windows are not aligned")
        self.stock, self.spy = stock, spy

    def __len__(self) -> int:
        return len(self.stock)

    def __getitem__(self, index: int) -> tuple[object, ...]:
        stock, target, *_ = self.stock[index]
        return (stock, self.spy[index][0]), target


class MarketContextTransformer(nn.Module):
    """Condition the existing encoder on SPY's last completed feature row."""

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.model = ForecastTransformer(config)

    def forward(
        self, stock: torch.Tensor, spy: torch.Tensor,
    ) -> torch.Tensor:
        expected = (
            isinstance(stock, torch.Tensor)
            and isinstance(spy, torch.Tensor)
            and stock.ndim == 3
            and stock.shape == spy.shape
            and stock.shape[1:] == (
                self.model.config.seq_len, self.model.config.in_dim,
            )
            and stock.dtype == spy.dtype
            and stock.device == spy.device
            and stock.dtype == self.model.embed_W.dtype
            and stock.device == self.model.embed_W.device
        )
        if not expected:
            raise ValueError("stock and SPY inputs have invalid shapes")
        return self.model(stock, spy[:, -1] @ self.model.embed_W)


def _retarget(window: Windows, targets: torch.Tensor) -> Windows:
    return Windows(
        window.features, targets, window.references, window.outcomes,
        window.seq_len, window.start, window.count,
        feature_starts=window.feature_starts, sample_rows=window.sample_rows,
    )


def spy_residual_data(
    stock: TrainingData, spy: TrainingData,
) -> TrainingData:
    """Build residual data after the caller authenticates one common grid."""
    stock_windows, spy_windows = _source_windows(stock), _source_windows(spy)
    if stock.feature_set != spy.feature_set or \
       stock.horizon_bars != spy.horizon_bars or any(
           not _aligned(left, right)
           for left, right in zip(stock_windows, spy_windows, strict=True)
       ):
        raise ValueError("stock and SPY preparations are not aligned")

    # Undo independent training scalers before subtracting the two labels.
    raw = (
        stock_windows[0].targets * stock.target_scale + stock.target_mean
        - spy_windows[0].targets * spy.target_scale - spy.target_mean
    )
    training = raw.narrow(
        0, stock_windows[0].start, stock_windows[0].count,
    )
    mean, scale = training.mean(), training.std(unbiased=False)
    if not _finite(raw) or not _finite(mean) or not _finite(scale) or \
       not bool(scale > torch.finfo(raw.dtype).eps):
        raise ValueError("SPY residuals require positive finite training scale")
    targets = (raw - mean) / scale
    residual = tuple(_retarget(window, targets) for window in stock_windows)
    paired = tuple(
        MarketContextWindows(left, right)
        for left, right in zip(residual, spy_windows, strict=True)
    )
    return replace(
        stock, train=paired[0], validation=paired[1], test=paired[2],
        target_mean=mean, target_scale=scale,
        target_kind=SPY_RESIDUAL_TARGET,
    )
