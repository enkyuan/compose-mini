#!/usr/bin/env python3
"""Verify leakage-safe SPY-relative targets and completed market context."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("relative-context tests require PyTorch") from error

from torch.utils.data import DataLoader

from tools.artifact_v1 import Config
from tools.data_v1 import CLOSE_RETURN_TARGET, EXECUTABLE_RETURN_TARGET
from tools.relative_context import (
    MarketContextTransformer, MarketContextWindows, SPY_RESIDUAL_TARGET,
    spy_residual_data,
)
from tools.session_samples import SampleRows
from tools.train import (
    TrainingData, Windows, evaluate, mean_loss, train_epoch,
)


def training_data(
    raw_targets: tuple[float, ...], *,
    ordinals: tuple[int, ...] = (10, 11, 12, 13),
    dtype: torch.dtype = torch.float32,
) -> TrainingData:
    """Build one indexed preparation with scalers fitted on two rows."""
    values = torch.tensor(raw_targets, dtype=dtype)
    mean, scale = values[:2].mean(), values[:2].std(unbiased=False)
    targets = (values - mean) / scale
    count, seq_len = len(values), 2
    features = torch.arange(
        (count + seq_len - 1) * 5, dtype=dtype,
    ).view(-1, 5)
    rows = tuple(
        SampleRows(index + 1, index + 2, index + 3, ordinal)
        for index, ordinal in enumerate(ordinals)
    )
    starts = tuple(range(count))
    references = torch.ones(count, dtype=dtype)
    outcomes = torch.ones(count, dtype=dtype)

    def windows(start: int, size: int) -> Windows:
        return Windows(
            features, targets, references, outcomes, seq_len, start, size,
            feature_starts=starts, sample_rows=rows,
        )

    return TrainingData(
        windows(0, 2), windows(2, 1), windows(3, 1),
        torch.zeros(5, dtype=dtype), torch.ones(5, dtype=dtype),
        mean, scale, "ohlcv", 13,
        EXECUTABLE_RETURN_TARGET,
    )


def rejects(action: Callable[[], object]) -> None:
    try:
        action()
    except ValueError:
        return
    raise AssertionError("invalid relative-context input was accepted")


def verify_residual_targets() -> None:
    stock_returns = (0.10, 0.20, 0.35, 0.40)
    spy_returns = (0.03, 0.07, 0.08, 0.06)
    stock, spy = training_data(stock_returns), training_data(spy_returns)
    residual = spy_residual_data(stock, spy)
    expected = torch.tensor(stock_returns) - torch.tensor(spy_returns)
    actual = (
        residual.train.stock.targets
        * residual.target_scale
        + residual.target_mean
    )

    assert residual.target_kind == SPY_RESIDUAL_TARGET
    assert isinstance(residual.train, MarketContextWindows)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(residual.target_mean, expected[:2].mean())
    torch.testing.assert_close(
        residual.target_scale, expected[:2].std(unbiased=False),
    )
    assert residual.train.stock.features is stock.train.features
    assert residual.train.stock.references is stock.train.references
    assert residual.train.stock.outcomes is stock.train.outcomes

    changed_later = spy_residual_data(
        stock, training_data((0.03, 0.07, 8.0, -6.0)),
    )
    torch.testing.assert_close(changed_later.target_mean, residual.target_mean)
    torch.testing.assert_close(changed_later.target_scale, residual.target_scale)


def verify_alignment_guards() -> None:
    stock = training_data((0.10, 0.20, 0.35, 0.40))
    spy = training_data((0.03, 0.07, 0.08, 0.06))
    other = training_data((0.10, 0.20, 0.35, 0.40))
    shifted = Windows(
        spy.validation.features, spy.validation.targets,
        spy.validation.references, spy.validation.outcomes,
        spy.validation.seq_len, 1, 1,
        feature_starts=spy.validation.feature_starts,
        sample_rows=spy.validation.sample_rows,
    )
    rejects(lambda: spy_residual_data(
        stock, training_data(
            (0.03, 0.07, 0.08, 0.06), ordinals=(10, 11, 99, 13),
        ),
    ))
    rejects(lambda: spy_residual_data(
        replace(stock, target_kind=CLOSE_RETURN_TARGET), spy,
    ))
    rejects(lambda: spy_residual_data(
        replace(stock, horizon_bars=1), spy,
    ))
    rejects(lambda: spy_residual_data(
        replace(stock, feature_set="stationary-v1"), spy,
    ))
    rejects(lambda: spy_residual_data(
        replace(stock, validation=other.validation), spy,
    ))
    rejects(lambda: spy_residual_data(
        stock, replace(spy, validation=shifted),
    ))
    rejects(lambda: spy_residual_data(
        replace(stock, target_scale=torch.tensor(float("nan"))), spy,
    ))
    rejects(lambda: spy_residual_data(
        replace(stock, target_mean=torch.zeros((1, 1))), spy,
    ))
    rejects(lambda: spy_residual_data(
        stock, training_data((0.03, 0.07, 0.08, 0.06), dtype=torch.float64),
    ))
    rejects(lambda: spy_residual_data(
        training_data((0.10, 0.20, 0.30, 0.40)),
        training_data((0.05, 0.15, 0.25, 0.35)),
    ))


def verify_causal_windows() -> None:
    residual = spy_residual_data(
        training_data((0.10, 0.20, 0.35, 0.40)),
        training_data((0.03, 0.07, 0.08, 0.06)),
    )
    first = residual.train[0][0][1].clone()
    later = residual.validation[0][0][1].clone()
    residual.train.spy.features[2:] += 1_000.0
    torch.testing.assert_close(residual.train[0][0][1], first)
    assert not torch.equal(residual.validation[0][0][1], later)
    assert len(residual.train[0]) == 2
    assert not hasattr(residual.train, "references")


def verify_context_model() -> None:
    torch.manual_seed(17)
    config = Config(
        model_dim=4, num_heads=2, ff_dim=6, num_layers=1, seq_len=2,
    )
    model = MarketContextTransformer(config).eval()
    stock = torch.randn(3, config.seq_len, config.in_dim)
    empty_market = torch.zeros_like(stock)
    completed_market = empty_market.clone()
    completed_market[:, -1, 0] = 1.0
    earlier_market = empty_market.clone()
    earlier_market[:, 0, 0] = 1.0

    with torch.no_grad():
        plain = model.model(stock)
        torch.testing.assert_close(model(stock, empty_market), plain)
        torch.testing.assert_close(model(stock, earlier_market), plain)
        assert not torch.equal(model(stock, completed_market), plain)
    for invalid in (
        completed_market[:1],
        completed_market[:, :-1],
        completed_market[:, :, :-1],
        completed_market.double(),
    ):
        rejects(lambda invalid=invalid: model(stock, invalid))

    residual = spy_residual_data(
        training_data((0.10, 0.20, 0.35, 0.40)),
        training_data((0.03, 0.07, 0.08, 0.06)),
    )
    loader, device = DataLoader(residual.train, batch_size=2), torch.device("cpu")
    loss = mean_loss(model, loader, device)
    assert math.isfinite(loss)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert math.isfinite(train_epoch(model, loader, optimizer, device))
    gradient = model.model.embed_W.grad
    assert gradient is not None and bool(torch.isfinite(gradient).all())
    rejects(lambda: evaluate(
        model, loader, residual.target_mean, residual.target_scale, device,
    ))


def main() -> None:
    verify_residual_targets()
    verify_alignment_guards()
    verify_causal_windows()
    verify_context_model()
    print("relative-context tests passed")


if __name__ == "__main__":
    main()
