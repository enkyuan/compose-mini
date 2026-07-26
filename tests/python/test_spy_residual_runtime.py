#!/usr/bin/env python3
"""Verify the ordered, label-free stock-minus-SPY model runtime."""

from __future__ import annotations

from array import array
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch
import hashlib
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    from torch.utils.data import DataLoader
except ModuleNotFoundError as error:
    raise SystemExit("SPY residual runtime tests require PyTorch") from error

from test_relative_context_contract import MASTER, source_phase
from tools.context_diagnostic_contract import (
    ContextFit, ContextPhase, _phase_value, context_phase_sha256,
)
from tools.experiment import stock_macro_linear_model
from tools.relative_context import (
    MarketContextForwardWindows, MarketContextWindows,
)
from tools.relative_context_contract import (
    SPY_RESIDUAL_TARGET, ResidualPhaseInput, expected_residual_fits,
    expected_residual_predictions,
)
from tools.session_samples import SampleRows
from tools.spy_residual_controller import (
    ResidualForwardSeries, ResidualPreparedPhase,
)
from tools.spy_residual_runtime import ResidualRuntime
from tools.train import (
    TrainingData, Windows, mean_loss, tail_training_data,
)
from tools.universe_forward_runner import ForwardFeatureWindows

RUNTIME = hashlib.sha256(b"residual-runtime").hexdigest()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def raises(function: object, *args: object) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except (TypeError, ValueError):
        return
    raise AssertionError("invalid residual runtime input was accepted")


def paired_training(count: int) -> MarketContextWindows:
    rows = count + 67
    stock = torch.randn(rows, 5, generator=torch.Generator().manual_seed(7))
    spy = torch.randn(rows, 5, generator=torch.Generator().manual_seed(19))
    targets = stock[67:, 0].clone()
    references = torch.full((count,), 100.0)
    outcomes = torch.full((count,), 101.0)
    starts = tuple(range(count))
    samples = tuple(
        SampleRows(67 + index, 68 + index, 81 + index, 67 + index)
        for index in range(count)
    )

    def windows(features: torch.Tensor, start: int, size: int) -> Windows:
        return Windows(
            features, targets, references, outcomes, 68, start, size,
            feature_starts=starts, sample_rows=samples,
        )

    return MarketContextWindows(
        windows(stock, 0, count), windows(spy, 0, count),
    )


def training_data(
    index: int, paired: MarketContextWindows,
) -> TrainingData:
    empty = MarketContextWindows(
        replace_window(paired.stock, len(paired), 0),
        replace_window(paired.spy, len(paired), 0),
    )
    return TrainingData(
        paired, empty, empty,
        torch.zeros(5), torch.ones(5),
        torch.tensor(0.01 + index / 1_000),
        torch.tensor(1.5 + index / 100),
        "ohlcv", 13, SPY_RESIDUAL_TARGET,
    )


def replace_window(window: Windows, start: int, count: int) -> Windows:
    return Windows(
        window.features, window.targets, window.references, window.outcomes,
        window.seq_len, start, count,
        feature_starts=window.feature_starts,
        sample_rows=window.sample_rows,
    )


def forward_rows(offset: float, count: int) -> array:
    values = array("f")
    for index in range(count + 16):
        open_ = 100.0 + offset + index
        values.extend((
            open_, open_ + 1.0, open_ - 1.0, open_ + 0.25,
            1_000.0 + index,
        ))
    return values


def forward_data(
    offset: float, count: int,
) -> tuple[ForwardFeatureWindows, tuple[SampleRows, ...]]:
    samples = tuple(
        SampleRows(16 + index, 17 + index, 30 + index, 16 + index)
        for index in range(count)
    )
    return (
        ForwardFeatureWindows(
            forward_rows(offset, count), samples, 17, "ohlcv",
            torch.zeros(5), torch.ones(5),
        ),
        samples,
    )


def prepared_phase(
    name: str = "fold-1",
    evaluation: tuple[str, ...] | None = None,
) -> ResidualPreparedPhase:
    source = source_phase(name)
    master = MASTER
    if evaluation is not None:
        if len(evaluation) != len(source.evaluation_rows):
            raise ValueError("synthetic evaluation universe changed")
        master = (*MASTER[:-len(evaluation)], *evaluation)
        value = _phase_value(source)
        for row, series in zip(
            value["evaluation_rows"], evaluation, strict=True,
        ):
            row["series"] = series
        source = ContextPhase.parse(value, master)
    count = source.training_rows[0][1]
    paired = paired_training(count)
    training = tuple(
        (series, training_data(index, paired))
        for index, series in enumerate(master)
    )
    forward = []
    for index, (series, size, _) in enumerate(source.evaluation_rows):
        stock, samples = forward_data(index, size)
        spy, _ = forward_data(100 + index, size)
        forward.append(ResidualForwardSeries(
            series=series,
            stock=stock,
            market=MarketContextForwardWindows(stock, spy),
            samples=samples,
            spy_feature_mean=spy.feature_mean.clone(),
            spy_feature_scale=spy.feature_scale.clone(),
        ))
    binding = ResidualPhaseInput(
        source.phase, context_phase_sha256(source),
        source.training_grid_sha256, source.evaluation_grid_sha256,
        digest("residual-scaler-inputs"),
    )
    return ResidualPreparedPhase(
        source, binding, training, tuple(forward),
    )


class StockProbe(torch.nn.Module):
    def __init__(self, value: float = 1.25) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(value))

    def forward(self, stock: torch.Tensor) -> torch.Tensor:
        return self.value.expand(len(stock))


class MarketProbe(torch.nn.Module):
    def __init__(self, _config: object, value: float = 1.25) -> None:
        super().__init__()
        self.config = _config
        self.num_heads = _config.num_heads
        self.head_dim = _config.model_dim // _config.num_heads
        self.value = torch.nn.Parameter(torch.tensor(value))
        self.register_buffer(
            "position", torch.tensor(0.0), persistent=False,
        )

    def forward(
        self, stock: torch.Tensor, spy: torch.Tensor,
    ) -> torch.Tensor:
        expected = self.config.num_heads / (
            self.config.model_dim // self.config.num_heads
        )
        scale = (self.num_heads / self.head_dim) / expected
        return ((self.value + self.position) * scale).expand(len(stock))


@dataclass
class Calls:
    ridges: list[tuple[TrainingData, ...]]
    neural: list[str]
    market: list[object]
    loaders: list[tuple[tuple[TrainingData, ...], int, int, int, bool, object]]
    updates: list[tuple[object, int]]
    seeds: list[int]


@contextmanager
def fake_training() -> object:
    calls = Calls([], [], [], [], [], [])

    def ridge(
        members: tuple[TrainingData, ...], _penalty: float,
    ) -> StockProbe:
        calls.ridges.append(tuple(members))
        return StockProbe()

    def neural(name: str, _candidate: object) -> StockProbe:
        calls.neural.append(name)
        return StockProbe()

    def market(config: object) -> MarketProbe:
        calls.market.append(config)
        return MarketProbe(config)

    def loader(
        members: tuple[TrainingData, ...], batch: int, samples: int,
        seed: int, *, drop_last: bool = False,
    ) -> object:
        token = object()
        calls.loaders.append((
            tuple(members), batch, samples, seed, drop_last, token,
        ))
        return token

    def train(
        _model: object, batches: object, updates: int,
        _learning_rate: float, _weight_decay: float, _device: object,
    ) -> float:
        calls.updates.append((batches, updates))
        return updates / 10

    with patch(
        "tools.spy_residual_runtime.stock_macro_linear_model",
        side_effect=ridge,
    ), patch(
        "tools.spy_residual_runtime._neural_model", side_effect=neural,
    ), patch(
        "tools.spy_residual_runtime.MarketContextTransformer",
        side_effect=market,
    ), patch(
        "tools.spy_residual_runtime._stock_uniform_loader",
        side_effect=loader,
    ), patch(
        "tools.spy_residual_runtime.fit_training_updates",
        side_effect=train,
    ), patch(
        "tools.spy_residual_runtime.mean_loss", return_value=0.25,
    ), patch(
        "tools.spy_residual_runtime.torch.manual_seed",
        side_effect=lambda seed: calls.seeds.append(seed),
    ):
        yield calls


def fit_all(
    runtime: ResidualRuntime, prepared: ResidualPreparedPhase,
) -> tuple[dict[ContextFit, str], dict[ContextFit, object]]:
    fingerprints, tokens = {}, {}
    master = tuple(series for series, _ in prepared.training)
    for fit in expected_residual_fits(master, prepared.source):
        fingerprint, loss, token = runtime.fit_one(fit)
        assert len(fingerprint) == 64 and math.isfinite(loss)
        fingerprints[fit], tokens[fit] = fingerprint, token
    return fingerprints, tokens


def test_fit_order_budgets_and_inputs() -> None:
    prepared = prepared_phase()
    runtime = ResidualRuntime(
        prepared, torch.device("cpu"), RUNTIME,
    )
    fits = expected_residual_fits(MASTER, prepared.source)
    raises(runtime.fit_one, fits[1])
    with fake_training() as calls:
        fingerprints, _ = fit_all(runtime, prepared)

    assert len(calls.ridges) == 1
    assert calls.neural == ["mlp"] * 5
    assert len(calls.market) == 5
    assert calls.seeds == [7, 19, 31, 43, 61] * 2
    assert len(set(fingerprints.values())) == 11
    assert all(
        isinstance(member.train, Windows)
        for members in (calls.ridges[0], *(
            call[0] for call in calls.loaders[:5]
        ))
        for member in members
    )
    assert all(
        isinstance(member.train, MarketContextWindows)
        for call in calls.loaders[5:] for member in call[0]
    )
    for fit, loader, update in zip(
        fits[1:], calls.loaders, calls.updates, strict=True,
    ):
        members, batch, samples, seed, drop_last, token = loader
        assert len(members) == len(fit.members) == 44
        assert (batch, samples, seed, drop_last) == (
            128, 128 * fit.optimizer_updates, fit.seed, True,
        )
        assert update == (token, fit.optimizer_updates)


def test_ordered_label_free_raw_predictions() -> None:
    prepared = prepared_phase()
    runtime = ResidualRuntime(
        prepared, torch.device("cpu"), RUNTIME,
    )
    with fake_training():
        _, tokens = fit_all(runtime, prepared)
        predictions = expected_residual_predictions(
            MASTER, prepared.source,
        )
        first = predictions[0]
        raises(runtime.predict_one, predictions[1], tokens[first.fit])
        raises(
            runtime.predict_one,
            replace(first, prediction_count=first.prediction_count + 1),
            tokens[first.fit],
        )
        raises(runtime.predict_one, first, tokens[predictions[11].fit])

        for _, data in prepared.training[44:]:
            data.train.stock.targets.fill_(10_000)
            data.train.stock.references.fill_(20_000)
            data.train.stock.outcomes.fill_(30_000)

        by_series = dict(prepared.training)
        for prediction in predictions:
            values = runtime.predict_one(
                prediction, tokens[prediction.fit],
            )
            data = by_series[prediction.series]
            expected = float(
                torch.tensor(1.25) * data.target_scale + data.target_mean
            )
            assert values == (expected,) * prediction.prediction_count
        raises(
            runtime.predict_one, predictions[-1],
            tokens[predictions[-1].fit],
        )


def fitted_runtime() -> tuple[
    ResidualPreparedPhase, ResidualRuntime, dict[ContextFit, object],
]:
    prepared = prepared_phase()
    runtime = ResidualRuntime(
        prepared, torch.device("cpu"), RUNTIME,
    )
    with fake_training():
        _, tokens = fit_all(runtime, prepared)
    return prepared, runtime, tokens


def test_model_specific_forward_binding() -> None:
    prepared, runtime, tokens = fitted_runtime()
    first = expected_residual_predictions(MASTER, prepared.source)[0]
    prepared.forward[0].stock.features.add_(1)
    raises(runtime.predict_one, first, tokens[first.fit])

    prepared, runtime, tokens = fitted_runtime()
    first = expected_residual_predictions(MASTER, prepared.source)[0]
    dict(prepared.training)[first.series].target_scale.add_(1)
    raises(runtime.predict_one, first, tokens[first.fit])

    prepared, runtime, tokens = fitted_runtime()
    predictions = expected_residual_predictions(MASTER, prepared.source)
    prepared.forward[0].market.spy.features.add_(1)
    for prediction in predictions[:66]:
        runtime.predict_one(prediction, tokens[prediction.fit])
    raises(
        runtime.predict_one, predictions[66],
        tokens[predictions[66].fit],
    )

    prepared, runtime, tokens = fitted_runtime()
    first = expected_residual_predictions(MASTER, prepared.source)[0]
    with torch.no_grad():
        tokens[first.fit].model.value.add_(1)
    raises(runtime.predict_one, first, tokens[first.fit])


def test_evidence_binds_evaluation_preprocessing() -> None:
    baseline, changed = prepared_phase(), prepared_phase()
    dict(changed.training)[MASTER[44]].target_scale.add_(1)
    with fake_training():
        left = ResidualRuntime(
            baseline, torch.device("cpu"), RUNTIME,
        ).fit_one(expected_residual_fits(MASTER, baseline.source)[0])[0]
        right = ResidualRuntime(
            changed, torch.device("cpu"), RUNTIME,
        ).fit_one(expected_residual_fits(MASTER, changed.source)[0])[0]
    assert left != right

    invalid = prepared_phase()
    invalid.forward[0].market.spy.feature_mean.add_(1)
    raises(ResidualRuntime, invalid, torch.device("cpu"), RUNTIME)


def test_stock_view_is_learnable() -> None:
    prepared = prepared_phase()
    data = prepared.training[0][1]
    stock = tail_training_data(replace(
        data,
        train=data.train.stock,
        validation=data.validation.stock,
        test=data.test.stock,
    ), 17)
    model = stock_macro_linear_model((stock,), 0.001)
    loss = mean_loss(
        model, DataLoader(stock.train, 128), torch.device("cpu"),
    )
    assert math.isfinite(loss) and loss < 2e-6


def main() -> None:
    test_fit_order_budgets_and_inputs()
    test_ordered_label_free_raw_predictions()
    test_model_specific_forward_binding()
    test_evidence_binds_evaluation_preprocessing()
    test_stock_view_is_learnable()
    print("SPY residual runtime tests passed")


if __name__ == "__main__":
    main()
