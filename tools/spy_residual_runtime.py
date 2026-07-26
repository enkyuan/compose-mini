"""Fit the frozen stock-minus-SPY model family without reading truth."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from statistics import fmean
import hashlib
import json
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from tools.context_diagnostic_contract import (
    BATCH_SIZE, ContextFit, ContextPrediction, expected_context_sweep,
)
from tools.context_diagnostic_runtime import _fit_job, _validate_scalers
from tools.experiment import (
    Candidate, _neural_model, _stock_uniform_loader,
    stock_macro_linear_model,
)
from tools.panel_contract import _sha256
from tools.relative_context import (
    MarketContextForwardWindows, MarketContextTransformer,
    MarketContextWindows,
)
from tools.relative_context_contract import (
    HISTORY_BARS, HORIZON_BARS, SPY_RESIDUAL_TARGET, ResidualPhaseInput,
    expected_residual_fits, expected_residual_predictions,
    residual_phase_sha256,
)
from tools.run_universe_scaling import _fingerprint_tensor, model_fingerprint
from tools.spy_residual_controller import (
    ResidualForwardSeries, ResidualPreparedPhase,
)
from tools.train import (
    TrainingData, Windows, _model_output, fit_training_updates, mean_loss,
    tail_training_data,
)
from tools.universe_forward_runner import ForwardFeatureWindows


@dataclass(frozen=True, slots=True)
class _Fitted:
    fit: ContextFit
    candidate: Candidate
    model: nn.Module
    members: tuple[tuple[str, TrainingData], ...]
    fingerprint: str


def _views(data: TrainingData, series: str) -> tuple[TrainingData, TrainingData]:
    """Return history-17 stock-only and stock-plus-SPY views."""
    splits = (data.train, data.validation, data.test)
    if type(data) is not TrainingData or \
       data.target_kind != SPY_RESIDUAL_TARGET or \
       data.horizon_bars != HORIZON_BARS or \
       data.feature_set != "ohlcv" or \
       any(type(split) is not MarketContextWindows for split in splits) or \
       not len(data.train) or len(data.validation) or len(data.test):
        raise ValueError(f"{series} residual training data is invalid")
    _validate_scalers(data, series)

    def projected(name: str) -> TrainingData:
        return replace(data, **{
            split: getattr(getattr(data, split), name)
            for split in ("train", "validation", "test")
        })

    stock = tail_training_data(projected("stock"), HISTORY_BARS)
    spy = tail_training_data(projected("spy"), HISTORY_BARS)
    market = replace(data, **{
        split: MarketContextWindows(
            getattr(stock, split), getattr(spy, split),
        )
        for split in ("train", "validation", "test")
    })
    return stock, market


def _prediction_input_fingerprint(
    series: str, data: TrainingData, forward: ResidualForwardSeries,
    market: bool,
) -> str:
    stock, spy = forward.stock, forward.market.spy
    digest = hashlib.sha256(json.dumps({
        "feature_set": stock.feature_set,
        "market": market,
        "seq_len": stock.seq_len,
        "series": series,
        "stock_starts": list(stock.starts),
    }, separators=(",", ":"), sort_keys=True).encode())
    values = [
        ("target_mean", data.target_mean),
        ("target_scale", data.target_scale),
        ("stock_feature_mean", stock.feature_mean),
        ("stock_feature_scale", stock.feature_scale),
        ("stock_features", stock.features),
    ]
    if market:
        digest.update(json.dumps({
            "spy_starts": list(spy.starts),
        }, separators=(",", ":"), sort_keys=True).encode())
        values.extend((
            ("spy_feature_mean", spy.feature_mean),
            ("spy_feature_scale", spy.feature_scale),
            ("spy_features", spy.features),
        ))
    for name, value in values:
        _fingerprint_tensor(digest, name, value)
    return digest.hexdigest()


class ResidualRuntime:
    """Own fresh causal residual fits and label-free evaluation inference."""

    def __init__(
        self, prepared: ResidualPreparedPhase, device: torch.device,
        runtime_sha256: str,
    ) -> None:
        if type(prepared) is not ResidualPreparedPhase or \
           not isinstance(device, torch.device) or \
           device != torch.device("cpu"):
            raise ValueError("residual runtime requires one CPU phase")
        self._runtime_sha256 = _sha256(
            runtime_sha256, "residual runtime identity",
        )
        source = prepared.source
        if ResidualPhaseInput.parse(
            asdict(prepared.binding), source,
        ) != prepared.binding:
            raise ValueError("residual phase binding changed")
        self._phase_sha256 = residual_phase_sha256(prepared.binding)
        master = (
            *(series for series, _ in source.training_rows),
            *(series for series, _, _ in source.evaluation_rows),
        )
        self._fits = expected_residual_fits(master, source)
        self._predictions = expected_residual_predictions(master, source)
        candidates = tuple(map(
            Candidate.parse, expected_context_sweep()["candidates"],
        ))
        self._candidate = next(
            item for item in candidates if item.seq_len == HISTORY_BARS
        )

        counts = dict(source.training_rows)
        views = {}
        for series, data in prepared.training:
            stock, market = _views(data, series)
            if series in counts and len(stock.train) != counts[series]:
                raise ValueError(f"{series} residual training count changed")
            views[series] = stock, market
        if tuple(views) != master:
            raise ValueError("residual training order changed")
        self._views = views

        expected_forward = tuple(
            series for series, _, _ in source.evaluation_rows
        )
        forward = {item.series: item for item in prepared.forward}
        if tuple(forward) != expected_forward:
            raise ValueError("residual prediction order changed")
        for item, (_, count, _) in zip(
            prepared.forward, source.evaluation_rows, strict=True,
        ):
            stock, spy = item.stock, item.market.spy
            data = views[item.series][0]
            if len(item.samples) != count or any(
                type(value) is not ForwardFeatureWindows
                for value in (stock, spy)
            ) or stock.seq_len != HISTORY_BARS or \
               spy.seq_len != HISTORY_BARS or \
               stock.feature_set != data.feature_set or \
               not torch.equal(
                   stock.feature_mean, data.feature_mean,
               ) or not torch.equal(
                   stock.feature_scale, data.feature_scale,
               ) or not torch.equal(
                   spy.feature_mean, item.spy_feature_mean,
               ) or not torch.equal(
                   spy.feature_scale, item.spy_feature_scale,
               ):
                raise ValueError(
                    f"{item.series} residual prediction input changed",
                )
        self._forward = forward
        self._prediction_inputs = {
            (series, market): _prediction_input_fingerprint(
                series, views[series][0], forward[series], market,
            )
            for series in expected_forward for market in (False, True)
        }
        self._prediction_closure = {
            market: hashlib.sha256(json.dumps({
                "inputs": [
                    self._prediction_inputs[series, market]
                    for series in expected_forward
                ],
                "market": market,
                "series": list(expected_forward),
            }, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
            for market in (False, True)
        }
        self._device = device
        self._fitted: dict[ContextFit, _Fitted] = {}
        self._fit_index = self._prediction_index = 0

    def _members(
        self, fit: ContextFit,
    ) -> tuple[tuple[str, TrainingData], ...]:
        index = int(fit.model == "panel_transformer")
        return tuple(
            (series, self._views[series][index])
            for series in fit.members
        )

    def _fingerprint(self, fitted: _Fitted) -> str:
        model = model_fingerprint(
            fitted.model, _fit_job(fitted.fit), fitted.candidate,
            fitted.members, self._runtime_sha256,
        )
        return hashlib.sha256(json.dumps({
            "model_fingerprint": model,
            "prediction_inputs_sha256": self._prediction_closure[
                fitted.fit.model == "panel_transformer"
            ],
            "residual_phase_sha256": self._phase_sha256,
        }, separators=(",", ":"), sort_keys=True).encode()).hexdigest()

    def _verify_fitted(self, fitted: _Fitted) -> None:
        if self._fingerprint(fitted) != fitted.fingerprint:
            raise ValueError("residual fitted state changed")

    def _verify_prediction_state(
        self, fitted: _Fitted, series: str,
    ) -> None:
        self._verify_fitted(fitted)
        market = fitted.fit.model == "panel_transformer"
        if _prediction_input_fingerprint(
            series, self._views[series][0], self._forward[series],
            market,
        ) != self._prediction_inputs[series, market]:
            raise ValueError("residual prediction inputs changed")

    def fit_one(self, fit: ContextFit) -> tuple[str, float, object]:
        """Fit the next declared residual state with its prior update budget."""
        if self._fit_index >= len(self._fits) or \
           fit != self._fits[self._fit_index]:
            raise ValueError("residual fit order changed")
        members = self._members(fit)
        data = tuple(value for _, value in members)
        if fit.model == "global_ridge":
            model = stock_macro_linear_model(
                data, self._candidate.ridge,
            ).to(self._device)
            loss = fmean(
                mean_loss(
                    model, DataLoader(value.train, BATCH_SIZE), self._device,
                )
                for value in data
            )
        else:
            if fit.seed is None or fit.optimizer_updates < 1:
                raise ValueError("residual neural fit budget is invalid")
            torch.manual_seed(fit.seed)
            model = (
                MarketContextTransformer(self._candidate.config())
                if fit.model == "panel_transformer"
                else _neural_model("mlp", self._candidate)
            ).to(self._device)
            loader = _stock_uniform_loader(
                data, BATCH_SIZE, BATCH_SIZE * fit.optimizer_updates,
                fit.seed, drop_last=True,
            )
            loss = fit_training_updates(
                model, loader, fit.optimizer_updates,
                self._candidate.learning_rate,
                self._candidate.weight_decay, self._device,
            )
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("residual training loss is invalid")
        token = _Fitted(fit, self._candidate, model, members, "")
        fitted = replace(token, fingerprint=self._fingerprint(token))
        self._fitted[fit] = fitted
        self._fit_index += 1
        return fitted.fingerprint, loss, fitted

    def predict_one(
        self, prediction: ContextPrediction, fitted: object,
    ) -> tuple[float, ...]:
        """Predict the next raw residual vector without reading its labels."""
        if self._fit_index != len(self._fits) or \
           self._prediction_index >= len(self._predictions) or \
           prediction != self._predictions[self._prediction_index] or \
           type(fitted) is not _Fitted or \
           fitted is not self._fitted.get(prediction.fit):
            raise ValueError("residual prediction order or fit changed")
        self._verify_prediction_state(fitted, prediction.series)
        item = self._forward[prediction.series]
        dataset = (
            item.market
            if prediction.fit.model == "panel_transformer"
            else item.stock
        )
        values = []
        fitted.model.eval()
        with torch.inference_mode():
            for features in DataLoader(dataset, BATCH_SIZE):
                outputs = _model_output(
                    fitted.model, features, self._device,
                )
                batch = len(
                    features if isinstance(features, torch.Tensor)
                    else features[0]
                )
                if not isinstance(outputs, torch.Tensor) or \
                   outputs.dtype != torch.float32 or \
                   outputs.shape != (batch,) or \
                   not torch.isfinite(outputs).all():
                    raise ValueError(
                        "residual prediction output is invalid",
                    )
                data = self._views[prediction.series][0]
                values.extend((
                    outputs.detach().cpu() * data.target_scale +
                    data.target_mean
                ).tolist())
        if len(values) != prediction.prediction_count:
            raise ValueError("residual prediction count changed")
        self._verify_prediction_state(fitted, prediction.series)
        self._prediction_index += 1
        return tuple(values)
