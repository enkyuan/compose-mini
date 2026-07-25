"""Fit and predict the frozen temporal-context family."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
import hashlib
import json
import math

import torch
from torch import nn
from torch.utils.data import DataLoader

from tools.context_diagnostic_contract import (
    BATCH_SIZE, HISTORY_LENGTHS, MAX_HISTORY, RUNTIME_TO_PUBLIC,
    ContextFit, ContextPhase, ContextPrediction,
    expected_context_fits, expected_context_predictions,
    expected_context_sweep,
)
from tools.data_v1 import FEATURE_COUNT
from tools.experiment import (
    Candidate, _neural_model, _stock_uniform_loader,
    stock_macro_linear_model,
)
from tools.panel_contract import _sha256
from tools.run_universe_scaling import _fingerprint_tensor, model_fingerprint
from tools.train import (
    TrainingData, Windows, fit_training_updates, mean_loss,
    tail_training_data,
)
from tools.universe_forward_runner import ForwardFeatureWindows
from tools.universe_scaling_contract import FitJob

_PUBLIC_TO_RUNTIME = {
    public: runtime for runtime, public in RUNTIME_TO_PUBLIC.items()
}


def _ordered(
    values: object, names: tuple[str, ...], label: str,
) -> tuple[object, ...]:
    if not isinstance(values, Mapping) or tuple(values) != names:
        raise ValueError(f"{label} must match the frozen series order")
    return tuple(values[name] for name in names)


def _validate_scalers(data: TrainingData, series: str) -> None:
    feature = (data.feature_mean, data.feature_scale)
    target = (data.target_mean, data.target_scale)
    tensors = (*feature, *target)
    if any(
        not isinstance(value, torch.Tensor) or
        value.dtype != torch.float32 or value.device.type != "cpu" or
        not torch.isfinite(value).all()
        for value in tensors
    ) or any(value.shape != (FEATURE_COUNT,) for value in feature) or \
       any(value.ndim for value in target) or \
       not torch.all(data.feature_scale > 0) or data.target_scale <= 0:
        raise ValueError(f"{series} training scalers are invalid")


def _fit_job(fit: ContextFit) -> FitJob:
    ridge = fit.model == "global_ridge"
    return FitJob(
        "ridge" if ridge else "pooled",
        None if ridge else "fixed-update",
        len(fit.members), fit.phase, fit.model, fit.seed, fit.members,
    )


def _prediction_input_fingerprint(
    series: str, data: TrainingData, forward: ForwardFeatureWindows,
) -> str:
    digest = hashlib.sha256(json.dumps({
        "feature_set": forward.feature_set,
        "seq_len": forward.seq_len,
        "series": series,
        "starts": list(forward.starts),
    }, separators=(",", ":"), sort_keys=True).encode())
    for name, value in (
        ("data_feature_mean", data.feature_mean),
        ("data_feature_scale", data.feature_scale),
        ("target_mean", data.target_mean),
        ("target_scale", data.target_scale),
        ("forward_feature_mean", forward.feature_mean),
        ("forward_feature_scale", forward.feature_scale),
        ("forward_features", forward.features),
    ):
        _fingerprint_tensor(digest, name, value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _Fitted:
    fit: ContextFit
    candidate: Candidate
    model: nn.Module
    fingerprint: str


class ContextRuntime:
    """Own fresh context fits and label-free unseen-stock inference."""

    def __init__(
        self, master: Sequence[str], phase: ContextPhase,
        max_data: Mapping[str, TrainingData],
        max_forward: Mapping[str, ForwardFeatureWindows],
        device: torch.device, runtime_sha256: str,
    ) -> None:
        names = tuple(master)
        fits = expected_context_fits(names, phase)
        predictions = expected_context_predictions(names, phase)
        if not isinstance(device, torch.device) or \
           device != torch.device("cpu"):
            raise ValueError("context diagnostic requires CPU")
        self._runtime_sha256 = _sha256(
            runtime_sha256, "context runtime identity",
        )
        sweep = expected_context_sweep()
        candidates = tuple(map(Candidate.parse, sweep["candidates"]))
        if tuple(item.seq_len for item in candidates) != HISTORY_LENGTHS:
            raise ValueError("context candidate family changed")
        by_history = {item.seq_len: item for item in candidates}
        feature_set = candidates[0].feature_set
        training = tuple(series for series, _ in phase.training_rows)
        evaluation = tuple(series for series, _, _ in phase.evaluation_rows)

        data_values = _ordered(max_data, names, "context training data")
        training_counts = dict(phase.training_rows)
        for series, data in zip(names, data_values, strict=True):
            if type(data) is not TrainingData or \
               any(type(split) is not Windows
                   for split in (data.train, data.validation, data.test)) or \
               len(data.validation) or len(data.test) or \
               len(data.train.sample_rows or ()) != len(data.train) or \
               any(
                   len(getattr(data.train, name)) != len(data.train)
                   for name in ("targets", "references", "outcomes")
               ) or \
               tail_training_data(data, MAX_HISTORY) is not data or \
               not len(data.train) or \
               series in training_counts and \
               len(data.train) != training_counts[series] or \
               data.feature_set != feature_set or \
               data.horizon_bars != sweep["target_horizon_bars"] or \
               data.target_kind != sweep["target_kind"]:
                raise ValueError(f"{series} max-history data is invalid")
            _validate_scalers(data, series)
        self._data = dict(zip(names, data_values, strict=True))

        forward_values = _ordered(
            max_forward, evaluation, "context forward data",
        )
        for (series, count, _), forward in zip(
            phase.evaluation_rows, forward_values, strict=True,
        ):
            if type(forward) is not ForwardFeatureWindows or \
               forward.seq_len != MAX_HISTORY or len(forward) != count or \
               forward.feature_set != self._data[series].feature_set or \
               not isinstance(forward.features, torch.Tensor) or \
               forward.features.ndim != 2 or \
               forward.features.shape[1] != FEATURE_COUNT or \
               forward.features.dtype != torch.float32 or \
               forward.features.device.type != "cpu" or \
               not torch.isfinite(forward.features).all() or any(
                   type(start) is not int or start < 0 or
                   start + MAX_HISTORY > len(forward.features)
                   for start in forward.starts
               ) or any(
                   not isinstance(value, torch.Tensor) or
                   value.dtype != torch.float32 or
                   value.device.type != "cpu" or
                   value.shape != (FEATURE_COUNT,) or
                   not torch.isfinite(value).all()
                   for value in (
                       forward.feature_mean, forward.feature_scale,
                   )
               ) or not torch.equal(
                   forward.feature_mean, self._data[series].feature_mean,
               ) or not torch.equal(
                   forward.feature_scale, self._data[series].feature_scale,
               ):
                raise ValueError(
                    f"{series} max-history forward data is invalid",
                )
        self._forward = dict(zip(evaluation, forward_values, strict=True))
        self._prediction_inputs = {
            series: _prediction_input_fingerprint(
                series, self._data[series], self._forward[series],
            )
            for series in evaluation
        }
        self._members = {
            history: tuple(
                (series, tail_training_data(self._data[series], history))
                for series in training
            )
            for history in HISTORY_LENGTHS
        }
        self._candidates = by_history
        self._device = device
        self._fits, self._predictions = fits, predictions
        self._fitted: dict[ContextFit, _Fitted] = {}
        self._fit_index = self._prediction_index = 0
        self._evaluation = evaluation

    def _fingerprint(
        self, fit: ContextFit, candidate: Candidate, model: nn.Module,
    ) -> str:
        return model_fingerprint(
            model, _fit_job(fit), candidate,
            self._members[fit.history], self._runtime_sha256,
        )

    def _verify_fitted(self, fitted: _Fitted) -> None:
        if self._fingerprint(
            fitted.fit, fitted.candidate, fitted.model,
        ) != fitted.fingerprint:
            raise ValueError("context fitted state changed")

    def _verify_prediction_state(
        self, fitted: _Fitted, series: str,
    ) -> None:
        self._verify_fitted(fitted)
        if _prediction_input_fingerprint(
            series, self._data[series], self._forward[series],
        ) != self._prediction_inputs[series]:
            raise ValueError("context prediction inputs changed")

    def fit_one(self, fit: ContextFit) -> tuple[str, float, object]:
        """Fit the next declared context state with its exact update budget."""
        if self._fit_index >= len(self._fits) or \
           fit != self._fits[self._fit_index]:
            raise ValueError("context fit order changed")
        candidate = self._candidates[fit.history]
        members = self._members[fit.history]
        data = tuple(value for _, value in members)
        if fit.model == "global_ridge":
            model = stock_macro_linear_model(data, candidate.ridge).to(
                self._device,
            )
            loss = fmean(
                mean_loss(
                    model, DataLoader(value.train, BATCH_SIZE), self._device,
                )
                for value in data
            )
        else:
            if fit.seed is None or fit.optimizer_updates < 1:
                raise ValueError("context neural fit budget is invalid")
            torch.manual_seed(fit.seed)
            model = _neural_model(
                _PUBLIC_TO_RUNTIME[fit.model], candidate,
            ).to(self._device)
            loader = _stock_uniform_loader(
                data, BATCH_SIZE, BATCH_SIZE * fit.optimizer_updates,
                fit.seed, drop_last=True,
            )
            loss = fit_training_updates(
                model, loader, fit.optimizer_updates,
                candidate.learning_rate, candidate.weight_decay, self._device,
            )
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("context training loss is invalid")
        fitted = _Fitted(
            fit, candidate, model, self._fingerprint(fit, candidate, model),
        )
        self._fitted[fit] = fitted
        self._fit_index += 1
        return fitted.fingerprint, loss, fitted

    def predict_one(
        self, prediction: ContextPrediction, fitted: object,
    ) -> Sequence[float]:
        """Predict the next declared unseen series without reading labels."""
        if self._fit_index != len(self._fits) or \
           self._prediction_index >= len(self._predictions) or \
           prediction != self._predictions[self._prediction_index] or \
           type(fitted) is not _Fitted or \
           fitted is not self._fitted.get(prediction.fit):
            raise ValueError("context prediction order or fit changed")
        self._verify_prediction_state(fitted, prediction.series)

        values = []
        fitted.model.eval()
        with torch.inference_mode():
            for features in DataLoader(
                self._forward[prediction.series], BATCH_SIZE,
            ):
                inputs = features[:, -prediction.fit.history:].to(
                    self._device,
                )
                outputs = fitted.model(inputs)
                if not isinstance(outputs, torch.Tensor) or \
                   outputs.shape != (len(inputs),) or \
                   not torch.isfinite(outputs).all():
                    raise ValueError("context prediction output is invalid")
                data = self._data[prediction.series]
                values.extend((
                    outputs.detach().cpu() * data.target_scale +
                    data.target_mean
                ).tolist())
        if len(values) != prediction.prediction_count:
            raise ValueError("context prediction count changed")
        self._verify_prediction_state(fitted, prediction.series)
        self._prediction_index += 1
        return values
