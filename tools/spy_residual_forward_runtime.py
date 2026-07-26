"""Transfer verified residual states to one later label-free panel."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import struct

import torch
from torch import nn

from tools.context_diagnostic_contract import ContextFit
from tools.data_v1 import FEATURE_COUNT
from tools.relative_context_contract import (
    HISTORY_BARS, SEEDS, ResidualFitEvidence, expected_residual_fits,
)
from tools.run_universe_scaling import _fingerprint_tensor
from tools.spy_residual_forward_contract import (
    FORWARD_UNIVERSE, STATE_FINGERPRINTS,
)
from tools.spy_residual_forward_inputs import (
    ForwardGrid, ForwardPredictions, ForwardRunBinding,
    ForwardSeriesPrediction, SeedPrediction, SpyResidualForwardInputs,
    _grid_sha256, _json_line,
)
from tools.spy_residual_controller import ResidualPreparedPhase
from tools.spy_residual_runtime import ResidualRuntime


@dataclass(frozen=True, slots=True)
class _State:
    fit: ContextFit
    fingerprint: str
    model: nn.Module
    model_sha256: str
    architecture_sha256: str


@dataclass(frozen=True, slots=True)
class _Scalers:
    series: str
    stock_mean: torch.Tensor
    stock_scale: torch.Tensor
    spy_mean: torch.Tensor
    spy_scale: torch.Tensor
    target_mean: torch.Tensor
    target_scale: torch.Tensor
    sha256: str


def _model_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for kind, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in sorted(values):
            _fingerprint_tensor(digest, f"{kind}:{name}", value)
    return digest.hexdigest()


def _architecture_sha256(model: nn.Module) -> str:
    modules = []
    for name, module in sorted(model.named_modules()):
        if "forward" in module.__dict__:
            raise ValueError("forward model override changed")
        value: dict[str, object] = {
            "name": name,
            "type": f"{type(module).__module__}.{type(module).__qualname__}",
        }
        config = getattr(module, "config", None)
        if config is not None:
            if not is_dataclass(config) or isinstance(config, type):
                raise ValueError("forward model config changed")
            value["config"] = asdict(config)
        for field in ("num_heads", "head_dim"):
            if hasattr(module, field):
                item = getattr(module, field)
                if type(item) is not int or item < 1:
                    raise ValueError("forward model architecture changed")
                value[field] = item
        modules.append(value)
    return hashlib.sha256(json.dumps(
        modules, allow_nan=False, separators=(",", ":"), sort_keys=True,
    ).encode()).hexdigest()


def _scaler_sha256(value: _Scalers) -> str:
    digest = hashlib.sha256(value.series.encode() + b"\0")
    for name in (
        "stock_mean", "stock_scale", "spy_mean", "spy_scale",
        "target_mean", "target_scale",
    ):
        _fingerprint_tensor(digest, name, getattr(value, name))
    return digest.hexdigest()


def _scalers(
    prepared: ResidualPreparedPhase,
) -> tuple[_Scalers, ...]:
    training = dict(prepared.training)
    forward = {
        item.series: item for item in prepared.forward
    }
    if tuple(forward) != FORWARD_UNIVERSE or any(
        series not in training for series in FORWARD_UNIVERSE
    ):
        raise ValueError("forward scaler order changed")
    result = []
    for series in FORWARD_UNIVERSE:
        data, item = training[series], forward[series]
        values = (
            data.feature_mean.clone(), data.feature_scale.clone(),
            item.spy_feature_mean.clone(), item.spy_feature_scale.clone(),
            data.target_mean.clone(), data.target_scale.clone(),
        )
        token = _Scalers(series, *values, "")
        result.append(_Scalers(
            series, *values, _scaler_sha256(token),
        ))
    return tuple(result)


class SpyResidualForwardRuntime:
    """Own only five verified models and their frozen inference scalers."""

    def __init__(
        self, states: tuple[_State, ...], scalers: tuple[_Scalers, ...],
        device: torch.device,
    ) -> None:
        self._states, self._scalers, self._device = states, scalers, device
        self._features, self._gates = hashlib.sha256(), hashlib.sha256()
        self._as_of: list[str] = []
        self._index, self._bound = 0, False

    @property
    def states(self) -> tuple[tuple[int, str], ...]:
        """Expose the ordered public identity of the retained states."""
        return tuple(
            (int(state.fit.seed), state.fingerprint)
            for state in self._states
        )

    @classmethod
    def reproduce(
        cls, prepared: ResidualPreparedPhase, device: torch.device,
        runtime_sha256: str, expected: Sequence[ResidualFitEvidence],
    ) -> SpyResidualForwardRuntime:
        """Run the full calibration schedule, then retain five states."""
        master = tuple(
            series for series, _ in prepared.training
        )
        fits = expected_residual_fits(
            master, prepared.source,
        )
        evidence = tuple(expected)
        if len(evidence) != len(fits) or any(
            type(item) is not ResidualFitEvidence or item.fit != fit
            for fit, item in zip(fits, evidence, strict=True)
        ):
            raise ValueError("calibration fit evidence changed")
        source = ResidualRuntime(prepared, device, runtime_sha256)
        states = []
        for fit, item in zip(fits, evidence, strict=True):
            fingerprint, _, token = source.fit_one(fit)
            if fingerprint != item.state_fingerprint:
                raise ValueError("calibration state fingerprint changed")
            if fit.model != "panel_transformer":
                continue
            expected_state = dict(zip(
                SEEDS, STATE_FINGERPRINTS, strict=True,
            )).get(fit.seed)
            if fingerprint != expected_state:
                raise ValueError("forward Transformer state changed")
            model = token.model
            states.append(_State(
                fit, fingerprint, model, _model_sha256(model),
                _architecture_sha256(model),
            ))
        scalers = _scalers(prepared)
        for state in states:
            for series in FORWARD_UNIVERSE:
                source._verify_prediction_state(
                    source._fitted[state.fit], series,
                )
        source._fitted.clear()
        ordered = tuple(states)
        if tuple(
            (state.fit.seed, state.fingerprint) for state in ordered
        ) != tuple(zip(SEEDS, STATE_FINGERPRINTS, strict=True)):
            raise ValueError("forward Transformer closure changed")
        return cls(ordered, scalers, device)

    def _verify(self) -> None:
        if any(
            _model_sha256(state.model) != state.model_sha256 or
            _architecture_sha256(state.model) != state.architecture_sha256
            for state in self._states
        ) or any(
            value.series != series or
            _scaler_sha256(value) != value.sha256
            for series, value in zip(
                FORWARD_UNIVERSE, self._scalers, strict=True,
            )
        ):
            raise ValueError("forward inference state changed")

    @staticmethod
    def _raw(values: Sequence[float]) -> torch.Tensor:
        result = torch.tensor(values, dtype=torch.float32).view(
            HISTORY_BARS, FEATURE_COUNT,
        )
        if not torch.isfinite(result).all():
            raise ValueError("forward feature values are invalid")
        return result

    def _digests(
        self, batch: SpyResidualForwardInputs,
    ) -> tuple[object, object]:
        features, gates = self._features.copy(), self._gates.copy()
        features.update(json.dumps(
            {
                "as_of": batch.as_of,
                "index": batch.index,
                "series": [item.series for item in batch.stocks],
                "spy": batch.spy.series,
            },
            separators=(",", ":"), sort_keys=True,
        ).encode())
        for item in (*batch.stocks, batch.spy):
            features.update(struct.pack(
                f"<{len(item.values)}f", *item.values,
            ))
        closes = (
            batch.spy.values[3],
            batch.spy.values[(HISTORY_BARS - 1) * FEATURE_COUNT + 3],
        )
        gates.update(json.dumps(
            {
                "as_of": batch.as_of,
                "index": batch.index,
                "regime": batch.regime,
            },
            separators=(",", ":"), sort_keys=True,
        ).encode())
        gates.update(struct.pack("<2f", *closes))
        return features, gates

    def predict(
        self, batch: SpyResidualForwardInputs,
    ) -> ForwardPredictions:
        """Run each retained seed once over one ordered 11-stock batch."""
        if type(batch) is not SpyResidualForwardInputs or \
           batch.index != self._index:
            raise ValueError("forward prediction batch order changed")
        self._verify()
        stock_raw = {
            item.series: self._raw(item.values) for item in batch.stocks
        }
        spy_raw = self._raw(batch.spy.values)
        stock = torch.stack(tuple(
            (stock_raw[series] - scale.stock_mean) / scale.stock_scale
            for series, scale in zip(
                FORWARD_UNIVERSE, self._scalers, strict=True,
            )
        )).to(self._device)
        spy = torch.stack(tuple(
            (spy_raw - scale.spy_mean) / scale.spy_scale
            for scale in self._scalers
        )).to(self._device)
        target_mean = torch.stack(tuple(
            scale.target_mean for scale in self._scalers
        ))
        target_scale = torch.stack(tuple(
            scale.target_scale for scale in self._scalers
        ))
        columns = []
        with torch.inference_mode():
            for state in self._states:
                state.model.eval()
                output = state.model(stock, spy)
                if not isinstance(output, torch.Tensor) or \
                   output.dtype != torch.float32 or \
                   output.shape != (len(FORWARD_UNIVERSE),) or \
                   not torch.isfinite(output).all():
                    raise ValueError("forward model output is invalid")
                columns.append(
                    output.cpu() * target_scale + target_mean,
                )
        self._verify()
        features, gates = self._digests(batch)
        matrix = torch.stack(columns, dim=1).tolist()
        result = tuple(
            ForwardSeriesPrediction(
                series,
                tuple(
                    SeedPrediction(seed, fingerprint, float(value))
                    for (seed, fingerprint), value in zip(
                        self.states, row, strict=True,
                    )
                ),
            )
            for series, row in zip(
                FORWARD_UNIVERSE, matrix, strict=True,
            )
        )
        self._features, self._gates = features, gates
        self._as_of.append(batch.as_of)
        self._index += 1
        return result

    def bind(
        self, provenance: Mapping[str, object],
    ) -> ForwardRunBinding:
        """Bind one immutable provenance map to this one-shot runtime."""
        reserved = {"future", "prediction_rows_sha256", "states"}
        try:
            value = json.loads(json.dumps(
                provenance, allow_nan=False, sort_keys=True,
            ))
        except (TypeError, ValueError) as error:
            raise ValueError("forward provenance is invalid") from error
        if self._bound or not isinstance(value, dict) or not value or \
           reserved.intersection(value):
            raise ValueError("forward provenance is invalid")
        self._bound = True

        def evidence(
            grid: ForwardGrid,
            records: Sequence[Mapping[str, object]],
        ) -> Mapping[str, object]:
            self._verify()
            if self._index != len(grid.triples) or tuple(self._as_of) != \
                    tuple(row[0] for row in grid.triples):
                raise ValueError("forward prediction grid changed")
            rows = hashlib.sha256()
            for record in records:
                rows.update(_json_line(record).encode())
            return {
                **value,
                "future": {
                    "feature_inputs_sha256":
                        self._features.hexdigest(),
                    "gate_inputs_sha256": self._gates.hexdigest(),
                    "timestamp_grid_sha256": _grid_sha256(grid),
                },
                "prediction_rows_sha256": rows.hexdigest(),
                "states": [
                    {
                        "seed": seed,
                        "state_fingerprint": fingerprint,
                    }
                    for seed, fingerprint in self.states
                ],
            }

        return ForwardRunBinding(self.predict, evidence)
