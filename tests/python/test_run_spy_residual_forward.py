#!/usr/bin/env python3
"""Verify authenticated state transfer and label-free forward publication."""

from __future__ import annotations

from array import array
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("SPY residual forward tests require PyTorch") from error

from test_spy_residual_forward_inputs import (
    fixture, frozen_files, prepare,
)
from test_spy_residual_runtime import (
    RUNTIME, fake_training, fit_all, prepared_phase,
)
from tools.arm_spy_residual_forward import (
    ForwardCalibration, ForwardRunContext, _provenance,
)
from tools.panel_contract import FileBinding, selected_source_tree
from tools.relative_context_contract import (
    SEEDS, ResidualFitEvidence, expected_residual_fits,
)
from tools.session_samples import SampleRows
from tools.spy_residual_forward_contract import (
    FORWARD_CONFIG, FORWARD_RUN_ID, FORWARD_SOURCE_PATHS, FORWARD_UNIVERSE,
    STATE_FINGERPRINTS,
)
from tools.spy_residual_forward_inputs import (
    CandidateLedger, ForwardPredictionSession, ForwardRunBinding,
    ForwardSeriesPrediction, SeedPrediction, SpyResidualForwardInputs,
)
from tools.spy_residual_forward_runtime import SpyResidualForwardRuntime
from tools.run_spy_residual_forward import publish_forward_candidate
from tools.universe_forward_runner import ForwardFeatureWindows


def digest(value: object) -> str:
    return hashlib.sha256(repr(value).encode()).hexdigest()


def rejects(function: object, *args: object) -> None:
    try:
        function(*args)  # type: ignore[operator]
    except (TypeError, ValueError):
        return
    raise AssertionError("invalid forward operation succeeded")


def calibration_evidence() -> tuple[
    object, tuple[ResidualFitEvidence, ...],
]:
    prepared = prepared_phase("calibration", FORWARD_UNIVERSE)
    training = dict(prepared.training)
    for index, item in enumerate(prepared.forward, 1):
        stock_mean = torch.arange(5, dtype=torch.float32) * index / 10
        stock_scale = torch.arange(1, 6, dtype=torch.float32) + index / 10
        spy_mean = -stock_mean
        spy_scale = stock_scale + 1
        data = training[item.series]
        for target, source in (
            (data.feature_mean, stock_mean),
            (data.feature_scale, stock_scale),
            (item.stock.feature_mean, stock_mean),
            (item.stock.feature_scale, stock_scale),
            (item.spy_feature_mean, spy_mean),
            (item.spy_feature_scale, spy_scale),
            (item.market.spy.feature_mean, spy_mean),
            (item.market.spy.feature_scale, spy_scale),
        ):
            target.copy_(source)
    from tools.spy_residual_runtime import ResidualRuntime

    runtime = ResidualRuntime(
        prepared, torch.device("cpu"), RUNTIME,
    )
    with fake_training():
        fingerprints, _ = fit_all(runtime, prepared)
    return prepared, tuple(
        ResidualFitEvidence(
            fit, digest(("provenance", fit)),
            fingerprints[fit], 0.0,
        )
        for fit in expected_residual_fits(
            tuple(series for series, _ in prepared.training),
            prepared.source,
        )
    )


@contextmanager
def reproduced() -> object:
    prepared, evidence = calibration_evidence()
    fingerprints = tuple(
        item.state_fingerprint
        for item in evidence if item.fit.model == "panel_transformer"
    )
    with patch(
        "tools.spy_residual_forward_runtime.STATE_FINGERPRINTS",
        fingerprints,
    ), patch(
        "tools.spy_residual_forward_inputs.STATE_FINGERPRINTS",
        fingerprints,
    ), fake_training() as calls:
        yield (
            SpyResidualForwardRuntime.reproduce(
                prepared, torch.device("cpu"), RUNTIME, evidence,
            ),
            calls,
            fingerprints,
        )


def batch(index: int = 0) -> SpyResidualForwardInputs:
    from tools.spy_residual_forward_inputs import ForwardSeriesInput

    def values(offset: float) -> tuple[float, ...]:
        return tuple(
            value
            for row in range(17)
            for value in (
                100.0 + offset + row,
                101.0 + offset + row,
                99.0 + offset + row,
                100.5 + offset + row,
                1_000.0 + row,
            )
        )

    return SpyResidualForwardInputs(
        index, f"2026-02-24T{10 + index:02d}:00:00Z",
        tuple(
            ForwardSeriesInput(series, values(member))
            for member, series in enumerate(FORWARD_UNIVERSE)
        ),
        ForwardSeriesInput("SPY", values(100.0)),
        "nonnegative",
    )


def test_reproduces_full_schedule_and_vectorizes_later_inference() -> None:
    with reproduced() as (runtime, calls, fingerprints):
        assert calls.seeds == [*SEEDS, *SEEDS]
        assert len(calls.ridges) == 1
        assert calls.neural == ["mlp"] * len(SEEDS)
        assert len(calls.market) == len(SEEDS)
        assert runtime.states == tuple(zip(
            SEEDS, fingerprints, strict=True,
        ))

        inputs: list[tuple[torch.Tensor, ...]] = []
        for state in runtime._states:
            state.model.register_forward_hook(
                lambda _model, values, _output: inputs.append(tuple(
                    item.detach().clone() for item in values
                )),
            )
        current = batch()
        predictions = runtime.predict(current)
        assert tuple(item.series for item in predictions) == FORWARD_UNIVERSE
        assert not torch.equal(
            runtime._scalers[0].stock_mean,
            runtime._scalers[1].stock_mean,
        )
        assert not torch.equal(
            runtime._scalers[0].spy_scale,
            torch.ones_like(runtime._scalers[0].spy_scale),
        )
        assert len(inputs) == len(SEEDS)
        assert all(
            tuple(item.shape for item in values) ==
            ((len(FORWARD_UNIVERSE), 17, 5),) * 2
            for values in inputs
        )
        sample = (SampleRows(16, 17, 29, 16),)
        expected_stock = torch.stack(tuple(
            ForwardFeatureWindows(
                array("f", item.values), sample, 17, "ohlcv",
                scale.stock_mean, scale.stock_scale,
            )[0]
            for item, scale in zip(
                current.stocks, runtime._scalers, strict=True,
            )
        ))
        expected_spy = torch.stack(tuple(
            ForwardFeatureWindows(
                array("f", current.spy.values), sample, 17, "ohlcv",
                scale.spy_mean, scale.spy_scale,
            )[0]
            for scale in runtime._scalers
        ))
        assert all(
            torch.equal(stock, expected_stock) and
            torch.equal(spy, expected_spy)
            for stock, spy in inputs
        )
        assert all(
            tuple(item.seed for item in prediction.values) == SEEDS and
            tuple(
                item.state_fingerprint for item in prediction.values
            ) == fingerprints
            for prediction in predictions
        )
        assert all(
            tuple(item.prediction for item in prediction.values) ==
            (float(
                torch.tensor(1.25) * scale.target_scale +
                scale.target_mean
            ),) * len(SEEDS)
            for prediction, scale in zip(
                predictions, runtime._scalers, strict=True,
            )
        )


def test_forward_inference_reauthenticates_state() -> None:
    with reproduced() as (runtime, _, _):
        with torch.no_grad():
            next(runtime._states[0].model.parameters()).add_(1.0)
        rejects(runtime.predict, batch())
    with reproduced() as (runtime, _, _):
        runtime._states[0].model.position.add_(1.0)
        rejects(runtime.predict, batch())
    with reproduced() as (runtime, _, _):
        model = runtime._states[0].model
        model.num_heads, model.head_dim = 1, model.config.model_dim
        rejects(runtime.predict, batch())
    with reproduced() as (runtime, _, _):
        runtime._states[0].model.forward = lambda *_args: torch.zeros(11)
        rejects(runtime.predict, batch())
    with reproduced() as (runtime, _, _):
        runtime._scalers[0].stock_mean.add_(1.0)
        rejects(runtime.predict, batch())
    with reproduced() as (runtime, _, _):
        runtime._scalers = (
            runtime._scalers[1], runtime._scalers[0],
            *runtime._scalers[2:],
        )
        rejects(runtime.predict, batch())


def test_candidate_is_one_runtime_bound_publication() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-runner-", dir=ROOT,
    ) as directory:
        value = fixture(Path(directory))
        with frozen_files(value) as (stocks, spy, verify):
            calls = 0

            def predict(
                current: SpyResidualForwardInputs,
            ) -> tuple[ForwardSeriesPrediction, ...]:
                nonlocal calls
                calls += 1
                return tuple(
                    ForwardSeriesPrediction(
                        series,
                        tuple(
                            SeedPrediction(
                                seed, fingerprint,
                                (current.index + 1) *
                                (member + 1) *
                                (seed_index + 1) / 1_000_000,
                            )
                            for seed_index, (seed, fingerprint) in enumerate(
                                zip(
                                    SEEDS, STATE_FINGERPRINTS,
                                    strict=True,
                                )
                            )
                        ),
                    )
                    for member, series in enumerate(FORWARD_UNIVERSE)
                )

            binding = ForwardRunBinding(
                predict,
                lambda _grid, rows: {
                    "prediction_rows_sha256": digest(tuple(rows)),
                    "run": {
                        "directory_identity": [7, 11],
                        "id": "test-forward-run",
                    },
                },
            )
            session, read_truth = prepare(
                value, stocks, spy, verify, binding,
            )
            result: SpyResidualForwardInputs | CandidateLedger = \
                session.current()
            while isinstance(result, SpyResidualForwardInputs):
                result = session.submit(result)

            assert calls == 780
            assert result.records == 780 * len(FORWARD_UNIVERSE)
            assert not value.receipt.exists()
            header, first = (
                json.loads(line)
                for line in value.ledger.read_text(
                    encoding="utf-8",
                ).splitlines()[:2]
            )
            assert header["type"] == "spy-residual-forward-candidate"
            assert header["evidence_role"] == "label-free-forward-candidate"
            assert "unbound-task-3-draft" not in value.ledger.read_text(
                encoding="utf-8",
            )
            assert [
                (item["seed"], item["state_fingerprint"])
                for item in first["raw_predictions"]
            ] == list(zip(SEEDS, STATE_FINGERPRINTS, strict=True))
            rejects(session.current)
            assert callable(read_truth)


def test_runner_stops_before_truth() -> None:
    events: list[str] = []
    current = batch()
    context = object()
    candidate = CandidateLedger(
        ROOT / "candidate.jsonl", "a" * 64, (7, 11), (7, 13), 11,
    )

    class Lease:
        def _prepare(self, value: object) -> object:
            assert value is context
            events.append("prepare")
            session = ForwardPredictionSession(
                lambda: events.append("current") or current,
                lambda value: events.append("submit") or (
                    candidate if value is current else None
                ),
            )

            def truth(_candidate: object) -> object:
                raise AssertionError("Task 4 opened truth")

            return session, truth

    result, truth = publish_forward_candidate(
        Lease(), context,  # type: ignore[arg-type]
    )
    assert result is candidate
    assert events == ["prepare", "current", "submit"]
    assert callable(truth)


def test_provenance_is_fixed_data_not_executable_code() -> None:
    prepared = prepared_phase("calibration", FORWARD_UNIVERSE)
    tree = selected_source_tree(ROOT, FORWARD_SOURCE_PATHS)
    context = ForwardRunContext(tree, FORWARD_RUN_ID, (7, 11))
    report = FileBinding("/tmp/fetch.json", "a" * 64)
    value = _provenance(
        context, ForwardCalibration(prepared, (), RUNTIME), report,
    )

    assert value["implementation_tree"]["sha256"] == tree.sha256
    assert value["protocol"] == {
        "path": FORWARD_CONFIG.path,
        "sha256": FORWARD_CONFIG.sha256,
    }
    assert value["run"] == {
        "directory_identity": [7, 11],
        "id": FORWARD_RUN_ID,
    }
    assert value["sources"]["future_bundle_report"] == {
        "path": report.path,
        "sha256": report.sha256,
    }
    json.dumps(value, allow_nan=False, sort_keys=True)


def main() -> None:
    test_reproduces_full_schedule_and_vectorizes_later_inference()
    test_forward_inference_reauthenticates_state()
    test_candidate_is_one_runtime_bound_publication()
    test_runner_stops_before_truth()
    test_provenance_is_fixed_data_not_executable_code()
    print("SPY residual forward runner tests passed")


if __name__ == "__main__":
    main()
