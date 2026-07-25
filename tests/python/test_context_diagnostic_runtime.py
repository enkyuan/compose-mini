#!/usr/bin/env python3
"""Verify causal fitting and prediction for the frozen context family."""

from __future__ import annotations

from array import array
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import patch
import hashlib
import math
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch
except ModuleNotFoundError as error:
    raise SystemExit("context runtime tests require PyTorch") from error

from tools.context_diagnostic_contract import (
    BATCH_SIZE, CONTROL_COHORT, MAX_HISTORY, PHASE_RANGES, SEEDS,
    ContextAttempt, ContextFit, ContextPhase, ContextPrediction,
    ContextPredictionEvidence, ContextScalerInput,
    context_scaler_inputs_sha256, expected_context_fits,
    expected_context_predictions,
)
from tools.arm_context_diagnostic import ContextLease, ContextSnapshots
from tools.context_diagnostic_controller import (
    _phase_rows as derive_phase_rows, prepare_context_phase,
)
from tools.context_diagnostic_inputs import (
    context_grid_sha256, timestamp_rows,
)
from tools.files import FrozenInput
from tools.context_diagnostic_runtime import ContextRuntime
from tools.session_samples import SampleRows
from tools.train import (
    TrainingData, Windows, feature_lookback, tail_training_data,
)
from tools.universe_contract import PackedRows
from tools.universe_forward_runner import ForwardFeatureWindows
from tools.universe_scaling_contract import (
    FitJob, fit_provenance_id, timestamp_grid_sha256,
)

MASTER = tuple(f"S{index:02d}" for index in range(55))
RUNTIME = hashlib.sha256(b"runtime").hexdigest()
CONTROLLER_TIMESTAMPS = tuple(
    f"2026-01-05T14:{30 + index:02d}:00Z" for index in range(8)
)
CONTROLLER_TRAIN = (
    SampleRows(0, 1, 2, 0),
    SampleRows(1, 2, 3, 1),
)
CONTROLLER_EVALUATION = (
    SampleRows(4, 5, 6, 2),
    SampleRows(5, 6, 7, 3),
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def raises(function: object, *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except (TypeError, ValueError):
        return
    raise AssertionError("expected context runtime failure")


def phase_for(master: tuple[str, ...] = MASTER) -> ContextPhase:
    training = [
        {"count": 2, "series": series}
        for series in master[:44]
    ]
    return ContextPhase.parse({
        "evaluation_grid_sha256": digest("evaluation"),
        "evaluation_rows": [
            {
                "count": 2,
                "grid_sha256": digest(series),
                "series": series,
            }
            for series in master[44:]
        ],
        "phase": "fold-1",
        "prior_selections": [
            {
                "model": model,
                "seed": seed,
                "selected_checkpoint": 1,
                "source_model_fingerprint": digest(
                    f"{model}-{seed}-state",
                ),
                "source_provenance_id": fit_provenance_id(FitJob(
                    "pooled", "fixed-update", 44, "fold-0", model, seed,
                    master[:44],
                )),
            }
            for model in ("global_mlp", "panel_transformer")
            for seed in SEEDS
        ],
        "source_ranges": list(map(list, PHASE_RANGES["fold-1"])),
        "scaler_inputs_sha256": digest("scalers"),
        "training_grid_sha256": digest("training"),
        "training_rows": training,
        "updates_per_checkpoint": (
            sum(row["count"] for row in training[:CONTROL_COHORT]) +
            BATCH_SIZE - 1
        ) // BATCH_SIZE,
    }, master)


def controller_fixture() -> tuple[
    ContextAttempt, ContextPhase, ContextLease,
    dict[str, PackedRows],
]:
    training_grid = timestamp_rows(
        CONTROLLER_TIMESTAMPS, CONTROLLER_TRAIN,
    )
    evaluation_grid = timestamp_rows(
        CONTROLLER_TIMESTAMPS, CONTROLLER_EVALUATION,
    )
    packed = {
        series: PackedRows(
            CONTROLLER_TRAIN + (
                CONTROLLER_EVALUATION if series in MASTER[44:] else ()
            ),
            (2, 2) if series in MASTER[44:] else (2, 0),
        )
        for series in MASTER
    }
    csv = tuple(
        (
            series,
            FrozenInput(
                Path(f"{series}.csv"), Path(f"{series}.snapshot"),
                digest(f"{series}-csv"),
            ),
        )
        for series in MASTER
    )
    phase = replace(
        phase_for(),
        evaluation_rows=tuple(
            (
                series, len(CONTROLLER_EVALUATION),
                timestamp_grid_sha256(evaluation_grid),
            )
            for series in MASTER[44:]
        ),
        training_grid_sha256=context_grid_sha256(
            "training", MASTER[:44],
            {series: training_grid for series in MASTER[:44]},
        ),
        evaluation_grid_sha256=context_grid_sha256(
            "evaluation", MASTER[44:],
            {series: evaluation_grid for series in MASTER[44:]},
        ),
        scaler_inputs_sha256=context_scaler_inputs_sha256(
            MASTER, tuple(
                ContextScalerInput(
                    series, frozen.sha256, len(CONTROLLER_TRAIN),
                    timestamp_grid_sha256(training_grid),
                )
                for series, frozen in csv
            ),
        ),
    )
    snapshots = ContextSnapshots(
        FrozenInput(Path("config"), Path("config.snapshot"), digest("config")),
        FrozenInput(
            Path("manifest"), Path("manifest.snapshot"), digest("manifest"),
        ),
        FrozenInput(
            Path("calendar"), Path("calendar.snapshot"), digest("calendar"),
        ),
        csv,
    )
    attempt = ContextAttempt(
        "experiments/context-attempt.json", "context", "reports/context",
        "0" * 40, (), None, (phase,), None, None, (), None, {},
    )
    return attempt, phase, ContextLease(snapshots, lambda: None), packed


def training_data(index: int) -> TrainingData:
    samples = tuple(
        SampleRows(67 + row, 68 + row, 81 + row, 67 + row)
        for row in range(2)
    )
    features = torch.arange(
        69 * 5, dtype=torch.float32,
    ).view(69, 5).add_(index)
    targets = torch.tensor((-0.5, 0.5), dtype=torch.float32)
    references = torch.tensor((100.0, 101.0), dtype=torch.float32)
    outcomes = torch.tensor((101.0, 100.0), dtype=torch.float32)
    starts = (0, 1)

    def windows(start: int, count: int) -> Windows:
        return Windows(
            features, targets, references, outcomes, 68, start, count,
            feature_starts=starts, sample_rows=samples,
        )

    return TrainingData(
        windows(0, 2), windows(2, 0), windows(2, 0),
        torch.full((5,), float(index)),
        torch.full((5,), 2.0),
        torch.tensor(index / 10.0),
        torch.tensor(2.0 + index / 100.0),
        "ohlcv", 13, "executable-return-v1",
    )


def forward_data(
    index: int, data: TrainingData, feature_set: str = "ohlcv",
) -> ForwardFeatureWindows:
    lookback = feature_lookback(feature_set)
    rows = array("f")
    for row in range(69 + lookback):
        open_ = 100.0 + index + row
        rows.extend((
            open_, open_ + 1.0, open_ - 1.0, open_ + 0.25,
            1_000.0 + row,
        ))
    samples = (
        SampleRows(67 + lookback, 68 + lookback, 80 + lookback, 67),
        SampleRows(68 + lookback, 69 + lookback, 81 + lookback, 68),
    )
    return ForwardFeatureWindows(
        rows, samples, 68, feature_set,
        data.feature_mean, data.feature_scale,
    )


def inputs(
    master: tuple[str, ...] = MASTER,
) -> tuple[dict[str, TrainingData], dict[str, ForwardFeatureWindows]]:
    data = {
        series: training_data(index)
        for index, series in enumerate(master)
    }
    forward = {
        series: forward_data(index, data[series])
        for index, series in enumerate(master[44:], 44)
    }
    return data, forward


class Probe(torch.nn.Module):
    """Expose one state value and map each window's final first feature."""

    def __init__(self, state: float = 1.0) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(state))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, -1, 0] * self.weight


@dataclass
class Calls:
    ridges: list[tuple[tuple[TrainingData, ...], float, Probe]]
    neural: list[tuple[tuple[object, ...], Probe]]
    loaders: list[
        tuple[tuple[TrainingData, ...], int, int, int, bool, object]
    ]
    updates: list[tuple[Probe, object, int, float, float, torch.device]]


@contextmanager
def fake_training(state: float = 1.0) -> Iterator[Calls]:
    calls = Calls([], [], [], [])

    def ridge(
        members: tuple[TrainingData, ...], penalty: float,
    ) -> Probe:
        model = Probe(state)
        calls.ridges.append((tuple(members), penalty, model))
        return model

    def neural(*args: object) -> Probe:
        model = Probe(state)
        calls.neural.append((args, model))
        return model

    def loader(
        members: tuple[TrainingData, ...], batch_size: int, samples: int,
        seed: int, *, drop_last: bool = False,
    ) -> object:
        result = object()
        calls.loaders.append((
            tuple(members), batch_size, samples, seed, drop_last, result,
        ))
        return result

    def train(
        model: Probe, batches: object, updates: int,
        learning_rate: float, weight_decay: float, device: torch.device,
    ) -> float:
        calls.updates.append((
            model, batches, updates, learning_rate, weight_decay, device,
        ))
        return updates / 10.0

    with patch(
        "tools.context_diagnostic_runtime.stock_macro_linear_model",
        side_effect=ridge,
    ), patch(
        "tools.context_diagnostic_runtime._neural_model",
        side_effect=neural,
    ), patch(
        "tools.context_diagnostic_runtime._stock_uniform_loader",
        side_effect=loader,
    ), patch(
        "tools.context_diagnostic_runtime.fit_training_updates",
        side_effect=train,
    ):
        yield calls


def fit_all(
    runtime: ContextRuntime, master: tuple[str, ...], phase: ContextPhase,
) -> tuple[
    dict[ContextFit, str], dict[ContextFit, object],
]:
    fingerprints, tokens = {}, {}
    for fit in expected_context_fits(master, phase):
        fingerprint, loss, token = runtime.fit_one(fit)
        assert len(fingerprint) == 64 and math.isfinite(loss)
        fingerprints[fit], tokens[fit] = fingerprint, token
    return fingerprints, tokens


def runtime_for(
    master: tuple[str, ...] = MASTER, *,
    data: dict[str, TrainingData] | None = None,
    forward: dict[str, ForwardFeatureWindows] | None = None,
    runtime_sha256: str = RUNTIME,
) -> tuple[
    ContextRuntime, ContextPhase, dict[str, TrainingData],
    dict[str, ForwardFeatureWindows],
]:
    prepared, predictions = inputs(master)
    prepared = prepared if data is None else data
    predictions = predictions if forward is None else forward
    phase = phase_for(master)
    return (
        ContextRuntime(
            master, phase, prepared, predictions,
            torch.device("cpu"), runtime_sha256,
        ),
        phase, prepared, predictions,
    )


def test_input_closure() -> None:
    data, forward = inputs()
    phase = phase_for()

    def construct(
        training: dict[str, TrainingData],
        prediction: dict[str, ForwardFeatureWindows],
    ) -> ContextRuntime:
        return ContextRuntime(
            MASTER, phase, training, prediction,
            torch.device("cpu"), RUNTIME,
        )

    invalid_data = (
        dict(tuple(data.items())[1:]),
        data | {"EXTRA": data[MASTER[0]]},
        {
            series: data[series]
            for series in (MASTER[1], MASTER[0], *MASTER[2:])
        },
        data | {MASTER[0]: tail_training_data(data[MASTER[0]], 34)},
        data | {MASTER[0]: replace(
            data[MASTER[0]], validation=data[MASTER[0]].train,
        )},
        data | {MASTER[0]: replace(
            data[MASTER[0]],
            train=Windows(
                data[MASTER[0]].train.features,
                data[MASTER[0]].train.targets,
                data[MASTER[0]].train.references,
                data[MASTER[0]].train.outcomes,
                68, 0, 1, feature_starts=(0, 1),
                sample_rows=data[MASTER[0]].train.sample_rows,
            ),
        )},
    )
    for changed in invalid_data:
        raises(construct, changed, forward)

    reordered = dict(reversed(tuple(forward.items())))
    shortened = ForwardFeatureWindows.__new__(ForwardFeatureWindows)
    shortened.features = forward[MASTER[44]].features
    shortened.starts = forward[MASTER[44]].starts
    shortened.seq_len = 34
    invalid_forward = (
        dict(tuple(forward.items())[1:]),
        forward | {"EXTRA": forward[MASTER[44]]},
        reordered,
        forward | {MASTER[44]: shortened},
    )
    for changed in invalid_forward:
        raises(construct, data, changed)
    raises(
        ContextRuntime, MASTER, phase, data, forward,
        torch.device("mps"), RUNTIME,
    )


def test_common_views_and_fit_budgets() -> None:
    runtime, phase, data, _ = runtime_for()
    fits = expected_context_fits(MASTER, phase)
    with fake_training() as calls:
        fit_all(runtime, MASTER, phase)

    neural_fits = tuple(fit for fit in fits if fit.seed is not None)
    assert len(calls.ridges) == 3
    assert len(calls.neural) == len(calls.loaders) == \
        len(calls.updates) == len(neural_fits)
    assert len({id(model) for _, model in calls.neural}) == len(neural_fits)
    for fit, loader_call, update_call in zip(
        neural_fits, calls.loaders, calls.updates, strict=True,
    ):
        members, batch, samples, seed, drop_last, loader = loader_call
        model, batches, updates, learning_rate, weight_decay, device = \
            update_call
        assert batch == BATCH_SIZE
        assert samples == fit.optimizer_updates * BATCH_SIZE
        assert seed == fit.seed and drop_last
        assert batches is loader and updates == fit.optimizer_updates
        assert model in tuple(item[1] for item in calls.neural)
        assert learning_rate == 0.0003 and weight_decay == 0.0001
        assert device == torch.device("cpu")
        for series, view in zip(fit.members, members, strict=True):
            source = data[series]
            assert view.train.features is source.train.features
            assert view.train.targets is source.train.targets
            assert view.train.references is source.train.references
            assert view.train.outcomes is source.train.outcomes
            assert view.train.sample_rows is source.train.sample_rows
            for name in (
                "feature_mean", "feature_scale",
                "target_mean", "target_scale",
            ):
                assert getattr(view, name) is getattr(source, name)
            assert view.train.seq_len == fit.history
            torch.testing.assert_close(
                view.train[0][0], source.train[0][0][-fit.history:],
            )

    for fit, (members, penalty, _model) in zip(
        (fit for fit in fits if fit.model == "global_ridge"),
        calls.ridges, strict=True,
    ):
        assert len(members) == len(fit.members) == 44
        assert all(
            (member is data[name]) == (fit.history == 68)
            for name, member in zip(fit.members, members, strict=True)
        )
        assert tuple(member.train.seq_len for member in members) == \
            (fit.history,) * 44
        assert penalty == 0.001


def test_label_free_ordered_predictions() -> None:
    runtime, phase, data, forward = runtime_for()
    with fake_training():
        _, tokens = fit_all(runtime, MASTER, phase)
        predictions = expected_context_predictions(MASTER, phase)
        first = predictions[0]
        raises(runtime.predict_one, predictions[1], tokens[first.fit])
        raises(
            runtime.predict_one,
            replace(first, prediction_count=first.prediction_count + 1),
            tokens[first.fit],
        )
        raises(runtime.predict_one, first, tokens[predictions[11].fit])

        for series in MASTER[44:]:
            source = data[series].train
            source.targets.fill_(10_000.0)
            source.references.fill_(20_000.0)
            source.outcomes.fill_(30_000.0)

        for prediction in predictions:
            actual = runtime.predict_one(
                prediction, tokens[prediction.fit],
            )
            source = forward[prediction.series]
            scaler = data[prediction.series]
            expected = tuple(
                float(
                    source[index][-1, 0] * scaler.target_scale +
                    scaler.target_mean
                )
                for index in range(len(source))
            )
            assert tuple(actual) == expected


def test_post_fit_mutation_is_rejected() -> None:
    runtime, phase, _, _ = runtime_for()
    with fake_training() as calls:
        _, tokens = fit_all(runtime, MASTER, phase)
        predictions = expected_context_predictions(MASTER, phase)
        first_group = predictions[:11]
        runtime.predict_one(first_group[0], tokens[first_group[0].fit])
        with torch.no_grad():
            calls.ridges[0][2].weight.add_(1.0)
        raises(
            runtime.predict_one, first_group[1],
            tokens[first_group[1].fit],
        )

    runtime, phase, data, _ = runtime_for()
    with fake_training():
        _, tokens = fit_all(runtime, MASTER, phase)
        first = expected_context_predictions(MASTER, phase)[0]
        data[MASTER[0]].target_scale.add_(1.0)
        raises(runtime.predict_one, first, tokens[first.fit])

    runtime, phase, data, _ = runtime_for()
    with fake_training():
        _, tokens = fit_all(runtime, MASTER, phase)
        first = expected_context_predictions(MASTER, phase)[0]
        data[first.series].target_scale.add_(1.0)
        raises(runtime.predict_one, first, tokens[first.fit])

    runtime, phase, _, forward = runtime_for()
    with fake_training():
        _, tokens = fit_all(runtime, MASTER, phase)
        first = expected_context_predictions(MASTER, phase)[0]
        forward[first.series].features.add_(1.0)
        raises(runtime.predict_one, first, tokens[first.fit])


def test_forward_scaler_provenance_is_required() -> None:
    data, forward = inputs()
    phase = phase_for()
    wrong = forward_data(44, data[MASTER[45]])
    stationary = forward_data(
        44, data[MASTER[44]], "stationary-v1",
    )
    raises(
        ContextRuntime, MASTER, phase,
        data, forward | {MASTER[44]: wrong},
        torch.device("cpu"), RUNTIME,
    )
    raises(
        ContextRuntime, MASTER, phase,
        data, forward | {MASTER[44]: stationary},
        torch.device("cpu"), RUNTIME,
    )


def first_fingerprint(
    master: tuple[str, ...], data: dict[str, TrainingData],
    forward: dict[str, ForwardFeatureWindows], runtime_sha256: str,
    state: float,
) -> str:
    runtime = ContextRuntime(
        master, phase_for(master), data, forward,
        torch.device("cpu"), runtime_sha256,
    )
    with fake_training(state):
        return runtime.fit_one(
            expected_context_fits(master, phase_for(master))[0],
        )[0]


def test_fingerprint_binds_context() -> None:
    data, forward = inputs()
    phase = phase_for()
    runtime = ContextRuntime(
        MASTER, phase, data, forward, torch.device("cpu"), RUNTIME,
    )
    with fake_training():
        fingerprints = {}
        for fit in expected_context_fits(MASTER, phase):
            fingerprint, _, _ = runtime.fit_one(fit)
            if fit.model == "global_ridge":
                fingerprints[fit.history] = fingerprint
            if len(fingerprints) == 2:
                break
    assert fingerprints[17] != fingerprints[34]

    baseline = first_fingerprint(MASTER, data, forward, RUNTIME, 1.0)
    alternate = tuple(f"T{index:02d}" for index in range(55))
    alternate_data = dict(zip(alternate, data.values(), strict=True))
    alternate_forward = dict(zip(
        alternate[44:], forward.values(), strict=True,
    ))
    changed_scaler = data | {MASTER[0]: replace(
        data[MASTER[0]], target_mean=torch.tensor(99.0),
    )}
    values = (
        first_fingerprint(
            alternate, alternate_data, alternate_forward, RUNTIME, 1.0,
        ),
        first_fingerprint(MASTER, changed_scaler, forward, RUNTIME, 1.0),
        first_fingerprint(MASTER, data, forward, RUNTIME, 2.0),
        first_fingerprint(MASTER, data, forward, digest("other"), 1.0),
    )
    assert len({baseline, *values}) == len(values) + 1


def test_controller_binds_ordered_phase_inputs() -> None:
    attempt, phase, lease, packed = controller_fixture()
    seen: list[str] = []
    snapshots = {
        frozen.snapshot: series for series, frozen in lease.snapshots.csv
    }

    def read(path: Path, stop: str) -> tuple[str, ...]:
        assert stop == CONTROLLER_TIMESTAMPS[-1]
        seen.append(snapshots[path])
        return CONTROLLER_TIMESTAMPS

    def pack(*args: object) -> PackedRows:
        series = MASTER[len(seen) - 1]
        assert args[0] == CONTROLLER_TIMESTAMPS
        return packed[series]

    sweep = SimpleNamespace(
        target_horizon_bars=13, alignment_horizon_bars=13,
    )
    manifest = SimpleNamespace(
        start=object(), end=object(), interval_minutes=30,
    )
    with patch(
        "tools.context_diagnostic_controller.context_cutoff_timestamp",
        return_value=CONTROLLER_TIMESTAMPS[-1],
    ), patch(
        "tools.context_diagnostic_controller.read_timestamps_until",
        side_effect=read,
    ), patch(
        "tools.context_diagnostic_controller.context_phase_rows",
        side_effect=pack,
    ):
        timestamps, actual = derive_phase_rows(
            attempt, phase, lease, sweep, manifest, object(),
        )

    assert tuple(timestamps) == tuple(actual) == MASTER
    assert seen == list(MASTER) and actual == packed
    with patch(
        "tools.context_diagnostic_controller.context_cutoff_timestamp",
        return_value=CONTROLLER_TIMESTAMPS[-1],
    ), patch(
        "tools.context_diagnostic_controller.read_timestamps_until",
        return_value=CONTROLLER_TIMESTAMPS,
    ), patch(
        "tools.context_diagnostic_controller.context_phase_rows",
        side_effect=packed.values(),
    ):
        raises(
            derive_phase_rows, attempt,
            replace(phase, evaluation_grid_sha256=digest("changed")),
            lease, sweep, manifest, object(),
        )


def test_controller_defers_truth_and_bounds_market_reads() -> None:
    attempt, phase, lease, packed = controller_fixture()
    reads: list[tuple[str, str]] = []
    prepared: list[PackedRows] = []
    forwarded: list[tuple[SampleRows, ...]] = []
    verified: list[None] = []
    snapshots = {
        frozen.snapshot: series for series, frozen in lease.snapshots.csv
    }
    lease = ContextLease(
        lease.snapshots, lambda: verified.append(None),
    )
    bars = array("f", (
        value
        for index in range(len(CONTROLLER_TIMESTAMPS))
        for value in (
            100.0 + index, 101.0 + index, 99.0 + index,
            100.25 + index, 1_000.0 + index,
        )
    ))
    data = SimpleNamespace(
        feature_mean=torch.zeros(5), feature_scale=torch.ones(5),
    )

    def read(
        snapshot: Path, timestamps: tuple[str, ...], stop: str,
    ) -> array:
        assert timestamps == CONTROLLER_TIMESTAMPS
        reads.append((snapshots[snapshot], stop))
        return bars

    def prepare(
        rows: array, _candidate: object, samples: PackedRows,
        history: int, _sweep: object,
    ) -> object:
        assert rows is bars and history == MAX_HISTORY
        assert samples == PackedRows(CONTROLLER_TRAIN, (2, 0))
        prepared.append(samples)
        return data

    def forward(
        rows: array, samples: tuple[SampleRows, ...], history: int,
        feature_set: str, mean: torch.Tensor, scale: torch.Tensor,
    ) -> object:
        assert rows is bars and history == MAX_HISTORY
        assert feature_set == "ohlcv"
        assert mean is data.feature_mean and scale is data.feature_scale
        forwarded.append(samples)
        return samples

    runtime = SimpleNamespace(
        fit_one=lambda _fit: None, predict_one=lambda _prediction, _fit: (),
    )
    sweep = SimpleNamespace(
        candidates=(SimpleNamespace(
            seq_len=MAX_HISTORY, feature_set="ohlcv",
        ),),
        target_kind="executable-return-v1",
    )
    with patch(
        "tools.context_diagnostic_controller.read_canonical_json",
        return_value={},
    ), patch(
        "tools.context_diagnostic_controller.validate_context_sweep",
    ), patch(
        "tools.context_diagnostic_controller.Sweep.read",
        return_value=sweep,
    ), patch(
        "tools.context_diagnostic_controller.UniverseManifest.read",
        return_value=object(),
    ), patch(
        "tools.context_diagnostic_controller.SessionCalendar.read",
        return_value=object(),
    ), patch(
        "tools.context_diagnostic_controller._phase_rows",
        return_value=(
            {series: CONTROLLER_TIMESTAMPS for series in MASTER}, packed,
        ),
    ), patch(
        "tools.context_diagnostic_controller.context_bar_prefix",
        side_effect=read,
    ), patch(
        "tools.context_diagnostic_controller.context_csv_prefix_sha256",
        side_effect=lambda _path, _stop: digest(
            f"{reads[-1][0]}-csv",
        ),
    ), patch(
        "tools.context_diagnostic_controller._prepare_packed",
        side_effect=prepare,
    ), patch(
        "tools.context_diagnostic_controller.ForwardFeatureWindows",
        side_effect=forward,
    ), patch(
        "tools.context_diagnostic_controller.ContextRuntime",
        return_value=runtime,
    ), patch(
        "tools.context_diagnostic_controller._runtime_sha256",
        return_value=RUNTIME,
    ), patch(
        "tools.context_diagnostic_controller.torch.use_deterministic_algorithms",
    ):
        fit_one, predict_one, truth = prepare_context_phase(
            attempt, phase, lease,
        )
        assert fit_one is runtime.fit_one and predict_one is runtime.predict_one
        expected_reads = [
            item
            for series in MASTER
            for item in (
                (
                    (series, CONTROLLER_TIMESTAMPS[3]),
                    (series, CONTROLLER_TIMESTAMPS[5]),
                )
                if series in MASTER[44:] else
                ((series, CONTROLLER_TIMESTAMPS[3]),)
            )
        ]
        assert reads == expected_reads, reads
        assert prepared == [
            PackedRows(CONTROLLER_TRAIN, (2, 0))
        ] * len(MASTER)
        assert forwarded == [
            CONTROLLER_EVALUATION
        ] * len(MASTER[44:])

        evidence = tuple(
            ContextPredictionEvidence(
                prediction, digest("provenance"), digest("state"), (0.0, 0.0),
            )
            for prediction in expected_context_predictions(MASTER, phase)
        )
        with patch(
            "tools.finalize_context_diagnostic.evaluate_context_phase",
            return_value={"status": "evaluated"},
        ) as evaluate:
            result = truth(evidence)

    assert result == {"status": "evaluated"}
    assert reads[-len(MASTER[44:]):] == [
        (series, CONTROLLER_TIMESTAMPS[7]) for series in MASTER[44:]
    ]
    assert len(verified) == 4
    master, received_phase, received, values = evaluate.call_args.args
    assert master == MASTER and received_phase == phase
    assert received == evidence and tuple(values) == MASTER[44:]
    for rows in values.values():
        assert tuple(
            (row.as_of, row.entry_time, row.target_time)
            for row in rows
        ) == timestamp_rows(
            CONTROLLER_TIMESTAMPS, CONTROLLER_EVALUATION,
        )
        assert tuple(row.reference_price for row in rows) == (105.0, 106.0)
        assert tuple(row.outcome_price for row in rows) == (106.25, 107.25)


def main() -> None:
    test_input_closure()
    test_common_views_and_fit_budgets()
    test_label_free_ordered_predictions()
    test_post_fit_mutation_is_rejected()
    test_forward_scaler_provenance_is_required()
    test_fingerprint_binds_context()
    test_controller_binds_ordered_phase_inputs()
    test_controller_defers_truth_and_bounds_market_reads()
    print("context diagnostic runtime tests passed")


if __name__ == "__main__":
    main()
