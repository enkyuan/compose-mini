#!/usr/bin/env python3
"""Validate and finalize one immutable universe-scaling development attempt."""

from __future__ import annotations

import sys

_BOOTSTRAP_PYTHON_FLAGS = ("-I", "-S", "-B")
_BOOTSTRAP_CACHE_PREFIX: str | None = None


def _require_isolated_execution(*, bootstrapped: bool = False) -> None:
    flags = sys.flags
    if not flags.isolated or not getattr(flags, "safe_path", False) or \
            not flags.no_user_site or not flags.no_site or \
            not flags.dont_write_bytecode or \
            not flags.ignore_environment or not sys.dont_write_bytecode:
        raise ValueError(
            "finalizer requires isolated no-site bytecode-free "
            "Python execution"
        )
    if bootstrapped and (
        _BOOTSTRAP_CACHE_PREFIX is None or
        sys.pycache_prefix != _BOOTSTRAP_CACHE_PREFIX
    ):
        raise ValueError("finalizer requires authenticated script bootstrap")


def _require_exact_launch(*, pristine: bool = False) -> None:
    if pristine and (
        "ctypes" in sys.modules or "_ctypes" in sys.modules
    ):
        raise ValueError("finalizer launch inspection is already loaded")

    from ctypes import (
        POINTER, byref, c_int, c_wchar_p, pythonapi,
    )
    import os

    argc = c_int()
    argv = POINTER(c_wchar_p)()
    get_argv = pythonapi.Py_GetArgcArgv
    get_argv.argtypes = (
        POINTER(c_int), POINTER(POINTER(c_wchar_p)),
    )
    get_argv.restype = None
    get_argv(byref(argc), byref(argv))
    observed = tuple(argv[index] for index in range(argc.value))
    canonical = lambda values: (
        os.path.realpath(values[0]), *values[1:],
    )
    expected = (
        os.path.realpath(sys.executable),
        *_BOOTSTRAP_PYTHON_FLAGS,
        *sys.argv,
    )
    if not observed or canonical(observed) != expected or \
            canonical(tuple(sys.orig_argv)) != expected or \
            os.path.realpath(sys.argv[0]) != \
            os.path.realpath(__file__):
        raise ValueError("finalizer requires the exact bound Python launch")


def _bootstrap_main() -> None:
    """Authenticate the import namespace before exposing repository code."""
    global _BOOTSTRAP_CACHE_PREFIX

    from importlib.machinery import PathFinder
    import os
    import stat
    import tempfile

    while True:
        prefix = os.path.join(
            tempfile.gettempdir(),
            f"compose-mini-finalizer-{os.urandom(32).hex()}",
        )
        if not os.path.lexists(prefix):
            break
    sys.pycache_prefix = prefix
    sys.dont_write_bytecode = True

    tools = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(tools)
    initializer = os.path.join(tools, "__init__.py")
    if not stat.S_ISDIR(os.lstat(tools).st_mode) or \
            not stat.S_ISREG(os.lstat(initializer).st_mode):
        raise ValueError("tools namespace is not a real package")
    for entry in os.scandir(tools):
        mode = entry.stat(follow_symlinks=False).st_mode
        if entry.name == "__pycache__":
            valid = stat.S_ISDIR(mode)
        else:
            valid = entry.name.endswith(".py") and stat.S_ISREG(mode)
        if not valid:
            raise ValueError("tools namespace contains an unsafe entry")
    if any(
        name == "tools" or name.startswith("tools.")
        for name in sys.modules
    ):
        raise ValueError("tools namespace is already loaded")
    spec = PathFinder.find_spec("tools", (*sys.path, root))
    locations = tuple(
        os.path.realpath(path)
        for path in (spec.submodule_search_locations or ())
    ) if spec is not None else ()
    if spec is None or \
            os.path.realpath(spec.origin or "") != \
            os.path.realpath(initializer) or \
            locations != (os.path.realpath(tools),):
        raise ValueError("tools namespace resolver is unsafe")
    sys.path.append(root)
    import tools as package
    if os.path.realpath(package.__file__ or "") != \
            os.path.realpath(initializer) or tuple(
                map(os.path.realpath, package.__path__)
            ) != locations:
        raise ValueError("tools namespace import is unsafe")
    _BOOTSTRAP_CACHE_PREFIX = prefix


if __name__ == "__main__":
    _require_isolated_execution()
    _require_exact_launch(pristine=True)
    _bootstrap_main()

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date, datetime
from math import isfinite, log
from pathlib import Path
from statistics import fmean
import argparse
import json
import os
import stat

ROOT = Path(__file__).resolve().parents[1]
if sys.flags.isolated and _BOOTSTRAP_CACHE_PREFIX is None:
    sys.path.append(str(ROOT))

from tools.files import (
    ExclusiveTemp, FrozenInput, freeze_inputs, verify_frozen,
    write_json_exclusive,
)
from tools.float32 import decode_f32le_base64
from tools.data_v1 import FEATURE_COUNT, read_bars
from tools.panel_contract import (
    NAME, FileBinding, _directory_identity, _open_directory,
    _regular_identity, _regular_inputs, _tree_digest, _verify_identities,
    iter_canonical_json_lines, read_canonical_json,
    read_canonical_json_lines, selected_source_tree, source_tree,
)
from tools.universe_scaling import (
    ForecastPoint, cohort_views, paired_comparison, stock_macro_metrics,
    unseen_view,
)
from tools.session_calendar import SessionCalendar
from tools.session_samples import session_samples
from tools.universe_contract import common_calendar, pack_rows
from tools.universe_scaling_contract import (
    CSV_ROOT, EXPECTED_BUDGETS, EXPECTED_FIT_COUNT, EXPECTED_MISSING,
    EXPECTED_PREDICTION_RECORDS, MODES, MODELS, PHASES, SEEDS,
    TRAINING_COHORTS, TRANSFER_COHORTS, FIXED_EPOCH_BUDGET, FitJob,
    PhaseCoverage, ScalingAttempt, ScalingCoverage, SeriesCoverage,
    expected_fit_jobs, fit_provenance_id, question_uses,
    required_prediction_series,
    timestamp_grid_sha256,
)

CONTROL_MODELS = (
    "zero", "global_ridge", "global_mlp", "local_transformer",
)
LOCKS = {
    "reserved_test_materialized_samples": 0,
    "policy_selected": False,
    "backtest_run": False,
    "trading_authorized": False,
}
TRANSITIONS = {
    "preflight-failure": ("preflight", lambda code: code != 0),
    "setup-failure": ("setup", lambda code: code != 0),
    "experiment-failure": ("experiment", lambda code: code != 0),
    "analysis-integrity-failure": (
        "analysis", lambda code: code not in (0, 3),
    ),
    "gate-failure": ("analysis", lambda code: code == 3),
    "pass": ("analysis", lambda code: code == 0),
}
PROVENANCE_STATUSES = frozenset(("gate-failure", "pass"))
FIT_FIELDS = {
    "schema", "provenance_id", "kind", "mode", "cohort", "phase",
    "model", "seed", "members", "budget", "coverage",
    "selected_checkpoint", "selected_epoch", "optimizer_updates",
    "epochs_trained", "model_fingerprint", "question_uses",
}
PREDICTION_FIELDS = {
    "schema", "provenance_id", "model_fingerprint", "phase", "series",
    "grid_sha256", "predictions",
}
MANIFEST_FIELDS = {
    "schema", "purpose", "declared_on", "eligibility_date", "start", "end",
    "interval_minutes", "adjusted", "session", "series",
}


@dataclass(frozen=True, slots=True)
class FitClosure:
    records: tuple[Mapping[str, object], ...]
    master: tuple[str, ...]
    evaluable: Mapping[str, tuple[str, ...]]
    rows: Mapping[tuple[str, str], tuple[int, int]]
    timestamp_sha256: Mapping[tuple[str, str], str]
    jobs: tuple[FitJob, ...]
    jobs_by_id: Mapping[str, FitJob]
    fingerprints: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class PredictionClosure:
    metrics: Mapping[tuple[object, ...], Mapping[str, float]]
    calibration: Mapping[tuple[object, ...], tuple[float, ...]]
    truth: MarketTruth
    records: int
    stored_values: int
    synthesized_zero_values: int


@dataclass(frozen=True, slots=True)
class PredictionTruth:
    as_of: str
    entry_time: str
    target_time: str
    reference_price: float
    outcome_price: float
    actual_return: float


@dataclass(frozen=True, slots=True)
class MarketTruth:
    coverage: ScalingCoverage
    rows: Mapping[tuple[str, str], tuple[PredictionTruth, ...]]


@dataclass(frozen=True, slots=True)
class MarketSpec:
    names: tuple[str, ...]
    start: date
    end: date
    interval_minutes: int


@dataclass(frozen=True, slots=True)
class OutputState:
    name: str
    path: Path
    identity: tuple[int, int] | None
    parent_identity: tuple[int, int] | None

    @property
    def present(self) -> bool:
        return self.identity is not None


@dataclass(frozen=True, slots=True)
class SuccessInputs:
    csv_names: tuple[str, ...]
    csv: tuple[FileBinding, ...]
    selection_paths: tuple[Path, ...]
    paths: tuple[Path, ...]


def _object(
    value: object, fields: set[str], label: str,
) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{label} fields are invalid")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is invalid")
    return value


def _integer(value: object, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} is invalid")
    return value


def _sha256(value: object, label: str) -> str:
    text = _string(value, label)
    if len(text) != 64 or any(byte not in "0123456789abcdef" for byte in text):
        raise ValueError(f"{label} is invalid")
    return text


def _finite(value: object, label: str) -> float:
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError(f"{label} is invalid")
    return float(value)


def _timestamp(value: object, label: str) -> str:
    text = _string(value, label)
    try:
        parsed = datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise ValueError(f"{label} must be a canonical UTC timestamp") from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != text:
        raise ValueError(f"{label} must be a canonical UTC timestamp")
    return text


def _master(value: Sequence[str]) -> tuple[str, ...]:
    names = tuple(value)
    if len(names) != 55 or len(set(names)) != 55 or any(
        not isinstance(name, str) or not name for name in names
    ):
        raise ValueError("finalizer requires 55 unique ordered series")
    return names


def _date(value: object, label: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO date") from error
    if str(parsed) != value:
        raise ValueError(f"{label} must be an ISO date")
    return parsed


def _market_spec(path: Path) -> MarketSpec:
    value = _object(
        read_canonical_json(path), MANIFEST_FIELDS, "master manifest",
    )
    raw_series = value["series"]
    if type(value["schema"]) is not int or value["schema"] != 1 or \
            value["session"] != "regular" or \
            type(value["adjusted"]) is not bool or \
            not isinstance(raw_series, list):
        raise ValueError("master manifest fields are invalid")
    names = _master(tuple(
        _string(
            _object(item, {"stratum", "ticker"}, "manifest series")[
                "ticker"
            ],
            "manifest ticker",
        )
        for item in raw_series
    ))
    start = _date(value["start"], "manifest start")
    end = _date(value["end"], "manifest end")
    interval = _integer(
        value["interval_minutes"], "manifest interval", 1,
    )
    if start > end or interval > 59:
        raise ValueError("master manifest bounds are invalid")
    return MarketSpec(names, start, end, interval)


def derive_market_truth(
    manifest_snapshot: Path, calendar_snapshot: Path,
    csv_snapshots: Mapping[str, Path], coverage: ScalingCoverage,
    protocol: Mapping[str, object],
) -> MarketTruth:
    """Derive development labels only from frozen market snapshots."""
    spec = _market_spec(manifest_snapshot)
    if tuple(csv_snapshots) != spec.names:
        raise ValueError("market snapshots do not follow manifest order")
    calendar = SessionCalendar.read(calendar_snapshot)
    history = _integer(protocol["history_bars"], "history bars", 1)
    horizon = _integer(
        protocol["target_horizon_bars"], "target horizon", 1,
    )
    alignment = _integer(
        protocol["alignment_horizon_bars"], "alignment horizon", 1,
    )
    folds = _integer(protocol["folds"], "folds", 1)
    fraction = protocol["fold_fraction"]
    calendar_contract = protocol["calendar"]
    if type(fraction) is not float or not 0 < fraction < 1 or \
            not isinstance(calendar_contract, Mapping):
        raise ValueError("development calendar protocol is invalid")
    opportunities = _integer(
        calendar_contract["opportunities"], "opportunities", 1,
    )
    blocks = common_calendar(
        opportunities, folds, fraction, alignment - 1,
    )
    phase_blocks = (*blocks.folds, blocks.holdout[:2])
    if len(phase_blocks) != len(PHASES) or not phase_blocks[-1]:
        raise ValueError("development phase blocks are invalid")
    development_stop = phase_blocks[-1][-1].stop

    rows: dict[tuple[str, str], tuple[PredictionTruth, ...]] = {}
    by_phase = {phase: [] for phase in PHASES}
    for name, csv_path in csv_snapshots.items():
        timestamps, bars = read_bars(csv_path)
        samples = session_samples(
            timestamps, spec.interval_minutes, calendar,
            spec.start, spec.end, history, horizon, alignment,
            opportunity_stop=development_stop,
        )
        if samples.opportunities != opportunities:
            raise ValueError("market opportunity count changed")
        for phase, ranges in zip(PHASES, phase_blocks, strict=True):
            packed = pack_rows(
                samples.rows, ranges, history, horizon, alignment,
            )
            train, validation = packed.counts
            truth = []
            for row in packed.rows[train:]:
                reference = float(bars[row.entry * FEATURE_COUNT])
                outcome = float(
                    bars[row.target * FEATURE_COUNT + 3],
                )
                if min(reference, outcome) <= 0:
                    raise ValueError("derived market prices must be positive")
                truth.append(PredictionTruth(
                    timestamps[row.as_of], timestamps[row.entry],
                    timestamps[row.target], reference, outcome,
                    log(outcome / reference),
                ))
            if len(truth) != validation:
                raise ValueError("derived validation rows are incomplete")
            result = tuple(truth)
            rows[(phase, name)] = result
            by_phase[phase].append(SeriesCoverage(
                name, train, validation,
                timestamp_grid_sha256(tuple(
                    (item.as_of, item.entry_time, item.target_time)
                    for item in result
                )),
            ))
    derived = ScalingCoverage(tuple(
        PhaseCoverage(phase, tuple(by_phase[phase]))
        for phase in PHASES
    ))
    if derived != coverage:
        raise ValueError("derived market coverage changed")
    return MarketTruth(derived, rows)


def _job_axes(record: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(record[name] for name in (
        "kind", "mode", "cohort", "phase", "model", "seed",
    ))


def _coverage(
    value: object, job: FitJob,
    expected: Mapping[tuple[str, str], tuple[int, int]],
) -> tuple[tuple[str, int, int], ...]:
    if not isinstance(value, list) or len(value) != len(job.members):
        raise ValueError("fit coverage is incomplete")
    result = []
    for index, (raw, name) in enumerate(zip(
        value, job.members, strict=True,
    )):
        item = _object(
            raw, {"series", "train_rows", "validation_rows"},
            f"fit coverage[{index}]",
        )
        if item["series"] != name:
            raise ValueError("fit coverage order is invalid")
        result.append((
            name,
            _integer(item["train_rows"], "fit train rows", 1),
            _integer(item["validation_rows"], "fit validation rows"),
        ))
    parsed = tuple(result)
    required = tuple(
        (name, *expected[(job.phase, name)]) for name in job.members
    )
    if parsed != required:
        raise ValueError("fit coverage does not match the armed attempt")
    return parsed


def _budget(value: object, job: FitJob) -> None:
    expected = (
        asdict(dict(EXPECTED_BUDGETS)[job.phase])
        if job.kind == "pooled" and job.mode == "fixed-update" else
        asdict(FIXED_EPOCH_BUDGET)
        if job.kind == "local" or job.mode == "fixed-epoch" else None
    )
    if value != expected:
        raise ValueError("fit budget is invalid")


def _selection(
    record: Mapping[str, object], job: FitJob,
    coverage: Sequence[tuple[str, int, int]],
) -> None:
    checkpoint = record["selected_checkpoint"]
    epoch = record["selected_epoch"]
    updates = _integer(record["optimizer_updates"], "optimizer updates")
    trained = _integer(record["epochs_trained"], "epochs trained")
    if job.kind == "ridge":
        valid = (
            checkpoint is None and epoch is None and trained == updates == 0
        )
    elif job.kind == "pooled" and job.mode == "fixed-update":
        valid = (
            type(checkpoint) is int and 1 <= checkpoint <=
            dict(EXPECTED_BUDGETS)[job.phase].checkpoints and
            epoch is None and trained == 0 and
            updates == dict(EXPECTED_BUDGETS)[job.phase].total_updates
        )
    else:
        rows_per_epoch = (
            sum(item[1] for item in coverage) +
            FIXED_EPOCH_BUDGET.batch_size - 1
        ) // FIXED_EPOCH_BUDGET.batch_size
        stale = trained - epoch if type(epoch) is int else -1
        valid = (
            checkpoint is None and type(epoch) is int and
            1 <= epoch <= trained <= FIXED_EPOCH_BUDGET.epochs and
            updates == trained * rows_per_epoch and
            (
                stale == FIXED_EPOCH_BUDGET.patience
                if trained < FIXED_EPOCH_BUDGET.epochs else
                stale <= FIXED_EPOCH_BUDGET.patience
            )
        )
    if not valid:
        raise ValueError("fit selection or optimizer count is invalid")


def validate_fit_ledger(
    values: Sequence[Mapping[str, object]], master: Sequence[str],
    coverage: ScalingCoverage,
) -> FitClosure:
    """Validate exact physical-fit closure and canonical fit reuse IDs."""
    records = tuple(values)
    names = _master(master)
    coverage.require_promotable()
    if coverage.master != names:
        raise ValueError("attempt coverage does not match the master manifest")
    if tuple(
        (phase.phase, phase.missing) for phase in coverage.phases
    ) != EXPECTED_MISSING:
        raise ValueError("attempt coverage misses changed")
    if len(records) != EXPECTED_FIT_COUNT:
        raise ValueError("fit ledger does not contain exactly 1,215 jobs")
    for index, raw in enumerate(records):
        record = _object(raw, FIT_FIELDS, f"fit[{index}]")
        if _integer(record["schema"], "fit schema", 1) != 1:
            raise ValueError("fit schema is invalid")

    evaluable = {
        phase.phase: phase.evaluable for phase in coverage.phases
    }
    jobs = expected_fit_jobs(names, evaluable)
    if len(jobs) != EXPECTED_FIT_COUNT or len(set(jobs)) != len(jobs):
        raise ValueError("physical fit schedule changed")
    row_counts = {
        (phase.phase, item.series): (
            item.train_rows, item.validation_rows,
        )
        for phase in coverage.phases for item in phase.series
    }
    timestamps = {
        (phase.phase, item.series): item.timestamp_sha256
        for phase in coverage.phases for item in phase.series
    }
    jobs_by_id: dict[str, FitJob] = {}
    fingerprints = {}
    for index, (record, job) in enumerate(zip(records, jobs, strict=True)):
        expected_axes = (
            job.kind, job.mode, job.cohort, job.phase, job.model, job.seed,
        )
        if _job_axes(record) != expected_axes or \
           record["members"] != list(job.members):
            raise ValueError(f"fit[{index}] is reordered or has invalid axes")
        provenance = fit_provenance_id(job)
        if _sha256(record["provenance_id"], "fit provenance") != provenance:
            raise ValueError("fit provenance identity changed")
        fingerprint = _sha256(
            record["model_fingerprint"], "model fingerprint",
        )
        _budget(record["budget"], job)
        fit_coverage = _coverage(record["coverage"], job, row_counts)
        _selection(record, job, fit_coverage)
        uses = [
            {"question": question, "cohort": cohort}
            for question, cohort in question_uses(job, names)
        ]
        if record["question_uses"] != uses:
            raise ValueError("fit question reuse is invalid")
        if provenance in jobs_by_id:
            raise ValueError("fit provenance identity is duplicated")
        jobs_by_id[provenance] = job
        fingerprints[provenance] = fingerprint
    return FitClosure(
        records, names, evaluable, row_counts, timestamps, jobs, jobs_by_id,
        fingerprints,
    )


def _family(job: FitJob) -> tuple[object, ...]:
    local = job.members[0] if job.kind == "local" else None
    return job.kind, job.mode, job.cohort, job.phase, job.model, local


def _points(
    rows: Sequence[PredictionTruth], values: Sequence[float],
) -> tuple[ForecastPoint, ...]:
    return tuple(
        ForecastPoint(
            row.target_time, row.actual_return, value,
            row.reference_price, row.outcome_price,
        )
        for row, value in zip(rows, values, strict=True)
    )


def _prediction_line_bytes(truth: MarketTruth) -> int:
    """Return the largest byte length of any valid physical ledger row."""
    count = max(map(len, truth.rows.values()), default=0)
    encoded = "A" * (4 * ((4 * count + 2) // 3))
    value = {
        "grid_sha256": "0" * 64,
        "model_fingerprint": "0" * 64,
        "phase": max(PHASES, key=len),
        "predictions": {
            "encoding": "f32le-base64",
            "count": count,
            "base64": encoded,
        },
        "provenance_id": "0" * 64,
        "schema": 2,
        "series": "X" * 64,
    }
    return len((
        json.dumps(value, allow_nan=False, sort_keys=True) + "\n"
    ).encode())


def validate_prediction_ledger(
    values: Iterable[Mapping[str, object]], closure: FitClosure,
    truth: MarketTruth,
) -> PredictionClosure:
    """Stream exact physical prediction records into bounded evidence."""
    expected_truth = set(closure.rows)
    if truth.coverage.master != closure.master or \
            set(truth.rows) != expected_truth or any(
                len(truth.rows[key]) != closure.rows[key][1] or
                timestamp_grid_sha256(tuple(
                    (row.as_of, row.entry_time, row.target_time)
                    for row in truth.rows[key]
                )) != closure.timestamp_sha256[key]
                for key in expected_truth
            ):
        raise ValueError("derived market truth does not match fit coverage")
    records = iter(values)
    metrics: dict[tuple[object, ...], Mapping[str, float]] = {}
    calibration: dict[tuple[object, ...], tuple[float, ...]] = {}
    record_count = stored_values = job_index = 0
    while job_index < len(closure.jobs):
        family = _family(closure.jobs[job_index])
        family_jobs = []
        while job_index < len(closure.jobs) and \
                _family(closure.jobs[job_index]) == family:
            family_jobs.append(closure.jobs[job_index])
            job_index += 1
        destinations = required_prediction_series(
            family_jobs[0], closure.master, closure.evaluable,
        )
        columns = {name: [] for name in destinations}
        for job in family_jobs:
            if required_prediction_series(
                job, closure.master, closure.evaluable,
            ) != destinations:
                raise ValueError("prediction family destinations changed")
            provenance = fit_provenance_id(job)
            for series in destinations:
                try:
                    raw = next(records)
                except StopIteration:
                    raise ValueError(
                        "prediction ledger is missing a physical record"
                    ) from None
                record_count += 1
                label = f"prediction[{record_count - 1}]"
                record = _object(raw, PREDICTION_FIELDS, label)
                if _integer(record["schema"], "prediction schema", 1) != 2:
                    raise ValueError("prediction schema is invalid")
                if _sha256(
                    record["provenance_id"], "prediction provenance",
                ) != provenance:
                    raise ValueError("prediction fit identity changed")
                if _sha256(
                    record["model_fingerprint"],
                    "prediction model fingerprint",
                ) != closure.fingerprints[provenance]:
                    raise ValueError(
                        "prediction model fingerprint is invalid"
                    )
                if record["phase"] != job.phase or \
                        record["series"] != series:
                    raise ValueError(
                        "prediction record is reordered or has invalid axes"
                    )
                if _sha256(
                    record["grid_sha256"], "prediction grid",
                ) != closure.timestamp_sha256[(job.phase, series)]:
                    raise ValueError("prediction timestamp grid changed")
                market = truth.rows[(job.phase, series)]
                predicted = decode_f32le_base64(
                    record["predictions"], expected_count=len(market),
                )
                stored_values += len(predicted)
                columns[series].append(predicted)
        for series, seeds in columns.items():
            ensemble = (
                seeds[0] if len(seeds) == 1 else
                tuple(fmean(row) for row in zip(*seeds, strict=True))
            )
            key = (*family, series)
            points = _points(
                truth.rows[(family[3], series)], ensemble,
            )
            metrics[key] = stock_macro_metrics({series: points})
            if family[3] == "calibration":
                calibration[key] = ensemble
    try:
        next(records)
    except StopIteration:
        pass
    else:
        raise ValueError("prediction ledger contains extra records")
    if record_count != EXPECTED_PREDICTION_RECORDS:
        raise ValueError("physical prediction record count changed")

    synthesized = 0
    for (phase, series), rows in truth.rows.items():
        if not rows:
            continue
        points = _points(rows, (0.0,) * len(rows))
        metrics[("zero", None, None, phase, "zero", None, series)] = \
            stock_macro_metrics({series: points})
        synthesized += len(rows)
    return PredictionClosure(
        metrics, calibration, truth, record_count, stored_values, synthesized,
    )


def _family_key(
    predictions: PredictionClosure, question: str, mode: str, cohort: int,
    phase: str, series: str, model: str,
) -> tuple[object, ...]:
    closure = predictions.truth.coverage
    evaluable = {
        item.phase: item.evaluable for item in closure.phases
    }
    if series not in evaluable[phase]:
        raise ValueError("prediction series is not evaluable")
    if model == "zero":
        return "zero", None, None, phase, model, None, series
    if model == "global_ridge":
        job = FitJob(
            "ridge", None, cohort, phase, model, None,
            closure.master[:cohort],
        )
    elif model == "local_transformer":
        job = FitJob("local", None, None, phase, model, SEEDS[0], (series,))
    else:
        job = FitJob(
            "pooled", mode, cohort, phase, model, SEEDS[0],
            closure.master[:cohort],
        )
    if (question, cohort) not in question_uses(job, closure.master) or \
            series not in required_prediction_series(
                job, closure.master, evaluable,
            ):
        raise ValueError("prediction references an inapplicable fit")
    return (*_family(job), series)


def _macro_metrics(
    predictions: PredictionClosure, question: str, mode: str, cohort: int,
    phase: str, members: Sequence[str], model: str,
) -> dict[str, float]:
    if not members:
        raise ValueError("macro metrics require at least one series")
    values = tuple(
        predictions.metrics[_family_key(
            predictions, question, mode, cohort, phase, series, model,
        )]
        for series in members
    )
    return {
        name: fmean(item[name] for item in values)
        for name in values[0]
    }


def _point_map(
    predictions: PredictionClosure, question: str, mode: str, cohort: int,
    phase: str, members: Sequence[str], model: str,
) -> dict[str, tuple[ForecastPoint, ...]]:
    if phase != "calibration":
        raise ValueError("prediction arrays are retained for calibration only")
    result = {}
    for series in members:
        key = _family_key(
            predictions, question, mode, cohort, phase, series, model,
        )
        result[series] = (
            _points(
                predictions.truth.rows[(phase, series)],
                (0.0,) * len(predictions.truth.rows[(phase, series)]),
            )
            if model == "zero" else
            _points(
                predictions.truth.rows[(phase, series)],
                predictions.calibration[key],
            )
        )
    return result


def _majority_accuracy(
    values: Mapping[str, Sequence[ForecastPoint]],
) -> float:
    stocks = []
    for points in values.values():
        signs = tuple(
            (point.actual_return > 0) - (point.actual_return < 0)
            for point in points
        )
        stocks.append(max(signs.count(sign) for sign in (-1, 0, 1)) /
                      len(signs))
    return fmean(stocks)


def _view_members(
    closure: FitClosure, question: str, cohort: int, phase: str,
) -> Mapping[str, tuple[str, ...]]:
    evaluable = set(closure.evaluable[phase])
    if question == "unseen-transfer":
        return {"unseen": tuple(
            name for name in unseen_view(closure.master) if name in evaluable
        )}
    return {
        name: tuple(member for member in members if member in evaluable)
        for name, members in cohort_views(closure.master, cohort).items()
    }


def _comparison(
    predictions: PredictionClosure, question: str, mode: str, cohort: int,
    members: Sequence[str], candidate_model: str, reference_model: str,
    *, reference_cohort: int | None = None,
) -> dict[str, object]:
    candidate = _point_map(
        predictions, question, mode, cohort, "calibration",
        members, candidate_model,
    )
    reference = _point_map(
        predictions, question, mode,
        cohort if reference_cohort is None else reference_cohort,
        "calibration", members, reference_model,
    )
    return paired_comparison(
        candidate, reference, block_days=(5, 10, 20),
    )


def _paired_calibration(
    closure: FitClosure, predictions: PredictionClosure,
) -> dict[str, object]:
    evaluable = set(closure.evaluable["calibration"])
    core = tuple(name for name in closure.master[:11] if name in evaluable)
    unseen = tuple(
        name for name in unseen_view(closure.master) if name in evaluable
    )
    result = {}
    for mode in MODES:
        result[mode] = {
            "candidate_vs_baselines": {
                label: {
                    model: _comparison(
                        predictions, question, mode, cohort, members,
                        "panel_transformer", model,
                    )
                    for model in CONTROL_MODELS
                }
                for label, question, cohort, members in (
                    ("core:55", "cohort-scaling", 55, core),
                    ("unseen:44", "unseen-transfer", 44, unseen),
                )
            },
            "breadth_vs_11": {
                "core": {
                    str(cohort): _comparison(
                        predictions, "cohort-scaling", mode, cohort, core,
                        "panel_transformer", "panel_transformer",
                        reference_cohort=11,
                    )
                    for cohort in (22, 33, 55)
                },
                "unseen": {
                    str(cohort): _comparison(
                        predictions, "unseen-transfer", mode, cohort, unseen,
                        "panel_transformer", "panel_transformer",
                        reference_cohort=11,
                    )
                    for cohort in (22, 33, 44)
                },
            },
            "unseen_44_vs_33": _comparison(
                predictions, "unseen-transfer", mode, 44, unseen,
                "panel_transformer", "panel_transformer",
                reference_cohort=33,
            ),
        }
    return result


def _gate_results(
    unseen_metrics: Mapping[str, float],
    unseen_control_metrics: Mapping[str, float],
    core_metrics: Mapping[str, float],
    core_control_metrics: Mapping[str, float],
    expansion: Mapping[str, object],
    marginal: Mapping[str, object],
    control_metrics: Mapping[
        str, Mapping[str, Mapping[str, float]],
    ],
    core_majority: float,
    unseen_majority: float,
) -> dict[str, object]:
    """Evaluate the exact eight frozen gate inequalities."""
    intervals = expansion["intervals"]
    gains = expansion["per_stock_mean_gain"]
    if not isinstance(intervals, Mapping) or not isinstance(gains, Mapping):
        raise ValueError("paired comparison result is invalid")
    gates = {
        "unseen_mae_improvement": {
            "pass": unseen_control_metrics["return_mae"] > 0 and
            unseen_metrics["return_mae"] <=
            0.99 * unseen_control_metrics["return_mae"],
            "candidate": unseen_metrics["return_mae"],
            "control": unseen_control_metrics["return_mae"],
            "required_relative_improvement": 0.01,
        },
        "positive_paired_intervals": {
            "pass": all(
                intervals[str(block)][0] > 0 for block in (5, 10, 20)
            ),
            "intervals": intervals,
        },
        "majority_unseen_improved": {
            "pass": expansion["wins"] >= 6 and len(gains) == 11,
            "wins": expansion["wins"],
            "required_wins": 6,
            "stocks": len(gains),
        },
        "core_degradation": {
            "pass": core_metrics["return_mae"] <=
            1.01 * core_control_metrics["return_mae"],
            "candidate": core_metrics["return_mae"],
            "control": core_control_metrics["return_mae"],
            "maximum_relative_degradation": 0.01,
        },
        "pooled_and_local_controls": {
            "pass": all(
                (
                    core_metrics if view == "core" else unseen_metrics
                )["return_mae"] < metrics[model]["return_mae"]
                for view, metrics in control_metrics.items()
                for model in CONTROL_MODELS
            ),
            "views": control_metrics,
        },
        "direction_majority": {
            "pass": core_metrics["direction_accuracy"] > core_majority and
            unseen_metrics["direction_accuracy"] > unseen_majority,
            "core_candidate": core_metrics["direction_accuracy"],
            "core_majority": core_majority,
            "unseen_candidate": unseen_metrics["direction_accuracy"],
            "unseen_majority": unseen_majority,
        },
        "close_mae": {
            "pass": core_metrics["close_mae"] <
            control_metrics["core"]["zero"]["close_mae"] and
            unseen_metrics["close_mae"] <
            control_metrics["unseen"]["zero"]["close_mae"],
            "core_candidate": core_metrics["close_mae"],
            "core_zero": control_metrics["core"]["zero"]["close_mae"],
            "unseen_candidate": unseen_metrics["close_mae"],
            "unseen_zero": control_metrics["unseen"]["zero"]["close_mae"],
        },
        "unseen_33_to_44_marginal": {
            "pass": marginal["mean_gain"] >= 0,
            "mean_gain": marginal["mean_gain"],
        },
    }
    gates["all_pass"] = all(item["pass"] for item in gates.values())
    return gates


def build_development_summary(
    closure: FitClosure, predictions: PredictionClosure,
) -> dict[str, object]:
    """Recompute development metrics and the eight frozen forecast gates."""
    results = []
    for question, cohorts in (
        ("cohort-scaling", TRAINING_COHORTS),
        ("unseen-transfer", TRANSFER_COHORTS),
    ):
        for mode in MODES:
            for cohort in cohorts:
                for phase in PHASES:
                    views = {}
                    for view, members in _view_members(
                        closure, question, cohort, phase,
                    ).items():
                        models = tuple(
                            model for model in MODELS
                            if not (
                                question == "unseen-transfer" and
                                model == "conditioned_panel_transformer"
                            )
                        )
                        views[view] = {
                            "members": list(members),
                            "metrics": {
                                model: _macro_metrics(
                                    predictions, question, mode, cohort,
                                    phase, members, model,
                                )
                                for model in models
                            } if members else {},
                        }
                    results.append({
                        "question": question, "mode": mode, "cohort": cohort,
                        "phase": phase, "views": views,
                    })

    core = closure.master[:11]
    unseen = unseen_view(closure.master)
    candidate_unseen = _point_map(
        predictions, "unseen-transfer", "fixed-update", 44,
        "calibration", unseen, "panel_transformer",
    )
    control_unseen = _point_map(
        predictions, "unseen-transfer", "fixed-update", 11,
        "calibration", unseen, "panel_transformer",
    )
    candidate_core = _point_map(
        predictions, "cohort-scaling", "fixed-update", 55,
        "calibration", core, "panel_transformer",
    )
    control_core = _point_map(
        predictions, "cohort-scaling", "fixed-update", 11,
        "calibration", core, "panel_transformer",
    )
    unseen_metrics = stock_macro_metrics(candidate_unseen)
    unseen_control_metrics = stock_macro_metrics(control_unseen)
    core_metrics = stock_macro_metrics(candidate_core)
    core_control_metrics = stock_macro_metrics(control_core)
    paired = _paired_calibration(closure, predictions)
    expansion = paired["fixed-update"]["breadth_vs_11"]["unseen"]["44"]
    marginal = paired["fixed-update"]["unseen_44_vs_33"]
    control_metrics = {
        view: {
            model: stock_macro_metrics(_point_map(
                predictions, question, "fixed-update", cohort,
                "calibration", members, model,
            ))
            for model in CONTROL_MODELS
        }
        for view, question, cohort, members in (
            ("core", "cohort-scaling", 55, core),
            ("unseen", "unseen-transfer", 44, unseen),
        )
    }
    gates = _gate_results(
        unseen_metrics, unseen_control_metrics, core_metrics,
        core_control_metrics, expansion, marginal, control_metrics,
        _majority_accuracy(candidate_core),
        _majority_accuracy(candidate_unseen),
    )
    return {
        "schema": 1,
        "status": "pass" if gates["all_pass"] else "gate-failure",
        "evidence_role": "development-diagnostic-not-forward-clean",
        "ensemble": "arithmetic-mean-of-five-neural-returns",
        "fold_role": "checkpoint-selection-audit-development-diagnostic",
        "fixed_epoch_role":
            "descriptive-cohort-sized-draw-data-plus-compute-curve",
        "gate_source": "fixed-update-calibration-only",
        "model_binding_role":
            "cross-ledger-consistency-not-independent-execution-proof",
        "prediction_evidence": {
            "schema": 2,
            "records": predictions.records,
            "stored_values": predictions.stored_values,
            "synthesized_zero_values":
                predictions.synthesized_zero_values,
        },
        "locks": dict(LOCKS),
        "results": results,
        "paired_calibration": paired,
        "gates": gates,
    }


def analyze_ledgers(
    fit_values: Sequence[Mapping[str, object]],
    prediction_values: Iterable[Mapping[str, object]],
    master: Sequence[str],
    coverage: ScalingCoverage,
    truth: MarketTruth,
) -> dict[str, object]:
    """Validate both ledgers once and recompute their canonical summary."""
    fit_closure = validate_fit_ledger(fit_values, master, coverage)
    prediction_closure = validate_prediction_ledger(
        prediction_values, fit_closure, truth,
    )
    return build_development_summary(fit_closure, prediction_closure)


def _transition(stage: str, code: int, status: str) -> None:
    if type(code) is not int or not 0 <= code <= 255 or \
       status not in TRANSITIONS:
        raise ValueError("terminal transition is invalid")
    expected_stage, accepts = TRANSITIONS[status]
    if stage != expected_stage or not accepts(code):
        raise ValueError("terminal stage, exit, and status disagree")


def _clean(path: Path, label: str) -> Path:
    absolute = Path(os.path.abspath(path))
    if absolute != path.resolve(strict=False):
        raise ValueError(f"{label} must not contain a symlink or alias")
    return absolute


def _binding_path(binding: FileBinding) -> Path:
    path = Path(binding.path)
    return path if path.is_absolute() else ROOT / path


def _observe(name: str, path: Path) -> OutputState:
    _clean(path, f"{name} output")
    try:
        parent_identity = _directory_identity(path.parent)
    except ValueError:
        if os.path.lexists(path.parent):
            raise
        parent_identity = None
    try:
        identity = _regular_identity(path)
    except ValueError:
        if os.path.lexists(path):
            raise
        identity = None
    return OutputState(name, path, identity, parent_identity)


def _verify_states(states: Sequence[OutputState]) -> None:
    for state in states:
        _clean(state.path, f"{state.name} output")
        if state.parent_identity is None:
            if os.path.lexists(state.path.parent):
                raise ValueError("an absent output directory appeared")
        elif _directory_identity(state.path.parent) != state.parent_identity:
            raise ValueError("declared output directory changed")
        if state.present:
            if _regular_identity(state.path) != state.identity:
                raise ValueError("declared output identity changed")
        elif os.path.lexists(state.path) or \
                os.path.lexists(state.path.resolve(strict=False)):
            raise ValueError("an absent declared output appeared")


def _validate_declared_paths(
    attempt_path: Path, states: Sequence[OutputState],
) -> None:
    resolved = (
        _clean(attempt_path, "attempt"),
        *(_clean(state.path, f"{state.name} output") for state in states),
    )
    if len(set(resolved)) != len(resolved):
        raise ValueError("attempt and output paths must be disjoint")
    present = (
        _regular_identity(attempt_path),
        *(state.identity for state in states if state.present),
    )
    if len(set(present)) != len(present):
        raise ValueError("attempt and output inodes must be disjoint")


def _entry_state(
    directory_fd: int, name: str,
) -> tuple[tuple[int, int], int, int] | None:
    try:
        value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return (
        (value.st_dev, value.st_ino),
        stat.S_IFMT(value.st_mode),
        value.st_nlink,
    )


def _owns_temp(directory_fd: int, binding: ExclusiveTemp) -> bool:
    return _entry_state(directory_fd, binding.name) == (
        binding.identity, stat.S_IFREG, 1,
    )


def _require_temp(directory_fd: int, binding: ExclusiveTemp) -> None:
    if Path(binding.name).name != binding.name or \
            not _owns_temp(directory_fd, binding):
        raise ValueError("exclusive output temporary changed")


def _cleanup_temp(directory_fd: int, binding: ExclusiveTemp | None) -> None:
    # Random private names make this cooperative check-then-unlink best effort;
    # Python has no portable conditional unlink for an inode.
    if binding is not None and _owns_temp(directory_fd, binding):
        os.unlink(binding.name, dir_fd=directory_fd)
        os.fsync(directory_fd)


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value, allow_nan=False, indent=2, sort_keys=True,
        ) + "\n"
    ).encode()


def _publish_exclusive(
    path: Path, value: Mapping[str, object], directory_fd: int,
    verify: Callable[[ExclusiveTemp], None],
) -> ExclusiveTemp:
    _canonical_json_bytes(value)
    binding: ExclusiveTemp | None = None
    completed = False

    def capture(temporary: ExclusiveTemp) -> None:
        nonlocal binding
        binding = temporary

    def before_link(temporary: ExclusiveTemp) -> None:
        if temporary != binding:
            raise OSError("exclusive output temporary identity changed")
        verify(temporary)

    try:
        write_json_exclusive(
            path, value, directory_fd,
            before_link_with_temp=before_link,
            on_temp_created=capture,
        )
        completed = True
    finally:
        if not completed:
            _cleanup_temp(directory_fd, binding)
    if binding is None:
        raise OSError("exclusive output callback was not invoked")
    return binding


def _run_directory(attempt: ScalingAttempt) -> Path:
    return _clean(ROOT / attempt.run_dir, "run directory")


def _validate_run_members(
    attempt: ScalingAttempt, states: Sequence[OutputState], status: str,
    temporary: ExclusiveTemp | None = None,
) -> None:
    state_by_name = {state.name: state for state in states}
    if set(state_by_name) != {"fits", "predictions", "summary", "outcome"}:
        raise ValueError("declared output states are incomplete")
    fits = state_by_name["fits"].present
    predictions = state_by_name["predictions"].present
    summary = state_by_name["summary"].present
    if state_by_name["outcome"].present:
        raise ValueError("outcome must be fresh and absent")
    if status in ("preflight-failure", "setup-failure"):
        if fits or predictions or summary:
            raise ValueError("early failure contains run outputs")
    elif status == "experiment-failure":
        if summary:
            raise ValueError("experiment failure contains a summary")
    elif status == "analysis-integrity-failure":
        if not fits or not predictions or summary:
            raise ValueError("analysis failure output closure is invalid")
    elif status in PROVENANCE_STATUSES:
        if not fits or not predictions:
            raise ValueError("development outcome requires both ledgers")
    else:
        raise ValueError("unknown terminal status")

    run_dir = _run_directory(attempt)
    exists = os.path.lexists(run_dir)
    if status == "preflight-failure":
        if exists:
            raise ValueError("preflight failure created the run directory")
        return
    if not exists:
        if status == "setup-failure":
            return
        raise ValueError("terminal status requires the run directory")
    _directory_identity(run_dir)
    members = {path.name for path in run_dir.iterdir()}
    expected = {
        state.path.name
        for state in states
        if state.name in ("fits", "predictions", "summary") and state.present
    }
    if temporary is not None:
        run_fd = os.open(
            run_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            _require_temp(run_fd, temporary)
        finally:
            os.close(run_fd)
        expected.add(temporary.name)
    if members != expected:
        raise ValueError("run directory members are invalid")


def _fetch_bindings(path: Path) -> tuple[tuple[str, ...], tuple[FileBinding, ...]]:
    value = read_canonical_json(path)
    raw = value.get("series")
    if not isinstance(raw, list):
        raise ValueError("fetch report series must be an array")
    names = []
    bindings = []
    for index, record in enumerate(raw):
        label = f"fetch series[{index}]"
        if not isinstance(record, Mapping):
            raise ValueError(f"{label} must be an object")
        name = _string(record.get("ticker"), f"{label}.ticker")
        raw_csv = record.get("csv")
        if not isinstance(raw_csv, Mapping):
            raise ValueError(f"{label}.csv must be an object")
        binding = FileBinding.parse(
            {key: raw_csv.get(key) for key in ("path", "sha256")},
            f"{label}.csv", relative=False,
        )
        _integer(raw_csv.get("rows"), f"{label}.rows", 1)
        expected = ROOT / CSV_ROOT / f"{name.lower()}-30m.csv"
        if not NAME.fullmatch(name) or Path(binding.path) != expected:
            raise ValueError("fetch CSV path is outside the frozen root")
        names.append(name)
        bindings.append(binding)
    if len(names) != 55 or len(set(names)) != 55 or \
            len({item.path for item in bindings}) != 55:
        raise ValueError("fetch report must bind 55 ordered unique series")
    return tuple(names), tuple(bindings)


def _selection_paths(attempt: ScalingAttempt) -> tuple[Path, ...]:
    root = _clean(ROOT / attempt.selection_tree.root, "selection root")
    identity = _directory_identity(root)
    paths = []
    for path in root.rglob("*"):
        metadata = path.stat(follow_symlinks=False)
        kind = stat.S_IFMT(metadata.st_mode)
        if kind == stat.S_IFREG:
            _regular_identity(path)
            paths.append(path)
        elif kind == stat.S_IFDIR:
            _clean(path, "selection directory")
            _directory_identity(path)
        else:
            raise ValueError("selection package contains a nonregular member")
    if _directory_identity(root) != identity:
        raise ValueError("selection package root changed")
    return tuple(sorted(paths))


def _selection_matches(
    attempt: ScalingAttempt, paths: Sequence[Path],
    by_path: Mapping[Path, FrozenInput],
) -> None:
    root = ROOT / attempt.selection_tree.root
    files = tuple(
        FileBinding(
            path.relative_to(root).as_posix(), by_path[path].sha256,
        )
        for path in paths
    )
    if len(files) != attempt.selection_tree.files or \
            _tree_digest(files) != attempt.selection_tree.sha256:
        raise ValueError("selection package closure changed")


def _master_from_snapshot(path: Path) -> tuple[str, ...]:
    value = read_canonical_json(path)
    raw = value.get("series")
    if not isinstance(raw, list):
        raise ValueError("master manifest series are invalid")
    return _master(tuple(
        _string(
            _object(item, {"stratum", "ticker"}, "manifest series")["ticker"],
            "manifest ticker",
        )
        for item in raw
    ))


def _trusted_paths(attempt: ScalingAttempt) -> tuple[Path, ...]:
    if Path(attempt.finalizer_tree.root) != ROOT.resolve(strict=True):
        raise ValueError("trusted finalizer root changed")
    executable = Path(attempt.primary_python.path)
    if executable.resolve(strict=True) != Path(sys.executable).resolve(strict=True):
        raise ValueError("finalizer is not running under bound primary Python")
    attempt.primary_python.validate_live("primary Python")
    return tuple(
        ROOT / item.path for item in attempt.finalizer_tree.files
    ) + (executable,)


def _validate_trusted(
    attempt: ScalingAttempt, by_path: Mapping[Path, FrozenInput],
) -> None:
    for binding in attempt.finalizer_tree.files:
        path = ROOT / binding.path
        if by_path[path].sha256 != binding.sha256:
            raise ValueError("trusted finalizer snapshot changed")
    executable = Path(attempt.primary_python.path)
    if by_path[executable].sha256 != attempt.primary_python.sha256:
        raise ValueError("primary Python snapshot changed")


def _success_inputs(attempt: ScalingAttempt) -> SuccessInputs:
    fetch = _binding_path(attempt.fetch_report)
    fetch_identity = _regular_inputs((fetch,))
    with freeze_inputs((fetch,)) as frozen:
        if frozen[0].sha256 != attempt.fetch_report.sha256:
            raise ValueError("fetch report snapshot changed")
        names, csv = _fetch_bindings(frozen[0].snapshot)
        _verify_identities(fetch_identity)
        verify_frozen(frozen)
    selection = _selection_paths(attempt)
    paths = tuple(dict.fromkeys((
        *(_binding_path(item.file) for item in attempt.manifests),
        fetch, _binding_path(attempt.session_calendar),
        _binding_path(attempt.config),
        *(Path(item.path) for item in csv),
        *(ROOT / item.path for item in attempt.source_tree.files),
        Path(attempt.torch_probe.python.path),
        *(
            Path(attempt.torch_probe.package_tree.root) / item.path
            for item in attempt.torch_probe.package_tree.files
        ),
        *selection,
    )))
    return SuccessInputs(names, csv, selection, paths)


def _validate_success_inputs(
    attempt: ScalingAttempt, success: SuccessInputs,
    by_path: Mapping[Path, FrozenInput],
) -> tuple[str, ...]:
    bindings = (
        *(item.file for item in attempt.manifests),
        attempt.fetch_report, attempt.session_calendar, attempt.config,
        *success.csv,
    )
    for binding in bindings:
        path = _binding_path(binding)
        if by_path[path].sha256 != binding.sha256:
            raise ValueError("successful input snapshot changed")
    for tree in (attempt.source_tree, attempt.torch_probe.package_tree):
        root = Path(tree.root)
        for binding in tree.files:
            path = root / binding.path
            if by_path[path].sha256 != binding.sha256:
                raise ValueError("successful source snapshot changed")
    torch_python = Path(attempt.torch_probe.python.path)
    if by_path[torch_python].sha256 != attempt.torch_probe.python.sha256:
        raise ValueError("Torch Python snapshot changed")
    names, csv = _fetch_bindings(
        by_path[_binding_path(attempt.fetch_report)].snapshot,
    )
    if names != success.csv_names or csv != success.csv:
        raise ValueError("fetch report bindings changed")
    _selection_matches(attempt, success.selection_paths, by_path)
    master_path = _binding_path(attempt.manifests[-1].file)
    master = _master_from_snapshot(by_path[master_path].snapshot)
    if master != success.csv_names:
        raise ValueError("master manifest and fetch order disagree")
    return master


def _validate_live_success(
    attempt: ScalingAttempt, success: SuccessInputs,
) -> None:
    if _selection_paths(attempt) != success.selection_paths:
        raise ValueError("selection package members changed")
    if selected_source_tree(
        ROOT, tuple(item.path for item in attempt.source_tree.files),
    ) != attempt.source_tree:
        raise ValueError("scaling source closure changed")
    attempt.torch_probe.python.validate_live("Torch Python")
    if source_tree(Path(attempt.torch_probe.package_tree.root)) != \
            attempt.torch_probe.package_tree:
        raise ValueError("Torch package closure changed")


def _finalizer_argv(
    attempt: ScalingAttempt, started: str, ended: str,
    stage: str, code: int, status: str,
) -> None:
    expected = (
        *attempt.commands["finalizer_prefix"],
        "--started", started, "--ended", ended, "--stage", stage,
        "--exit", str(code), "--status", status,
    )
    if tuple(sys.argv) != expected:
        raise ValueError("finalizer arguments do not match the armed attempt")


def _output_records(
    states: Sequence[OutputState],
    by_path: Mapping[Path, FrozenInput],
) -> dict[str, object]:
    return {
        state.name: {
            "path": str(state.path),
            "state": "present" if state.present else "absent",
            "sha256": by_path[state.path].sha256 if state.present else None,
        }
        for state in states
    }


def _publish_outcome(
    attempt: ScalingAttempt, attempt_path: Path, outcome: Path,
    started: str, ended: str, stage: str, code: int, status: str,
    states: Sequence[OutputState],
    by_path: Mapping[Path, FrozenInput],
    frozen: Sequence[FrozenInput],
    identities: Sequence[tuple[Path, tuple[int, int]]],
    discovery: Sequence[FrozenInput],
    success: SuccessInputs | None,
) -> dict[str, object]:
    output = {
        "schema": 1,
        "attempt": {
            "path": str(attempt_path),
            "sha256": by_path[attempt_path].sha256,
            "run_id": attempt.run_id,
        },
        "started": started, "ended": ended,
        "stage": stage, "exit": code, "status": status,
        "outputs": _output_records(states, by_path),
        "integrity": {
            "trusted_finalizer_tree": attempt.finalizer_tree.sha256,
            "primary_python": {
                "path": attempt.primary_python.path,
                "sha256": attempt.primary_python.sha256,
            },
        },
    }
    outcome_fd, outcome_parent = _open_directory(outcome.parent)
    try:
        def verify_inputs(
            checked_states: Sequence[OutputState],
        ) -> None:
            _verify_identities(identities)
            verify_frozen(frozen)
            verify_frozen(discovery)
            _verify_states(checked_states)
            _validate_run_members(attempt, states, status)
            if success is not None:
                _validate_live_success(attempt, success)

        def verify_outcome(temporary: ExclusiveTemp) -> None:
            _require_temp(outcome_fd, temporary)
            verify_inputs(states)
            if _directory_identity(outcome.parent) != outcome_parent:
                raise ValueError("outcome parent changed")

        published = _publish_exclusive(
            outcome, output, outcome_fd, verify_outcome,
        )
        os.fsync(outcome_fd)
        if _directory_identity(outcome.parent) != outcome_parent or \
                _regular_identity(outcome) != published.identity:
            raise ValueError("published outcome identity changed")
        # The captured outcome state is intentionally absent: publication is
        # the transition. Revalidate every other declared output afterward.
        verify_inputs(tuple(
            state for state in states if state.name != "outcome"
        ))
    finally:
        os.close(outcome_fd)
    return output


def finalize(
    attempt_path: Path, outcome: Path, started: str, ended: str,
    stage: str, code: int, status: str,
) -> dict[str, object]:
    """Publish an exclusive summary and terminal outcome for one attempt."""
    _require_isolated_execution(bootstrapped=True)
    _require_exact_launch()
    _transition(stage, code, status)
    if _timestamp(ended, "ended") < _timestamp(started, "started"):
        raise ValueError("ended precedes started")
    attempt_path = _clean(attempt_path, "attempt")
    discovery_identity = _regular_inputs((attempt_path,))
    with freeze_inputs((attempt_path,)) as discovery:
        logical_attempt = Path(os.path.relpath(attempt_path, ROOT))
        attempt = ScalingAttempt.read(
            discovery[0].snapshot, logical_attempt, ROOT,
        )
        expected_outcome = ROOT / attempt.outputs["outcome"]
        if _clean(outcome, "outcome") != expected_outcome:
            raise ValueError("outcome path does not match the armed attempt")
        outcome = expected_outcome
        _finalizer_argv(attempt, started, ended, stage, code, status)
        trusted = _trusted_paths(attempt)
        declared = tuple(
            (name, ROOT / attempt.outputs[name])
            for name in ("fits", "predictions", "summary", "outcome")
        )
        states = tuple(_observe(name, path) for name, path in declared)
        state_by_name = {state.name: state for state in states}
        _validate_declared_paths(attempt_path, states)
        _validate_run_members(attempt, states, status)
        summary = state_by_name["summary"].path
        fits = state_by_name["fits"].path
        predictions = state_by_name["predictions"].path
        if state_by_name["outcome"].present:
            raise ValueError("outcome must be fresh and absent")
        development = status in PROVENANCE_STATUSES
        success = _success_inputs(attempt) if development else None
        present = tuple(state.path for state in states if state.present)
        sources = tuple(dict.fromkeys((
            attempt_path, *trusted, *present,
            *(success.paths if success is not None else ()),
        )))
        identities = _regular_inputs(sources)
        _verify_identities(discovery_identity)
        with freeze_inputs(sources) as frozen:
            by_path = dict(zip(sources, frozen, strict=True))
            verify_frozen(discovery)
            _verify_identities(discovery_identity)
            _verify_states(states)
            _validate_run_members(attempt, states, status)
            frozen_attempt = ScalingAttempt.read(
                by_path[attempt_path].snapshot,
                Path(attempt.attempt_path),
                ROOT,
            )
            if frozen_attempt != attempt:
                raise ValueError("attempt changed during finalization")
            _validate_trusted(attempt, by_path)
            summary_value = None
            if development:
                if success is None:
                    raise ValueError("development input closure is missing")
                master = _validate_success_inputs(attempt, success, by_path)
                fit_values = read_canonical_json_lines(
                    by_path[fits].snapshot,
                )
                truth = derive_market_truth(
                    by_path[
                        _binding_path(attempt.manifests[-1].file)
                    ].snapshot,
                    by_path[_binding_path(attempt.session_calendar)].snapshot,
                    {
                        name: by_path[Path(binding.path)].snapshot
                        for name, binding in zip(
                            success.csv_names, success.csv, strict=True,
                        )
                    },
                    attempt.coverage, attempt.protocol,
                )
                prediction_values = iter_canonical_json_lines(
                    by_path[predictions].snapshot,
                    max_line_bytes=_prediction_line_bytes(truth),
                )
                summary_value = analyze_ledgers(
                    fit_values, prediction_values, master, attempt.coverage,
                    truth,
                )
                if summary_value["status"] != status:
                    raise ValueError("declared status disagrees with gates")
                summary_value["inputs"] = {
                    "attempt": {
                        "path": str(attempt_path),
                        "sha256": by_path[attempt_path].sha256,
                    },
                    "fits": {
                        "path": str(fits), "sha256": by_path[fits].sha256,
                    },
                    "predictions": {
                        "path": str(predictions),
                        "sha256": by_path[predictions].sha256,
                    },
                }
            expected_summary = (
                _canonical_json_bytes(summary_value)
                if summary_value is not None else None
            )

            def verify_base(
                checked_states: Sequence[OutputState],
                checked_frozen: Sequence[FrozenInput],
                temporary: ExclusiveTemp | None = None,
            ) -> None:
                _verify_identities(identities)
                verify_frozen(checked_frozen)
                verify_frozen(discovery)
                _verify_states(checked_states)
                _validate_run_members(
                    attempt, checked_states, status, temporary,
                )
                if success is not None:
                    _validate_live_success(attempt, success)

            outcome_frozen = frozen
            outcome_by_path = by_path
            outcome_states = states
            if development and not state_by_name["summary"].present:
                summary_fd, summary_parent = _open_directory(summary.parent)
                try:
                    def verify_summary(temporary: ExclusiveTemp) -> None:
                        if _directory_identity(summary.parent) != \
                                summary_parent:
                            raise ValueError("summary parent changed")
                        _require_temp(summary_fd, temporary)
                        verify_base(states, frozen, temporary)

                    published = _publish_exclusive(
                        summary, summary_value, summary_fd, verify_summary,
                    )
                    os.fsync(summary_fd)
                    if _regular_identity(summary) != published.identity:
                        raise ValueError("published summary identity changed")
                finally:
                    os.close(summary_fd)
                outcome_states = tuple(
                    _observe(name, path) if name == "summary" else state
                    for state, (name, path) in zip(
                        states, declared, strict=True,
                    )
                )
                _validate_declared_paths(attempt_path, outcome_states)
                _validate_run_members(attempt, outcome_states, status)
                with freeze_inputs((summary,)) as frozen_summary:
                    if frozen_summary[0].snapshot.read_bytes() != \
                            expected_summary:
                        raise ValueError("published summary bytes changed")
                    outcome_frozen = (*frozen, *frozen_summary)
                    outcome_by_path = {
                        **by_path, summary: frozen_summary[0],
                    }
                    output = _publish_outcome(
                        attempt, attempt_path, outcome, started, ended,
                        stage, code, status, outcome_states,
                        outcome_by_path, outcome_frozen, identities,
                        discovery, success,
                    )
                    return output
            if development:
                if expected_summary is None:
                    raise ValueError("development summary is missing")
                if by_path[summary].snapshot.read_bytes() != expected_summary:
                    raise ValueError("existing summary is not recoverable")
                _verify_states(outcome_states)
                _validate_run_members(attempt, outcome_states, status)
            return _publish_outcome(
                attempt, attempt_path, outcome, started, ended,
                stage, code, status, outcome_states,
                outcome_by_path, outcome_frozen, identities, discovery,
                success,
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("outcome", type=Path)
    parser.add_argument("--started", required=True)
    parser.add_argument("--ended", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--exit", dest="exit_code", required=True, type=int)
    parser.add_argument("--status", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        result = finalize(
            args.attempt, args.outcome, args.started, args.ended,
            args.stage, args.exit_code, args.status,
        )
    except (
        KeyError, OSError, OverflowError, TypeError, UnicodeError, ValueError,
    ) as error:
        print(f"integrity error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    print(json.dumps({
        "outcome": str(args.outcome), "status": result["status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
