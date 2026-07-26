"""Authenticate the fixed source run and one later Massive data bundle."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from threading import Lock
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in tuple(map(os.path.realpath, sys.path)):
    sys.path.insert(0, str(ROOT))

from tools.analyze_spy_residual_shrinkage import (
    ANALYSIS_SOURCE_PATHS, _completed_run,
)
from tools.arm_spy_residual import (
    ResidualLease, _directory_members, _single_link_inputs,
    authenticate_residual_attempt,
)
from tools.context_diagnostic_contract import ContextAttempt
from tools.context_diagnostic_inputs import (
    context_cutoff_timestamp, context_grid_sha256, context_phase_rows,
    timestamp_rows,
)
from tools.data_v1 import read_timestamps, read_timestamps_until
from tools.files import FrozenInput, freeze_inputs, verify_frozen
from tools.panel_contract import (
    _exact_json, _verify_identities, read_canonical_json,
    read_canonical_json_lines,
)
from tools.relative_context_contract import (
    HORIZON_BARS, INTERVAL_MINUTES, RESIDUAL_SOURCE, SEEDS, SPY_END,
    SPY_START, ResidualAttempt, validate_residual_fit_records,
)
from tools.run_spy_residual import phase_artifacts
from tools.session_calendar import SessionCalendar, expected_bins
from tools.spy_residual_forward_contract import (
    FORWARD_CALENDAR, FORWARD_SOURCES, FORWARD_UNIVERSE,
    STATE_FINGERPRINTS,
)
from tools.spy_residual_forward_inputs import (
    ForwardGrid, ForwardPredictionSession, ForwardSeriesFiles, TruthReader,
    _prepare_forward_inputs, derive_forward_grid,
)

Verify = Callable[[], None]
PURPOSE = "Authenticate the fixed SPY-residual forward holdout."
FORWARD_RUN_DIR = ROOT / "reports/h13-spy-direction-forward-20260726-01"
FORWARD_DRAFT = FORWARD_RUN_DIR / "prediction-draft.jsonl"
FORWARD_TRUTH_RECEIPT = FORWARD_RUN_DIR / "truth-access.json"


@dataclass(frozen=True, slots=True)
class ForwardLease:
    """Hold Task 4's authenticated, one-use internal input handoff."""

    grid: ForwardGrid
    _open: Callable[
        [], tuple[ForwardPredictionSession, TruthReader],
    ] = field(repr=False, compare=False)
    _verify: Verify = field(repr=False, compare=False)

    def __call__(self) -> None:
        self._verify()

    def _prepare(
        self,
    ) -> tuple[ForwardPredictionSession, TruthReader]:
        """Hand the fixed one-shot session to the authenticated runner."""
        self()
        return self._open()


@dataclass(frozen=True, slots=True)
class _Historical:
    calendar: SessionCalendar
    boundary: str
    stocks: tuple[tuple[str, FrozenInput], ...]
    spy: FrozenInput
    verify: Verify = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class _Future:
    calendar: SessionCalendar
    stocks: tuple[tuple[str, FrozenInput], ...]
    spy: FrozenInput
    verify: Verify = field(repr=False, compare=False)


def _source_context(snapshot: FrozenInput) -> ContextAttempt:
    binding = RESIDUAL_SOURCE["context_attempt"]
    if snapshot.sha256 != binding.sha256:
        raise ValueError("forward source context changed")
    return ContextAttempt.read(
        snapshot.snapshot, Path(binding.path), ROOT,
    )


def _source_boundary(
    context: ContextAttempt, lease: ResidualLease,
    calendar: SessionCalendar,
) -> str:
    phase = next(
        item for item in context.phases if item.phase == "calibration"
    )
    expected = {
        series: (count, grid)
        for series, count, grid in phase.evaluation_rows
    }
    if tuple(expected) != FORWARD_UNIVERSE:
        raise ValueError("forward source universe changed")
    cutoff = context_cutoff_timestamp(
        calendar, date.fromisoformat(SPY_START), date.fromisoformat(SPY_END),
        INTERVAL_MINUTES, HORIZON_BARS, HORIZON_BARS,
    )
    grids, targets = {}, []
    for series, snapshot in lease.context.snapshots.csv:
        if series not in expected:
            continue
        timestamps = read_timestamps_until(snapshot.snapshot, cutoff)
        packed = context_phase_rows(
            timestamps, INTERVAL_MINUTES, calendar,
            date.fromisoformat(SPY_START), date.fromisoformat(SPY_END),
            phase.phase, HORIZON_BARS, HORIZON_BARS,
        )
        rows = packed.rows[packed.counts[0]:]
        grid = timestamp_rows(timestamps, rows)
        if len(rows) != expected[series][0] or \
           not grid or grid[-1][2] > cutoff:
            raise ValueError("forward calibration rows changed")
        grids[series] = grid
        targets.append(grid[-1][2])
    if context_grid_sha256(
        "evaluation", FORWARD_UNIVERSE, grids,
    ) != phase.evaluation_grid_sha256:
        raise ValueError("forward calibration grid changed")
    boundary = max(targets)
    boundary_day = date.fromisoformat(boundary[:10])
    source_times = {
        item.timestamp for item in expected_bins(
            calendar, boundary_day, boundary_day, INTERVAL_MINUTES,
        )
    }
    if boundary != cutoff or boundary not in source_times:
        raise ValueError("forward source boundary changed")
    return boundary


def _validate_alignment(value: object) -> None:
    source = {
        name: {"path": path, "sha256": sha256}
        for name, path, sha256 in FORWARD_SOURCES
    }
    if not isinstance(value, Mapping) or \
       value.get("schema") != 1 or \
       value.get("evidence_role") != \
            "development-post-hoc-not-forward-clean" or \
       value.get("truth_phases_read") != ["fold-1"] or \
       value.get("subject") != {
           "model": "panel_transformer",
           "seed_aggregation": "arithmetic-mean",
       } or value.get("market_regime") != {
           "feature": "log(spy.close[as_of] / spy.close[as_of - 16])",
           "history_bars": 17,
           "labels": {
               "negative": "feature < 0",
               "nonnegative": "feature >= 0",
           },
           "timing": "completed-as-of-bars-only",
       }:
        raise ValueError("forward alignment evidence changed")
    inputs = value.get("inputs")
    if not isinstance(inputs, Mapping) or \
       inputs.get("attempt") != source["residual_attempt"] or \
       inputs.get("outcome") != source["residual_outcome"]:
        raise ValueError("forward alignment provenance changed")


def _validate_fits(
    path: Path, context: ContextAttempt, attempt: ResidualAttempt,
) -> None:
    source = next(
        item for item in context.phases if item.phase == "calibration"
    )
    phase = next(
        item for item in attempt.phases if item.phase == "calibration"
    )
    records = validate_residual_fit_records(
        read_canonical_json_lines(path), context.master, source, phase,
    )
    observed = tuple(
        (item.fit.seed, item.state_fingerprint)
        for item in records if item.fit.model == "panel_transformer"
    )
    if observed != tuple(zip(SEEDS, STATE_FINGERPRINTS, strict=True)):
        raise ValueError("forward transformer states changed")


@contextmanager
def _bound_historical() -> Iterator[_Historical]:
    """Hold the completed source run and its original market inputs."""
    source_paths = {
        name: ROOT / path for name, path, _ in FORWARD_SOURCES
    }
    context_path = ROOT / RESIDUAL_SOURCE["context_attempt"].path
    initial = (context_path, *source_paths.values())
    initial_identities = _single_link_inputs(initial, "forward sources")
    with freeze_inputs(initial) as first:
        frozen = dict(zip(initial, first, strict=True))
        expected_hashes = {
            name: sha256 for name, _, sha256 in FORWARD_SOURCES
        }
        if any(
            frozen[path].sha256 != expected_hashes[name]
            for name, path in source_paths.items()
        ):
            raise ValueError("forward fixed source changed")
        context = _source_context(frozen[context_path])
        attempt = ResidualAttempt.read(
            frozen[source_paths["residual_attempt"]].snapshot,
            Path(FORWARD_SOURCES[0][1]), ROOT, context,
        )

    artifacts = tuple(
        path
        for source in context.phases
        for path in phase_artifacts(
            ROOT, source_paths["residual_attempt"], source,
        )
    )
    run = ROOT / attempt.run_dir
    run_identity = _directory_members(
        run, tuple(path.name for path in artifacts),
    )
    analysis = tuple(ROOT / path for path in ANALYSIS_SOURCE_PATHS)
    paths = tuple(dict.fromkeys((
        *initial, *artifacts, *analysis,
    )))
    identities = _single_link_inputs(paths, "forward source closure")
    alignment_dir = source_paths["alignment_report"].parent
    alignment_identity = _directory_members(
        alignment_dir, ("alignment.json",),
    )
    with freeze_inputs(paths) as snapshots:
        frozen = dict(zip(paths, snapshots, strict=True))
        if any(
            frozen[path].sha256 != expected_hashes[name]
            for name, path in source_paths.items()
        ):
            raise ValueError("forward fixed source changed")
        context = _source_context(frozen[context_path])
        authenticated, _, _ = _completed_run(
            source_paths["residual_attempt"],
            Path(FORWARD_SOURCES[0][1]), frozen, run_identity, context,
        )
        if authenticated != attempt:
            raise ValueError("forward residual run changed")
        _validate_alignment(read_canonical_json(
            frozen[source_paths["alignment_report"]].snapshot,
        ))
        _validate_fits(
            frozen[source_paths["calibration_fits"]].snapshot,
            context, attempt,
        )
        with authenticate_residual_attempt(attempt) as lease:
            calendar_input = lease.context.snapshots.calendar
            calendar = SessionCalendar.read(calendar_input.snapshot)
            csv = dict(lease.context.snapshots.csv)
            stocks = tuple(
                (series, csv[series])
                for series in FORWARD_UNIVERSE
            )
            spy = dict(lease.benchmark)["spy_csv"]
            boundary = _source_boundary(context, lease, calendar)

            def verify() -> None:
                lease()
                if _directory_members(
                    run, tuple(path.name for path in artifacts),
                ) != run_identity or _directory_members(
                    alignment_dir, ("alignment.json",),
                ) != alignment_identity:
                    raise ValueError("forward source directory changed")
                _verify_identities(identities)
                _single_link_inputs(paths, "forward source closure")
                verify_frozen(snapshots)

            verify()
            yield _Historical(calendar, boundary, stocks, spy, verify)
            verify()
    _verify_identities(initial_identities)


def _bundle_record(
    ticker: str, path: Path, sha256: str, rows: int,
    source_rows: object, calendar: SessionCalendar,
) -> dict[str, object]:
    if type(source_rows) is not int or source_rows < rows:
        raise ValueError("forward Massive source row count changed")
    return {
        "aggregate": {
            "path": (
                f"/v2/aggs/ticker/{ticker}/range/{INTERVAL_MINUTES}/minute/"
                f"{calendar.start}/{calendar.end}"
            ),
            "query": {
                "adjusted": "true", "limit": "50000", "sort": "asc",
            },
        },
        "csv": {
            "path": str(path),
            "rows": rows,
            "session_audit": {
                "expected_bins": rows,
                "missing_bins": 0,
                "scope": "all-expected-session-bins",
            },
            "sha256": sha256,
            "source_rows": source_rows,
        },
        "ticker": ticker,
    }


def _validate_bundle_report(
    value: object, calendar_path: Path, calendar_input: FrozenInput,
    calendar: SessionCalendar,
    future: Sequence[tuple[str, FrozenInput]],
) -> None:
    bins = tuple(expected_bins(
        calendar, calendar.start, calendar.end, INTERVAL_MINUTES,
    ))
    expected_times = tuple(item.timestamp for item in bins)
    records = []
    for ticker, snapshot in future:
        timestamps = read_timestamps(snapshot.snapshot)
        if timestamps != expected_times:
            raise ValueError(f"{ticker} forward grid is incomplete")
        supplied = value.get("series") if isinstance(value, Mapping) else None
        index = len(records)
        source_rows = (
            supplied[index].get("csv", {}).get("source_rows")
            if isinstance(supplied, list) and index < len(supplied) and
            isinstance(supplied[index], Mapping) and
            isinstance(supplied[index].get("csv"), Mapping)
            else None
        )
        records.append(_bundle_record(
            ticker, snapshot.source, snapshot.sha256, len(timestamps),
            source_rows, calendar,
        ))
    expected = {
        "adjusted": True,
        "calendar": {
            "path": str(calendar_path), "sha256": calendar_input.sha256,
        },
        "end": str(calendar.end),
        "interval_minutes": INTERVAL_MINUTES,
        "provider": "massive",
        "purpose": PURPOSE,
        "schema": 1,
        "series": records,
        "session": "regular",
        "start": str(calendar.start),
    }
    if not _exact_json(value, expected):
        raise ValueError("forward Massive bundle report changed")


@contextmanager
def _bound_future(
    calendar_path: Path, bundle: Path, calendar_sha256: str,
) -> Iterator[_Future]:
    names = (
        "fetch.json",
        *(f"{ticker.lower()}-30m.csv"
          for ticker in (*FORWARD_UNIVERSE, "SPY")),
    )
    bundle_identity = _directory_members(bundle, names)
    paths = (
        calendar_path, bundle / "fetch.json",
        *(bundle / name for name in names[1:]),
    )
    identities = _single_link_inputs(paths, "forward Massive bundle")
    with freeze_inputs(paths) as snapshots:
        calendar_input, report_input, *csv_inputs = snapshots
        if calendar_input.sha256 != calendar_sha256:
            raise ValueError("forward calendar changed")
        calendar = SessionCalendar.read(calendar_input.snapshot)
        future = tuple(zip(
            (*FORWARD_UNIVERSE, "SPY"), csv_inputs, strict=True,
        ))
        _validate_bundle_report(
            read_canonical_json(report_input.snapshot),
            calendar_path, calendar_input, calendar, future,
        )

        def verify() -> None:
            if _directory_members(bundle, names) != bundle_identity:
                raise ValueError("forward Massive bundle directory changed")
            _verify_identities(identities)
            _single_link_inputs(paths, "forward Massive bundle")
            verify_frozen(snapshots)

        verify()
        yield _Future(
            calendar, future[:-1], future[-1][1], verify,
        )
        verify()


@contextmanager
def arm_forward_inputs(
    massive_bundle: Path,
) -> Iterator[ForwardLease]:
    """Bind one exact future bundle to the preregistered source closure."""
    if massive_bundle != Path(os.path.abspath(massive_bundle)):
        raise ValueError("forward input paths must be absolute")
    _, calendar_path, calendar_sha256 = FORWARD_CALENDAR
    forward_calendar = ROOT / calendar_path
    with _bound_historical() as source:
        with _bound_future(
            forward_calendar, massive_bundle, calendar_sha256,
        ) as future:
            grid = derive_forward_grid(
                source.calendar, future.calendar, source.boundary,
            )
            future_by_series = dict(future.stocks)
            stocks = tuple(
                ForwardSeriesFiles(
                    series, historical, future_by_series[series],
                )
                for series, historical in source.stocks
            )
            spy = ForwardSeriesFiles("SPY", source.spy, future.spy)

            def verify() -> None:
                source.verify()
                future.verify()

            lock, opened = Lock(), False

            def prepare() -> tuple[ForwardPredictionSession, TruthReader]:
                nonlocal opened
                with lock:
                    if opened:
                        raise ValueError("forward input lease was already used")
                    opened = True
                verify()
                return _prepare_forward_inputs(
                    grid, source.calendar, future.calendar, stocks, spy,
                    FORWARD_DRAFT, FORWARD_TRUTH_RECEIPT, verify,
                )

            verify()
            yield ForwardLease(grid, prepare, verify)
            verify()
