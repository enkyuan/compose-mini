"""Stream the fixed August 5 diagnostic without exposing future labels."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import chain
from math import fsum, isfinite, log
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import TextIO
import hashlib
import json
import os
import stat

from tools.context_diagnostic_inputs import timestamp_rows
from tools.data_v1 import (
    FEATURE_COUNT, _records, read_timestamps_until,
)
from tools.files import (
    ExclusiveTemp, FrozenInput, _owns_entry, exclusive_text, file_sha256,
    freeze_inputs, verify_frozen,
)
from tools.panel_contract import (
    _absent, _directory_identity, _exact_json, _open_directory,
    _regular_inputs, read_canonical_json, read_canonical_json_lines,
)
from tools.relative_context_contract import (
    HISTORY_BARS, HORIZON_BARS, INTERVAL_MINUTES, SEEDS, ResidualTruthRow,
)
from tools.relative_context_inputs import align_spy_rows
from tools.session_calendar import (
    SessionBin, SessionCalendar, expected_bins,
)
from tools.session_samples import SampleRows, SessionSamples, session_samples
from tools.spy_residual_forward_contract import (
    FORWARD_TARGET_SESSIONS, FORWARD_UNIVERSE, STATE_FINGERPRINTS,
)
from tools.spy_residual_gate import gate_mean_predictions, market_regimes
from tools.universe_contract import PackedRows

TARGET_SESSIONS = len(FORWARD_TARGET_SESSIONS)
TimestampTriple = tuple[str, str, str]
Verify = Callable[[], None]


@dataclass(frozen=True, slots=True)
class ForwardGrid:
    """Describe the six fixed target sessions and their causal triples."""

    boundary: str
    target_sessions: tuple[date, ...]
    triples: tuple[TimestampTriple, ...]

    def __post_init__(self) -> None:
        sessions = tuple(map(str, self.target_sessions))
        if type(self.boundary) is not str or not self.boundary or \
           sessions != FORWARD_TARGET_SESSIONS or \
           tuple(sorted(set(self.target_sessions))) != self.target_sessions or \
           not self.triples or any(
               type(row) is not tuple or len(row) != 3 or
               any(type(value) is not str or not value for value in row) or
               not row[0] < row[1] <= row[2] or row[1] <= self.boundary or
               row[2][:10] not in sessions
               for row in self.triples
           ) or tuple(dict.fromkeys(row[2][:10] for row in self.triples)) != \
           sessions:
            raise ValueError("forward grid is invalid")


@dataclass(frozen=True, slots=True)
class ForwardSeriesFiles:
    """Pair one historical snapshot with its strictly later continuation."""

    series: str
    source: FrozenInput
    future: FrozenInput

    def __post_init__(self) -> None:
        if type(self.series) is not str or not self.series or \
           type(self.source) is not FrozenInput or \
           type(self.future) is not FrozenInput or \
           self.source == self.future:
            raise ValueError("forward series files are invalid")


@dataclass(frozen=True, slots=True)
class ForwardSeriesInput:
    """Hold one immutable 17-bar OHLCV feature window."""

    series: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.series) is not str or not self.series or \
           len(self.values) != HISTORY_BARS * FEATURE_COUNT or any(
               type(value) is not float or not isfinite(value)
               for value in self.values
           ):
            raise ValueError("forward series input is invalid")


@dataclass(frozen=True, slots=True)
class SpyResidualForwardInputs:
    """Expose one causal cross-section and no entry or target prices."""

    index: int
    as_of: str
    stocks: tuple[ForwardSeriesInput, ...]
    spy: ForwardSeriesInput
    regime: str

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0 or \
           type(self.as_of) is not str or not self.as_of or \
           tuple(item.series for item in self.stocks) != FORWARD_UNIVERSE or \
           self.spy.series != "SPY" or self.regime not in (
               "negative", "nonnegative",
           ) or any(
               type(item) is not ForwardSeriesInput
               for item in (*self.stocks, self.spy)
           ):
            raise ValueError("SPY residual forward inputs are invalid")


@dataclass(frozen=True, slots=True)
class SeedPrediction:
    """Bind one raw prediction to its authenticated fitted state."""

    seed: int
    state_fingerprint: str
    prediction: float

    def __post_init__(self) -> None:
        expected = dict(zip(SEEDS, STATE_FINGERPRINTS, strict=True))
        if expected.get(self.seed) != self.state_fingerprint or \
           type(self.prediction) is not float or \
           not isfinite(self.prediction):
            raise ValueError("forward seed prediction is invalid")


@dataclass(frozen=True, slots=True)
class ForwardSeriesPrediction:
    """Hold the five ordered authenticated predictions for one stock."""

    series: str
    values: tuple[SeedPrediction, ...]

    def __post_init__(self) -> None:
        if type(self.series) is not str or not self.series or \
           type(self.values) is not tuple or \
           tuple(item.seed for item in self.values) != SEEDS or any(
               type(item) is not SeedPrediction for item in self.values
           ):
            raise ValueError("forward series prediction is invalid")


ForwardPredictions = tuple[ForwardSeriesPrediction, ...]
CandidateEvidence = Callable[
    [ForwardGrid, Sequence[Mapping[str, object]]], Mapping[str, object],
]
Predict = Callable[[SpyResidualForwardInputs], ForwardPredictions]


@dataclass(frozen=True, slots=True)
class ForwardRunBinding:
    """Join one verified predictor to its final provenance closure."""

    predict: Predict
    evidence: CandidateEvidence

    def __post_init__(self) -> None:
        if not callable(self.predict) or not callable(self.evidence):
            raise ValueError("forward run binding is invalid")


@dataclass(frozen=True, slots=True)
class CandidateLedger:
    """Bind one final label-free candidate to its published inode."""

    path: Path
    sha256: str
    identity: tuple[int, int]
    directory_identity: tuple[int, int]
    records: int


@dataclass(frozen=True, slots=True)
class ForwardPredictionSession:
    """Advance only after retaining predictions for the current as-of."""

    _current: Callable[[], SpyResidualForwardInputs]
    _submit: Callable[
        [SpyResidualForwardInputs],
        SpyResidualForwardInputs | CandidateLedger,
    ]

    def current(self) -> SpyResidualForwardInputs:
        """Return the only currently authorized feature batch."""
        return self._current()

    def submit(
        self, batch: SpyResidualForwardInputs,
    ) -> SpyResidualForwardInputs | CandidateLedger:
        """Retain one cross-section before authorizing the next cutoff."""
        return self._submit(batch)


TruthReader = Callable[
    [CandidateLedger],
    Mapping[str, tuple[ResidualTruthRow, ...]],
]


@dataclass(frozen=True, slots=True)
class _SeriesRows:
    files: ForwardSeriesFiles
    timestamps: tuple[str, ...]
    samples: tuple[SampleRows, ...]
    source_stop: str


class _SeriesStream:
    """Decode bars in timestamp order and retain only causal state."""

    def __init__(self, value: _SeriesRows) -> None:
        self.value = value
        self.position = -1
        self.window: deque[tuple[float, ...]] = deque(maxlen=HISTORY_BARS)
        self.truth: dict[str, tuple[float, float]] = {}
        needed = {
            value.timestamps[index]
            for row in value.samples for index in (row.entry, row.target)
        }

        def records() -> Iterator[tuple[str, list[float]]]:
            yield from _records(
                value.files.source.snapshot, value.source_stop,
            )
            yield from _records(
                value.files.future.snapshot,
                value.timestamps[value.samples[-1].target],
            )

        self._records = records()
        self._needed = needed

    def advance(self, stop: int) -> tuple[float, ...]:
        if type(stop) is not int or not self.position <= stop < \
           len(self.value.timestamps):
            raise ValueError("forward stream cutoff is invalid")
        while self.position < stop:
            try:
                timestamp, values = next(self._records)
            except StopIteration as error:
                raise ValueError("forward stream ended before its cutoff") \
                    from error
            self.position += 1
            if timestamp != self.value.timestamps[self.position]:
                raise ValueError("forward numeric stream changed")
            row = tuple(values)
            self.window.append(row)
            if timestamp in self._needed:
                self.truth[timestamp] = row[0], row[3]
        if len(self.window) != HISTORY_BARS:
            raise ValueError("forward stream lacks its feature history")
        return tuple(chain.from_iterable(self.window))

    def close(self) -> None:
        self._records.close()


def _calendar(
    source: SessionCalendar, future: SessionCalendar, boundary: str,
) -> SessionCalendar:
    if type(source) is not SessionCalendar or \
       type(future) is not SessionCalendar:
        raise ValueError("forward calendars are invalid")
    try:
        boundary_day = date.fromisoformat(boundary[:10])
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("forward calendar boundary is invalid") from error
    overlap_end = min(source.end, future.end)
    overlap = (
        range((overlap_end - future.start).days + 1)
        if future.start <= overlap_end else ()
    )
    if not source.start <= boundary_day <= source.end or \
       future.start != boundary_day + timedelta(days=1) or \
       (
           source.open_minute, source.close_minute, source.venues
       ) != (
           future.open_minute, future.close_minute, future.venues
       ) or any(
           source.session(future.start + timedelta(days=offset)) !=
           future.session(future.start + timedelta(days=offset))
           for offset in overlap
       ):
        raise ValueError("forward calendars are not equivalent continuations")
    return SessionCalendar(
        source.start, future.end, source.open_minute, source.close_minute,
        source.venues,
        tuple(day for day in source.closed_dates if day < future.start) +
        future.closed_dates,
        tuple(
            item for item in source.early_closes
            if item[0] < future.start
        ) + future.early_closes,
    )


def _bins(calendar: SessionCalendar) -> tuple[SessionBin, ...]:
    return tuple(expected_bins(
        calendar, calendar.start, calendar.end, INTERVAL_MINUTES,
    ))


def derive_forward_grid(
    source_calendar: SessionCalendar,
    forward_calendar: SessionCalendar,
    source_boundary: str,
) -> ForwardGrid:
    """Freeze the six post-preregistration target sessions."""
    calendar = _calendar(
        source_calendar, forward_calendar, source_boundary,
    )
    bins = _bins(calendar)
    try:
        boundary_day = date.fromisoformat(source_boundary[:10])
    except (TypeError, ValueError) as error:
        raise ValueError("source boundary is invalid") from error
    source = tuple(
        item.timestamp for item in expected_bins(
            source_calendar, source_calendar.start, boundary_day,
            INTERVAL_MINUTES,
        )
        if item.timestamp <= source_boundary
    )
    forward_bins = _bins(forward_calendar)
    future = tuple(item.timestamp for item in forward_bins)
    future_sessions = {item.session for item in forward_bins}
    if type(source_boundary) is not str or not source or \
       source_boundary != source[-1]:
        raise ValueError("source boundary is not an expected source bin")

    timestamps = source + future
    samples = session_samples(
        timestamps, INTERVAL_MINUTES, calendar, calendar.start, calendar.end,
        HISTORY_BARS, HORIZON_BARS, HORIZON_BARS,
    ).rows
    eligible = tuple(
        row for row in samples if timestamps[row.entry] > source_boundary
    )
    triples = timestamp_rows(timestamps, eligible)
    bin_by_time = {item.timestamp: item for item in bins}
    session_bins: dict[date, list[str]] = {}
    observed: dict[date, list[TimestampTriple]] = {}
    for item in bins:
        if item.session in future_sessions:
            session_bins.setdefault(item.session, []).append(item.timestamp)
    for triple in triples:
        observed.setdefault(
            bin_by_time[triple[2]].session, [],
        ).append(triple)

    selected_sessions = tuple(map(date.fromisoformat, FORWARD_TARGET_SESSIONS))
    if any(
        session not in session_bins or
        tuple(row[2] for row in observed.get(session, ())) !=
        tuple(session_bins[session])
        for session in selected_sessions
    ):
        raise ValueError("forward calendar lacks the fixed target sessions")
    selected = frozenset(selected_sessions)
    return ForwardGrid(
        source_boundary, selected_sessions,
        tuple(
            row for row in triples
            if bin_by_time[row[2]].session in selected
        ),
    )


def _future_timestamps(
    path: Path, calendar: SessionCalendar, stop: str, boundary: str,
) -> tuple[str, ...]:
    observed = read_timestamps_until(path, stop)
    expected = tuple(
        item.timestamp for item in expected_bins(
            calendar, calendar.start, date.fromisoformat(stop[:10]),
            INTERVAL_MINUTES,
        )
        if item.timestamp <= stop
    )
    if observed != expected or not observed or observed[0] <= boundary:
        raise ValueError("future CSV differs from its fixed calendar prefix")
    return observed


def _series_rows(
    files: ForwardSeriesFiles, grid: ForwardGrid,
    source_calendar: SessionCalendar, future_calendar: SessionCalendar,
) -> _SeriesRows:
    calendar = _calendar(
        source_calendar, future_calendar, grid.boundary,
    )
    source = read_timestamps_until(
        files.source.snapshot, grid.boundary,
    )
    future = _future_timestamps(
        files.future.snapshot, future_calendar, grid.triples[-1][2],
        grid.boundary,
    )
    if not source or source[-1] != grid.boundary or \
       source[-1] >= future[0]:
        raise ValueError("source and future CSVs do not form one timeline")
    timestamps = source + future
    candidates = session_samples(
        timestamps, INTERVAL_MINUTES, calendar, calendar.start, calendar.end,
        HISTORY_BARS, HORIZON_BARS, HORIZON_BARS,
    ).rows
    targets = {row[2] for row in grid.triples}
    samples = tuple(
        row for row in candidates
        if timestamps[row.entry] > grid.boundary and
        timestamps[row.target] in targets
    )
    if timestamp_rows(timestamps, samples) != grid.triples:
        raise ValueError("series differs from the canonical forward grid")
    return _SeriesRows(files, timestamps, samples, grid.boundary)


def _single_link_inputs(
    frozen: Sequence[FrozenInput],
) -> tuple[tuple[Path, tuple[int, int]], ...]:
    if any(
        type(item) is not FrozenInput or item.snapshot_identity is None
        for item in frozen
    ):
        raise ValueError("forward inputs must be genuine frozen snapshots")
    paths = tuple(item.source for item in frozen)
    identities = _regular_inputs(paths)
    if any(
        path.stat(follow_symlinks=False).st_nlink != 1 for path in paths
    ):
        raise ValueError("forward inputs must be single-link files")
    return identities


def _verify_inputs(
    frozen: Sequence[FrozenInput],
    identities: Sequence[tuple[Path, tuple[int, int]]],
    verify: Verify,
) -> None:
    verify()
    if _single_link_inputs(frozen) != tuple(identities):
        raise ValueError("forward input identity changed")
    verify_frozen(frozen)


def _grid_sha256(grid: ForwardGrid) -> str:
    value = {
        "boundary": grid.boundary,
        "target_sessions": list(map(str, grid.target_sessions)),
        "triples": [list(row) for row in grid.triples],
    }
    payload = json.dumps(
        value, allow_nan=False, sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _json_line(value: Mapping[str, object]) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True) + "\n"


def _header(
    run: ForwardRunBinding, grid: ForwardGrid,
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    evidence = run.evidence(grid, records)
    if not isinstance(evidence, Mapping) or not evidence:
        raise ValueError("forward candidate evidence is invalid")
    return {
        "batches": len(grid.triples),
        "closure": dict(evidence),
        "evidence_role": "label-free-forward-candidate",
        "grid_sha256": _grid_sha256(grid),
        "records": len(grid.triples) * len(FORWARD_UNIVERSE),
        "schema": 1,
        "type": "spy-residual-forward-candidate",
    }


def _prediction_record(
    value: ForwardSeriesPrediction, triple: TimestampTriple,
    regime: str,
) -> dict[str, object]:
    if type(value) is not ForwardSeriesPrediction:
        raise ValueError("forward prediction is invalid")
    predictions = tuple(item.prediction for item in value.values)
    mean = fsum(predictions) / len(predictions)
    gated = gate_mean_predictions((mean,), (regime,))[0]
    return {
        "as_of": triple[0],
        "entry": triple[1],
        "gate_active": regime == "nonnegative",
        "gated_prediction": gated,
        "mean_prediction": mean,
        "raw_predictions": [
            {
                "prediction": item.prediction,
                "seed": item.seed,
                "state_fingerprint": item.state_fingerprint,
            }
            for item in value.values
        ],
        "regime": regime,
        "schema": 1,
        "series": value.series,
        "target": triple[2],
    }


def _validate_ledger(
    path: Path, grid: ForwardGrid, run: ForwardRunBinding,
) -> tuple[Mapping[str, object], ...]:
    observed = read_canonical_json_lines(path)
    header, rows = observed[0], observed[1:]
    if not _exact_json(header, _header(run, grid, rows)) or \
       len(rows) != len(grid.triples) * len(FORWARD_UNIVERSE):
        raise ValueError("forward prediction ledger closure changed")
    expected = (
        (series, triple)
        for triple in grid.triples for series in FORWARD_UNIVERSE
    )
    for row, (series, triple) in zip(rows, expected, strict=True):
        raw = row.get("raw_predictions") \
            if isinstance(row, Mapping) else None
        if not isinstance(raw, list) or len(raw) != len(SEEDS):
            raise ValueError("forward prediction ledger row changed")
        try:
            values = tuple(
                SeedPrediction(
                    item["seed"], item["state_fingerprint"],
                    item["prediction"],
                )
                for item in raw if isinstance(item, Mapping)
            )
            regime = row["regime"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("forward prediction ledger row changed") \
                from error
        if len(values) != len(SEEDS) or \
           not isinstance(regime, str) or \
           not _exact_json(
               row, _prediction_record(
                   ForwardSeriesPrediction(series, values),
                   triple, regime,
               ),
           ):
            raise ValueError("forward prediction ledger row changed")
    return observed


def _private_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or \
       stat.S_IMODE(metadata.st_mode) != 0o600:
        raise ValueError("forward output is not a private regular file")
    return metadata.st_dev, metadata.st_ino


def _exclusive_output(
    path: Path, write: Callable[[TextIO], None], verify: Verify,
    directory_identity: tuple[int, int],
) -> tuple[int, int]:
    binding: ExclusiveTemp | None = None
    directory, identity = _open_directory(path.parent)
    if identity != directory_identity:
        os.close(directory)
        raise ValueError("forward output directory changed")

    def capture(value: ExclusiveTemp) -> None:
        nonlocal binding
        binding = value

    def before_link(value: ExclusiveTemp) -> None:
        if value is not binding or \
           _directory_identity(path.parent) != directory_identity:
            raise ValueError("forward output temporary changed")
        verify()

    try:
        exclusive_text(
            path, write, directory,
            before_link_with_temp=before_link,
            on_temp_created=capture,
        )
        os.fsync(directory)
    except BaseException:
        if binding is not None and _owns_entry(directory, binding, (1,)):
            os.unlink(binding.name, dir_fd=directory)
            os.fsync(directory)
        raise
    finally:
        os.close(directory)
    if binding is None or _private_identity(path) != binding.identity:
        raise ValueError("forward output publication changed")
    return binding.identity


def _publish(
    path: Path, grid: ForwardGrid,
    records: Sequence[Mapping[str, object]], run: ForwardRunBinding,
    verify: Verify,
    directory_identity: tuple[int, int],
) -> CandidateLedger:
    def write(file: TextIO) -> None:
        for record in chain((_header(run, grid, records),), records):
            file.write(_json_line(record))

    identity = _exclusive_output(
        path, write, verify, directory_identity,
    )
    with freeze_inputs((path,)) as frozen:
        verify()
        verify_frozen(frozen)
        _validate_ledger(frozen[0].snapshot, grid, run)
        if _private_identity(path) != identity:
            raise ValueError("prediction ledger changed after publication")
        return CandidateLedger(
            path, frozen[0].sha256, identity, directory_identity,
            len(records),
        )


def _receipt(
    candidate: CandidateLedger, grid: ForwardGrid,
) -> dict[str, object]:
    return {
        "candidate": {
            "directory_identity": list(candidate.directory_identity),
            "identity": list(candidate.identity),
            "path": str(candidate.path),
            "records": candidate.records,
            "sha256": candidate.sha256,
        },
        "grid_sha256": _grid_sha256(grid),
        "schema": 1,
        "type": "spy-residual-forward-truth-access",
    }


def _publish_receipt(
    path: Path, candidate: CandidateLedger, grid: ForwardGrid,
    verify: Verify,
    directory_identity: tuple[int, int],
) -> tuple[tuple[int, int], str]:
    expected = _receipt(candidate, grid)

    def write(file: TextIO) -> None:
        json.dump(
            expected, file, allow_nan=False, indent=2, sort_keys=True,
        )
        file.write("\n")

    identity = _exclusive_output(
        path, write, verify, directory_identity,
    )
    with freeze_inputs((path,)) as frozen:
        verify()
        verify_frozen(frozen)
        if not _exact_json(
            read_canonical_json(frozen[0].snapshot), expected,
        ) or _private_identity(path) != identity:
            raise ValueError("truth access receipt changed")
        return identity, frozen[0].sha256


def _input(stream: _SeriesStream, row: SampleRows) -> ForwardSeriesInput:
    return ForwardSeriesInput(
        stream.value.files.series, stream.advance(row.as_of),
    )


def _batch(
    index: int, stocks: Sequence[_SeriesStream], spy: _SeriesStream,
) -> SpyResidualForwardInputs:
    stock_inputs = tuple(
        _input(stream, stream.value.samples[index]) for stream in stocks
    )
    spy_input = _input(spy, spy.value.samples[index])
    regime = market_regimes(
        spy_input.values, (HISTORY_BARS - 1,),
    )[HISTORY_BARS - 1]
    return SpyResidualForwardInputs(
        index, spy.value.timestamps[spy.value.samples[index].as_of],
        stock_inputs, spy_input, regime,
    )


def _truth(
    stocks: Sequence[_SeriesStream], spy: _SeriesStream,
) -> Mapping[str, tuple[ResidualTruthRow, ...]]:
    final = spy.value.samples[-1].target
    spy.advance(final)
    for stock in stocks:
        stock.advance(stock.value.samples[-1].target)
    result = {}
    for stock in stocks:
        rows = []
        for row, spy_row in zip(
            stock.value.samples, spy.value.samples, strict=True,
        ):
            entry = stock.value.timestamps[row.entry]
            target = stock.value.timestamps[row.target]
            spy_entry = spy.value.timestamps[spy_row.entry]
            spy_target = spy.value.timestamps[spy_row.target]
            stock_open, stock_close = \
                stock.truth[entry][0], stock.truth[target][1]
            spy_open, spy_close = \
                spy.truth[spy_entry][0], spy.truth[spy_target][1]
            rows.append(ResidualTruthRow(
                stock.value.timestamps[row.as_of], entry, target,
                log(stock_close / stock_open) -
                log(spy_close / spy_open),
            ))
        result[stock.value.files.series] = tuple(rows)
    return MappingProxyType(result)


def _prepare_forward_inputs(
    grid: ForwardGrid,
    source_calendar: SessionCalendar,
    forward_calendar: SessionCalendar,
    stocks: Sequence[ForwardSeriesFiles],
    spy: ForwardSeriesFiles,
    prediction_ledger_path: Path,
    truth_receipt_path: Path,
    run: ForwardRunBinding,
    verify_inputs: Verify,
    expected_directory_identity: tuple[int, int],
) -> tuple[ForwardPredictionSession, TruthReader]:
    """Build one runtime-bound candidate stream and deferred truth gate."""
    stock_files = tuple(stocks)
    outputs = (prediction_ledger_path, truth_receipt_path)
    if type(grid) is not ForwardGrid or \
       derive_forward_grid(
           source_calendar, forward_calendar, grid.boundary,
       ) != grid or \
       tuple(item.series for item in stock_files) != FORWARD_UNIVERSE or \
       any(type(item) is not ForwardSeriesFiles for item in stock_files) or \
       type(spy) is not ForwardSeriesFiles or spy.series != "SPY" or \
       any(
           not isinstance(path, Path) or
           path != Path(os.path.abspath(path))
           for path in outputs
       ) or len(set(outputs)) != len(outputs) or \
       len({path.parent for path in outputs}) != 1 or \
       type(run) is not ForwardRunBinding or \
       not callable(verify_inputs) or \
       type(expected_directory_identity) is not tuple or \
       len(expected_directory_identity) != 2 or any(
           type(value) is not int or value < 0
           for value in expected_directory_identity
       ):
        raise ValueError("forward preparation inputs are invalid")
    _absent(prediction_ledger_path, "prediction ledger")
    _absent(truth_receipt_path, "truth access receipt")
    directory_identity = _directory_identity(prediction_ledger_path.parent)
    if directory_identity != expected_directory_identity:
        raise ValueError("forward output directory changed")
    frozen = tuple(
        item for files in (*stock_files, spy)
        for item in (files.source, files.future)
    )
    identities = _single_link_inputs(frozen)

    def verify() -> None:
        _verify_inputs(frozen, identities, verify_inputs)

    verify()
    stock_rows = tuple(
        _series_rows(
            files, grid, source_calendar, forward_calendar,
        )
        for files in stock_files
    )
    spy_rows = _series_rows(
        spy, grid, source_calendar, forward_calendar,
    )
    spy_samples = SessionSamples(
        spy_rows.samples, len(spy_rows.samples),
    )
    for stock in stock_rows:
        aligned = align_spy_rows(
            stock.timestamps,
            PackedRows(stock.samples, (len(stock.samples), 0)),
            spy_rows.timestamps, spy_samples,
        )
        if aligned.rows != spy_rows.samples:
            raise ValueError("stock and SPY forward rows differ")
    stock_streams = tuple(map(_SeriesStream, stock_rows))
    spy_stream = _SeriesStream(spy_rows)
    streams = (*stock_streams, spy_stream)
    lock, state, index = Lock(), "ready", 0
    records: list[Mapping[str, object]] = []
    publication: CandidateLedger | None = None
    current = _batch(index, stock_streams, spy_stream)

    def fail() -> None:
        for stream in streams:
            stream.close()

    def read_current() -> SpyResidualForwardInputs:
        with lock:
            if state != "ready":
                raise ValueError("forward feature session is not ready")
            return current

    def submit(
        batch: SpyResidualForwardInputs,
    ) -> SpyResidualForwardInputs | CandidateLedger:
        nonlocal state, index, current, publication
        with lock:
            if state != "ready" or batch is not current:
                raise ValueError("forward prediction session is not ready")
            state = "busy"
        try:
            predictions = run.predict(batch)
            if type(predictions) is not tuple or \
               tuple(item.series for item in predictions) != \
                    FORWARD_UNIVERSE or any(
                        type(item) is not ForwardSeriesPrediction
                        for item in predictions
                    ):
                raise ValueError("forward prediction order changed")
            triple, regime = grid.triples[index], current.regime
            records.extend(
                _prediction_record(
                    prediction, triple, regime,
                )
                for prediction in predictions
            )
            index += 1
            if index < len(grid.triples):
                current = _batch(index, stock_streams, spy_stream)
                result: SpyResidualForwardInputs | CandidateLedger = \
                    current
                next_state = "ready"
            else:
                verify()
                publication = _publish(
                    prediction_ledger_path, grid, records, run, verify,
                    directory_identity,
                )
                result, next_state = publication, "published"
        except BaseException:
            with lock:
                state = "failed"
            fail()
            raise
        with lock:
            state = next_state
        return result

    def read_truth(
        claim: CandidateLedger,
    ) -> Mapping[str, tuple[ResidualTruthRow, ...]]:
        nonlocal state
        with lock:
            if state != "published" or claim is not publication:
                raise ValueError("forward truth is not authorized")
            state = "consumed"
        try:
            if _directory_identity(claim.path.parent) != \
                    claim.directory_identity or \
               _private_identity(claim.path) != claim.identity or \
               file_sha256(claim.path) != claim.sha256:
                raise ValueError("prediction ledger changed")
            with freeze_inputs((claim.path,)) as frozen_ledger:
                verify()
                verify_frozen(frozen_ledger)
                rows = _validate_ledger(
                    frozen_ledger[0].snapshot, grid, run,
                )
                if len(rows) - 1 != claim.records or \
                   frozen_ledger[0].sha256 != claim.sha256 or \
                   _directory_identity(claim.path.parent) != \
                        claim.directory_identity or \
                   _private_identity(claim.path) != claim.identity:
                    raise ValueError("prediction publication changed")
                receipt_identity, receipt_sha256 = _publish_receipt(
                    truth_receipt_path, claim, grid, verify,
                    directory_identity,
                )
                with freeze_inputs((truth_receipt_path,)) as frozen_receipt:
                    verify()
                    verify_frozen((*frozen_ledger, *frozen_receipt))
                    if frozen_receipt[0].sha256 != receipt_sha256 or \
                       _private_identity(claim.path) != claim.identity or \
                       _private_identity(truth_receipt_path) != \
                            receipt_identity or not _exact_json(
                                read_canonical_json(
                                    frozen_receipt[0].snapshot,
                                ),
                                _receipt(claim, grid),
                            ):
                        raise ValueError("truth access receipt changed")
                    truth = _truth(stock_streams, spy_stream)
                    verify()
                    verify_frozen((*frozen_ledger, *frozen_receipt))
                    if _private_identity(claim.path) != claim.identity or \
                       _directory_identity(claim.path.parent) != \
                            claim.directory_identity or \
                       _private_identity(truth_receipt_path) != \
                            receipt_identity:
                        raise ValueError(
                            "forward output changed during truth access",
                        )
                    return truth
        finally:
            fail()

    return ForwardPredictionSession(read_current, submit), read_truth
