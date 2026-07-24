#!/usr/bin/env python3
"""Verify Massive downloads and strict universe fetching without network."""

from datetime import date, datetime
from decimal import Decimal
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qsl, parse_qs, unquote, urlsplit
from unittest.mock import patch
import hashlib
import json
import math
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.data_v1 import FEATURE_COUNT, read_csv
from tools.fetch_massive import (
    aggregate_url, api_key, authorized_url, fetch_bars, regular_bars,
    request_gate, request_json, write_csv,
)
from tools.fetch_universe import (
    SeriesSpec, UniverseManifest, fetch_universe, parse_args as universe_args,
    reference_url,
)
from tools.files import file_sha256, write_json
from tools.select_universe import (
    Candidate, DailyRow, Reference, SelectionPolicy, select_candidates,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def timestamp(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def aggregate(value: str, close: float) -> dict[str, object]:
    return {"t": timestamp(value), "o": close - 0.25, "h": close + 0.5,
            "l": close - 0.5, "c": close, "v": 1000}


def raises(
    kind: type[BaseException] | tuple[type[BaseException], ...],
    function: object,
    *args: object,
    **kwargs: object,
) -> BaseException:
    try:
        function(*args, **kwargs)  # type: ignore[operator]
    except kind as error:
        return error
    names = (
        ", ".join(item.__name__ for item in kind)
        if isinstance(kind, tuple) else kind.__name__
    )
    raise AssertionError(f"{names} was not raised")


def selection_policy_value() -> dict[str, object]:
    return {
        "adjusted": True,
        "anchor_date": "2024-10-31",
        "cohort_sizes": [11, 22, 33, 55],
        "declared_on": "2026-07-24",
        "end": "2026-07-21",
        "formation_end": "2024-10-31",
        "formation_start": "2024-08-01",
        "interval_minutes": 30,
        "liquidity_strata": 5,
        "minimum_coverage": "0.95",
        "minimum_formation_sessions": 60,
        "minimum_median_close_usd": "5",
        "minimum_median_dollar_volume_usd": "25000000",
        "primary_cohort_size": 55,
        "purpose": (
            "Select nested point-in-time U.S. common-stock cohorts before "
            "fetching model data."
        ),
        "schema": 1,
        "selection_seed": "compose-mini-massive-universe-v1",
        "session": "regular",
        "start": "2024-11-01",
    }


def write_selection_policy(
    path: Path, value: dict[str, object] | None = None,
) -> None:
    path.write_text(
        json.dumps(selection_policy_value() if value is None else value) + "\n",
        encoding="ascii",
    )


def assert_selection_policy_error(path: Path) -> None:
    raises((OSError, ValueError), SelectionPolicy.read, path)


def test_selection_policy(directory: Path) -> None:
    path = directory / "selection-policy.json"
    write_selection_policy(path)
    assert SelectionPolicy.read(
        ROOT / "universes/liquid-common-ladder.example.json",
    ) == SelectionPolicy(
        1,
        "Select nested point-in-time U.S. common-stock cohorts before "
        "fetching model data.",
        date(2026, 7, 24), date(2024, 10, 31), date(2024, 8, 1),
        date(2024, 10, 31), date(2024, 11, 1), date(2026, 7, 21), 30, True,
        "regular", (11, 22, 33, 55), 55,
        "compose-mini-massive-universe-v1", 5, 60, Decimal("0.95"),
        Decimal("5"), Decimal("25000000"),
    )

    for field in selection_policy_value():
        value = selection_policy_value()
        del value[field]
        write_selection_policy(path, value)
        assert_selection_policy_error(path)
    write_selection_policy(path, selection_policy_value() | {"extra": True})
    assert_selection_policy_error(path)

    raw = json.dumps(selection_policy_value())
    path.write_text(
        raw.replace('"schema": 1', '"schema": 1, "schema": 1', 1),
        encoding="ascii",
    )
    assert_selection_policy_error(path)

    for field in (
        "schema", "interval_minutes", "liquidity_strata",
        "minimum_formation_sessions", "primary_cohort_size",
    ):
        value = selection_policy_value()
        value[field] = True
        write_selection_policy(path, value)
        assert_selection_policy_error(path)

    for field, replacement in (
        ("anchor_date", "2024-10-31T00:00:00"),
        ("minimum_coverage", "0.950"),
        ("minimum_median_close_usd", "05"),
        ("minimum_median_dollar_volume_usd", "2.5e7"),
        ("formation_start", "2024-11-01"),
        ("formation_end", "2024-10-30"),
        ("start", "2024-10-31"),
        ("end", "2024-10-31"),
        ("declared_on", "2024-10-30"),
        ("cohort_sizes", [11, 22, 22, 55]),
        ("primary_cohort_size", 33),
        ("liquidity_strata", 1),
        ("minimum_coverage", "0"),
        ("minimum_coverage", "1.01"),
        ("minimum_formation_sessions", 0),
        ("minimum_median_close_usd", "0"),
        ("minimum_median_dollar_volume_usd", "0"),
    ):
        value = selection_policy_value()
        value[field] = replacement
        write_selection_policy(path, value)
        assert_selection_policy_error(path)

    precise = "0.12345678901234567890123456789"
    value = selection_policy_value() | {"minimum_coverage": precise}
    write_selection_policy(path, value)
    assert SelectionPolicy.read(path).minimum_coverage == Decimal(precise)
    for nonfinite in ("NaN", "sNaN", "Infinity", "-Infinity"):
        value["minimum_coverage"] = nonfinite
        write_selection_policy(path, value)
        raises(ValueError, SelectionPolicy.read, path)


def selection_reference(
    ticker: str, share_class_figi: str, **changes: object,
) -> Reference:
    value: dict[str, object] = {
        "active": True,
        "market": "stocks",
        "locale": "us",
        "type": "CS",
        "currency_name": "usd",
        "primary_exchange": "XNYS",
        "composite_figi": f"BBG{ticker}",
        "share_class_figi": share_class_figi,
    }
    value.update(changes)
    return Reference(ticker=ticker, **value)  # type: ignore[arg-type]


def test_pure_selection() -> None:
    policy = SelectionPolicy.read(
        ROOT / "universes/liquid-common-ladder.example.json",
    )
    dollar_volumes = (
        Decimal("100000000"), Decimal("80000000"), Decimal("60000000"),
        Decimal("40000000"), Decimal("30000000"),
    )
    references = []
    for band, dollar_volume in enumerate(dollar_volumes):
        for member in range(12):
            ticker = f"B{band}-{member:02d}"
            share_class_figi = (
                "SC0-00" if ticker == "B0-01" else f"SC{band}-{member:02d}"
            )
            references.append(selection_reference(ticker, share_class_figi))
    references.append(selection_reference(
        "META", "", active=False, market="otc", locale="ca", type="ETF",
        currency_name="cad", primary_exchange="OTC", composite_figi="",
    ))

    sessions = []
    first = date(2024, 8, 1)
    for day in range(60):
        rows = []
        for band, dollar_volume in enumerate(dollar_volumes):
            for member in range(12):
                ticker = f"B{band}-{member:02d}"
                close = vwap = Decimal("50")
                volume = dollar_volume / close
                if ticker == "B4-02" and day == 0:
                    vwap = Decimal("NaN")
                if ticker == "B4-03" and day < 4:
                    volume = Decimal(0)
                if ticker == "B4-04":
                    close = Decimal((day * 17) % 60 + 1)
                    vwap = Decimal(1)
                    volume = close * Decimal("1000000")
                rows.append(DailyRow(ticker, close, volume, vwap))
        sessions.append((
            date.fromordinal(first.toordinal() + day), tuple(rows),
        ))

    selection = select_candidates(
        policy, tuple(reversed(references)), tuple(reversed(sessions)),
    )
    repeated = select_candidates(policy, tuple(references), tuple(sessions))
    assert selection == repeated
    assert selection.candidates == tuple(sorted(
        selection.candidates, key=lambda item: item.reference.ticker,
    ))
    assert len(selection.master) == 55
    assert all(isinstance(item, Candidate) for item in selection.master)

    candidates = {
        item.reference.ticker: item for item in selection.candidates
    }
    assert candidates["B4-02"].observed == 59
    assert candidates["B4-02"].coverage == Decimal(59) / Decimal(60)
    assert candidates["B4-02"].median_close == Decimal("50")
    assert candidates["B4-02"].median_dollar_volume == Decimal("30000000")
    assert candidates["B4-03"].rejection_reasons == (
        "coverage-below-minimum",
    )
    assert candidates["B4-04"].median_close == Decimal("30.5")
    assert candidates["B4-04"].median_dollar_volume == Decimal("30500000")
    assert candidates["B0-01"].rejection_reasons == (
        "duplicate-share-class-figi",
    )
    assert candidates["META"].rejection_reasons == (
        "inactive", "market-not-stocks", "locale-not-us",
        "type-not-common-stock", "currency-not-usd", "exchange-not-listed",
        "missing-composite-figi", "missing-share-class-figi",
        "coverage-below-minimum", "median-close-below-minimum",
        "median-dollar-volume-below-minimum",
    )

    assert [item.stratum for item in selection.master[:5]] == [1, 2, 3, 4, 5]
    for size in policy.cohort_sizes:
        cohort = selection.master[:size]
        assert len(cohort) == size
        counts = [
            sum(item.stratum == stratum for item in cohort)
            for stratum in range(1, 6)
        ]
        assert max(counts) - min(counts) <= 1
    assert tuple(item.master_rank for item in selection.master) == tuple(
        range(55),
    )
    assert len({item.reference.ticker for item in selection.master}) == 55
    assert "B0-01" not in {
        item.reference.ticker for item in selection.master
    }

    for stratum in range(1, 6):
        members = sorted(
            (
                item for item in selection.candidates
                if item.stratum == stratum
            ),
            key=lambda item: item.within_stratum_rank,
        )
        expected = sorted(
            members,
            key=lambda item: (
                hashlib.sha256(
                    (
                        policy.selection_seed + "\0" +
                        item.reference.share_class_figi
                    ).encode()
                ).digest(),
                item.reference.composite_figi,
                item.reference.ticker,
            ),
        )
        assert members == expected

    raises(
        ValueError, select_candidates, policy, tuple(references[:55]),
        tuple(sessions),
    )


def test_request_gate() -> None:
    clock = FakeClock()
    gate = request_gate(5, clock=clock, sleeper=clock.sleep)
    starts = []
    for work in (4.0, 0.0, 0.0, 0.0):
        gate()
        starts.append(clock())
        clock.advance(work)
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(starts, (0.0, 12.2, 24.4, 36.6),
                                    strict=True)
    )
    assert math.isclose(clock.sleeps[0], 8.2)

    failed = FakeClock()
    consumed = request_gate(5, clock=failed, sleeper=failed.sleep)

    def transport_failure() -> None:
        consumed()
        raise OSError("offline failure")

    raises(OSError, transport_failure)
    consumed()
    assert math.isclose(failed(), 12.2)

    def reached(*_args: object) -> object:
        raise AssertionError("zero-rate gate read the clock or sleeper")

    request_gate(0, clock=reached, sleeper=reached)()
    for invalid in (True, False, -1, 1.0, "5", None, 61):
        error = raises(ValueError, request_gate, invalid)
        assert "fake-secret" not in str(error)
    raises(TypeError, gate, "fake-secret")

    for invalid_time in (math.nan, math.inf, -math.inf):
        invalid_clock = request_gate(
            5, clock=lambda value=invalid_time: value,
        )
        error = raises(ValueError, invalid_clock)
        assert "fake-secret" not in str(error)

    backward = FakeClock()
    guarded = request_gate(5, clock=backward, sleeper=backward.sleep)
    guarded()
    backward.advance(-1.0)
    error = raises(ValueError, guarded)
    assert "fake-secret" not in str(error)

    for fraction in (0.0, 0.5):
        short = FakeClock()

        def undersleep(seconds: float, share: float = fraction) -> None:
            short.advance(seconds * share)

        guarded = request_gate(5, clock=short, sleeper=undersleep)
        guarded()
        error = raises(ValueError, guarded)
        assert "fake-secret" not in str(error)


def manifest_value() -> dict[str, object]:
    return {
        "adjusted": True,
        "declared_on": "2026-07-23",
        "eligibility_date": "2024-07-22",
        "end": "2026-07-21",
        "interval_minutes": 30,
        "purpose": "Offline universe test",
        "schema": 1,
        "series": [
            {"stratum": "generic", "ticker": "AAPL"},
            {"stratum": "generic", "ticker": "MSFT"},
        ],
        "session": "regular",
        "start": "2024-07-22",
    }


def write_manifest(path: Path,
                   value: dict[str, object] | None = None) -> None:
    path.write_text(
        json.dumps(manifest_value() if value is None else value) + "\n",
        encoding="ascii",
    )


def assert_manifest_error(path: Path) -> None:
    raises((OSError, ValueError), UniverseManifest.read, path)
    output, report = path.parent / "new-output", path.parent / "new-report.json"

    def reached(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("key or requester reached for invalid input")

    with patch("tools.fetch_universe.api_key", side_effect=reached):
        raises((OSError, ValueError), fetch_universe, path, output, report,
               requester=reached)
    assert not os.path.lexists(output) and not os.path.lexists(report)


def test_manifest_contract(directory: Path) -> None:
    path = directory / "manifest.json"
    write_manifest(path)
    manifest = UniverseManifest.read(path)
    assert manifest == UniverseManifest(
        1, "Offline universe test", date(2026, 7, 23), date(2024, 7, 22),
        date(2024, 7, 22), date(2026, 7, 21), 30, True, "regular",
        (SeriesSpec("AAPL", "generic"), SeriesSpec("MSFT", "generic")),
    )
    assert manifest.series[0].stratum == manifest.series[1].stratum

    top = set(manifest_value())
    for field in top:
        value = manifest_value()
        del value[field]
        write_manifest(path, value)
        assert_manifest_error(path)
    write_manifest(path, manifest_value() | {"extra": True})
    assert_manifest_error(path)

    write_manifest(path)
    raw = path.read_text(encoding="ascii")
    path.write_text(
        raw.replace('"schema": 1', '"schema": 1, "schema": 1', 1),
        encoding="ascii",
    )
    assert_manifest_error(path)
    path.write_text(
        raw.replace(
            '"ticker": "AAPL"',
            '"ticker": "AAPL", "ticker": "AAPL"',
            1,
        ),
        encoding="ascii",
    )
    assert_manifest_error(path)

    for field in ("ticker", "stratum"):
        value = manifest_value()
        del value["series"][0][field]  # type: ignore[index]
        write_manifest(path, value)
        assert_manifest_error(path)
    value = manifest_value()
    value["series"][0]["extra"] = True  # type: ignore[index]
    write_manifest(path, value)
    assert_manifest_error(path)

    invalid = (
        ("schema", True), ("schema", 2), ("purpose", ""), ("purpose", " "),
        ("declared_on", "2024-07-21"), ("declared_on", "not-a-date"),
        ("eligibility_date", "2024-07-21"), ("start", "2024-07-23"),
        ("end", "2024-07-21"), ("interval_minutes", 0),
        ("interval_minutes", 60), ("interval_minutes", True),
        ("interval_minutes", 30.0), ("adjusted", 1),
        ("adjusted", "true"), ("session", "extended"),
        ("session", ""),
    )
    for field, replacement in invalid:
        value = manifest_value()
        value[field] = replacement
        write_manifest(path, value)
        assert_manifest_error(path)

    invalid_series: tuple[object, ...] = (
        [], {}, [{"ticker": "aapl", "stratum": "generic"}],
        [{"ticker": "../AAPL", "stratum": "generic"}],
        [{"ticker": "AAPL", "stratum": ""}],
        [{"ticker": "AAPL", "stratum": " "}],
        [
            {"ticker": "AAPL", "stratum": "one"},
            {"ticker": "AAPL", "stratum": "two"},
        ],
    )
    for series in invalid_series:
        value = manifest_value()
        value["series"] = series
        write_manifest(path, value)
        assert_manifest_error(path)
    for ticker in (".", "..", "-", ".-"):
        value = manifest_value()
        value["series"] = [{"ticker": ticker, "stratum": "generic"}]
        write_manifest(path, value)
        assert_manifest_error(path)

    value = manifest_value()
    value["series"] = [
        {"ticker": "BRK.B", "stratum": "generic"},
        {"ticker": "BF-B", "stratum": "generic"},
    ]
    write_manifest(path, value)
    assert [item.ticker for item in UniverseManifest.read(path).series] == [
        "BRK.B", "BF-B",
    ]

    missing = directory / "missing.json"
    assert_manifest_error(missing)
    malformed = directory / "malformed.json"
    malformed.write_text("{", encoding="ascii")
    assert_manifest_error(malformed)
    linked = directory / "linked.json"
    linked.symlink_to(path)
    assert_manifest_error(linked)
    nonregular = directory / "manifest-directory"
    nonregular.mkdir()
    assert_manifest_error(nonregular)

    tracked = UniverseManifest.read(ROOT / "universes/liquid-common-11.json")
    assert len(tracked.series) == 11
    assert len({item.ticker for item in tracked.series}) == 11
    assert len({item.stratum for item in tracked.series}) == 11


def test_target_rejections(directory: Path) -> None:
    manifest = directory / "manifest.json"
    write_manifest(manifest)

    def reached(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("key or requester reached for invalid targets")

    def rejected(output: Path, report: Path) -> None:
        report_existed = os.path.lexists(report)
        with patch("tools.fetch_universe.api_key", side_effect=reached):
            raises((OSError, ValueError), fetch_universe, manifest, output,
                   report, requester=reached)
        assert os.path.lexists(report) == report_existed

    existing_output = directory / "existing-output"
    existing_output.mkdir()
    rejected(existing_output, directory / "existing-output-report.json")

    target_directory = directory / "target-directory"
    target_directory.mkdir()
    output_link = directory / "output-link"
    output_link.symlink_to(target_directory, target_is_directory=True)
    rejected(output_link, directory / "output-link-report.json")

    broken_output = directory / "broken-output"
    broken_output.symlink_to(directory / "missing-output")
    rejected(broken_output, directory / "broken-output-report.json")

    existing_report = directory / "existing-report.json"
    existing_report.write_text("{}\n", encoding="ascii")
    rejected(directory / "for-existing-report", existing_report)

    report_target = directory / "report-target.json"
    report_target.write_text("{}\n", encoding="ascii")
    report_link = directory / "report-link.json"
    report_link.symlink_to(report_target)
    rejected(directory / "for-report-link", report_link)

    broken_report = directory / "broken-report.json"
    broken_report.symlink_to(directory / "missing-report.json")
    rejected(directory / "for-broken-report", broken_report)

    equal = directory / "equal"
    rejected(equal, equal)
    output_parent = directory / "output-parent"
    rejected(output_parent, output_parent / "report.json")
    report_parent = directory / "report-parent"
    rejected(report_parent / "output", report_parent)

    normalized_output = directory / "normalized-existing-output"
    normalized_output.mkdir()
    rejected(
        directory / "missing-output-parent" / ".." / normalized_output.name,
        directory / "normalized-output-report.json",
    )

    normalized_report = directory / "normalized-existing-report.json"
    normalized_report.write_text("{}\n", encoding="ascii")
    rejected(
        directory / "normalized-report-output",
        directory / "missing-report-parent" / ".." / normalized_report.name,
    )
    assert normalized_report.read_text(encoding="ascii") == "{}\n"

    normalized_broken_output = directory / "normalized-broken-output"
    normalized_broken_output.symlink_to(directory / "missing-output-target")
    rejected(
        directory / "missing-broken-output-parent" / ".." /
        normalized_broken_output.name,
        directory / "normalized-broken-output-report.json",
    )
    assert normalized_broken_output.is_symlink()

    normalized_broken_report = directory / "normalized-broken-report.json"
    normalized_broken_report.symlink_to(directory / "missing-report-target")
    rejected(
        directory / "normalized-broken-report-output",
        directory / "missing-broken-report-parent" / ".." /
        normalized_broken_report.name,
    )
    assert normalized_broken_report.is_symlink()

    collision_output = directory / "late-collision-output"
    collision_report = directory / "late-collision-report.json"
    real_read = UniverseManifest.read.__func__

    def create_collision(
        cls: type[UniverseManifest], path: Path,
    ) -> UniverseManifest:
        value = real_read(cls, path)
        collision_output.mkdir()
        (collision_output / "aapl-30m.csv").write_text(
            "collision", encoding="ascii",
        )
        return value

    with patch.object(
        UniverseManifest, "read", classmethod(create_collision),
    ), patch("tools.fetch_universe.api_key", side_effect=reached):
        error = raises(
            (OSError, ValueError), fetch_universe, manifest,
            collision_output, collision_report, requester=reached,
        )
    assert "universe CSV" in str(error)
    assert not os.path.lexists(collision_report)


def test_rate_validation(directory: Path) -> None:
    manifest = directory / "rate-manifest.json"
    write_manifest(manifest)

    def reached(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("credential, clock, sleeper, or request reached")

    for index, rate in enumerate((True, False, -1, 1.0, "5", None, 61)):
        output = directory / f"rate-output-{index}"
        report = directory / f"rate-report-{index}.json"
        with patch("tools.fetch_universe.api_key", side_effect=reached):
            raises(
                ValueError, fetch_universe, manifest, output, report,
                requests_per_minute=rate, clock=reached, sleeper=reached,
                requester=reached,
            )
        assert not os.path.lexists(output) and not os.path.lexists(report)

    assert universe_args(["manifest", "output", "report"]).requests_per_minute == 0
    assert universe_args([
        "manifest", "output", "report", "--requests-per-minute", "5",
    ]).requests_per_minute == 5


def fake_requester(requested: list[str],
                   on_request: object | None = None) -> object:
    def request(url: str) -> dict[str, object]:
        requested.append(url)
        if on_request is not None:
            on_request(url)  # type: ignore[operator]
        parts = urlsplit(url)
        if parts.path.startswith("/v3/reference/tickers/"):
            ticker = unquote(parts.path.rsplit("/", 1)[1])
            return {
                "status": "OK",
                "results": {
                    "ticker": ticker, "active": True, "market": "stocks",
                    "locale": "us", "type": "CS", "currency_name": "usd",
                },
            }
        ticker = unquote(parts.path.split("/ticker/", 1)[1].split("/", 1)[0])
        return {
            "status": "OK", "ticker": ticker,
            "results": [aggregate("2024-07-22T13:30:00+00:00", 100.0)],
        }
    return request


def test_universe_fetch(directory: Path) -> None:
    manifest_path = directory / "manifest.json"
    write_manifest(manifest_path)
    frozen_bytes = manifest_path.read_bytes()
    output = directory / "output"
    report_path = directory / "report.json"
    resolved_output = output.resolve(strict=False)
    resolved_report = report_path.resolve(strict=False)
    requested: list[str] = []
    starts: list[float] = []
    clock = FakeClock()
    events: list[tuple[str, Path]] = []
    real_read = UniverseManifest.read.__func__

    def read_frozen(cls: type[UniverseManifest], path: Path) -> UniverseManifest:
        assert path != manifest_path and path.read_bytes() == frozen_bytes
        events.append(("parse", path))
        return real_read(cls, path)

    def checked_hash(path: Path) -> str:
        events.append(("hash", path))
        return file_sha256(path)

    def checked_write(path: Path, value: dict[str, object]) -> None:
        csvs = tuple(
            resolved_output / f"{ticker}-30m.csv"
            for ticker in ("aapl", "msft")
        )
        assert events[-3:] == [
            ("hash", manifest_path),
            ("hash", csvs[0]),
            ("hash", csvs[1]),
        ]
        assert path == resolved_report
        events.append(("report", path))
        write_json(path, value)

    with patch.object(UniverseManifest, "read", classmethod(read_frozen)), \
         patch("tools.fetch_universe.file_sha256", side_effect=checked_hash), \
         patch("tools.fetch_universe.write_json", side_effect=checked_write):
        report = fetch_universe(
            manifest_path, output, report_path, key="fake-secret",
            requester=fake_requester(
                requested, lambda _url: starts.append(clock()),
            ),
            requests_per_minute=5, clock=clock, sleeper=clock.sleep,
        )

    assert set(report) == {
        "schema", "purpose", "declared_on", "eligibility_date", "start", "end",
        "interval_minutes", "adjusted", "session", "manifest", "series",
    }
    assert report["manifest"] == {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(frozen_bytes).hexdigest(),
    }
    series = report["series"]
    assert isinstance(series, list)
    assert [item["ticker"] for item in series] == ["AAPL", "MSFT"]
    assert [item["stratum"] for item in series] == ["generic", "generic"]
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert events[-1] == ("report", resolved_report)

    for item in series:
        ticker = item["ticker"]
        reference = item["reference"]
        aggregate_contract = item["aggregate"]
        assert reference == {
            "path": f"/v3/reference/tickers/{ticker}",
            "query": {"date": "2024-07-22"},
            "active": True, "market": "stocks", "locale": "us",
            "type": "CS", "currency_name": "usd",
        }
        assert aggregate_contract["query"] == {
            "adjusted": "true", "sort": "asc", "limit": "50000",
        }
        assert aggregate_contract["path"] == (
            f"/v2/aggs/ticker/{ticker}/range/30/minute/"
            "2024-07-22/2026-07-21"
        )
        csv = item["csv"]
        path = Path(csv["path"])
        assert csv == {
            "path": str(resolved_output / f"{ticker.lower()}-30m.csv"),
            "rows": 1, "sessions": 1, "source_rows": 1,
            "sha256": file_sha256(path),
        }
        assert len(read_csv(path)) == FEATURE_COUNT

        actual = next(
            url for url in requested
            if urlsplit(url).path.startswith(f"/v2/aggs/ticker/{ticker}/")
        )
        parts = urlsplit(actual)
        pairs = parse_qsl(parts.query, keep_blank_values=True)
        assert [value for name, value in pairs if name == "apiKey"] == [
            "fake-secret"
        ]
        sanitized = dict((name, value) for name, value in pairs
                         if name != "apiKey")
        assert parts.path == aggregate_contract["path"]
        assert sanitized == aggregate_contract["query"]

        actual_reference = next(
            url for url in requested
            if urlsplit(url).path == reference["path"]
        )
        reference_parts = urlsplit(actual_reference)
        reference_pairs = parse_qsl(
            reference_parts.query, keep_blank_values=True,
        )
        assert [value for name, value in reference_pairs if name == "apiKey"] == [
            "fake-secret"
        ]
        assert dict(
            (name, value) for name, value in reference_pairs
            if name != "apiKey"
        ) == reference["query"]

    assert len(requested) == 4
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(starts, (0.0, 12.2, 24.4, 36.6),
                                    strict=True)
    )
    rendered = json.dumps(report)
    assert all(secret not in rendered
               for secret in ("apiKey", "MASSIVE_API_KEY", "fake-secret"))


def test_universe_pagination_gate(directory: Path) -> None:
    manifest = directory / "pagination-manifest.json"
    output = directory / "pagination-output"
    report_path = directory / "pagination-report.json"
    write_manifest(manifest)
    clock = FakeClock()
    starts: list[float] = []
    paths: list[str] = []

    def request(url: str) -> dict[str, object]:
        starts.append(clock())
        parts = urlsplit(url)
        paths.append(parts.path)
        if parts.path.startswith("/v3/reference/tickers/"):
            ticker = unquote(parts.path.rsplit("/", 1)[1])
            return {
                "status": "OK",
                "results": {
                    "ticker": ticker, "active": True, "market": "stocks",
                    "locale": "us", "type": "CS", "currency_name": "usd",
                },
            }
        ticker = unquote(parts.path.split("/ticker/", 1)[1].split("/", 1)[0])
        paginated = parts.path.endswith("/page")
        result = aggregate(
            "2024-07-22T14:00:00+00:00" if paginated
            else "2024-07-22T13:30:00+00:00",
            101.0 if paginated else 100.0,
        )
        response: dict[str, object] = {
            "status": "OK", "ticker": ticker, "results": [result],
        }
        if ticker == "AAPL" and not paginated:
            response["next_url"] = (
                "https://api.massive.com/v2/aggs/ticker/AAPL/page?cursor=next"
            )
        return response

    report = fetch_universe(
        manifest, output, report_path, key="fake-secret", requester=request,
        requests_per_minute=5, clock=clock, sleeper=clock.sleep,
    )
    assert paths == [
        "/v3/reference/tickers/AAPL",
        "/v2/aggs/ticker/AAPL/range/30/minute/2024-07-22/2026-07-21",
        "/v2/aggs/ticker/AAPL/page",
        "/v3/reference/tickers/MSFT",
        "/v2/aggs/ticker/MSFT/range/30/minute/2024-07-22/2026-07-21",
    ]
    assert all(
        math.isclose(actual, index * 12.2)
        for index, actual in enumerate(starts)
    )
    assert report["series"][0]["csv"]["source_rows"] == 2
    assert "fake-secret" not in json.dumps(report)


def test_reference_identity(directory: Path) -> None:
    assert reference_url("AAPL", date(2024, 7, 22)) == (
        "https://api.massive.com/v3/reference/tickers/AAPL?date=2024-07-22"
    )
    for ticker in ("../AAPL", ".", "..", "-", ".-"):
        raises(ValueError, reference_url, ticker, date(2024, 7, 22))
    for ticker in ("BRK.B", "BF-B"):
        assert urlsplit(reference_url(ticker, date(2024, 7, 22))).path == (
            f"/v3/reference/tickers/{ticker}"
        )

    manifest = directory / "manifest.json"
    value = manifest_value()
    value["series"] = [{"ticker": "AAPL", "stratum": "generic"}]
    write_manifest(manifest, value)
    valid = {
        "ticker": "AAPL", "active": True, "market": "stocks",
        "locale": "us", "type": "CS", "currency_name": "usd",
    }
    invalid = (
        {"status": "ERROR", "results": valid},
        {"status": "OK", "results": [valid]},
        {"status": "OK", "results": valid | {"ticker": "MSFT"}},
        {"status": "OK", "results": valid | {"active": False}},
        {"status": "OK", "results": valid | {"active": 1}},
        {"status": "OK", "results": valid | {"market": "otc"}},
        {"status": "OK", "results": valid | {"locale": "global"}},
        {"status": "OK", "results": valid | {"type": "ETF"}},
        {"status": "OK", "results": valid | {"currency_name": "eur"}},
    )
    for index, response in enumerate(invalid):
        output = directory / f"identity-output-{index}"
        report = directory / f"identity-report-{index}.json"
        raises(ValueError, fetch_universe, manifest, output, report,
               key="fake-secret", requester=lambda _url, item=response: item)
        assert not os.path.lexists(report)


def test_mutations(directory: Path) -> None:
    manifest = directory / "mutable-manifest.json"
    value = manifest_value()
    value["series"] = [{"ticker": "AAPL", "stratum": "generic"}]
    write_manifest(manifest, value)
    output = directory / "manifest-mutation-output"
    report = directory / "manifest-mutation-report.json"
    mutated = False

    def mutate_manifest(_url: str) -> None:
        nonlocal mutated
        if not mutated:
            manifest.write_text('{"mutated":true}\n', encoding="ascii")
            mutated = True

    raises(ValueError, fetch_universe, manifest, output, report,
           key="fake-secret",
           requester=fake_requester([], mutate_manifest))
    assert mutated and not os.path.lexists(report)

    write_manifest(manifest, value)
    output = directory / "csv-mutation-output"
    report = directory / "csv-mutation-report.json"
    csv = output.resolve(strict=False) / "aapl-30m.csv"
    mutated = False

    def mutate_after_hash(path: Path) -> str:
        nonlocal mutated
        digest = file_sha256(path)
        if path == csv and not mutated:
            path.write_text("mutated\n", encoding="ascii")
            mutated = True
        return digest

    with patch("tools.fetch_universe.file_sha256",
               side_effect=mutate_after_hash):
        raises(ValueError, fetch_universe, manifest, output, report,
               key="fake-secret", requester=fake_requester([]))
    assert mutated and not os.path.lexists(report)


def test_fetch_helpers(initial: str) -> None:
    cycle = lambda _url: {
        "status": "OK", "ticker": "AAPL",
        "results": [aggregate("2026-07-01T13:30:00+00:00", 100.0)],
        "next_url": initial,
    }
    raises(ValueError, fetch_bars, initial, "secret", "AAPL", cycle)

    pages = iter((
        {
            "status": "OK", "ticker": "AAPL",
            "results": [aggregate("2026-07-01T13:30:00+00:00", 100.0)],
            "next_url": "https://example.com/page",
        },
    ))
    raises(ValueError, fetch_bars, initial, "secret", "AAPL",
           lambda _url: next(pages))
    raises(ValueError, fetch_bars, initial, "secret", "AAPL",
           lambda _url: {"status": "OK", "ticker": "AAPL",
                         "results": [{"t": "invalid"}]})
    raises(ValueError, fetch_bars, initial, "secret", "AAPL",
           lambda _url: {"status": "ERROR", "results": []})


def test_existing_downloader(directory: Path) -> None:
    env = directory / ".env"
    env.write_text("MASSIVE_API_KEY=file-key\n", encoding="ascii")
    prior = os.environ.pop("MASSIVE_API_KEY", None)
    try:
        assert api_key(env) == "file-key"
        os.environ["MASSIVE_API_KEY"] = "process-key"
        assert api_key(env) == "process-key"
    finally:
        if prior is None:
            os.environ.pop("MASSIVE_API_KEY", None)
        else:
            os.environ["MASSIVE_API_KEY"] = prior

    initial = aggregate_url("AAPL", date(2026, 7, 1),
                            date(2026, 7, 2), 30, True)
    pages = iter((
        {"status": "OK", "ticker": "AAPL",
         "results": [aggregate("2026-07-01T13:00:00+00:00", 100.0),
                     aggregate("2026-07-01T13:30:00+00:00", 101.0)],
         "next_url": "https://api.massive.com/page?cursor=next"},
        {"status": "OK", "ticker": "AAPL",
         "results": [aggregate("2026-07-01T14:00:00+00:00", 102.0),
                     aggregate("2026-07-01T20:00:00+00:00", 103.0)]},
    ))
    requested: list[str] = []

    def request(url: str) -> dict[str, object]:
        requested.append(url)
        return next(pages)

    source = fetch_bars(initial, "secret", "AAPL", request)
    bars, sessions = regular_bars(source, 30)
    assert len(source) == 4 and len(bars) == 2 and sessions == 1
    assert all(parse_qs(urlsplit(url).query)["apiKey"] == ["secret"]
               for url in requested)
    raises(ValueError, authorized_url, "https://example.com/page", "secret")

    headers = Message()
    headers["Retry-After"] = "0"
    responses = iter((HTTPError("https://api.massive.com", 429, "rate limited",
                                headers, BytesIO()),
                      BytesIO(b'{"status":"OK"}'),
                      BytesIO(b'{"status":"OK","page":2}')))
    clock = FakeClock()
    gate = request_gate(5, clock=clock, sleeper=clock.sleep)
    starts: list[float] = []

    def urlopen_once(*_args: object, **_kwargs: object) -> BytesIO:
        starts.append(clock())
        response = next(responses)
        if isinstance(response, HTTPError):
            raise response
        return response

    with patch("tools.fetch_massive.urlopen", side_effect=urlopen_once), \
         patch("tools.fetch_massive.time.sleep", side_effect=clock.sleep) as sleep:
        assert request_json(
            "https://api.massive.com", before_request=gate,
        ) == {"status": "OK"}
        assert request_json(
            "https://api.massive.com/page", before_request=gate,
        ) == {"status": "OK", "page": 2}
        sleep.assert_called_once_with(1.0)
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(starts, (0.0, 12.2, 24.4), strict=True)
    )

    path = directory / "bars.csv"
    write_csv(path, bars)
    assert len(read_csv(path)) == len(bars) * FEATURE_COUNT
    test_fetch_helpers(initial)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="compose-mini-massive-") as name:
        directory = Path(name)
        test_request_gate()
        test_selection_policy(directory)
        test_pure_selection()
        test_existing_downloader(directory)
        test_manifest_contract(directory)
        test_target_rejections(directory)
        test_rate_validation(directory)
        test_universe_fetch(directory)
        test_universe_pagination_gate(directory)
        test_reference_identity(directory)
        test_mutations(directory)
    print("Massive downloader and universe tests passed")


if __name__ == "__main__":
    main()
