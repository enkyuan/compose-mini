#!/usr/bin/env python3
"""Verify Massive downloads and strict universe fetching without network."""

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal, localcontext
from email.message import Message
from functools import cache
from io import BytesIO, StringIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import (
    parse_qsl, parse_qs, unquote, urlencode, urlsplit, urlunsplit,
)
from unittest.mock import patch
import hashlib
import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tools.fetch_benchmark as benchmark_fetch
from tools.data_v1 import FEATURE_COUNT, read_bars, read_csv
from tools.apply_universe_coverage_overlay import (
    apply_overlay, replacement_candidate, revised_manifest,
)
from tools.fetch_benchmark import (
    END as BENCHMARK_END, START as BENCHMARK_START, fetch_benchmark,
)
from tools.fetch_massive import (
    Bar, Requester, aggregate_url, api_key, authorized_url, fetch_bars,
    regular_bars, request_gate, request_json, scan_regular_bars,
    session_grid_audit, write_csv,
)
from tools.fetch_universe import (
    SeriesSpec, UniverseManifest, fetch_universe, parse_args as universe_args,
    reference_url,
)
from tools.files import (
    ExclusiveTemp, file_sha256, rename_noreplace, write_json,
    write_json_exclusive,
)
from tools.session_calendar import (
    DEFAULT_CALENDAR, SessionCalendar, expected_bins,
)
from tools.select_universe import (
    Candidate, DailyRow, Reference, SelectionPolicy, SourceArchive,
    SourceBundle, _candidate_value, _canonical_sha256, _decimal_text,
    _manifest_value, _member_value, _source_archive_value,
    _source_binding_value, _transport, daily_summary_url, fetch_sources,
    main as selection_main, parse_args as selection_args,
    reference_universe_url, select_candidates, select_universe,
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


def calendar_value() -> dict[str, object]:
    return json.loads(DEFAULT_CALENDAR.read_text(encoding="utf-8"))


def test_session_calendar(directory: Path) -> None:
    calendar = SessionCalendar.read(DEFAULT_CALENDAR)
    assert tuple(map(str, calendar.closed_dates)) == (
        "2024-09-02", "2024-11-28", "2024-12-25", "2025-01-01",
        "2025-01-09",
        "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
        "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27",
        "2025-12-25", "2026-01-01", "2026-01-19", "2026-02-16",
        "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03",
    )
    assert tuple((str(day), close) for day, close in calendar.early_closes) == (
        ("2024-11-29", 780), ("2024-12-24", 780),
        ("2025-07-03", 780), ("2025-11-28", 780),
        ("2025-12-24", 780),
    )
    assert calendar.session(date(2024, 11, 1)) == (570, 960)
    assert calendar.session(date(2024, 11, 29)) == (570, 780)
    assert calendar.session(date(2025, 1, 9)) is None
    assert calendar.session(date(2026, 7, 4)) is None
    raises(ValueError, calendar.session, date(2024, 7, 21))

    day, sessions, bins = calendar.start, 0, 0
    while day <= calendar.end:
        bounds = calendar.session(day)
        if bounds is not None:
            sessions += 1
            bins += (bounds[1] - bounds[0]) // 30
        day += timedelta(days=1)
    assert (sessions, bins) == (501, 6483)

    path = directory / "session-calendar.json"
    for change in (
        lambda value: value.update({"timezone": "UTC"}),
        lambda value: value.update({"open_minute": True}),
        lambda value: value.update({"start": "2024-11-01T00:00:00"}),
        lambda value: value.update({"venues": ["XNYS", "XNAS"]}),
        lambda value: value["closed_dates"].reverse(),
        lambda value: value["early_closes"].update({"2025-01-09": 780}),
        lambda value: value["early_closes"].update({"2025-07-05": 780}),
        lambda value: value.update({"sources": []}),
        lambda value: value.update({"sources": [[]]}),
    ):
        value = calendar_value()
        change(value)
        write_json(path, value)
        raises(ValueError, SessionCalendar.read, path)
    path.write_text('{"schema":1,"schema":1}\n', encoding="ascii")
    raises(ValueError, SessionCalendar.read, path)


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


def decimal_ratio(numerator: int, denominator: int) -> Decimal:
    with localcontext() as context:
        context.prec = 64
        return Decimal(numerator) / Decimal(denominator)


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
                if ticker == "B4-05":
                    volume = Decimal("1999999.9999999998")
                    vwap = Decimal("12.5")
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
    assert candidates["B4-02"].coverage == decimal_ratio(59, 60)
    assert candidates["B4-02"].median_close == Decimal("50")
    assert candidates["B4-02"].median_dollar_volume == Decimal("30000000")
    assert candidates["B4-03"].rejection_reasons == (
        "coverage-below-minimum",
    )
    assert candidates["B4-04"].median_close == Decimal("30.5")
    assert candidates["B4-04"].median_dollar_volume == Decimal("30500000")
    assert candidates["B4-05"].median_dollar_volume == Decimal(
        "24999999.99999999750"
    )
    assert candidates["B4-05"].rejection_reasons == (
        "median-dollar-volume-below-minimum",
    )
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

    missing = select_candidates(
        policy, tuple(references),
        (*sessions, (date(2024, 9, 30), ())),
    )
    missing_candidate = next(
        item for item in missing.candidates
        if item.reference.ticker == "B0-00"
    )
    assert missing_candidate.observed == 60
    assert missing_candidate.coverage == decimal_ratio(60, 61)

    with localcontext() as context:
        context.prec = 8
        low_precision = select_candidates(
            policy, tuple(references), tuple(sessions),
        )
    with localcontext() as context:
        context.prec = 60
        high_precision = select_candidates(
            policy, tuple(references), tuple(sessions),
        )
    assert low_precision == high_precision == repeated

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


def source_reference(ticker: str) -> dict[str, object]:
    return {
        "active": True,
        "composite_figi": f"BBG{ticker}",
        "currency_name": "usd",
        "locale": "us",
        "market": "stocks",
        "primary_exchange": "XNYS",
        "share_class_figi": f"SC{ticker}",
        "ticker": ticker,
        "type": "CS",
    }


def source_daily(ticker: str, close: object = 10.25) -> dict[str, object]:
    return {"T": ticker, "c": close, "v": 2_000_000, "vw": 10}


def weekdays(start: date, end: date) -> tuple[date, ...]:
    days = []
    while start <= end:
        if start.weekday() < 5:
            days.append(start)
        start += timedelta(days=1)
    return tuple(days)


def test_universe_sources() -> None:
    policy = SelectionPolicy.read(
        ROOT / "universes/liquid-common-ladder.example.json",
    )
    formation_days = weekdays(policy.formation_start, policy.formation_end)
    empty_day = formation_days[0]
    filtered_day = formation_days[1]
    next_page = (
        "https://api.massive.com/v3/reference/tickers"
        "?apiKey=provider-value&cursor=next"
    )
    requested: list[str] = []
    gated: list[int] = []

    def before_request() -> None:
        gated.append(len(requested))

    def requester(url: str) -> dict[str, object]:
        gated_index = len(requested)
        requested.append(url)
        assert gated[-1] == gated_index
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query))
        assert query.pop("apiKey") == "source-secret"
        if parts.path == "/v3/reference/tickers":
            return (
                {
                    "status": "OK",
                    "results": [
                        source_reference("AAPL"),
                        source_reference("ECGw"),
                        source_reference("MSFT"),
                    ],
                    "next_url": next_page,
                }
                if "cursor" not in query
                else {
                    "status": "OK",
                    "results": [
                        source_reference("SPY") |
                        {"share_class_figi": None}
                    ],
                }
            )
        day = date.fromisoformat(parts.path.rsplit("/", 1)[-1])
        assert query == {"adjusted": "false", "include_otc": "false"}
        if day == empty_day:
            return {"status": "OK", "resultsCount": 0}
        if day == filtered_day:
            return {
                "status": "OK",
                "resultsCount": 2,
                "results": [{"T": "ECGw"}, {"T": "ZZZZ"}],
            }
        return {
            "status": "OK",
            "resultsCount": 3,
            "results": [
                {"T": "ZZZZ"},
                source_daily("MSFT", 20.5),
                source_daily("AAPL", 0.1 + 0.2),
            ],
        }

    bundle = fetch_sources(
        policy, "source-secret", requester, before_request,
    )
    assert isinstance(bundle, SourceBundle)
    assert all(isinstance(item, SourceArchive) for item in bundle.archives)
    assert tuple(item.ticker for item in bundle.references) == (
        "AAPL", "MSFT", "SPY",
    )
    assert bundle.archives[0].records[1]["ticker"] == "ECGw"
    assert bundle.references[2].share_class_figi is None
    assert len(bundle.sessions) == len(formation_days) - 1
    assert empty_day not in {day for day, _ in bundle.sessions}
    assert bundle.sessions[0] == (filtered_day, ())
    assert all(
        tuple(row.ticker for row in rows) == ("AAPL", "MSFT")
        for _, rows in bundle.sessions[1:]
    )
    assert bundle.sessions[1][1][0].close == Decimal(
        "0.30000000000000004"
    )
    assert len(requested) == len(gated) == len(formation_days) + 2
    assert tuple(
        date.fromisoformat(urlsplit(url).path.rsplit("/", 1)[-1])
        for url in requested[2:]
    ) == formation_days

    first = bundle.archives[0]
    assert first.name == "tickers-0001"
    assert first.request == {
        "path": "/v3/reference/tickers",
        "query": {
            "active": "true",
            "date": "2024-10-31",
            "limit": "1000",
            "market": "stocks",
            "order": "asc",
            "sort": "ticker",
            "type": "CS",
        },
    }
    assert bundle.archives[1].name == "tickers-0002"
    assert bundle.archives[2].name == f"daily-{empty_day}"
    assert bundle.archives[2].records == ()
    assert bundle.archives[3].name == f"daily-{filtered_day}"
    assert bundle.archives[3].records == ()
    populated = bundle.archives[4]
    assert populated.records == (
        {
            "ticker": "AAPL",
            "c": "0.30000000000000004",
            "v": "2000000",
            "vw": "10",
        },
        {"ticker": "MSFT", "c": "20.5", "v": "2000000", "vw": "10"},
    )
    for archive in bundle.archives:
        assert tuple(archive.request["query"]) == tuple(
            sorted(archive.request["query"])
        )
        value = {
            "name": archive.name,
            "records": archive.records,
            "request": archive.request,
            "schema": 1,
        }
        serialized = json.dumps(value, sort_keys=True)
        assert "apiKey" not in serialized
        assert "provider-value" not in serialized
        assert "source-secret" not in serialized


def test_universe_source_rejections() -> None:
    tracked = SelectionPolicy.read(
        ROOT / "universes/liquid-common-ladder.example.json",
    )
    policy = replace(
        tracked,
        anchor_date=tracked.formation_end,
        formation_start=tracked.formation_end,
        minimum_formation_sessions=1,
    )
    daily_ok = {
        "status": "OK",
        "results": [source_daily("AAPL")],
        "resultsCount": 1,
    }

    def attempt(
        reference: object,
        daily: object = daily_ok,
    ) -> BaseException:
        def requester(url: str) -> object:
            return (
                reference
                if urlsplit(url).path == "/v3/reference/tickers"
                else daily
            )

        return raises(
            ValueError, fetch_sources, policy, "source-secret",
            requester, lambda: None,
        )

    for payload in (
        [],
        {"status": "ERROR", "results": []},
        {"status": "OK", "results": {}},
        {"status": "OK", "results": []},
        {"status": "OK", "results": [{}]},
        *(
            {
                "status": "OK",
                "results": [source_reference(ticker)],
            }
            for ticker in ("/", "@@@", "\0")
        ),
        {
            "status": "OK",
            "results": [source_reference("AAPL") | {"active": 1}],
        },
        {
            "status": "OK",
            "results": [
                source_reference("MSFT"),
                source_reference("AAPL"),
            ],
        },
        {
            "status": "OK",
            "results": [
                source_reference("AAPL"),
                source_reference("AAPL"),
            ],
        },
        {
            "status": "OK",
            "results": [source_reference("AAPL")],
            "next_url": 1,
        },
    ):
        attempt(payload)

    initial = reference_universe_url(policy.anchor_date)
    attempt({
        "status": "OK",
        "results": [source_reference("AAPL")],
        "next_url": "https://example.com/v3/reference/tickers",
    })
    parts = urlsplit(initial)
    cycle = urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        urlencode(
            tuple(reversed(parse_qsl(parts.query))) +
            (("apiKey", "different-provider-value"),)
        ),
        "",
    ))
    attempt({
        "status": "OK",
        "results": [source_reference("AAPL")],
        "next_url": cycle,
    })

    reference_ok = {
        "status": "OK",
        "results": [source_reference("AAPL")],
    }
    for payload in (
        [],
        {"status": "ERROR", "results": []},
        {"status": "OK", "results": {}},
        {"status": "OK", "results": [{}]},
        *(
            {"status": "OK", "results": [{"T": ticker}]}
            for ticker in ("/", "@@@", "\0")
        ),
        {
            "status": "OK",
            "results": [source_daily("AAPL"), source_daily("AAPL")],
        },
        {
            "status": "OK",
            "results": [{"T": "ZZZZ"}, {"T": "ZZZZ"}],
        },
        {
            "status": "OK",
            "results": [{"T": "AAPL"}],
        },
        {
            "status": "OK",
            "results": [source_daily("AAPL", float("nan"))],
        },
        {
            "status": "OK",
            "results": [
                source_daily("AAPL") | {"c": True}
            ],
        },
        {"status": "OK", "results": [], "resultsCount": 1},
        {"status": "OK", "results": [], "resultsCount": False},
        {"status": "OK", "resultsCount": 0},
    ):
        attempt(reference_ok, payload)

    def leaking(url: str) -> object:
        raise OSError(url)

    error = raises(
        ValueError, fetch_sources, policy, "source-secret",
        leaking, lambda: None,
    )
    assert "source-secret" not in str(error)
    for invalid_key in ("", " ", "two words"):
        raises(
            ValueError, fetch_sources, policy, invalid_key,
            lambda _url: reference_ok, lambda: None,
        )

    assert reference_universe_url(policy.anchor_date) == (
        "https://api.massive.com/v3/reference/tickers"
        "?active=true&date=2024-10-31&limit=1000&market=stocks"
        "&order=asc&sort=ticker&type=CS"
    )
    assert daily_summary_url(policy.formation_end) == (
        "https://api.massive.com/v2/aggs/grouped/locale/us/market/stocks"
        "/2024-10-31?adjusted=false&include_otc=false"
    )


def test_universe_values() -> None:
    policy = SelectionPolicy.read(
        ROOT / "universes/liquid-common-ladder.example.json",
    )
    selected = Candidate(
        selection_reference("AAPL", "SCAAPL"),
        60,
        Decimal("0.9500"),
        Decimal("5.000"),
        Decimal("25000000.000"),
        (),
        liquidity_rank=0,
        stratum=1,
        within_stratum_rank=0,
        master_rank=0,
    )
    assert _candidate_value(selected, "AAPL") == {
        "ticker": "AAPL",
        "active": True,
        "market": "stocks",
        "locale": "us",
        "type": "CS",
        "currency_name": "usd",
        "primary_exchange": "XNYS",
        "composite_figi": "BBGAAPL",
        "share_class_figi": "SCAAPL",
        "observed": 60,
        "coverage": "0.95",
        "median_close_usd": "5",
        "median_dollar_volume_usd": "25000000",
        "rejection_reasons": [],
        "decision": "selected",
        "share_class_representative": "AAPL",
        "liquidity_rank": 0,
        "stratum": 1,
        "within_stratum_rank": 0,
        "master_rank": 0,
    }
    eligible = replace(selected, master_rank=None)
    assert _candidate_value(
        eligible, "AAPL",
    )["decision"] == "eligible-not-selected"
    duplicate = replace(
        selected,
        reference=selection_reference("AAPL.A", "SCAAPL"),
        rejection_reasons=("duplicate-share-class-figi",),
        liquidity_rank=None,
        stratum=None,
        within_stratum_rank=None,
        master_rank=None,
    )
    duplicate_value = _candidate_value(duplicate, "AAPL")
    assert duplicate_value["decision"] == "rejected"
    assert duplicate_value["share_class_representative"] == "AAPL"
    rejected = replace(
        duplicate, rejection_reasons=("inactive",),
    )
    assert _candidate_value(
        rejected, None,
    )["share_class_representative"] is None

    member = _member_value(selected)
    second = _member_value(replace(
        selected,
        reference=selection_reference("MSFT", "SCMSFT"),
        liquidity_rank=1,
        stratum=2,
        within_stratum_rank=0,
        master_rank=1,
    ))
    assert member == {
        "ticker": "AAPL",
        "composite_figi": "BBGAAPL",
        "share_class_figi": "SCAAPL",
        "stratum": 1,
    }
    members = [member, second]
    assert [item["ticker"] for item in members[:1]] == ["AAPL"]
    assert [item["ticker"] for item in members[:2]] == ["AAPL", "MSFT"]
    manifest = _manifest_value(policy, members)
    assert manifest == {
        "schema": 1,
        "purpose": policy.purpose,
        "declared_on": "2026-07-24",
        "eligibility_date": "2024-10-31",
        "start": "2024-11-01",
        "end": "2026-07-21",
        "interval_minutes": 30,
        "adjusted": True,
        "session": "regular",
        "series": [
            {"stratum": "liquidity-1", "ticker": "AAPL"},
            {"stratum": "liquidity-2", "ticker": "MSFT"},
        ],
    }
    for size in (1, 2):
        assert _manifest_value(
            policy, members[:size],
        )["series"] == manifest["series"][:size]

    semantic = {"members": members, "size": 2}
    reordered = {"size": 2, "members": members}
    expected = hashlib.sha256(
        (
            json.dumps(
                semantic, allow_nan=False, indent=2, sort_keys=True,
            ) + "\n"
        ).encode("utf-8")
    ).hexdigest()
    assert _canonical_sha256(semantic) == expected
    assert _canonical_sha256(reordered) == expected
    assert _decimal_text(Decimal("-0.000")) == "0"
    assert _decimal_text(Decimal("1.2300")) == "1.23"
    assert _decimal_text(Decimal("1E+3")) == "1000"
    for nonfinite in (
        math.nan, math.inf, -math.inf, Decimal("NaN"),
        Decimal("Infinity"), Decimal("-Infinity"),
    ):
        raises((TypeError, ValueError), _canonical_sha256, [nonfinite])
    for nonfinite in (
        Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity"),
        Decimal("-Infinity"),
    ):
        raises(ValueError, _decimal_text, nonfinite)
        raises(
            ValueError, _candidate_value,
            replace(selected, coverage=nonfinite), "AAPL",
        )

    request = {"path": "/v3/reference/tickers", "query": {}}
    reference = SourceArchive(
        "tickers-0001", request, ({"ticker": "AAPL"},),
    )
    provider_empty = SourceArchive(
        "daily-2024-08-01", {"path": "/daily/2024-08-01", "query": {}}, (),
    )
    filtered_empty = SourceArchive(
        "daily-2024-08-02", {"path": "/daily/2024-08-02", "query": {}}, (),
    )
    assert _source_archive_value(reference) == {
        "schema": 1,
        "request": request,
        "records": [{"ticker": "AAPL"}],
    }
    sessions = frozenset({"2024-08-02"})
    assert _source_binding_value(
        reference, sessions, "a" * 64,
    ) == {
        "name": "tickers-0001",
        "path": "sources/tickers-0001.json",
        "sha256": "a" * 64,
        "records": 1,
        "formation_session": None,
    }
    assert _source_binding_value(
        provider_empty, sessions, "b" * 64,
    )["formation_session"] is False
    filtered_binding = _source_binding_value(
        filtered_empty, sessions, "c" * 64,
    )
    assert set(filtered_binding) == {
        "name", "path", "sha256", "records", "formation_session",
    }
    assert filtered_binding["records"] == 0
    assert filtered_binding["formation_session"] is True


def test_exclusive_writer(directory: Path) -> None:
    rename_dir = directory / "exclusive-rename"
    rename_dir.mkdir()
    source = rename_dir / "source"
    target = rename_dir / "target"
    source.write_text("source\n", encoding="ascii")
    descriptor = os.open(rename_dir, os.O_RDONLY)
    try:
        rename_noreplace(descriptor, source.name, descriptor, target.name)
        assert not source.exists() and target.read_text(
            encoding="ascii",
        ) == "source\n"
        source.write_text("second\n", encoding="ascii")
        raises(
            FileExistsError,
            rename_noreplace,
            descriptor,
            source.name,
            descriptor,
            target.name,
        )
    finally:
        os.close(descriptor)
    assert source.read_text(encoding="ascii") == "second\n"
    assert target.read_text(encoding="ascii") == "source\n"

    open_failure = directory / "exclusive-open-failure"
    opened: list[int] = []
    real_open = os.open

    def fail_private_open(
        path: object, *args: object, **kwargs: object,
    ) -> int:
        if kwargs.get("dir_fd") is not None:
            raise OSError("private open failed")
        descriptor = real_open(path, *args, **kwargs)
        opened.append(descriptor)
        return descriptor

    with patch(
        "tools.files.os.open", side_effect=fail_private_open,
    ):
        raises(
            OSError, write_json_exclusive, open_failure / "value.json",
            {"value": 1},
        )
    assert len(opened) == 1
    raises(OSError, os.fstat, opened[0])

    stat_failure = directory / "exclusive-stat-failure"
    opened.clear()

    def record_open(
        path: object, *args: object, **kwargs: object,
    ) -> int:
        descriptor = real_open(path, *args, **kwargs)
        opened.append(descriptor)
        return descriptor

    with patch(
        "tools.files.os.open", side_effect=record_open,
    ), patch("tools.files.os.fstat", side_effect=OSError("stat failed")):
        raises(
            OSError, write_json_exclusive, stat_failure / "value.json",
            {"value": 1},
        )
    assert len(opened) == 2
    for descriptor in opened:
        raises(OSError, os.fstat, descriptor)
    assert len(list(stat_failure.glob(".value.json.*.tmp"))) == 1

    close_failure = directory / "exclusive-close-failure"
    opened.clear()
    real_close = os.close
    failed_close = False

    def fail_private_close(descriptor: int) -> None:
        nonlocal failed_close
        if len(opened) == 2 and descriptor == opened[1] and \
           not failed_close:
            failed_close = True
            raise OSError("private close failed")
        real_close(descriptor)

    with patch(
        "tools.files.os.open", side_effect=record_open,
    ), patch("tools.files.os.close", side_effect=fail_private_close):
        raises(
            OSError, write_json_exclusive, close_failure / "value.json",
            {"value": 1},
        )
    assert failed_close and len(opened) == 2
    raises(OSError, os.fstat, opened[0])
    os.fstat(opened[1])
    real_close(opened[1])


def publication_policy(path: Path) -> bytes:
    value = selection_policy_value() | {
        "formation_start": "2024-10-31",
        "minimum_coverage": "1",
        "minimum_formation_sessions": 1,
        "minimum_median_close_usd": "1",
        "minimum_median_dollar_volume_usd": "1",
    }
    write_selection_policy(path, value)
    return path.read_bytes()


def publication_requester(
    requested: list[str] | None = None,
) -> object:
    references = [source_reference(f"S{index:03d}") for index in range(60)]
    daily = [source_daily(f"S{index:03d}", index + 10) for index in range(60)]

    def request(url: str) -> dict[str, object]:
        if requested is not None:
            requested.append(url)
        results = (
            references
            if urlsplit(url).path == "/v3/reference/tickers"
            else daily
        )
        return {
            "status": "OK",
            "results": results,
            "resultsCount": len(results),
        }

    return request


def test_selection_transport() -> None:
    starts: list[str] = []

    def gate_factory(rate: int) -> object:
        assert rate == 5
        return lambda: starts.append("gate")

    def retried(
        _url: str, *, before_request: object | None = None,
    ) -> dict[str, object]:
        assert before_request is not None
        before_request()  # type: ignore[operator]
        before_request()  # type: ignore[operator]
        return {}

    with patch(
        "tools.select_universe.request_gate", side_effect=gate_factory,
    ), patch("tools.select_universe.request_json", side_effect=retried):
        requester, before_request = _transport(None, 5)
        before_request()
        assert requester("https://api.massive.com") == {}
    assert starts == ["gate", "gate"]

    starts.clear()
    injected = lambda _url: {}
    with patch(
        "tools.select_universe.request_gate", side_effect=gate_factory,
    ):
        requester, before_request = _transport(injected, 5)
        before_request()
        assert requester("https://api.massive.com") == {}
    assert requester is injected and starts == ["gate"]
    raises(ValueError, _transport, None, -1)


def test_select_universe_publication(directory: Path) -> None:
    policy_path = directory / "publish-policy.json"
    policy_bytes = publication_policy(policy_path)
    output = directory / "published-selection"
    requested: list[str] = []
    writes: list[str] = []

    def recorded_write(
        path: Path,
        value: dict[str, object],
        directory_fd: int | None = None,
        before_link: object | None = None,
        *,
        before_link_with_temp: object | None = None,
    ) -> None:
        write_json_exclusive(
            path, value, directory_fd,
            before_link,  # type: ignore[arg-type]
            before_link_with_temp=before_link_with_temp,  # type: ignore[arg-type]
        )
        writes.append(path.name)

    with patch(
        "tools.select_universe.write_json_exclusive",
        side_effect=recorded_write,
    ):
        report = select_universe(
            policy_path, output, key="fake-secret",
            requester=publication_requester(requested),
        )

    marker = output / "selection.json"
    assert json.loads(marker.read_bytes()) == report
    assert marker.read_bytes() == (
        json.dumps(
            report, allow_nan=False, indent=2, sort_keys=True,
        ).encode() + b"\n"
    )
    assert writes[-1] == "selection.json"
    assert set(report) == {
        "schema", "purpose", "declared_on", "anchor_date",
        "formation_start", "formation_end", "start", "end",
        "primary_cohort_size", "policy", "source_closure", "sources",
        "formation_sessions", "candidates", "master", "master_sha256",
        "cohorts",
    }
    assert report["policy"] == {
        "path": policy_path.resolve().as_posix(),
        "sha256": hashlib.sha256(policy_bytes).hexdigest(),
    }
    assert report["source_closure"] == [
        {
            "path": name,
            "sha256": file_sha256(ROOT / name),
        }
        for name in (
            "tools/select_universe.py",
            "tools/fetch_massive.py",
            "tools/files.py",
        )
    ]
    assert report["formation_sessions"] == ["2024-10-31"]
    assert set(path.name for path in output.iterdir()) == {
        "sources", "manifests", "selection.json",
    }

    source_values = report["sources"]
    assert isinstance(source_values, list)
    assert [item["name"] for item in source_values] == [
        "tickers-0001", "daily-2024-10-31",
    ]
    assert [item["formation_session"] for item in source_values] == [
        None, True,
    ]
    assert {
        item["path"] for item in source_values
    } == {
        "sources/tickers-0001.json",
        "sources/daily-2024-10-31.json",
    }
    for item in source_values:
        path = output / item["path"]
        value = json.loads(path.read_bytes())
        assert set(value) == {"schema", "request", "records"}
        assert item["records"] == len(value["records"])
        assert item["sha256"] == file_sha256(path)

    master = report["master"]
    candidates = report["candidates"]
    assert isinstance(master, list) and len(master) == 55
    assert isinstance(candidates, list) and len(candidates) == 60
    assert all(set(member) == {
        "ticker", "composite_figi", "share_class_figi", "stratum",
    } for member in master)
    assert all(set(candidate) == {
        "ticker", "active", "market", "locale", "type", "currency_name",
        "primary_exchange", "composite_figi", "share_class_figi",
        "observed", "coverage", "median_close_usd",
        "median_dollar_volume_usd", "rejection_reasons", "decision",
        "share_class_representative", "liquidity_rank", "stratum",
        "within_stratum_rank", "master_rank",
    } for candidate in candidates)
    assert sum(
        candidate["decision"] == "selected" for candidate in candidates
    ) == 55
    assert sum(
        candidate["decision"] == "eligible-not-selected"
        for candidate in candidates
    ) == 5
    assert report["master_sha256"] == _canonical_sha256(master)

    cohorts = report["cohorts"]
    assert isinstance(cohorts, dict)
    assert tuple(cohorts) == ("11", "22", "33", "55")
    for size_text, cohort in cohorts.items():
        size = int(size_text)
        members = cohort["members"]
        manifest_path = output / cohort["manifest"]
        assert set(cohort) == {
            "size", "primary", "members", "members_sha256", "manifest",
            "manifest_sha256",
        }
        assert cohort["size"] == size
        assert cohort["primary"] is (size == 55)
        assert members == master[:size]
        assert cohort["members_sha256"] == _canonical_sha256(members)
        assert cohort["manifest"] == (
            f"manifests/liquid-common-{size}.json"
        )
        assert cohort["manifest_sha256"] == file_sha256(manifest_path)
        parsed = UniverseManifest.read(manifest_path)
        assert parsed.eligibility_date == date(2024, 10, 31)
        assert parsed.start == date(2024, 11, 1)
        assert [item.ticker for item in parsed.series] == [
            item["ticker"] for item in members
        ]

    assert len(requested) == 2
    for url in requested:
        assert parse_qs(urlsplit(url).query)["apiKey"] == ["fake-secret"]
    for path in output.rglob("*"):
        if path.is_file():
            rendered = path.read_text(encoding="utf-8")
            assert all(value not in rendered for value in (
                "apiKey", "fake-secret", "NaN", "Infinity",
            ))


def mutate_final_write(
    action: object,
) -> object:
    def write(
        path: Path,
        value: dict[str, object],
        directory_fd: int | None = None,
        before_link: object | None = None,
        *,
        before_link_with_temp: object | None = None,
    ) -> None:
        callback = before_link_with_temp
        if path.name == "selection.json":
            assert callback is not None
            original = callback

            def changed(binding: ExclusiveTemp) -> None:
                action()  # type: ignore[operator]
                original(binding)  # type: ignore[operator]

            callback = changed
        write_json_exclusive(
            path, value, directory_fd, before_link,  # type: ignore[arg-type]
            before_link_with_temp=callback,  # type: ignore[arg-type]
        )

    return write


def test_selection_publication_mutations(directory: Path) -> None:
    actions = (
        (
            "policy",
            lambda policy, _output: policy.write_text(
                '{"mutated":true}\n', encoding="ascii",
            ),
        ),
        (
            "source",
            lambda _policy, output: (
                output / "sources/tickers-0001.json"
            ).write_text('{"mutated":true}\n', encoding="ascii"),
        ),
        (
            "manifest",
            lambda _policy, output: (
                output / "manifests/liquid-common-11.json"
            ).write_text('{"mutated":true}\n', encoding="ascii"),
        ),
        (
            "identity",
            lambda _policy, output: replace_file(
                output / "sources/tickers-0001.json",
            ),
        ),
        (
            "directory",
            lambda _policy, output: replace_directory(
                output, "sources",
            ),
        ),
        (
            "membership",
            lambda _policy, output: (
                output / "undeclared"
            ).write_text("extra\n", encoding="ascii"),
        ),
    )
    for name, action in actions:
        policy = directory / f"selection-{name}-mutation-policy.json"
        output = directory / f"selection-{name}-mutation-output"
        publication_policy(policy)
        with patch(
            "tools.select_universe.write_json_exclusive",
            side_effect=mutate_final_write(
                lambda item=action: item(policy, output),
            ),
        ):
            raises(
                (OSError, ValueError), select_universe,
                policy, output, key="fake-secret",
                requester=publication_requester(),
            )
        assert output.is_dir()
        assert not os.path.lexists(output / "selection.json")

    closure_root = directory / "mutable-closure"
    closure_tools = closure_root / "tools"
    closure_tools.mkdir(parents=True)
    sources = tuple(
        closure_tools / name
        for name in ("select_universe.py", "fetch_massive.py", "files.py")
    )
    for path in sources:
        path.write_text(f"# {path.name}\n", encoding="ascii")
    policy = directory / "closure-mutation-policy.json"
    output = directory / "closure-mutation-output"
    publication_policy(policy)
    with patch("tools.select_universe.ROOT", closure_root), \
         patch("tools.select_universe.SOURCE_PATHS", sources), \
         patch(
             "tools.select_universe.write_json_exclusive",
             side_effect=mutate_final_write(
                 lambda: sources[1].write_text(
                     "# changed\n", encoding="ascii",
                 ),
             ),
         ):
        raises(
            (OSError, ValueError), select_universe,
            policy, output, key="fake-secret",
            requester=publication_requester(),
        )
    assert output.is_dir()
    assert not os.path.lexists(output / "selection.json")


def replace_file(path: Path) -> None:
    contents = path.read_bytes()
    path.unlink()
    path.write_bytes(contents)


def replace_directory(root: Path, name: str) -> None:
    original = root / name
    original.rename(root / f"{name}-replaced")
    original.mkdir()


def test_selection_target_rejections(directory: Path) -> None:
    policy = directory / "target-policy.json"
    publication_policy(policy)
    reached_calls: list[str] = []

    def reached(url: str) -> object:
        reached_calls.append(url)
        raise AssertionError("request reached for invalid output")

    existing_file = directory / "existing-selection-file"
    existing_file.write_text("occupied\n", encoding="ascii")
    existing_directory = directory / "existing-selection-directory"
    existing_directory.mkdir()
    target = directory / "selection-target"
    target.mkdir()
    valid_link = directory / "selection-link"
    valid_link.symlink_to(target, target_is_directory=True)
    broken_link = directory / "selection-broken-link"
    broken_link.symlink_to(directory / "missing-selection-target")
    normalized = directory / "normalized-selection"
    normalized.mkdir()
    parent_target = directory / "parent-target"
    parent_target.mkdir()
    parent_link = directory / "parent-link"
    parent_link.symlink_to(parent_target, target_is_directory=True)
    rejected = (
        existing_file,
        existing_directory,
        valid_link,
        broken_link,
        directory / "missing-parent" / ".." / normalized.name,
        policy,
        parent_link / "child",
        directory / "absent-parent" / "child",
    )
    for output in rejected:
        before = len(reached_calls)
        raises(
            (OSError, ValueError), select_universe,
            policy, output, key="fake-secret", requester=reached,
        )
        assert len(reached_calls) == before

    linked_policy = directory / "linked-target-policy.json"
    linked_policy.symlink_to(policy)
    before = len(reached_calls)
    raises(
        ValueError, select_universe,
        linked_policy, directory / "linked-policy-output",
        key="fake-secret", requester=reached,
    )
    aliased_policy = directory / "aliased-target-policy.json"
    os.link(ROOT / "tools/files.py", aliased_policy)
    raises(
        ValueError, select_universe,
        aliased_policy, directory / "aliased-policy-output",
        key="fake-secret", requester=reached,
    )
    assert len(reached_calls) == before

    network_output = directory / "network-failure-output"
    raises(
        ValueError, select_universe, policy, network_output,
        key="fake-secret",
        requester=lambda _url: (_ for _ in ()).throw(OSError("offline")),
    )
    assert not os.path.lexists(network_output)

    selection_output = directory / "selection-failure-output"

    def too_small(url: str) -> dict[str, object]:
        result = (
            [source_reference("AAPL")]
            if urlsplit(url).path == "/v3/reference/tickers"
            else [source_daily("AAPL")]
        )
        return {
            "status": "OK", "results": result, "resultsCount": 1,
        }

    raises(
        ValueError, select_universe, policy, selection_output,
        key="fake-secret", requester=too_small,
    )
    assert not os.path.lexists(selection_output)

    original_parent = directory / "substituted-parent"
    original_parent.mkdir()
    renamed_parent = directory / "substituted-parent-original"
    output = original_parent / "selection"
    changed = False
    requester = publication_requester()

    def substitute(url: str) -> object:
        nonlocal changed
        if not changed:
            original_parent.rename(renamed_parent)
            original_parent.mkdir()
            changed = True
        return requester(url)  # type: ignore[operator]

    raises(
        (OSError, ValueError), select_universe,
        policy, output, key="fake-secret", requester=substitute,
    )
    assert changed
    assert not os.path.lexists(output)
    assert not os.path.lexists(renamed_parent / output.name)


def test_selection_commit_failures(directory: Path) -> None:
    policy = directory / "commit-policy.json"
    publication_policy(policy)

    parent_failure = directory / "parent-sync-output"
    parent_identity = os.stat(directory)
    real_fsync = os.fsync
    real_rename = rename_noreplace

    def fail_parent_sync(descriptor: int) -> None:
        identity = os.fstat(descriptor)
        if (identity.st_dev, identity.st_ino) == (
            parent_identity.st_dev, parent_identity.st_ino,
        ):
            raise OSError("parent sync failed")
        real_fsync(descriptor)

    with patch("tools.select_universe.os.fsync", side_effect=fail_parent_sync):
        raises(
            OSError, select_universe, policy, parent_failure,
            key="fake-secret", requester=publication_requester(),
        )
    assert parent_failure.is_dir()
    assert not os.path.lexists(parent_failure / "selection.json")

    linked_failure = directory / "after-link-output"

    def fail_after_link(
        path: Path,
        value: dict[str, object],
        directory_fd: int | None = None,
        before_link: object | None = None,
        *,
        before_link_with_temp: object | None = None,
    ) -> None:
        write_json_exclusive(
            path, value, directory_fd,
            before_link,  # type: ignore[arg-type]
            before_link_with_temp=before_link_with_temp,  # type: ignore[arg-type]
        )
        if path.name == "selection.json":
            raise OSError("after link")

    with patch(
        "tools.select_universe.write_json_exclusive",
        side_effect=fail_after_link,
    ):
        raises(
            OSError, select_universe, policy, linked_failure,
            key="fake-secret", requester=publication_requester(),
        )
    assert not os.path.lexists(linked_failure / "selection.json")

    rollback_unavailable = directory / "rollback-unavailable-output"

    def fail_rollback(
        source_fd: int,
        source: str,
        target_fd: int,
        target: str,
    ) -> None:
        if source == target == "selection.json" and \
           source_fd != target_fd:
            raise OSError("rollback unavailable")
        real_rename(source_fd, source, target_fd, target)

    with patch(
        "tools.select_universe.write_json_exclusive",
        side_effect=fail_after_link,
    ), patch(
        "tools.select_universe.rename_noreplace",
        side_effect=fail_rollback,
    ):
        error = raises(
            OSError, select_universe, policy, rollback_unavailable,
            key="fake-secret", requester=publication_requester(),
        )
    assert str(error) == "universe marker rollback failed"
    assert error.__cause__ is not None
    assert str(error.__cause__) == "rollback unavailable"
    assert error.__cause__.__context__ is not None
    assert str(error.__cause__.__context__) == "after link"
    assert json.loads(
        (rollback_unavailable / "selection.json").read_bytes(),
    )["schema"] == 1
    assert len(list(rollback_unavailable.glob(
        ".selection-rollback.*",
    ))) == 2

    foreign_failure = directory / "foreign-marker-output"
    foreign_marker = foreign_failure / "selection.json"
    foreign_bytes = b'{"owner":"foreign"}\n'
    foreign_identity: tuple[int, int] | None = None

    def inject_foreign_marker(
        path: Path,
        value: dict[str, object],
        directory_fd: int | None = None,
        before_link: object | None = None,
        *,
        before_link_with_temp: object | None = None,
    ) -> None:
        callback = before_link_with_temp
        if path.name == "selection.json":
            assert callback is not None
            original = callback

            def occupy_marker(binding: ExclusiveTemp) -> None:
                nonlocal foreign_identity
                original(binding)  # type: ignore[operator]
                foreign_marker.write_bytes(foreign_bytes)
                identity = foreign_marker.stat()
                foreign_identity = identity.st_dev, identity.st_ino

            callback = occupy_marker
        write_json_exclusive(
            path, value, directory_fd, before_link,  # type: ignore[arg-type]
            before_link_with_temp=callback,  # type: ignore[arg-type]
        )

    with patch(
        "tools.select_universe.write_json_exclusive",
        side_effect=inject_foreign_marker,
    ):
        raises(
            OSError, select_universe, policy, foreign_failure,
            key="fake-secret", requester=publication_requester(),
        )
    identity = foreign_marker.stat()
    assert foreign_marker.read_bytes() == foreign_bytes
    assert (identity.st_dev, identity.st_ino) == foreign_identity

    pre_callback_failure = directory / "pre-callback-temp-output"
    pre_callback_bytes = b"foreign pre-callback marker\n"
    pre_callback_name: str | None = None
    pre_callback_identity: tuple[int, int] | None = None
    real_unlink = os.unlink

    def replace_before_callback(
        path: Path,
        value: dict[str, object],
        directory_fd: int | None = None,
        before_link: object | None = None,
        *,
        before_link_with_temp: object | None = None,
    ) -> None:
        callback = before_link_with_temp
        if path.name == "selection.json":
            assert callback is not None and directory_fd is not None
            original = callback

            def replace(binding: ExclusiveTemp) -> None:
                nonlocal pre_callback_name, pre_callback_identity
                names = [
                    name for name in os.listdir(directory_fd)
                    if name.startswith(".selection.json.")
                ]
                assert len(names) == 1
                pre_callback_name = names[0]
                real_unlink(pre_callback_name, dir_fd=directory_fd)
                descriptor = os.open(
                    pre_callback_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    os.write(descriptor, pre_callback_bytes)
                    metadata = os.fstat(descriptor)
                finally:
                    os.close(descriptor)
                pre_callback_identity = metadata.st_dev, metadata.st_ino
                original(binding)  # type: ignore[operator]

            callback = replace
        write_json_exclusive(
            path, value, directory_fd, before_link,  # type: ignore[arg-type]
            before_link_with_temp=callback,  # type: ignore[arg-type]
        )

    with patch(
        "tools.select_universe.write_json_exclusive",
        side_effect=replace_before_callback,
    ):
        raises(
            (OSError, ValueError), select_universe, policy,
            pre_callback_failure, key="fake-secret",
            requester=publication_requester(),
        )
    assert pre_callback_name is not None
    pre_callback_marker = pre_callback_failure / pre_callback_name
    metadata = pre_callback_marker.stat()
    assert pre_callback_marker.read_bytes() == pre_callback_bytes
    assert (metadata.st_dev, metadata.st_ino) == pre_callback_identity
    assert not os.path.lexists(pre_callback_failure / "selection.json")

    rollback_failure = directory / "rollback-race-output"
    rollback_bytes = b'{"owner":"rollback-race"}\n'
    rollback_identity: tuple[int, int] | None = None
    raced = False

    def replace_during_rollback(
        source_fd: int,
        source: str,
        target_fd: int,
        target: str,
    ) -> None:
        nonlocal raced, rollback_identity
        if source == target == "selection.json" and not raced and \
           source_fd != target_fd:
            raced = True
            real_unlink(source, dir_fd=source_fd)
            descriptor = os.open(
                source, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600, dir_fd=source_fd,
            )
            try:
                os.write(descriptor, rollback_bytes)
                metadata = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            rollback_identity = metadata.st_dev, metadata.st_ino
        real_rename(source_fd, source, target_fd, target)

    with patch(
        "tools.select_universe.write_json_exclusive",
        side_effect=fail_after_link,
    ), patch(
        "tools.select_universe.rename_noreplace",
        side_effect=replace_during_rollback,
    ):
        raises(
            OSError, select_universe, policy, rollback_failure,
            key="fake-secret", requester=publication_requester(),
        )
    rollback_marker = rollback_failure / "selection.json"
    metadata = rollback_marker.stat()
    assert raced and rollback_marker.read_bytes() == rollback_bytes
    assert (metadata.st_dev, metadata.st_ino) == rollback_identity

    foreign_private_failure = directory / "foreign-private-output"
    foreign_private_bytes = b"foreign private marker\n"
    foreign_private_name: str | None = None
    foreign_private_identity: tuple[int, int] | None = None

    def replace_at_commit(
        source_fd: int,
        source: str,
        target_fd: int,
        target: str,
    ) -> None:
        nonlocal foreign_private_name, foreign_private_identity
        if target == "selection.json" and foreign_private_name is None:
            real_unlink(source, dir_fd=source_fd)
            descriptor = os.open(
                source, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600, dir_fd=source_fd,
            )
            try:
                os.write(descriptor, foreign_private_bytes)
                identity = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            foreign_private_name = source
            foreign_private_identity = identity.st_dev, identity.st_ino
        real_rename(source_fd, source, target_fd, target)

    with patch(
        "tools.files.rename_noreplace",
        side_effect=replace_at_commit,
    ):
        raises(
            OSError, select_universe, policy, foreign_private_failure,
            key="fake-secret", requester=publication_requester(),
        )
    assert foreign_private_name is not None
    foreign_private = foreign_private_failure / "selection.json"
    identity = foreign_private.stat()
    assert foreign_private.read_bytes() == foreign_private_bytes
    assert (identity.st_dev, identity.st_ino) == foreign_private_identity

    interrupted_failure = directory / "interrupted-output"

    def interrupt_after_link(
        path: Path,
        value: dict[str, object],
        directory_fd: int | None = None,
        before_link: object | None = None,
        *,
        before_link_with_temp: object | None = None,
    ) -> None:
        write_json_exclusive(
            path, value, directory_fd,
            before_link,  # type: ignore[arg-type]
            before_link_with_temp=before_link_with_temp,  # type: ignore[arg-type]
        )
        if path.name == "selection.json":
            raise KeyboardInterrupt

    with patch(
        "tools.select_universe.write_json_exclusive",
        side_effect=interrupt_after_link,
    ):
        raises(
            KeyboardInterrupt, select_universe, policy,
            interrupted_failure, key="fake-secret",
            requester=publication_requester(),
        )
    assert not os.path.lexists(
        interrupted_failure / "selection.json"
    )

    ambiguous_success = directory / "ambiguous-success-output"

    def commit_then_report_error(
        source_fd: int,
        source: str,
        target_fd: int,
        target: str,
    ) -> None:
        real_rename(source_fd, source, target_fd, target)
        if target == "selection.json":
            raise OSError("ambiguous commit result")

    with patch(
        "tools.files.rename_noreplace",
        side_effect=commit_then_report_error,
    ):
        report = select_universe(
            policy, ambiguous_success, key="fake-secret",
            requester=publication_requester(),
        )
    assert json.loads(
        (ambiguous_success / "selection.json").read_bytes(),
    ) == report
    assert not any(
        path.name.startswith(".selection.json.")
        for path in ambiguous_success.iterdir()
    )

    commit_failure = directory / "commit-failure-output"

    def fail_commit(
        source_fd: int,
        source: str,
        target_fd: int,
        target: str,
    ) -> None:
        if target == "selection.json":
            raise OSError("commit failed")
        real_rename(source_fd, source, target_fd, target)

    with patch(
        "tools.files.rename_noreplace",
        side_effect=fail_commit,
    ):
        raises(
            OSError, select_universe, policy, commit_failure,
            key="fake-secret", requester=publication_requester(),
        )
    assert not os.path.lexists(commit_failure / "selection.json")
    temporary = [
        path for path in commit_failure.iterdir()
        if path.name.startswith(".selection.json.")
    ]
    assert len(temporary) == 1
    assert json.loads(temporary[0].read_bytes())["schema"] == 1

    collision_failure = directory / "rollback-target-collision-output"
    collision_bytes = b"unrelated rollback entry\n"
    collision_identity: tuple[int, int] | None = None

    def occupy_rollback_target(
        source_fd: int,
        source: str,
        target_fd: int,
        target: str,
    ) -> None:
        nonlocal collision_identity
        if source == target == "selection.json" and \
           source_fd != target_fd and collision_identity is None:
            descriptor = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600, dir_fd=target_fd,
            )
            try:
                os.write(descriptor, collision_bytes)
                metadata = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            collision_identity = metadata.st_dev, metadata.st_ino
        real_rename(source_fd, source, target_fd, target)

    with patch(
        "tools.files.rename_noreplace",
        side_effect=fail_commit,
    ), patch(
        "tools.select_universe.rename_noreplace",
        side_effect=occupy_rollback_target,
    ):
        raises(
            OSError, select_universe, policy, collision_failure,
            key="fake-secret", requester=publication_requester(),
        )
    assert collision_identity is not None
    assert not os.path.lexists(collision_failure / "selection.json")
    collision = list(
        collision_failure.glob(
            ".selection-rollback.*/selection.json",
        )
    )
    assert len(collision) == 1
    metadata = collision[0].stat()
    assert collision[0].read_bytes() == collision_bytes
    assert (metadata.st_dev, metadata.st_ino) == collision_identity

    sync_failure = directory / "root-sync-output"

    def fail_commit_sync(descriptor: int) -> None:
        marker = sync_failure / "selection.json"
        identity = os.fstat(descriptor)
        if marker.exists():
            root = os.stat(sync_failure)
            if (identity.st_dev, identity.st_ino) == (
                root.st_dev, root.st_ino,
            ):
                raise OSError("root commit sync failed")
        real_fsync(descriptor)

    with patch(
        "tools.select_universe.os.fsync", side_effect=fail_commit_sync,
    ):
        raises(
            OSError, select_universe, policy, sync_failure,
            key="fake-secret", requester=publication_requester(),
        )
    assert not os.path.lexists(sync_failure / "selection.json")

    validation_failure = directory / "post-link-validation-output"

    def mutate_after_link(
        path: Path,
        value: dict[str, object],
        directory_fd: int | None = None,
        before_link: object | None = None,
        *,
        before_link_with_temp: object | None = None,
    ) -> None:
        write_json_exclusive(
            path, value, directory_fd,
            before_link,  # type: ignore[arg-type]
            before_link_with_temp=before_link_with_temp,  # type: ignore[arg-type]
        )
        if path.name == "selection.json":
            (validation_failure / "extra").write_text(
                "extra\n", encoding="ascii",
            )

    with patch(
        "tools.select_universe.write_json_exclusive",
        side_effect=mutate_after_link,
    ):
        raises(
            (OSError, ValueError), select_universe,
            policy, validation_failure,
            key="fake-secret", requester=publication_requester(),
        )
    assert not os.path.lexists(validation_failure / "selection.json")


def test_selection_cli(directory: Path) -> None:
    arguments = selection_args(["policy.json", "output"])
    assert arguments.policy == Path("policy.json")
    assert arguments.output_dir == Path("output")
    assert arguments.requests_per_minute == 0
    assert selection_args([
        "policy.json", "output", "--requests-per-minute", "5",
    ]).requests_per_minute == 5

    with patch.object(
        sys, "argv", ["select_universe.py", "policy.json", "output"],
    ), patch(
        "tools.select_universe.select_universe",
        return_value={"z": 1, "a": 2},
    ), patch("sys.stdout", new_callable=StringIO) as stdout:
        selection_main()
    assert stdout.getvalue() == '{"a": 2, "z": 1}\n'

    for error in (
        ValueError("fake-secret"),
        OSError("fake-secret"),
    ):
        with patch.object(
            sys, "argv", ["select_universe.py", "policy.json", "output"],
        ), patch(
            "tools.select_universe.select_universe", side_effect=error,
        ):
            failure = raises(SystemExit, selection_main)
        assert str(failure) == "universe selection failed"
        assert "fake-secret" not in str(failure)
    with patch.object(
        sys, "argv", ["select_universe.py", "policy.json", "output"],
    ), patch(
        "tools.select_universe.select_universe",
        side_effect=RuntimeError("not caught"),
    ):
        raises(RuntimeError, selection_main)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/select_universe.py"),
            "--help",
        ],
        cwd=directory,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "POLICY OUTPUT_DIR" in completed.stdout
    assert "ModuleNotFoundError" not in completed.stderr


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
        "eligibility_date": "2024-10-31",
        "end": "2026-07-21",
        "interval_minutes": 30,
        "purpose": "Offline universe test",
        "schema": 1,
        "series": [
            {"stratum": "generic", "ticker": "AAPL"},
            {"stratum": "generic", "ticker": "MSFT"},
        ],
        "session": "regular",
        "start": "2024-11-01",
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
    original = path.read_bytes()
    manifest = UniverseManifest.read(path)
    assert path.read_bytes() == original
    assert manifest == UniverseManifest(
        1, "Offline universe test", date(2026, 7, 23), date(2024, 10, 31),
        date(2024, 11, 1), date(2026, 7, 21), 30, True, "regular",
        (SeriesSpec("AAPL", "generic"), SeriesSpec("MSFT", "generic")),
    )
    assert manifest.series[0].stratum == manifest.series[1].stratum

    value = manifest_value()
    value["eligibility_date"] = "2024-10-30"
    write_manifest(path, value)
    assert UniverseManifest.read(path).eligibility_date == date(2024, 10, 30)

    value["eligibility_date"] = "2024-11-02"
    write_manifest(path, value)
    assert_manifest_error(path)
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
        ("declared_on", "2024-10-30"), ("declared_on", "not-a-date"),
        ("end", "2024-10-31"), ("interval_minutes", 0),
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


def test_universe_coverage_overlay(directory: Path) -> None:
    policy_path = (
        ROOT / "universes/liquid-common-55-coverage-v2.example.json"
    )
    selection_path = (
        ROOT / "reports/universe-selection-20260724-06/selection.json"
    )
    base_path = (
        ROOT / "reports/universe-selection-20260724-06/manifests/"
        "liquid-common-55.json"
    )
    policy_value = json.loads(policy_path.read_bytes())
    selection = json.loads(selection_path.read_bytes())
    base = json.loads(base_path.read_bytes())
    failed = next(
        item for item in selection["candidates"] if item["ticker"] == "ENLC"
    )
    replacement = replacement_candidate(selection, "ENLC")
    assert replacement["ticker"] == "AAON"
    revised = revised_manifest(
        base, failed, replacement,
        purpose=policy_value["purpose"],
        declared_on=policy_value["declared_on"],
    )
    assert revised["series"][:49] == base["series"][:49]
    assert revised["series"][49] == {
        "stratum": "liquidity-5", "ticker": "AAON",
    }
    assert revised["series"][50:] == base["series"][50:]

    output = directory / "coverage-overlay.json"
    assert apply_overlay(policy_path, output) == revised
    parsed = UniverseManifest.read(output)
    assert [item.ticker for item in parsed.series[:33]] == [
        item["ticker"] for item in base["series"][:33]
    ]
    assert parsed.series[49].ticker == "AAON"
    original = output.read_bytes()
    raises(ValueError, apply_overlay, policy_path, output)
    assert output.read_bytes() == original
    nested_output = selection_path.parent / "coverage-overlay-test.json"
    raises(ValueError, apply_overlay, policy_path, nested_output)
    assert not os.path.lexists(nested_output)

    fixture = (directory / "overlay-fixture").resolve()
    fixture.mkdir()
    tree = fixture / "selection-tree"
    shutil.copytree(selection_path.parent, tree)
    selection_copy, base_copy = (
        tree / "selection.json", tree / "manifests/liquid-common-55.json",
    )

    def tree_binding() -> dict[str, object]:
        entries = sorted(
            (
                path.relative_to(tree).as_posix(),
                file_sha256(path),
            )
            for path in tree.rglob("*") if path.is_file()
        )
        digest = hashlib.sha256()
        for path, sha256 in entries:
            digest.update(
                path.encode() + b"\0" + sha256.encode() + b"\n"
            )
        return {
            "root": "selection-tree", "files": len(entries),
            "sha256": digest.hexdigest(),
        }

    policy = json.loads(policy_path.read_bytes())
    policy["selection"] = {
        "path": "selection-tree/selection.json",
        "sha256": file_sha256(selection_copy),
    }
    policy["base_manifest"] = {
        "path": "selection-tree/manifests/liquid-common-55.json",
        "sha256": file_sha256(base_copy),
    }
    policy["selection_tree"] = tree_binding()
    local_policy = fixture / "policy.json"
    write_json(local_policy, policy)

    race_output = fixture / "race-output.json"
    mutated = False

    def mutate_before_publish(
        path: Path, value: dict[str, object], **kwargs: object,
    ) -> None:
        nonlocal mutated
        mutated = True
        base_copy.write_bytes(base_copy.read_bytes() + b" ")
        write_json_exclusive(path, value, **kwargs)

    with patch(
        "tools.apply_universe_coverage_overlay.write_json_exclusive",
        side_effect=mutate_before_publish,
    ):
        raises(
            ValueError, apply_overlay, local_policy,
            race_output, root=fixture,
        )
    assert mutated
    assert not os.path.lexists(race_output)
    base_copy.write_bytes(base_path.read_bytes())

    for name, change in (
        ("replacement", lambda item: item["replacement"].update(
            {"ticker": "POR"},
        )),
        ("wrong-stratum", lambda item: item["failed_member"].update(
            {"stratum": 4},
        )),
        ("extra-field", lambda item: item.update({"metrics": {}})),
        ("selection-hash", lambda item: item["selection"].update(
            {"sha256": "0" * 64},
        )),
    ):
        candidate = json.loads(json.dumps(policy))
        change(candidate)
        candidate_path = fixture / f"{name}.json"
        write_json(candidate_path, candidate)
        rejected_output = fixture / f"{name}-output.json"
        raises(
            ValueError, apply_overlay, candidate_path,
            rejected_output, root=fixture,
        )
        assert not os.path.lexists(rejected_output)

    changed_base = json.loads(base_copy.read_bytes())
    changed_base["series"][0], changed_base["series"][1] = (
        changed_base["series"][1], changed_base["series"][0]
    )
    write_json(base_copy, changed_base)
    changed_selection = json.loads(selection_copy.read_bytes())
    changed_selection["cohorts"]["55"]["manifest_sha256"] = \
        file_sha256(base_copy)
    write_json(selection_copy, changed_selection)
    policy["selection"]["sha256"] = file_sha256(selection_copy)
    policy["base_manifest"]["sha256"] = file_sha256(base_copy)
    policy["selection_tree"] = tree_binding()
    write_json(local_policy, policy)
    changed_base_output = fixture / "changed-base-output.json"
    raises(
        ValueError, apply_overlay, local_policy,
        changed_base_output, root=fixture,
    )
    assert not os.path.lexists(changed_base_output)
    base_copy.write_bytes(base_path.read_bytes())
    selection_copy.write_bytes(selection_path.read_bytes())
    policy["selection"]["sha256"] = file_sha256(selection_copy)
    policy["base_manifest"]["sha256"] = file_sha256(base_copy)
    policy["selection_tree"] = tree_binding()

    invalid_master = json.loads(selection_copy.read_bytes())
    invalid_master["master_sha256"] = "0" * 64
    write_json(selection_copy, invalid_master)
    policy["selection"]["sha256"] = file_sha256(selection_copy)
    policy["selection_tree"] = tree_binding()
    write_json(local_policy, policy)
    invalid_master_output = fixture / "invalid-master-output.json"
    raises(
        ValueError, apply_overlay, local_policy,
        invalid_master_output, root=fixture,
    )
    assert not os.path.lexists(invalid_master_output)
    selection_copy.write_bytes(selection_path.read_bytes())
    policy["selection"]["sha256"] = file_sha256(selection_copy)
    policy["selection_tree"] = tree_binding()

    extra_selected = json.loads(selection_copy.read_bytes())
    extra = next(
        item for item in extra_selected["candidates"]
        if item["ticker"] == "POR"
    )
    extra.update({"decision": "selected", "master_rank": 55})
    write_json(selection_copy, extra_selected)
    policy["selection"]["sha256"] = file_sha256(selection_copy)
    policy["selection_tree"] = tree_binding()
    write_json(local_policy, policy)
    extra_selected_output = fixture / "extra-selected-output.json"
    raises(
        ValueError, apply_overlay, local_policy,
        extra_selected_output, root=fixture,
    )
    assert not os.path.lexists(extra_selected_output)
    selection_copy.write_bytes(selection_path.read_bytes())
    policy["selection"]["sha256"] = file_sha256(selection_copy)
    policy["selection_tree"] = tree_binding()

    changed_selection = json.loads(selection_copy.read_bytes())
    next(
        item for item in changed_selection["candidates"]
        if item["ticker"] == "AAON"
    )["decision"] = "rejected"
    write_json(selection_copy, changed_selection)
    policy["selection"]["sha256"] = file_sha256(selection_copy)
    policy["selection_tree"] = tree_binding()
    write_json(local_policy, policy)
    changed_selection_output = fixture / "changed-selection-output.json"
    raises(
        ValueError, apply_overlay, local_policy,
        changed_selection_output, root=fixture,
    )
    assert not os.path.lexists(changed_selection_output)

    selection_copy.unlink()
    selection_copy.symlink_to(selection_path)
    policy["selection"]["sha256"] = file_sha256(selection_path)
    write_json(local_policy, policy)
    symlink_output = fixture / "symlink-output.json"
    raises(
        ValueError, apply_overlay, local_policy,
        symlink_output, root=fixture,
    )
    assert not os.path.lexists(symlink_output)


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
                    "primary_exchange": "XNYS",
                },
            }
        ticker = unquote(parts.path.split("/ticker/", 1)[1].split("/", 1)[0])
        return {
            "status": "OK", "ticker": ticker,
            "results": [aggregate("2024-11-01T13:30:00+00:00", 100.0)],
        }
    return request


def benchmark_reference(**changes: object) -> dict[str, object]:
    value: dict[str, object] = {
        "ticker": "SPY", "active": True, "market": "stocks",
        "locale": "us", "type": "ETF", "currency_name": "usd",
        "primary_exchange": "ARCX",
    }
    value.update(changes)
    return value


@cache
def benchmark_aggregates() -> tuple[dict[str, object], ...]:
    calendar = SessionCalendar.read(DEFAULT_CALENDAR)
    rows = [
        aggregate(item.timestamp, 100.0 + index / 10_000)
        for index, item in enumerate(expected_bins(
            calendar, BENCHMARK_START, BENCHMARK_END, 30,
        ))
    ]
    rows.append(aggregate("2024-11-29T18:00:00+00:00", 99.0))
    return tuple(sorted(rows, key=lambda item: item["t"]))


def benchmark_requester(
    requested: list[str],
    *,
    reference: object | None = None,
    rows: object | None = None,
) -> Requester:
    def request(url: str) -> dict[str, object]:
        requested.append(url)
        if urlsplit(url).path == "/v3/reference/tickers/SPY":
            return {
                "status": "OK",
                "results": benchmark_reference()
                if reference is None else reference,
            }
        return {
            "adjusted": True, "status": "OK", "ticker": "SPY",
            "results": list(benchmark_aggregates())
            if rows is None else rows,
        }
    return request


def test_benchmark_fetch(directory: Path) -> None:
    bundle = directory.resolve() / "spy-bundle"
    csv_path, report_path = bundle / "spy.csv", bundle / "fetch.json"
    requested: list[str] = []

    report = fetch_benchmark(
        bundle, key="fake-secret",
        requester=benchmark_requester(requested),
    )
    timestamps_, _ = read_bars(csv_path)
    assert len(timestamps_) == 5_534
    assert "2024-11-29T18:00:00Z" not in timestamps_
    assert report["csv"]["session_audit"] == {
        "scope": "all-expected-session-bins",
        "expected_sessions": 428,
        "affected_sessions": 0,
        "missing_sessions": [],
        "expected_bins": 5_534,
        "missing_bins": 0,
        "ranges": [],
    }
    assert report["calendar"]["applicability"] == {
        "benchmark": "SPY",
        "calendar_venue": "XNYS",
        "exchange_source": (
            "https://massive.com/docs/rest/stocks/"
            "market-operations/exchanges"
        ),
        "market_group": "NYSE Group",
        "operating_mic": "XNYS",
        "primary_exchange": "ARCX",
        "session": "core",
        "session_source": "https://www.nyse.com/trade/hours-calendars",
    }
    assert report["csv"]["sha256"] == file_sha256(csv_path)
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert [
        (urlsplit(url).path, parse_qs(urlsplit(url).query))
        for url in requested
    ] == [
        (
            "/v3/reference/tickers/SPY",
            {"apiKey": ["fake-secret"], "date": ["2024-10-31"]},
        ),
        (
            "/v2/aggs/ticker/SPY/range/30/minute/"
            "2024-11-01/2026-07-21",
            {
                "adjusted": ["true"], "apiKey": ["fake-secret"],
                "limit": ["50000"], "sort": ["asc"],
            },
        ),
    ]
    assert report["reference"]["request"] == {
        "path": "/v3/reference/tickers/SPY",
        "query": {"date": "2024-10-31"},
    }
    assert report["aggregate"]["request"] == {
        "path": (
            "/v2/aggs/ticker/SPY/range/30/minute/"
            "2024-11-01/2026-07-21"
        ),
        "query": {
            "adjusted": "true", "limit": "50000", "sort": "asc",
        },
    }
    assert "fake-secret" not in csv_path.read_text(encoding="ascii")
    assert "fake-secret" not in report_path.read_text(encoding="utf-8")


def test_benchmark_import_boundary() -> None:
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
        "import tools.fetch_benchmark; "
        "blocked=('tools.experiment','tools.relative_context','tools.train'); "
        "assert not any(name == 'torch' or name.startswith('torch.') "
        "for name in sys.modules); "
        "assert not any(name in sys.modules for name in blocked)"
    )
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )


def test_benchmark_source_rejections(directory: Path) -> None:
    directory = directory.resolve()

    def unreachable(*_args: object) -> object:
        raise AssertionError("benchmark request was reached")

    for index, (field, value) in enumerate((
        ("ticker", "QQQ"),
        ("active", False),
        ("active", 1),
        ("market", "crypto"),
        ("locale", "global"),
        ("type", "CS"),
        ("currency_name", "eur"),
        ("primary_exchange", "XNAS"),
    )):
        bundle = directory / f"spy-identity-{index}"
        requested: list[str] = []
        raises(
            ValueError, fetch_benchmark, bundle, key="fake-secret",
            requester=benchmark_requester(
                requested,
                reference=benchmark_reference(**{field: value}),
            ),
        )
        assert len(requested) == 1
        assert not bundle.exists()

    for index, (field, value, remove) in enumerate((
        ("ticker", "QQQ", False),
        ("ticker", None, True),
        ("adjusted", False, False),
        ("adjusted", None, True),
    )):
        direct = benchmark_requester([])

        def changed_request(
            url: str, *, name: str = field,
            replacement: object = value, absent: bool = remove,
        ) -> Mapping[str, object]:
            payload = dict(direct(url))
            if urlsplit(url).path.startswith("/v2/aggs/"):
                if absent:
                    payload.pop(name, None)
                else:
                    payload[name] = replacement
            return payload

        bundle = directory / f"spy-aggregate-{index}"
        raises(
            ValueError, fetch_benchmark, bundle, key="fake-secret",
            requester=changed_request,
        )
        assert not bundle.exists()

    for index, response in enumerate((
        {"status": "ERROR", "results": benchmark_reference()},
        {"status": "OK", "results": [benchmark_reference()]},
    )):
        bundle = directory / f"spy-reference-payload-{index}"
        raises(
            ValueError, fetch_benchmark, bundle, key="fake-secret",
            requester=lambda _url, item=response: item,
        )
        assert not bundle.exists()

    pages = iter((
        {
            "adjusted": True, "status": "OK", "ticker": "SPY",
            "results": [aggregate(
                "2024-11-01T13:30:00+00:00", 100.0,
            )],
            "next_url": "https://api.massive.com/page?cursor=next",
        },
        {
            "status": "OK", "ticker": "SPY",
            "results": [aggregate(
                "2024-11-01T14:00:00+00:00", 101.0,
            )],
        },
    ))

    def paginated(url: str) -> Mapping[str, object]:
        return (
            {"status": "OK", "results": benchmark_reference()}
            if urlsplit(url).path == "/v3/reference/tickers/SPY"
            else next(pages)
        )

    pagination_bundle = directory / "spy-pagination-fields"
    raises(
        ValueError, fetch_benchmark, pagination_bundle,
        key="fake-secret", requester=paginated,
    )
    assert not pagination_bundle.exists()

    source = list(benchmark_aggregates())
    changed = (
        source[1:],
        [source[0], source[0], *source[1:]],
        [dict(source[0]) | {"t": source[0]["t"] + 15 * 60_000}, *source[1:]],
        [aggregate("2024-10-31T13:30:00+00:00", 100.0), *source],
        tuple(source),
    )
    for index, rows in enumerate(changed):
        bundle = directory / f"spy-grid-{index}"
        raises(
            ValueError, fetch_benchmark, bundle, key="fake-secret",
            requester=benchmark_requester([], rows=rows),
        )
        assert not bundle.exists()

    calendar = directory / "changed-calendar.json"
    value = calendar_value()
    value["purpose"] = "changed"
    write_json(calendar, value)
    raises(
        ValueError, fetch_benchmark, directory / "wrong-calendar",
        calendar_path=calendar, key="fake-secret",
        requester=lambda _url: unreachable(),
    )
    for index, key in enumerate(("", "contains whitespace", 1)):
        raises(
            ValueError, fetch_benchmark, directory / f"wrong-key-{index}",
            key=key, requester=lambda _url: unreachable(),
        )


def test_benchmark_target_rejections(directory: Path) -> None:
    directory = directory.resolve()

    def unreachable(*_args: object) -> object:
        raise AssertionError("benchmark request was reached")

    def rejected(bundle: Path,
                 calendar_path: Path = DEFAULT_CALENDAR) -> None:
        raises(
            ValueError, fetch_benchmark, bundle,
            calendar_path=calendar_path, key="fake-secret",
            requester=lambda _url: unreachable(),
        )

    existing = directory / "existing-spy"
    existing.mkdir()
    rejected(existing)

    broken = directory / "broken-spy"
    broken.symlink_to(directory / "missing-spy")
    rejected(broken)

    hardlink_source = directory / "hardlink-source"
    hardlink_source.write_text("occupied", encoding="ascii")
    hardlink = directory / "hardlink-spy"
    os.link(hardlink_source, hardlink)
    rejected(hardlink)

    real_parent = directory / "real-spy-parent"
    real_parent.mkdir()
    alias_parent = directory / "alias-spy-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    rejected(alias_parent / "spy")

    child = real_parent / "child"
    child.mkdir()
    rejected(alias_parent / "child" / "spy")

    calendar_alias = directory / "calendar-alias.json"
    calendar_alias.symlink_to(DEFAULT_CALENDAR)
    rejected(directory / "calendar-alias-spy", calendar_alias)

    calendar_parent = directory / "calendar-parent"
    calendar_parent.mkdir()
    calendar_copy = calendar_parent / "calendar.json"
    shutil.copyfile(DEFAULT_CALENDAR, calendar_copy)
    calendar_parent_alias = directory / "calendar-parent-alias"
    calendar_parent_alias.symlink_to(
        calendar_parent, target_is_directory=True,
    )
    rejected(
        directory / "calendar-ancestor-spy",
        calendar_parent_alias / "calendar.json",
    )

    rejected(directory / "missing-parent" / "spy")


def test_benchmark_publication_failures(directory: Path) -> None:
    directory = directory.resolve()

    def attempt(name: str, *, calendar: Path = DEFAULT_CALENDAR) -> Path:
        bundle = directory / name
        fetch_benchmark(
            bundle, calendar_path=calendar,
            key="fake-secret", requester=benchmark_requester([]),
        )
        return bundle

    with patch(
        "tools.fetch_benchmark.rename_noreplace",
        side_effect=OSError("bundle publication failed"),
    ):
        raises(OSError, attempt, "rename-failure")
    assert not (directory / "rename-failure").exists()

    write = benchmark_fetch.write_csv

    def mutate_stage(path: Path, bars: Sequence[Bar]) -> None:
        write(path, bars)
        text = path.read_text(encoding="ascii")
        path.write_text(text.replace(",100,", ",101,", 1), encoding="ascii")

    with patch(
        "tools.fetch_benchmark.write_csv", side_effect=mutate_stage,
    ):
        raises(ValueError, attempt, "stage-mutation")
    assert not (directory / "stage-mutation").exists()

    publish = benchmark_fetch.rename_noreplace
    calendar = directory / "mutable-spy-calendar.json"
    shutil.copyfile(DEFAULT_CALENDAR, calendar)

    def mutate_calendar(
        source_fd: int, source: str, target_fd: int, target: str,
    ) -> None:
        calendar.write_bytes(calendar.read_bytes() + b" ")
        publish(source_fd, source, target_fd, target)

    with patch(
        "tools.fetch_benchmark.rename_noreplace",
        side_effect=mutate_calendar,
    ):
        raises(ValueError, attempt, "calendar-mutation", calendar=calendar)
    assert tuple(sorted(
        path.name for path in (directory / "calendar-mutation").iterdir()
    )) == ("fetch.json", "spy.csv")

    def mutate_csv(
        source_fd: int, source: str, target_fd: int, target: str,
    ) -> None:
        path = directory / source / "spy.csv"
        path.write_bytes(path.read_bytes() + b"\n")
        publish(source_fd, source, target_fd, target)

    with patch(
        "tools.fetch_benchmark.rename_noreplace", side_effect=mutate_csv,
    ):
        raises(ValueError, attempt, "csv-mutation")
    assert (directory / "csv-mutation").exists()

    def occupy_target(
        source_fd: int, source: str, target_fd: int, target: str,
    ) -> None:
        (directory / target).mkdir()
        publish(source_fd, source, target_fd, target)

    with patch(
        "tools.fetch_benchmark.rename_noreplace",
        side_effect=occupy_target,
    ):
        raises(OSError, attempt, "target-race")
    assert not any((directory / "target-race").iterdir())

    fsync = benchmark_fetch.os.fsync

    def fail_stage_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("stage fsync failed")
        fsync(descriptor)

    with patch(
        "tools.fetch_benchmark.os.fsync", side_effect=fail_stage_fsync,
    ):
        raises(OSError, attempt, "stage-fsync-failure")
    assert not (directory / "stage-fsync-failure").exists()

    directory_fsyncs = 0

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_fsyncs += 1
            if directory_fsyncs == 2:
                raise OSError("parent fsync failed")
        fsync(descriptor)

    with patch(
        "tools.fetch_benchmark.os.fsync", side_effect=fail_parent_fsync,
    ):
        raises(OSError, attempt, "fsync-failure")
    assert tuple(sorted(
        path.name for path in (directory / "fsync-failure").iterdir()
    )) == ("fetch.json", "spy.csv")


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
        assert events[-4:] == [
            ("hash", manifest_path),
            ("hash", DEFAULT_CALENDAR),
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
        "fetch_schema", "schema", "purpose", "declared_on",
        "eligibility_date", "start", "end", "interval_minutes", "adjusted",
        "session", "gap_policy", "manifest", "session_calendar", "series",
    }
    assert report["fetch_schema"] == 4
    assert report["manifest"] == {
        "path": str(manifest_path),
        "sha256": hashlib.sha256(frozen_bytes).hexdigest(),
    }
    assert report["session_calendar"] == {
        "path": str(DEFAULT_CALENDAR),
        "sha256": file_sha256(DEFAULT_CALENDAR),
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
            "query": {"date": "2024-10-31"},
            "active": True, "market": "stocks", "locale": "us",
            "type": "CS", "currency_name": "usd",
            "primary_exchange": "XNYS",
        }
        assert aggregate_contract["query"] == {
            "adjusted": "true", "sort": "asc", "limit": "50000",
        }
        assert aggregate_contract["path"] == (
            f"/v2/aggs/ticker/{ticker}/range/30/minute/"
            "2024-11-01/2026-07-21"
        )
        csv = item["csv"]
        path = Path(csv["path"])
        audit = csv["session_audit"]
        assert {name: value for name, value in csv.items()
                if name != "session_audit"} == {
            "path": str(resolved_output / f"{ticker.lower()}-30m.csv"),
            "rows": 1, "sessions": 1, "source_rows": 1,
            "sha256": file_sha256(path),
        }
        assert {
            name: audit[name] for name in (
                "scope", "expected_sessions", "affected_sessions",
                "expected_bins", "missing_bins",
            )
        } == {
            "scope": "all-expected-session-bins",
            "expected_sessions": 428,
            "affected_sessions": 428,
            "expected_bins": 5_534,
            "missing_bins": 5_533,
        }
        assert len(audit["missing_sessions"]) == 427
        assert audit["missing_sessions"][0] == "2024-11-04"
        assert audit["missing_sessions"][-1] == "2026-07-21"
        assert len(audit["ranges"]) == 428
        assert sum(item["absent_bins"] for item in audit["ranges"]) == 5_533
        assert audit["ranges"][0] == {
            "session": "2024-11-01",
            "start_timestamp": "2024-11-01T14:00:00Z",
            "end_timestamp": "2024-11-01T20:00:00Z",
            "absent_bins": 12,
        }
        assert audit["ranges"][-1] == {
            "session": "2026-07-21",
            "start_timestamp": "2026-07-21T13:30:00Z",
            "end_timestamp": "2026-07-21T20:00:00Z",
            "absent_bins": 13,
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


def test_observed_bar_gaps(directory: Path) -> None:
    manifest = directory / "gap-manifest.json"
    value = manifest_value()
    value["series"] = [{"ticker": "AAPL", "stratum": "generic"}]
    value["end"] = "2024-11-04"
    write_manifest(manifest, value)
    results = [
        aggregate("2024-11-01T13:00:00+00:00", 99.0),
        aggregate("2024-11-01T13:30:00+00:00", 100.0),
        aggregate("2024-11-01T14:30:00+00:00", 101.0),
        aggregate("2024-11-01T16:00:00+00:00", 102.0),
        aggregate("2024-11-01T20:00:00+00:00", 999.0),
        aggregate("2024-11-04T14:30:00+00:00", 103.0),
        aggregate("2024-11-04T16:00:00+00:00", 104.0),
    ]
    source = fetch_bars(
        aggregate_url("AAPL", date(2024, 7, 22), date(2024, 7, 23), 30, True),
        "fake-secret", "AAPL",
        lambda _url: {"status": "OK", "ticker": "AAPL", "results": results},
    )
    raises(ValueError, regular_bars, source, 30)
    bars, sessions, gaps = scan_regular_bars(source, 30)
    assert len(bars) == 5 and sessions == 2
    assert gaps == [
        {
            "session": "2024-11-01",
            "left_timestamp": "2024-11-01T13:30:00Z",
            "right_timestamp": "2024-11-01T14:30:00Z",
            "absent_bins": 1,
        },
        {
            "session": "2024-11-01",
            "left_timestamp": "2024-11-01T14:30:00Z",
            "right_timestamp": "2024-11-01T16:00:00Z",
            "absent_bins": 2,
        },
        {
            "session": "2024-11-04",
            "left_timestamp": "2024-11-04T14:30:00Z",
            "right_timestamp": "2024-11-04T16:00:00Z",
            "absent_bins": 2,
        },
    ]
    misaligned = list(bars)
    misaligned[1] = (
        timestamp("2024-11-01T14:45:00+00:00"), *misaligned[1][1:],
    )
    for invalid in (
        misaligned, [bars[0], bars[0]], [bars[1], bars[0]],
        [bars[3], bars[0]],
    ):
        raises(ValueError, scan_regular_bars, invalid, 30)

    output = directory / "gap-output"
    report_path = directory / "gap-report.json"

    def request(url: str) -> dict[str, object]:
        if urlsplit(url).path.startswith("/v3/reference/tickers/"):
            return {
                "status": "OK",
                "results": {
                    "ticker": "AAPL", "active": True, "market": "stocks",
                    "locale": "us", "type": "CS", "currency_name": "usd",
                    "primary_exchange": "XNYS",
                },
            }
        return {"status": "OK", "ticker": "AAPL", "results": results}

    report = fetch_universe(
        manifest, output, report_path, key="fake-secret", requester=request,
    )
    assert report["fetch_schema"] == 4
    assert report["gap_policy"] == "retain-observed-bars"
    csv = report["series"][0]["csv"]
    assert csv["session_audit"] == {
        "scope": "all-expected-session-bins",
        "expected_sessions": 2,
        "affected_sessions": 2,
        "missing_sessions": [],
        "expected_bins": 26,
        "missing_bins": 21,
        "ranges": [
            {
                "session": "2024-11-01",
                "start_timestamp": "2024-11-01T14:00:00Z",
                "end_timestamp": "2024-11-01T14:30:00Z",
                "absent_bins": 1,
            },
            {
                "session": "2024-11-01",
                "start_timestamp": "2024-11-01T15:00:00Z",
                "end_timestamp": "2024-11-01T16:00:00Z",
                "absent_bins": 2,
            },
            {
                "session": "2024-11-01",
                "start_timestamp": "2024-11-01T16:30:00Z",
                "end_timestamp": "2024-11-01T20:00:00Z",
                "absent_bins": 7,
            },
            {
                "session": "2024-11-04",
                "start_timestamp": "2024-11-04T15:00:00Z",
                "end_timestamp": "2024-11-04T16:00:00Z",
                "absent_bins": 2,
            },
            {
                "session": "2024-11-04",
                "start_timestamp": "2024-11-04T16:30:00Z",
                "end_timestamp": "2024-11-04T21:00:00Z",
                "absent_bins": 9,
            },
        ],
    }
    timestamps_, values = read_bars(Path(csv["path"]))
    assert timestamps_ == (
        "2024-11-01T13:30:00Z", "2024-11-01T14:30:00Z",
        "2024-11-01T16:00:00Z",
        "2024-11-04T14:30:00Z", "2024-11-04T16:00:00Z",
    )
    expected = tuple(
        value
        for close in (100.0, 101.0, 102.0, 103.0, 104.0)
        for value in (close - 0.25, close + 0.5, close - 0.5, close, 1000.0)
    )
    assert tuple(values) == expected


def test_session_grid_audit() -> None:
    calendar = SessionCalendar.read(DEFAULT_CALENDAR)
    observed = [
        (
            timestamp(value), 100.0, 100.0, 100.0, 100.0, 1.0,
        )
        for value in (
            "2024-11-01T14:00:00+00:00",
            "2024-11-01T15:00:00+00:00",
        )
    ]
    audit = session_grid_audit(
        observed, 30, calendar, date(2024, 11, 1), date(2024, 11, 4),
    )
    assert audit == {
        "scope": "all-expected-session-bins",
        "expected_sessions": 2,
        "affected_sessions": 2,
        "missing_sessions": ["2024-11-04"],
        "expected_bins": 26,
        "missing_bins": 24,
        "ranges": [
            {
                "session": "2024-11-01",
                "start_timestamp": "2024-11-01T13:30:00Z",
                "end_timestamp": "2024-11-01T14:00:00Z",
                "absent_bins": 1,
            },
            {
                "session": "2024-11-01",
                "start_timestamp": "2024-11-01T14:30:00Z",
                "end_timestamp": "2024-11-01T15:00:00Z",
                "absent_bins": 1,
            },
            {
                "session": "2024-11-01",
                "start_timestamp": "2024-11-01T15:30:00Z",
                "end_timestamp": "2024-11-01T20:00:00Z",
                "absent_bins": 9,
            },
            {
                "session": "2024-11-04",
                "start_timestamp": "2024-11-04T14:30:00Z",
                "end_timestamp": "2024-11-04T21:00:00Z",
                "absent_bins": 13,
            },
        ],
    }
    assert session_grid_audit(
        (), 30, calendar, date(2024, 11, 29), date(2024, 11, 29),
    ) == {
        "scope": "all-expected-session-bins",
        "expected_sessions": 1,
        "affected_sessions": 1,
        "missing_sessions": ["2024-11-29"],
        "expected_bins": 7,
        "missing_bins": 7,
        "ranges": [{
            "session": "2024-11-29",
            "start_timestamp": "2024-11-29T14:30:00Z",
            "end_timestamp": "2024-11-29T18:00:00Z",
            "absent_bins": 7,
        }],
    }
    outside = [
        (
            timestamp("2024-11-05T14:30:00+00:00"),
            100.0, 100.0, 100.0, 100.0, 1.0,
        )
    ]
    assert str(raises(
        ValueError, session_grid_audit,
        outside, 30, calendar, date(2024, 11, 1), date(2024, 11, 4),
    )) == "observed bar is outside the requested session grid"
    assert str(raises(
        ValueError, session_grid_audit,
        [observed[0], observed[0]], 30, calendar,
        date(2024, 11, 1), date(2024, 11, 4),
    )) == "observed session-grid starts are not unique"
    for minutes in (True, 0, 60):
        raises(
            ValueError, session_grid_audit,
            (), minutes, calendar, date(2024, 11, 1), date(2024, 11, 4),
        )
    short = SessionCalendar(
        date(2024, 11, 1), date(2024, 11, 1), 570, 960,
        ("XNAS", "XNYS"), (), ((date(2024, 11, 1), 580),),
    )
    assert session_grid_audit(
        (), 30, short, short.start, short.end,
    ) == {
        "scope": "all-expected-session-bins",
        "expected_sessions": 0,
        "affected_sessions": 0,
        "missing_sessions": [],
        "expected_bins": 0,
        "missing_bins": 0,
        "ranges": [],
    }


def test_calendar_filtering() -> None:
    results = [
        aggregate("2024-11-04T15:00:00+00:00", 100.0),
        aggregate("2024-11-04T16:00:00+00:00", 101.0),
        aggregate("2024-11-29T17:30:00+00:00", 102.0),
        aggregate("2024-11-29T18:00:00+00:00", 999.0),
        aggregate("2024-11-29T18:30:00+00:00", 999.0),
        aggregate("2025-01-09T14:30:00+00:00", 999.0),
    ]
    source = fetch_bars(
        aggregate_url(
            "AAPL", date(2024, 11, 4), date(2025, 1, 9), 30, True,
        ),
        "fake-secret", "AAPL",
        lambda _url: {
            "status": "OK", "ticker": "AAPL", "results": results,
        },
    )
    bars, sessions, gaps = scan_regular_bars(
        source, 30, SessionCalendar.read(DEFAULT_CALENDAR),
    )
    assert [bar[0] for bar in bars] == [
        timestamp("2024-11-04T15:00:00+00:00"),
        timestamp("2024-11-04T16:00:00+00:00"),
        timestamp("2024-11-29T17:30:00+00:00"),
    ]
    cross_close = [bar for bar in source if bar[0] == bars[-1][0]]
    before_close = (
        timestamp("2024-11-29T16:45:00+00:00"), *cross_close[0][1:],
    )
    assert scan_regular_bars(
        [before_close, *cross_close], 45,
        SessionCalendar.read(DEFAULT_CALENDAR),
    ) == ([before_close], 1, [])
    assert sessions == 2
    assert gaps == [{
        "session": "2024-11-04",
        "left_timestamp": "2024-11-04T15:00:00Z",
        "right_timestamp": "2024-11-04T16:00:00Z",
        "absent_bins": 1,
    }]


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
                    "primary_exchange": "XNYS",
                },
            }
        ticker = unquote(parts.path.split("/ticker/", 1)[1].split("/", 1)[0])
        paginated = parts.path.endswith("/page")
        result = aggregate(
            "2024-11-01T14:00:00+00:00" if paginated
            else "2024-11-01T13:30:00+00:00",
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
        "/v2/aggs/ticker/AAPL/range/30/minute/2024-11-01/2026-07-21",
        "/v2/aggs/ticker/AAPL/page",
        "/v3/reference/tickers/MSFT",
        "/v2/aggs/ticker/MSFT/range/30/minute/2024-11-01/2026-07-21",
    ]
    assert all(
        math.isclose(actual, index * 12.2)
        for index, actual in enumerate(starts)
    )
    assert report["series"][0]["csv"]["source_rows"] == 2
    assert "fake-secret" not in json.dumps(report)


def test_reference_identity(directory: Path) -> None:
    assert reference_url("AAPL", date(2024, 10, 31)) == (
        "https://api.massive.com/v3/reference/tickers/AAPL?date=2024-10-31"
    )
    for ticker in ("../AAPL", ".", "..", "-", ".-"):
        raises(ValueError, reference_url, ticker, date(2024, 10, 31))
    for ticker in ("BRK.B", "BF-B"):
        assert urlsplit(reference_url(ticker, date(2024, 10, 31))).path == (
            f"/v3/reference/tickers/{ticker}"
        )

    manifest = directory / "manifest.json"
    value = manifest_value()
    value["series"] = [{"ticker": "AAPL", "stratum": "generic"}]
    write_manifest(manifest, value)
    valid = {
        "ticker": "AAPL", "active": True, "market": "stocks",
        "locale": "us", "type": "CS", "currency_name": "usd",
        "primary_exchange": "XNYS",
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
        {"status": "OK", "results": valid | {"primary_exchange": "XASE"}},
        {"status": "OK", "results": valid | {"primary_exchange": 1}},
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
    calendar = directory / "mutable-calendar.json"
    calendar.write_bytes(DEFAULT_CALENDAR.read_bytes())
    output = directory / "calendar-mutation-output"
    report = directory / "calendar-mutation-report.json"
    mutated = False

    def mutate_calendar(_url: str) -> None:
        nonlocal mutated
        if not mutated:
            calendar.write_text('{"mutated":true}\n', encoding="ascii")
            mutated = True

    raises(
        ValueError, fetch_universe, manifest, output, report,
        calendar_path=calendar, key="fake-secret",
        requester=fake_requester([], mutate_calendar),
    )
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
        test_session_calendar(directory)
        test_selection_policy(directory)
        test_pure_selection()
        test_universe_sources()
        test_universe_source_rejections()
        test_universe_values()
        test_exclusive_writer(directory)
        test_selection_transport()
        test_select_universe_publication(directory)
        test_selection_publication_mutations(directory)
        test_selection_target_rejections(directory)
        test_selection_commit_failures(directory)
        test_selection_cli(directory)
        test_existing_downloader(directory)
        test_manifest_contract(directory)
        test_universe_coverage_overlay(directory)
        test_target_rejections(directory)
        test_rate_validation(directory)
        test_benchmark_import_boundary()
        test_benchmark_fetch(directory)
        test_benchmark_source_rejections(directory)
        test_benchmark_target_rejections(directory)
        test_benchmark_publication_failures(directory)
        test_universe_fetch(directory)
        test_observed_bar_gaps(directory)
        test_session_grid_audit()
        test_calendar_filtering()
        test_universe_pagination_gate(directory)
        test_reference_identity(directory)
        test_mutations(directory)
    print("Massive downloader and universe tests passed")


if __name__ == "__main__":
    main()
