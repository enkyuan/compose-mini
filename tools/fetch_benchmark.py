#!/usr/bin/env python3
"""Fetch one atomic, audited SPY bundle for residual calibration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from types import MappingProxyType
from urllib.parse import parse_qsl, urlsplit
import argparse
import json
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in tuple(map(os.path.realpath, sys.path)):
    sys.path.insert(0, str(ROOT))

from tools.data_v1 import read_bars
from tools.fetch_massive import (
    API_HOST, Bar, Requester, aggregate_url, api_key, authorized_url,
    fetch_bars, request_json, scan_regular_bars, session_grid_audit, write_csv,
)
from tools.fetch_universe import reference_url
from tools import atomic_bundle
from tools.files import file_sha256, freeze_inputs, require_disjoint, verify_frozen, write_json
from tools.relative_context_contract import validate_spy_session_audit
from tools.session_calendar import DEFAULT_CALENDAR, SessionCalendar

TICKER = "SPY"
REFERENCE_DATE = date(2024, 10, 31)
START = date(2024, 11, 1)
END = date(2026, 7, 21)
INTERVAL_MINUTES = 30
CALENDAR_SHA256 = \
    "b1e0835a60624a67e21f7941ac00ece6c488937989560bbd4d0333afd869e5f8"
REFERENCE = MappingProxyType({
    "ticker": TICKER,
    "active": True,
    "market": "stocks",
    "locale": "us",
    "type": "ETF",
    "currency_name": "usd",
    "primary_exchange": "ARCX",
})
APPLICABILITY = MappingProxyType({
    "benchmark": TICKER,
    "calendar_venue": "XNYS",
    "exchange_source":
        "https://massive.com/docs/rest/stocks/market-operations/exchanges",
    "market_group": "NYSE Group",
    "operating_mic": "XNYS",
    "primary_exchange": "ARCX",
    "session": "core",
    "session_source": "https://www.nyse.com/trade/hours-calendars",
})
PURPOSE = "Authenticate SPY bars for development-only residual calibration."
RETURN_BASIS = "split-adjusted-price-return-not-dividend-adjusted"
BUNDLE_FILES = ("fetch.json", "spy.csv")
Verify = Callable[[], None]
Identity = atomic_bundle.Identity


def _paths(bundle: Path, calendar: Path) -> tuple[Path, Path]:
    bundle = atomic_bundle.without_symlinks(bundle, "benchmark bundle")
    calendar = atomic_bundle.without_symlinks(calendar, "session calendar")
    atomic_bundle.absent(bundle, "benchmark bundle")
    atomic_bundle.path_identity(bundle.parent, "benchmark output parent", directory=True)
    atomic_bundle.path_identity(calendar, "session calendar")
    require_disjoint((calendar,), (bundle,))
    if bundle in calendar.parents or calendar in bundle.parents:
        raise ValueError("benchmark bundle must not contain its calendar")
    return bundle, calendar


def _contract(url: str) -> dict[str, object]:
    parts = urlsplit(url)
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    names = tuple(name for name, _ in pairs)
    if parts.scheme != "https" or parts.netloc != API_HOST or \
       "apiKey" in names or len(names) != len(set(names)):
        raise ValueError("benchmark request contract is invalid")
    return {"path": parts.path, "query": dict(pairs)}


def _reference(key: str, requester: Requester) -> dict[str, object]:
    url = reference_url(TICKER, REFERENCE_DATE)
    payload = requester(authorized_url(url, key))
    result = payload.get("results") if isinstance(payload, Mapping) else None
    if not isinstance(payload, Mapping) or payload.get("status") != "OK" or \
       not isinstance(result, Mapping) or \
       result.get("active") is not True or any(
           result.get(name) != value for name, value in REFERENCE.items()
       ):
        raise ValueError("Massive returned the wrong benchmark identity")
    return {"identity": dict(REFERENCE), "request": _contract(url)}


def _aggregate_requester(requester: Requester) -> Requester:
    def request(url: str) -> Mapping[str, object]:
        payload = requester(url)
        if not isinstance(payload, Mapping) or \
           payload.get("ticker") != TICKER or \
           payload.get("adjusted") is not True:
            raise ValueError("Massive returned the wrong benchmark aggregate")
        return payload

    return request


def _matches_source(path: Path, bars: Sequence[Bar]) -> bool:
    timestamps, values = read_bars(path)
    expected_timestamps = tuple(
        datetime.fromtimestamp(
            bar[0] / 1000, timezone.utc,
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        for bar in bars
    )
    return timestamps == expected_timestamps and tuple(values) == tuple(
        value for bar in bars for value in bar[1:]
    )


def _verify_bundle(
    bundle: Path,
    bundle_identity: Identity,
    bindings: Mapping[str, tuple[Identity, str]],
    report: Mapping[str, object],
) -> None:
    atomic_bundle.verify_identity(
        bundle, bundle_identity, "benchmark bundle", directory=True,
    )
    if tuple(sorted(path.name for path in bundle.iterdir())) != BUNDLE_FILES:
        raise ValueError("benchmark bundle contents changed")
    for name, (identity, sha256) in bindings.items():
        path = bundle / name
        atomic_bundle.verify_identity(path, identity, f"benchmark {name}")
        if file_sha256(path) != sha256:
            raise ValueError(f"benchmark {name} changed")
    try:
        value = json.loads((bundle / "fetch.json").read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("benchmark report changed") from error
    if value != report:
        raise ValueError("benchmark report changed")


def fetch_benchmark(
    bundle: Path,
    *,
    calendar_path: Path = DEFAULT_CALENDAR,
    env_file: Path = ROOT / ".env",
    key: str | None = None,
    requester: Requester | None = None,
) -> Mapping[str, object]:
    """Publish SPY only after its identity and complete grid are authenticated."""
    bundle, calendar_path = _paths(bundle, calendar_path)
    parent_identity = atomic_bundle.path_identity(
        bundle.parent, "benchmark output parent", directory=True,
    )
    calendar_identity = atomic_bundle.path_identity(calendar_path, "session calendar")

    with freeze_inputs((calendar_path,)) as (calendar_input,):
        if calendar_input.sha256 != CALENDAR_SHA256:
            raise ValueError("benchmark session calendar changed")
        calendar = SessionCalendar.read(calendar_input.snapshot)
        if START < calendar.start or END > calendar.end or \
           APPLICABILITY["calendar_venue"] not in calendar.venues:
            raise ValueError("session calendar does not cover SPY")
        secret = api_key(env_file) if key is None else key
        if not isinstance(secret, str) or not secret or any(
            character.isspace() for character in secret
        ):
            raise ValueError("MASSIVE_API_KEY is missing or invalid")
        transport = request_json if requester is None else requester
        reference = _reference(secret, transport)
        aggregate = aggregate_url(
            TICKER, START, END, INTERVAL_MINUTES, True,
        )
        source = fetch_bars(
            aggregate, secret, TICKER, _aggregate_requester(transport),
        )
        bars, sessions, _ = scan_regular_bars(
            source, INTERVAL_MINUTES, calendar,
        )
        audit = validate_spy_session_audit(session_grid_audit(
            bars, INTERVAL_MINUTES, calendar, START, END,
        ))

        with tempfile.TemporaryDirectory(
            prefix=".compose-mini-spy-", dir=bundle.parent,
        ) as directory:
            stage = Path(directory)
            stage_identity = atomic_bundle.path_identity(
                stage, "staged benchmark bundle", directory=True,
            )
            csv_path, report_path = stage / "spy.csv", stage / "fetch.json"
            write_csv(csv_path, bars)
            with freeze_inputs((csv_path,)) as (csv_input,):
                if not _matches_source(csv_input.snapshot, bars):
                    raise ValueError("staged benchmark CSV changed")
                report: dict[str, object] = {
                    "adjusted": True,
                    "aggregate": {"request": _contract(aggregate)},
                    "calendar": {
                        "applicability": dict(APPLICABILITY),
                        "path": str(calendar_path),
                        "sha256": calendar_input.sha256,
                    },
                    "csv": {
                        "path": str(bundle / "spy.csv"),
                        "rows": len(bars),
                        "sessions": sessions,
                        "session_audit": dict(audit),
                        "sha256": csv_input.sha256,
                        "source_rows": len(source),
                    },
                    "end": str(END),
                    "gap_policy": "require-complete-core-session-grid",
                    "interval_minutes": INTERVAL_MINUTES,
                    "purpose": PURPOSE,
                    "reference": reference,
                    "reference_date": str(REFERENCE_DATE),
                    "return_basis": RETURN_BASIS,
                    "schema": 1,
                    "session": "regular",
                    "start": str(START),
                    "ticker": TICKER,
                }
                write_json(report_path, report)
                with freeze_inputs((report_path,)) as (report_input,):
                    csv_identity = atomic_bundle.path_identity(
                        csv_path, "staged benchmark CSV",
                    )
                    report_identity = atomic_bundle.path_identity(
                        report_path, "staged benchmark report",
                    )

                    def verify() -> None:
                        verify_frozen((
                            calendar_input, csv_input, report_input,
                        ))
                        atomic_bundle.verify_identity(
                            bundle.parent, parent_identity,
                            "benchmark output parent", directory=True,
                        )
                        atomic_bundle.verify_identity(
                            calendar_path, calendar_identity,
                            "session calendar",
                        )
                        atomic_bundle.verify_identity(
                            stage, stage_identity,
                            "staged benchmark bundle", directory=True,
                        )
                        atomic_bundle.verify_identity(
                            csv_path, csv_identity, "staged benchmark CSV",
                        )
                        atomic_bundle.verify_identity(
                            report_path, report_identity,
                            "staged benchmark report",
                        )
                        atomic_bundle.absent(bundle, "benchmark bundle")

                    bundle_identity = atomic_bundle.publish_directory(
                        stage, bundle, verify,
                    )
                    verify_frozen((calendar_input,))
                    atomic_bundle.verify_identity(
                        calendar_path, calendar_identity, "session calendar",
                    )
                    atomic_bundle.verify_identity(
                        bundle.parent, parent_identity,
                        "benchmark output parent", directory=True,
                    )
                    _verify_bundle(
                        bundle, bundle_identity, {
                            "fetch.json": (
                                report_identity, report_input.sha256,
                            ),
                            "spy.csv": (
                                csv_identity, csv_input.sha256,
                            ),
                        }, report,
                    )
                    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        report = fetch_benchmark(
            args.bundle,
            calendar_path=args.calendar, env_file=args.env_file,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
