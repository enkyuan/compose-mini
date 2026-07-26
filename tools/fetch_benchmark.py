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
import stat
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
from tools.files import (
    file_sha256, freeze_inputs, rename_may_have_committed, rename_noreplace,
    require_disjoint, verify_frozen, write_json,
)
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
Identity = tuple[int, int]
Verify = Callable[[], None]


def _absent(path: Path, name: str) -> None:
    if os.path.lexists(path):
        raise ValueError(f"{name} must not already exist")


def _identity(path: Path, name: str, *, directory: bool = False) -> Identity:
    try:
        value = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{name} must be a regular path") from error
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not kind(value.st_mode) or (
        not directory and value.st_nlink != 1
    ):
        raise ValueError(f"{name} must be a regular path")
    return value.st_dev, value.st_ino


def _verify_identity(
    path: Path, expected: Identity, name: str, *, directory: bool = False,
) -> None:
    if _identity(path, name, directory=directory) != expected:
        raise ValueError(f"{name} changed during the fetch")


def _without_symlinks(path: Path, name: str) -> Path:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if os.path.lexists(current) and current.is_symlink():
            raise ValueError(f"{name} must not traverse symlinks")
    return absolute


def _paths(bundle: Path, calendar: Path) -> tuple[Path, Path]:
    bundle = _without_symlinks(bundle, "benchmark bundle")
    calendar = _without_symlinks(calendar, "session calendar")
    _absent(bundle, "benchmark bundle")
    _identity(bundle.parent, "benchmark output parent", directory=True)
    _identity(calendar, "session calendar")
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


def _entry(
    directory_fd: int, name: str,
) -> tuple[Identity, int] | None:
    try:
        value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    return (value.st_dev, value.st_ino), stat.S_IFMT(value.st_mode)


def _publish_bundle(stage: Path, bundle: Path, verify: Verify) -> Identity:
    parent_fd = os.open(
        bundle.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
        getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        stage_identity = _identity(
            stage, "staged benchmark bundle", directory=True,
        )
        verify()
        stage_fd = os.open(
            stage.name,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            os.fsync(stage_fd)
        finally:
            os.close(stage_fd)

        failure: OSError | None = None
        try:
            rename_noreplace(
                parent_fd, stage.name, parent_fd, bundle.name,
            )
        except OSError as error:
            failure = error
        committed = rename_may_have_committed(failure) and \
            _entry(parent_fd, stage.name) is None and \
            _entry(parent_fd, bundle.name) == (
                stage_identity, stat.S_IFDIR,
            )
        if not committed:
            if failure is not None:
                raise failure
            raise OSError("benchmark bundle publication failed")
        os.fsync(parent_fd)
        return stage_identity
    finally:
        os.close(parent_fd)


def _verify_bundle(
    bundle: Path,
    bundle_identity: Identity,
    bindings: Mapping[str, tuple[Identity, str]],
    report: Mapping[str, object],
) -> None:
    _verify_identity(
        bundle, bundle_identity, "benchmark bundle", directory=True,
    )
    if tuple(sorted(path.name for path in bundle.iterdir())) != BUNDLE_FILES:
        raise ValueError("benchmark bundle contents changed")
    for name, (identity, sha256) in bindings.items():
        path = bundle / name
        _verify_identity(path, identity, f"benchmark {name}")
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
    parent_identity = _identity(
        bundle.parent, "benchmark output parent", directory=True,
    )
    calendar_identity = _identity(calendar_path, "session calendar")

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
            stage_identity = _identity(
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
                    csv_identity = _identity(
                        csv_path, "staged benchmark CSV",
                    )
                    report_identity = _identity(
                        report_path, "staged benchmark report",
                    )

                    def verify() -> None:
                        verify_frozen((
                            calendar_input, csv_input, report_input,
                        ))
                        _verify_identity(
                            bundle.parent, parent_identity,
                            "benchmark output parent", directory=True,
                        )
                        _verify_identity(
                            calendar_path, calendar_identity,
                            "session calendar",
                        )
                        _verify_identity(
                            stage, stage_identity,
                            "staged benchmark bundle", directory=True,
                        )
                        _verify_identity(
                            csv_path, csv_identity, "staged benchmark CSV",
                        )
                        _verify_identity(
                            report_path, report_identity,
                            "staged benchmark report",
                        )
                        _absent(bundle, "benchmark bundle")

                    bundle_identity = _publish_bundle(
                        stage, bundle, verify,
                    )
                    verify_frozen((calendar_input,))
                    _verify_identity(
                        calendar_path, calendar_identity, "session calendar",
                    )
                    _verify_identity(
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
