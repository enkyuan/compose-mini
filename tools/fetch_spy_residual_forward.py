#!/usr/bin/env python3
"""Fetch the fixed complete SPY-residual forward holdout bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit
import argparse
import json
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in tuple(map(os.path.realpath, sys.path)):
    sys.path.insert(0, str(ROOT))

from tools.atomic_bundle import (
    Identity, absent, path_identity, publish_directory, verify_identity,
    without_symlinks,
)
from tools.data_v1 import read_bars
from tools.fetch_massive import (
    API_HOST, Bar, Requester, aggregate_url, api_key, fetch_bars,
    request_gate, request_json, scan_regular_bars, session_grid_audit, write_csv,
)
from tools.files import file_sha256, freeze_inputs, require_disjoint, verify_frozen, write_json
from tools.relative_context_contract import INTERVAL_MINUTES
from tools.session_calendar import SessionCalendar, expected_bins
from tools.spy_residual_forward_contract import FORWARD_CALENDAR, FORWARD_UNIVERSE

FORWARD_CALENDAR_PATH = ROOT / FORWARD_CALENDAR[1]
TICKERS = (*FORWARD_UNIVERSE, "SPY")
FILES = tuple(sorted((
    "fetch.json", *(f"{ticker.lower()}-30m.csv" for ticker in TICKERS),
)))
PURPOSE = "Authenticate the fixed SPY-residual forward holdout."


def _paths(bundle: Path) -> Path:
    bundle = without_symlinks(bundle, "forward bundle")
    absent(bundle, "forward bundle")
    path_identity(bundle.parent, "forward output parent", directory=True)
    calendar = without_symlinks(FORWARD_CALENDAR_PATH, "forward calendar")
    if calendar != FORWARD_CALENDAR_PATH:
        raise ValueError("forward calendar path changed")
    path_identity(calendar, "forward calendar")
    require_disjoint((calendar,), (bundle,))
    if bundle in calendar.parents or calendar in bundle.parents:
        raise ValueError("forward bundle must not contain its calendar")
    return bundle


def _contract(url: str) -> dict[str, object]:
    parts = urlsplit(url)
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    names = tuple(name for name, _ in pairs)
    if parts.scheme != "https" or parts.netloc != API_HOST or "apiKey" in names or \
       len(names) != len(set(names)):
        raise ValueError("forward aggregate request contract is invalid")
    return {"path": parts.path, "query": dict(pairs)}


def _aggregate_requester(ticker: str, requester: Requester) -> Requester:
    def request(url: str) -> Mapping[str, object]:
        payload = requester(url)
        if not isinstance(payload, Mapping) or payload.get("ticker") != ticker or \
           payload.get("adjusted") is not True:
            raise ValueError("Massive returned the wrong forward aggregate")
        return payload

    return request


def _matches_source(path: Path, bars: Sequence[Bar]) -> bool:
    timestamps, values = read_bars(path)
    expected = tuple(
        datetime.fromtimestamp(bar[0] / 1000, timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ",
        ) for bar in bars
    )
    return timestamps == expected and tuple(values) == tuple(
        value for bar in bars for value in bar[1:]
    )


def _verify_bundle(
    bundle: Path, identity: Identity, bindings: Mapping[str, tuple[Identity, str]],
    report: Mapping[str, object], report_bytes: bytes,
) -> None:
    verify_identity(bundle, identity, "forward bundle", directory=True)
    if tuple(sorted(path.name for path in bundle.iterdir())) != FILES:
        raise ValueError("forward bundle contents changed")
    for name, (file_identity, sha256) in bindings.items():
        path = bundle / name
        verify_identity(path, file_identity, f"forward {name}")
        if file_sha256(path) != sha256:
            raise ValueError(f"forward {name} changed")
    if (bundle / "fetch.json").read_bytes() != report_bytes:
        raise ValueError("forward report changed")
    try:
        value = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("forward report changed") from error
    if value != report:
        raise ValueError("forward report changed")


def fetch_forward_bundle(
    bundle: Path, *, env_file: Path = ROOT / ".env", key: str | None = None,
    requester: Requester | None = None, requests_per_minute: int = 5,
    current_time: datetime | None = None,
) -> Mapping[str, object]:
    """Publish all fixed forward series only after the final bar is available."""
    bundle = _paths(bundle)
    parent_identity = path_identity(bundle.parent, "forward output parent", directory=True)
    calendar_identity = path_identity(FORWARD_CALENDAR_PATH, "forward calendar")
    with freeze_inputs((FORWARD_CALENDAR_PATH,)) as (calendar_input,):
        if calendar_input.sha256 != FORWARD_CALENDAR[2]:
            raise ValueError("forward calendar changed")
        calendar = SessionCalendar.read(calendar_input.snapshot)
        expected_times = tuple(item.timestamp for item in expected_bins(
            calendar, calendar.start, calendar.end, INTERVAL_MINUTES,
        ))
        ready = datetime.fromisoformat(expected_times[-1].replace("Z", "+00:00")) + \
            timedelta(minutes=INTERVAL_MINUTES)
        now = datetime.now(timezone.utc) if requester is None or \
            requester is request_json or current_time is None else current_time
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("forward current time must be timezone-aware")
        if now.astimezone(timezone.utc) < ready:
            raise ValueError("forward window is not complete")
        secret: str | None = None
        direct: Requester | None = None
        transport: Requester | None = None
        fetch_failed = False
        fetched: list[tuple[str, str, list[Bar], int]] = []
        try:
            secret = api_key(env_file) if key is None else key
            if not isinstance(secret, str) or not secret or \
               any(c.isspace() for c in secret):
                raise ValueError("MASSIVE_API_KEY is missing or invalid")
            gate = request_gate(requests_per_minute)
            direct = request_json if requester is None else requester

            def transport(url: str) -> Mapping[str, object]:
                gate()
                return direct(url)

            for ticker in TICKERS:
                aggregate = aggregate_url(
                    ticker, calendar.start, calendar.end, INTERVAL_MINUTES, True,
                )
                try:
                    source = fetch_bars(
                        aggregate, secret, ticker,
                        _aggregate_requester(ticker, transport),
                    )
                except Exception:
                    fetch_failed = True
                    break
                bars, _, _ = scan_regular_bars(source, INTERVAL_MINUTES, calendar)
                timestamps = tuple(datetime.fromtimestamp(
                    bar[0] / 1000, timezone.utc,
                ).strftime("%Y-%m-%dT%H:%M:%SZ") for bar in bars)
                if timestamps != expected_times:
                    raise ValueError(f"{ticker} forward grid is incomplete")
                audit = session_grid_audit(
                    bars, INTERVAL_MINUTES, calendar, calendar.start, calendar.end,
                )
                if audit != {
                    "scope": "all-expected-session-bins",
                    "expected_sessions": audit["expected_sessions"],
                    "affected_sessions": 0,
                    "missing_sessions": [],
                    "expected_bins": len(expected_times),
                    "missing_bins": 0,
                    "ranges": [],
                }:
                    raise ValueError(f"{ticker} forward session grid is incomplete")
                fetched.append((ticker, aggregate, bars, len(source)))
        finally:
            key = None
            secret = None
            requester = None
            direct = None
            transport = None
        if fetch_failed:
            raise ValueError("Massive forward request failed") from None

        with tempfile.TemporaryDirectory(
            prefix=".compose-mini-forward-", dir=bundle.parent,
        ) as directory:
            stage = Path(directory)
            stage_identity = path_identity(stage, "staged forward bundle", directory=True)
            csv_paths = []
            for ticker, _aggregate, bars, _source_rows in fetched:
                path = stage / f"{ticker.lower()}-30m.csv"
                write_csv(path, bars)
                csv_paths.append(path)

            with freeze_inputs(csv_paths) as csv_inputs:
                records: list[dict[str, object]] = []
                for (
                    ticker, aggregate, bars, source_rows,
                ), path, csv_input in zip(
                    fetched, csv_paths, csv_inputs, strict=True,
                ):
                    if not _matches_source(csv_input.snapshot, bars):
                        raise ValueError("staged forward CSV changed")
                    records.append({
                        "aggregate": _contract(aggregate),
                        "csv": {
                            "path": str(bundle / path.name),
                            "rows": len(expected_times),
                            "session_audit": {
                                "expected_bins": len(expected_times),
                                "missing_bins": 0,
                                "scope": "all-expected-session-bins",
                            },
                            "sha256": csv_input.sha256,
                            "source_rows": source_rows,
                        },
                        "ticker": ticker,
                    })
                report: dict[str, object] = {
                    "adjusted": True,
                    "calendar": {
                        "path": str(FORWARD_CALENDAR_PATH),
                        "sha256": calendar_input.sha256,
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
                report_path = stage / "fetch.json"
                write_json(report_path, report)
                with freeze_inputs((report_path,)) as (report_input,):
                    snapshots = (*csv_inputs, report_input)
                    bindings = {
                        path.name: (
                            path_identity(path, f"staged forward {path.name}"),
                            frozen.sha256,
                        )
                        for path, frozen in zip(
                            (*csv_paths, report_path), snapshots, strict=True,
                        )
                    }

                    def verify() -> None:
                        verify_frozen((calendar_input, *snapshots))
                        verify_identity(
                            bundle.parent, parent_identity,
                            "forward output parent", directory=True,
                        )
                        verify_identity(
                            FORWARD_CALENDAR_PATH, calendar_identity,
                            "forward calendar",
                        )
                        verify_identity(
                            stage, stage_identity,
                            "staged forward bundle", directory=True,
                        )
                        for path, (identity, _) in bindings.items():
                            verify_identity(
                                stage / path, identity, f"staged forward {path}",
                            )
                        absent(bundle, "forward bundle")

                    bundle_identity = publish_directory(stage, bundle, verify)
                    verify_frozen((calendar_input,))
                    verify_identity(
                        FORWARD_CALENDAR_PATH, calendar_identity,
                        "forward calendar",
                    )
                    verify_identity(
                        bundle.parent, parent_identity,
                        "forward output parent", directory=True,
                    )
                    _verify_bundle(
                        bundle, bundle_identity, bindings, report,
                        report_input.snapshot.read_bytes(),
                    )
                    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--requests-per-minute", type=int, default=5)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        report = fetch_forward_bundle(
            args.bundle, env_file=args.env_file,
            requests_per_minute=args.requests_per_minute,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
