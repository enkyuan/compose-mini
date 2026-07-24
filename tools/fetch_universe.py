#!/usr/bin/env python3
"""Fetch a frozen point-in-time stock universe into strict CSV files."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.data_v1 import FEATURE_COUNT, read_csv
from tools.fetch_massive import (
    API_HOST, TICKER, Requester, aggregate_url, api_key, authorized_url,
    fetch_bars, regular_bars, request_gate, request_json, write_csv,
)
from tools.files import file_sha256, freeze_inputs, write_json

MANIFEST_FIELDS = {
    "schema", "purpose", "declared_on", "eligibility_date", "start", "end",
    "interval_minutes", "adjusted", "session", "series",
}
SERIES_FIELDS = {"ticker", "stratum"}


def _ticker_valid(value: object) -> bool:
    return isinstance(value, str) and TICKER.fullmatch(value) is not None and \
        any(character.isascii() and character.isalnum() for character in value)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"universe {name} must be nonempty text")
    return value


def _date(value: object, name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"universe {name} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"universe {name} must be an ISO date") from error
    if str(parsed) != value:
        raise ValueError(f"universe {name} must be an ISO date")
    return parsed


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("universe manifest contains a duplicate field")
        value[name] = item
    return value


@dataclass(frozen=True)
class SeriesSpec:
    ticker: str
    stratum: str


@dataclass(frozen=True)
class UniverseManifest:
    schema: int
    purpose: str
    declared_on: date
    eligibility_date: date
    start: date
    end: date
    interval_minutes: int
    adjusted: bool
    session: str
    series: tuple[SeriesSpec, ...]

    @classmethod
    def read(cls, path: Path) -> UniverseManifest:
        if not os.path.lexists(path) or path.is_symlink() or not path.is_file():
            raise ValueError("universe manifest must be a regular file")
        try:
            value = json.loads(
                path.read_bytes().decode("utf-8"),
                object_pairs_hook=_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("universe manifest is not valid JSON") from error
        if not isinstance(value, dict) or set(value) != MANIFEST_FIELDS:
            raise ValueError("universe manifest fields are invalid")
        if type(value["schema"]) is not int or value["schema"] != 1:
            raise ValueError("universe schema must be 1")
        purpose = _text(value["purpose"], "purpose")
        declared_on = _date(value["declared_on"], "declared_on")
        eligibility_date = _date(value["eligibility_date"], "eligibility_date")
        start = _date(value["start"], "start")
        end = _date(value["end"], "end")
        interval = value["interval_minutes"]
        if type(interval) is not int or not 1 <= interval <= 59:
            raise ValueError("universe interval_minutes must be from 1 to 59")
        if type(value["adjusted"]) is not bool:
            raise ValueError("universe adjusted must be boolean")
        if value["session"] != "regular":
            raise ValueError("universe session must be regular")
        series_value = value["series"]
        if not isinstance(series_value, list) or not series_value:
            raise ValueError("universe series must be a nonempty list")
        series = []
        for item in series_value:
            if not isinstance(item, dict) or set(item) != SERIES_FIELDS:
                raise ValueError("universe series fields are invalid")
            ticker = item["ticker"]
            if not _ticker_valid(ticker):
                raise ValueError("universe ticker is invalid")
            series.append(SeriesSpec(ticker, _text(item["stratum"], "stratum")))
        if len({item.ticker for item in series}) != len(series):
            raise ValueError("universe tickers must be unique")
        if declared_on < eligibility_date or eligibility_date != start or start > end:
            raise ValueError("universe date relationship is invalid")
        return cls(
            value["schema"], purpose, declared_on, eligibility_date, start, end,
            interval, value["adjusted"], value["session"], tuple(series),
        )


def reference_url(ticker: str, eligibility_date: date) -> str:
    if not _ticker_valid(ticker) or type(eligibility_date) is not date:
        raise ValueError("invalid ticker or eligibility date")
    path = f"/v3/reference/tickers/{quote(ticker)}"
    return urlunsplit((
        "https", API_HOST, path,
        urlencode({"date": str(eligibility_date)}), "",
    ))


def _absent(path: Path, name: str) -> None:
    if os.path.lexists(path):
        raise ValueError(f"{name} must not already exist")


def _validate_initial_targets(
    output_dir: Path,
    report_path: Path,
) -> tuple[Path, Path]:
    _absent(output_dir, "output directory")
    _absent(report_path, "report")
    lexical_output = Path(os.path.abspath(output_dir))
    lexical_report = Path(os.path.abspath(report_path))
    _absent(lexical_output, "lexically normalized output directory")
    _absent(lexical_report, "lexically normalized report")
    try:
        output = output_dir.resolve(strict=False)
        report = report_path.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError("output and report paths are invalid") from error
    _absent(output, "resolved output directory")
    _absent(report, "resolved report")
    if output == report or output in report.parents or report in output.parents:
        raise ValueError("output directory and report must not be nested")
    return output, report


def _recheck_targets(
    output_dir: Path,
    report_path: Path,
    csv_paths: Sequence[Path],
) -> None:
    for path in csv_paths:
        _absent(path, "universe CSV")
    _absent(output_dir, "resolved output directory")
    _absent(report_path, "resolved report")


def _contract(url: str) -> dict[str, object]:
    parts = urlsplit(url)
    return {
        "path": parts.path,
        "query": dict(parse_qsl(parts.query, keep_blank_values=True)),
    }


def _reference(
    ticker: str,
    eligibility_date: date,
    key: str,
    requester: Requester,
) -> dict[str, object]:
    url = reference_url(ticker, eligibility_date)
    payload = requester(authorized_url(url, key))
    result = payload.get("results") if isinstance(payload, Mapping) else None
    expected: dict[str, object] = {
        "ticker": ticker,
        "active": True,
        "market": "stocks",
        "locale": "us",
        "type": "CS",
        "currency_name": "usd",
    }
    if not isinstance(payload, Mapping) or payload.get("status") != "OK" or \
       not isinstance(result, Mapping) or result.get("active") is not True or \
       any(result.get(name) != value for name, value in expected.items()):
        raise ValueError("Massive returned an ineligible reference ticker")
    contract = _contract(url)
    contract.update((name, result[name]) for name in expected if name != "ticker")
    return contract


def _regular_file(path: Path, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} changed during the fetch")


def fetch_universe(
    manifest_path: Path,
    output_dir: Path,
    report_path: Path,
    *,
    key: str | None = None,
    requester: Requester | None = None,
    requests_per_minute: int = 0,
    clock: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> dict[str, object]:
    output_dir, report_path = _validate_initial_targets(
        output_dir, report_path,
    )
    if not os.path.lexists(manifest_path) or manifest_path.is_symlink() or \
       not manifest_path.is_file():
        raise ValueError("universe manifest must be a regular file")

    with freeze_inputs((manifest_path,)) as inputs:
        frozen = inputs[0]
        manifest = UniverseManifest.read(frozen.snapshot)
        csv_paths = tuple(
            output_dir / (
                f"{item.ticker.lower()}-{manifest.interval_minutes}m.csv"
            )
            for item in manifest.series
        )
        _recheck_targets(output_dir, report_path, csv_paths)

        gate = request_gate(
            requests_per_minute, clock=clock, sleeper=sleeper,
        )
        if requests_per_minute == 0:
            transport = request_json if requester is None else requester
        elif requester is None:
            transport = lambda url: request_json(
                url, before_request=gate,
            )
        else:
            direct = requester

            def transport(url: str) -> Mapping[str, object]:
                gate()
                return direct(url)

        secret = api_key(ROOT / ".env") if key is None else key
        if not secret or any(character.isspace() for character in secret):
            raise ValueError("MASSIVE_API_KEY is missing or invalid")

        records = []
        for item, path in zip(manifest.series, csv_paths, strict=True):
            reference = _reference(
                item.ticker, manifest.eligibility_date, secret, transport,
            )
            aggregate = aggregate_url(
                item.ticker, manifest.start, manifest.end,
                manifest.interval_minutes, manifest.adjusted,
            )
            aggregate_contract = _contract(aggregate)
            source = fetch_bars(aggregate, secret, item.ticker, transport)
            bars, sessions = regular_bars(source, manifest.interval_minutes)
            write_csv(path, bars)
            rows = len(read_csv(path)) // FEATURE_COUNT
            if rows != len(bars):
                raise ValueError("written universe CSV row count changed")
            records.append({
                "ticker": item.ticker,
                "stratum": item.stratum,
                "reference": reference,
                "aggregate": aggregate_contract,
                "csv": {
                    "path": str(path),
                    "rows": rows,
                    "sessions": sessions,
                    "source_rows": len(source),
                    "sha256": file_sha256(path),
                },
            })

        report: dict[str, object] = {
            "schema": manifest.schema,
            "purpose": manifest.purpose,
            "declared_on": str(manifest.declared_on),
            "eligibility_date": str(manifest.eligibility_date),
            "start": str(manifest.start),
            "end": str(manifest.end),
            "interval_minutes": manifest.interval_minutes,
            "adjusted": manifest.adjusted,
            "session": manifest.session,
            "manifest": {
                "path": str(manifest_path),
                "sha256": frozen.sha256,
            },
            "series": records,
        }

        _absent(report_path, "report")
        _regular_file(manifest_path, "universe manifest")
        if file_sha256(manifest_path) != frozen.sha256:
            raise ValueError("universe manifest changed during the fetch")
        for path, record in zip(csv_paths, records, strict=True):
            _regular_file(path, "universe CSV")
            csv = record["csv"]
            if not isinstance(csv, Mapping) or \
               file_sha256(path) != csv["sha256"]:
                raise ValueError("universe CSV changed during the fetch")
        write_json(report_path, report)
        return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--requests-per-minute", type=int, default=0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        report = fetch_universe(
            args.manifest, args.output_dir, args.report,
            requests_per_minute=args.requests_per_minute,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
