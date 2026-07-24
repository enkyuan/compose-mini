#!/usr/bin/env python3
"""Fetch Massive stock aggregates into compose-mini's strict CSV format."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo
import argparse
import csv
import json
import math
import os
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.data_v1 import CSV_HEADER, read_csv
from tools.float32 import f32
from tools.session_calendar import DEFAULT_CALENDAR, SessionCalendar

API_HOST = "api.massive.com"
EASTERN = ZoneInfo("America/New_York")
TICKER = re.compile(r"[A-Z0-9.-]{1,32}")
Bar = tuple[int, float, float, float, float, float]
Gap = dict[str, str | int]
Requester = Callable[[str], Mapping[str, object]]


def request_gate(
    requests_per_minute: int,
    *,
    clock: Callable[[], float] | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> Callable[[], None]:
    """Return a monotonic gate for physical request starts."""
    if type(requests_per_minute) is not int or \
       not 0 <= requests_per_minute <= 60:
        raise ValueError("requests_per_minute must be an integer from 0 to 60")
    if requests_per_minute == 0:
        return lambda: None
    current_time = time.monotonic if clock is None else clock
    sleep = time.sleep if sleeper is None else sleeper
    interval, next_start, last_time = 61.0 / requests_per_minute, None, None

    def read_time() -> float:
        nonlocal last_time
        now = current_time()
        if not math.isfinite(now) or \
           last_time is not None and now < last_time:
            raise ValueError("request clock must be finite and monotonic")
        last_time = now
        return now

    def gate() -> None:
        nonlocal next_start
        now = read_time()
        if next_start is not None and next_start > now:
            sleep(next_start - now)
            now = read_time()
            if now < next_start:
                raise ValueError("request sleeper returned before its deadline")
        next_start = now + interval

    return gate


def api_key(path: Path) -> str:
    """Read the key from the process first, then a local environment file."""
    value = os.environ.get("MASSIVE_API_KEY", "")
    if not value and path.exists():
        for line in path.read_text(encoding="ascii").splitlines():
            name, separator, candidate = line.partition("=")
            if separator and name == "MASSIVE_API_KEY":
                value = candidate.strip()
                break
    if not value or any(character.isspace() for character in value):
        raise ValueError("MASSIVE_API_KEY is missing or invalid")
    return value


def aggregate_url(ticker: str, start: date, end: date, minutes: int,
                  adjusted: bool) -> str:
    if not TICKER.fullmatch(ticker) or not 1 <= minutes <= 59 or start > end:
        raise ValueError("invalid ticker, minute interval, or date range")
    path = f"/v2/aggs/ticker/{quote(ticker)}/range/{minutes}/minute/{start}/{end}"
    query = urlencode({"adjusted": str(adjusted).lower(), "sort": "asc",
                       "limit": 50000})
    return urlunsplit(("https", API_HOST, path, query, ""))


def authorized_url(url: str, key: str) -> str:
    """Attach the secret only after verifying Massive owns the destination."""
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != API_HOST:
        raise ValueError("Massive pagination returned an untrusted URL")
    query = [(name, value) for name, value in parse_qsl(parts.query)
             if name != "apiKey"]
    query.append(("apiKey", key))
    return urlunsplit((parts.scheme, parts.netloc, parts.path,
                       urlencode(query), parts.fragment))


def request_json(
    url: str,
    *,
    before_request: Callable[[], None] | None = None,
) -> Mapping[str, object]:
    for attempt in range(5):
        try:
            request = Request(url, headers={"User-Agent": "compose-mini/1"})
            if before_request is not None:
                before_request()
            with urlopen(request, timeout=60) as response:
                payload = json.load(response)
        except HTTPError as error:
            if error.code != 429 or attempt == 4:
                raise ValueError(
                    f"Massive request failed with HTTP {error.code}"
                ) from error
            try:
                delay = float(error.headers.get("Retry-After", "13"))
            except ValueError:
                delay = 13.0
            error.close()
            time.sleep(min(60.0, max(1.0, delay)))
            continue
        except (OSError, URLError, json.JSONDecodeError) as error:
            raise ValueError("Massive request failed") from error
        if not isinstance(payload, dict):
            raise ValueError("Massive returned a non-object response")
        return payload
    raise AssertionError("unreachable")


def _bar(value: object) -> Bar:
    if not isinstance(value, dict) or type(value.get("t")) is not int:
        raise ValueError("Massive returned an invalid aggregate")
    timestamp = value["t"]
    try:
        prices = tuple(float(value[name]) for name in ("o", "h", "l", "c", "v"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Massive returned an invalid aggregate") from error
    open_, high, low, close, volume = prices
    if timestamp < 0 or timestamp % 1000 or \
       not all(math.isfinite(number) for number in prices) or \
       min(open_, high, low, close) <= 0 or volume < 0 or \
       low > min(open_, close) or high < max(open_, close) or low > high:
        raise ValueError("Massive returned an invalid aggregate")
    rounded = tuple(f32(number) for number in prices)
    if not all(math.isfinite(number) for number in rounded):
        raise ValueError("Massive aggregate exceeds binary32")
    return (timestamp, *rounded)


def fetch_bars(url: str, key: str, ticker: str,
               requester: Requester = request_json) -> list[Bar]:
    bars, seen, previous = [], set(), -1
    while url:
        if url in seen:
            raise ValueError("Massive pagination contains a cycle")
        seen.add(url)
        payload = requester(authorized_url(url, key))
        if payload.get("status") != "OK" or \
           payload.get("ticker", ticker) != ticker or \
           not isinstance(payload.get("results", []), list):
            raise ValueError("Massive returned an unsuccessful response")
        for value in payload.get("results", []):
            bar = _bar(value)
            if bar[0] <= previous:
                raise ValueError("Massive bars are not strictly chronological")
            bars.append(bar)
            previous = bar[0]
        next_url = payload.get("next_url", "")
        if not isinstance(next_url, str):
            raise ValueError("Massive returned an invalid pagination URL")
        url = next_url
    if not bars:
        raise ValueError("Massive returned no bars")
    return bars


def _timestamp(timestamp: int) -> str:
    return datetime.fromtimestamp(
        timestamp / 1000, timezone.utc,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")


def scan_regular_bars(
    bars: Sequence[Bar], minutes: int,
    calendar: SessionCalendar | None = None,
) -> tuple[list[Bar], int, list[Gap]]:
    """Return observed regular bars and a deterministic internal-gap audit."""
    sessions: dict[date, list[tuple[int, Bar]]] = {}
    previous = -1
    for bar in bars:
        local = datetime.fromtimestamp(bar[0] / 1000, timezone.utc).astimezone(EASTERN)
        minute = local.hour * 60 + local.minute
        bounds = (
            (570, 960) if calendar is None and local.weekday() < 5
            else calendar.session(local.date()) if calendar is not None
            else None
        )
        if bounds is not None and bounds[0] <= minute and \
           minute + minutes <= bounds[1]:
            if bar[0] <= previous:
                raise ValueError("Massive regular-session bars are not chronological")
            previous = bar[0]
            if local.second or local.microsecond or \
               (minute - bounds[0]) % minutes:
                raise ValueError("Massive bar is not aligned to the requested interval")
            sessions.setdefault(local.date(), []).append((minute, bar))
    selected: list[Bar] = []
    gaps: list[Gap] = []
    for day, session in sessions.items():
        if calendar is None and session[0][0] != 570:
            raise ValueError("Massive regular session has an internal gap")
        for left, right in zip(session, session[1:]):
            distance = right[0] - left[0]
            if distance > minutes:
                gaps.append({
                    "session": str(day),
                    "left_timestamp": _timestamp(left[1][0]),
                    "right_timestamp": _timestamp(right[1][0]),
                    "absent_bins": distance // minutes - 1,
                })
        selected.extend(bar for _, bar in session)
    if not selected:
        raise ValueError("Massive returned no regular-session bars")
    return selected, len(sessions), gaps


def regular_bars(
    bars: Sequence[Bar], minutes: int,
    calendar: SessionCalendar | None = None,
) -> tuple[list[Bar], int]:
    """Keep regular hours and reject internal gaps in each observed session."""
    selected, sessions, gaps = scan_regular_bars(bars, minutes, calendar)
    if gaps:
        raise ValueError("Massive regular session has an internal gap")
    return selected, sessions


def write_csv(path: Path, bars: Sequence[Bar]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="ascii", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(CSV_HEADER.split(","))
            for timestamp, *values in bars:
                writer.writerow((_timestamp(timestamp),
                                 *(format(value, ".9g") for value in values)))
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    read_csv(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", type=str.upper)
    parser.add_argument("start", type=date.fromisoformat)
    parser.add_argument("end", type=date.fromisoformat)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--calendar", type=Path, default=DEFAULT_CALENDAR)
    parser.add_argument("--unadjusted", action="store_true")
    parser.add_argument("--all-sessions", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        calendar = (
            None if args.all_sessions else SessionCalendar.read(args.calendar)
        )
        if calendar is not None and (
            args.start < calendar.start or args.end > calendar.end
        ):
            raise ValueError("session calendar does not cover the request")
        key = api_key(args.env_file)
        url = aggregate_url(args.ticker, args.start, args.end,
                            args.minutes, not args.unadjusted)
        source = fetch_bars(url, key, args.ticker)
        bars, sessions = (
            (list(source), 0) if args.all_sessions else
            regular_bars(source, args.minutes, calendar)
        )
        write_csv(args.output, bars)
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    report = {"adjusted": not args.unadjusted, "interval_minutes": args.minutes,
              "output": str(args.output), "rows": len(bars),
              "sessions": sessions, "source_rows": len(source),
              "ticker": args.ticker}
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
