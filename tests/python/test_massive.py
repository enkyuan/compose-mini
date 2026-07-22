#!/usr/bin/env python3
"""Verify Massive pagination, filtering, and CSV conversion without network."""

from datetime import datetime, timezone
from email.message import Message
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.data_v1 import FEATURE_COUNT, read_csv
from tools.fetch_massive import (
    aggregate_url, api_key, authorized_url, fetch_bars, regular_bars,
    request_json, write_csv,
)


def timestamp(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp() * 1000)


def aggregate(value: str, close: float) -> dict[str, object]:
    return {"t": timestamp(value), "o": close - 0.25, "h": close + 0.5,
            "l": close - 0.5, "c": close, "v": 1000}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="compose-mini-massive-") as directory:
        env = Path(directory) / ".env"
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

        initial = aggregate_url("AAPL", datetime(2026, 7, 1).date(),
                                datetime(2026, 7, 2).date(), 30, True)
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
        try:
            authorized_url("https://example.com/page", "secret")
        except ValueError:
            pass
        else:
            raise AssertionError("untrusted pagination URL was accepted")

        headers = Message()
        headers["Retry-After"] = "0"
        responses = iter((HTTPError("https://api.massive.com", 429, "rate limited",
                                    headers, BytesIO()),
                          BytesIO(b'{"status":"OK"}')))

        def urlopen_once(*_args: object, **_kwargs: object) -> BytesIO:
            response = next(responses)
            if isinstance(response, HTTPError):
                raise response
            return response

        with patch("tools.fetch_massive.urlopen", side_effect=urlopen_once), \
             patch("tools.fetch_massive.time.sleep") as sleep:
            assert request_json("https://api.massive.com") == {"status": "OK"}
            sleep.assert_called_once_with(1.0)

        path = Path(directory) / "bars.csv"
        write_csv(path, bars)
        assert len(read_csv(path)) == len(bars) * FEATURE_COUNT
    print("Massive downloader tests passed")


if __name__ == "__main__":
    main()
