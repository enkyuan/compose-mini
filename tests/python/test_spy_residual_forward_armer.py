#!/usr/bin/env python3
"""Verify the fixed-source, exact-bundle SPY-residual forward armer."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import patch
import json
import os
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import arm_spy_residual_forward as armer
from tools.data_v1 import CSV_HEADER
from tools.files import file_sha256, freeze_inputs, write_json
from tools.session_calendar import (
    SessionBin, SessionCalendar, expected_bins,
)
from tools.spy_residual_forward_contract import FORWARD_UNIVERSE


def rejects(function: Callable[..., object], *args: object) -> BaseException:
    try:
        function(*args)
    except (OSError, TypeError, ValueError) as error:
        return error
    raise AssertionError("invalid forward arming operation succeeded")


def calendar(start: date, end: date) -> SessionCalendar:
    return SessionCalendar(
        start, end, 570, 960, ("XNAS", "XNYS"), (), (),
    )


def write_calendar(path: Path, value: SessionCalendar) -> None:
    path.write_text(json.dumps({
        "close_minute": value.close_minute,
        "closed_dates": list(map(str, value.closed_dates)),
        "early_closes": {
            str(day): minute for day, minute in value.early_closes
        },
        "end": str(value.end),
        "open_minute": value.open_minute,
        "purpose": "Test the fixed forward calendar.",
        "schema": 1,
        "sources": ["https://example.com/calendar"],
        "start": str(value.start),
        "timezone": "America/New_York",
        "venues": list(value.venues),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_text(bins: Sequence[SessionBin], base: float) -> str:
    rows = [CSV_HEADER]
    for index, item in enumerate(bins, 1):
        open_ = base + index
        rows.append(",".join(map(str, (
            item.timestamp, open_, open_ + 2, open_ - 1, open_ + 1,
            1_000 + index,
        ))))
    return "\n".join(rows) + "\n"


def report(
    calendar_path: Path, value: SessionCalendar, bundle: Path,
) -> dict[str, object]:
    bins = tuple(expected_bins(
        value, value.start, value.end, 30,
    ))
    records = []
    for ticker in (*FORWARD_UNIVERSE, "SPY"):
        path = bundle / f"{ticker.lower()}-30m.csv"
        records.append({
            "aggregate": {
                "path": (
                    f"/v2/aggs/ticker/{ticker}/range/30/minute/"
                    f"{value.start}/{value.end}"
                ),
                "query": {
                    "adjusted": "true",
                    "limit": "50000",
                    "sort": "asc",
                },
            },
            "csv": {
                "path": str(path),
                "rows": len(bins),
                "session_audit": {
                    "expected_bins": len(bins),
                    "missing_bins": 0,
                    "scope": "all-expected-session-bins",
                },
                "sha256": file_sha256(path),
                "source_rows": len(bins),
            },
            "ticker": ticker,
        })
    return {
        "adjusted": True,
        "calendar": {
            "path": str(calendar_path),
            "sha256": file_sha256(calendar_path),
        },
        "end": str(value.end),
        "interval_minutes": 30,
        "provider": "massive",
        "purpose": armer.PURPOSE,
        "schema": 1,
        "series": records,
        "session": "regular",
        "start": str(value.start),
    }


def future_bundle(
    parent: Path,
) -> tuple[Path, Path, SessionCalendar]:
    parent.mkdir(parents=True)
    value = calendar(date(2026, 2, 21), date(2026, 5, 18))
    calendar_path, bundle = parent / "forward-calendar.json", parent / "data"
    bundle.mkdir()
    write_calendar(calendar_path, value)
    bins = tuple(expected_bins(value, value.start, value.end, 30))
    for index, ticker in enumerate((*FORWARD_UNIVERSE, "SPY"), 1):
        (bundle / f"{ticker.lower()}-30m.csv").write_text(
            csv_text(bins, 100.0 * index), encoding="ascii",
        )
    write_json(bundle / "fetch.json", report(calendar_path, value, bundle))
    return calendar_path, bundle, value


@contextmanager
def historical(
    parent: Path,
) -> Iterator[armer._Historical]:
    parent.mkdir(parents=True)
    value = calendar(date(2026, 2, 2), date(2026, 2, 20))
    bins = tuple(expected_bins(value, value.start, value.end, 30))
    paths = []
    for index, ticker in enumerate((*FORWARD_UNIVERSE, "SPY"), 1):
        path = parent / f"source-{ticker.lower()}.csv"
        path.write_text(csv_text(bins, 100.0 * index), encoding="ascii")
        paths.append(path)
    with freeze_inputs(paths) as frozen:
        yield armer._Historical(
            value, bins[-1].timestamp,
            tuple(zip(FORWARD_UNIVERSE, frozen[:-1], strict=True)),
            frozen[-1], lambda: None,
        )


def enter_future(
    calendar_path: Path, bundle: Path, sha256: str | None = None,
) -> armer._Future:
    with armer._bound_future(
        calendar_path, bundle, sha256 or file_sha256(calendar_path),
    ) as value:
        value.verify()
        return value


def test_requires_one_exact_massive_bundle() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-armer-bundle-", dir=ROOT,
    ) as directory:
        root = Path(directory)
        calendar_path, bundle, _ = future_bundle(root / "valid")
        value = enter_future(calendar_path, bundle)
        assert tuple(series for series, _ in value.stocks) == \
            FORWARD_UNIVERSE
        assert value.spy.source == bundle / "spy-30m.csv"
        rejects(enter_future, calendar_path, bundle, "0" * 64)

        extra_calendar, extra, _ = future_bundle(root / "extra")
        (extra / "unexpected.csv").write_text("x", encoding="ascii")
        rejects(enter_future, extra_calendar, extra)

        linked_calendar, linked, _ = future_bundle(root / "linked")
        left = linked / "krys-30m.csv"
        right = linked / "tgt-30m.csv"
        right.unlink()
        os.link(left, right)
        rejects(enter_future, linked_calendar, linked)


def test_rejects_report_and_grid_changes() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-armer-report-", dir=ROOT,
    ) as directory:
        root = Path(directory)
        calendar_path, bundle, _ = future_bundle(root / "report")
        report_path = bundle / "fetch.json"
        value = json.loads(report_path.read_text(encoding="utf-8"))
        value["series"] = list(reversed(value["series"]))
        write_json(report_path, value)
        rejects(enter_future, calendar_path, bundle)

        calendar_path, bundle, _ = future_bundle(root / "gap")
        csv = bundle / "krys-30m.csv"
        rows = csv.read_text(encoding="ascii").splitlines()
        csv.write_text("\n".join((rows[0], *rows[2:])) + "\n", encoding="ascii")
        value = report(calendar_path, calendar(
            date(2026, 2, 21), date(2026, 5, 18),
        ), bundle)
        write_json(bundle / "fetch.json", value)
        rejects(enter_future, calendar_path, bundle)


def test_public_lease_binds_the_calendar_first_grid() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-armer-public-", dir=ROOT,
    ) as directory:
        root = Path(directory)
        calendar_path, bundle, _ = future_bundle(root / "future")
        with historical(root / "historical") as source:
            run = root / "run"
            run.mkdir()

            @contextmanager
            def bound() -> Iterator[armer._Historical]:
                yield source

            with patch.object(armer, "_bound_historical", bound):
                binding = (
                    "forward_calendar", str(calendar_path),
                    file_sha256(calendar_path),
                )
                outputs = {
                    "FORWARD_CALENDAR": binding,
                    "FORWARD_DRAFT": run / "prediction-draft.jsonl",
                    "FORWARD_TRUTH_RECEIPT": run / "truth-access.json",
                }
                with patch.multiple(armer, **outputs):
                    with armer.arm_forward_inputs(bundle) as lease:
                        lease()
                        assert not hasattr(lease, "prepare")
                        assert len(lease.grid.target_sessions) == 60
                        assert len(lease.grid.triples) == 780
                        assert lease.grid.target_sessions[0] == \
                            date(2026, 2, 24)
                        session, _ = lease._prepare()
                        current = session.current()
                        rejects(lease._prepare)
                        rejects(session.submit, current, {})


def main() -> None:
    test_requires_one_exact_massive_bundle()
    test_rejects_report_and_grid_changes()
    test_public_lease_binds_the_calendar_first_grid()
    print("SPY residual forward armer tests passed")


if __name__ == "__main__":
    main()
