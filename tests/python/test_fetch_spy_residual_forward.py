#!/usr/bin/env python3
"""Verify the strict, atomic SPY-residual forward bundle fetcher."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import arm_spy_residual_forward as armer
from tools.panel_contract import SourceTree, selected_source_tree
from tools.session_calendar import SessionCalendar, expected_bins
from tools.spy_residual_forward_contract import (
    FORWARD_CALENDAR, FORWARD_SOURCE_PATHS, FORWARD_UNIVERSE,
)


def rejects(function: Callable[..., object], *args: object, **kwargs: object) -> None:
    try:
        function(*args, **kwargs)
    except (OSError, TypeError, ValueError):
        return
    raise AssertionError("invalid forward bundle operation succeeded")


FORWARD_CALENDAR_PATH = ROOT / FORWARD_CALENDAR[1]


def aggregate(timestamp: str, close: float) -> dict[str, object]:
    return {
        "t": int(datetime.fromisoformat(timestamp).timestamp() * 1000),
        "o": close - 0.25,
        "h": close + 0.5,
        "l": close - 0.5,
        "c": close,
        "v": 1_000,
    }


def requester(*, missing: bool = False) -> Callable[[str], Mapping[str, object]]:
    calendar = SessionCalendar.read(FORWARD_CALENDAR_PATH)
    bins = tuple(expected_bins(calendar, calendar.start, calendar.end, 30))

    def request(url: str) -> Mapping[str, object]:
        ticker = url.split("/ticker/", 1)[1].split("/", 1)[0]
        rows = [aggregate(item.timestamp, 100.0 + index)
                for index, item in enumerate(bins, 1)]
        if missing and ticker == FORWARD_UNIVERSE[0]:
            rows.pop()
        return {
            "status": "OK", "ticker": ticker, "adjusted": True,
            "results": rows,
        }

    return request


def test_import_boundary() -> None:
    code = (
        f"import sys; sys.path.insert(0, {str(ROOT)!r}); "
        "import tools.fetch_spy_residual_forward; "
        "blocked=('torch','tools.train','tools.experiment',"
        "'tools.run_spy_residual_forward','tools.finalize_spy_residual_forward'); "
        "assert not any(name == item or name.startswith(item + '.') "
        "for item in blocked for name in sys.modules)"
    )
    subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )


def test_fetches_one_exact_bundle(directory: Path) -> None:
    from tools.fetch_spy_residual_forward import fetch_forward_bundle

    bundle = directory / "bundle"
    report = fetch_forward_bundle(
        bundle, key="fake-secret", requester=requester(),
        requests_per_minute=0,
        current_time=datetime(2026, 8, 6, tzinfo=timezone.utc),
    )
    assert [item["ticker"] for item in report["series"]] == [
        *FORWARD_UNIVERSE, "SPY",
    ]
    assert all(item["csv"]["rows"] == 702 for item in report["series"])
    assert SourceTree.parse(
        report["implementation_tree"], "forward fetch implementation",
        FORWARD_SOURCE_PATHS,
    ) == selected_source_tree(ROOT, FORWARD_SOURCE_PATHS)
    assert "fake-secret" not in (bundle / "fetch.json").read_text()
    with armer._bound_future(FORWARD_CALENDAR_PATH, bundle, FORWARD_CALENDAR[2]):
        pass


def test_rejects_early_or_incomplete_bundle(directory: Path) -> None:
    from tools.fetch_spy_residual_forward import fetch_forward_bundle

    early_calls = []

    def record_early_call(url: str) -> Mapping[str, object]:
        early_calls.append(url)
        return {}

    early_bundle = directory / "early"
    rejects(
        fetch_forward_bundle, early_bundle, key="fake-secret",
        requester=record_early_call, requests_per_minute=0,
        current_time=datetime(2026, 7, 27, tzinfo=timezone.utc),
    )
    assert early_calls == []
    assert not early_bundle.exists()
    incomplete_bundle = directory / "incomplete"
    rejects(
        fetch_forward_bundle, incomplete_bundle, key="fake-secret",
        requester=requester(missing=True), requests_per_minute=0,
        current_time=datetime(2026, 8, 6, tzinfo=timezone.utc),
    )
    assert not incomplete_bundle.exists()


def test_default_transport_rejects_forged_future_time(directory: Path) -> None:
    from tools import fetch_spy_residual_forward as forward

    calendar = SessionCalendar.read(FORWARD_CALENDAR_PATH)
    bins = tuple(expected_bins(calendar, calendar.start, calendar.end, 30))
    ready = datetime.fromisoformat(
        bins[-1].timestamp.replace("Z", "+00:00"),
    ) + timedelta(minutes=30)
    before_ready = ready - timedelta(microseconds=1)
    calls = 0

    class BeforeReadiness(datetime):
        @classmethod
        def now(cls, tz: timezone | None = None) -> datetime:
            assert tz is timezone.utc
            return before_ready

    def record_call(_url: str) -> Mapping[str, object]:
        nonlocal calls
        calls += 1
        return {}

    with patch.object(forward, "datetime", BeforeReadiness), \
         patch.object(forward, "request_json", record_call):
        for name, arguments in (
            ("default", {}),
            ("explicit", {"requester": forward.request_json}),
        ):
            bundle = directory / f"forged-time-{name}"
            try:
                forward.fetch_forward_bundle(
                    bundle, key="fake-secret", requests_per_minute=0,
                    current_time=datetime(2026, 8, 6, tzinfo=timezone.utc),
                    **arguments,
                )
            except ValueError as error:
                assert str(error) == "forward window is not complete"
            else:
                raise AssertionError("forged future time bypassed readiness")
            assert calls == 0
            assert not bundle.exists()
            assert not tuple(directory.glob(".compose-mini-forward-*"))


def test_sanitizes_authenticated_request_failure(directory: Path) -> None:
    from tools.fetch_spy_residual_forward import fetch_forward_bundle

    calls = 0
    authenticated = False

    def leaking(url: str) -> Mapping[str, object]:
        nonlocal calls, authenticated
        calls += 1
        authenticated = "apiKey=fake-secret" in url and \
            f"/ticker/{FORWARD_UNIVERSE[0]}/" in url
        raise RuntimeError(url)

    bundle = directory / "request-failure"
    try:
        fetch_forward_bundle(
            bundle, key="fake-secret", requester=leaking,
            requests_per_minute=0,
            current_time=datetime(2026, 8, 6, tzinfo=timezone.utc),
        )
    except ValueError as error:
        assert str(error) == "Massive forward request failed"
        assert "fake-secret" not in str(error)
        assert "fake-secret" not in repr(error)
        assert error.__context__ is None
        assert error.__cause__ is None
        traceback = error.__traceback__
        frame_names = []
        while traceback is not None:
            frame_names.append(traceback.tb_frame.f_code.co_name)
            local_values = tuple(traceback.tb_frame.f_locals.values())
            for value in local_values:
                assert "fake-secret" not in str(value)
                assert "fake-secret" not in repr(value)
            traceback = traceback.tb_next
        assert not {"leaking", "request", "transport", "fetch_bars"} & \
            set(frame_names)
    else:
        raise AssertionError("secret-bearing requester failure escaped")
    assert calls == 1
    assert authenticated
    assert not bundle.exists()
    assert not tuple(directory.glob(".compose-mini-forward-*"))


def test_validates_frozen_csv_before_reporting(directory: Path) -> None:
    from tools import fetch_spy_residual_forward as forward

    freeze = forward.freeze_inputs
    mutations = 0

    def mutate_then_freeze(paths: Sequence[Path]) -> object:
        nonlocal mutations
        csv_paths = tuple(
            path for path in paths
            if isinstance(path, Path) and path.suffix == ".csv"
        )
        if csv_paths:
            mutations += 1
            path = csv_paths[0]
            text = path.read_text(encoding="ascii")
            path.write_text(
                text.replace(",101,", ",102,", 1), encoding="ascii",
            )
        return freeze(paths)

    bundle = directory / "csv-mutation"
    with patch.object(forward, "freeze_inputs", mutate_then_freeze):
        rejects(
            forward.fetch_forward_bundle, bundle, key="fake-secret",
            requester=requester(), requests_per_minute=0,
            current_time=datetime(2026, 8, 6, tzinfo=timezone.utc),
        )
    assert mutations == 1
    assert not bundle.exists()
    assert not tuple(directory.glob(".compose-mini-forward-*"))


def test_cleans_private_stage_when_publication_fails(directory: Path) -> None:
    from tools.fetch_spy_residual_forward import fetch_forward_bundle

    bundle = directory / "failure"

    def fail(stage: Path, _target: Path, _verify: Callable[[], None]) -> object:
        assert stage.is_dir()
        raise OSError("publication failed")

    with patch("tools.fetch_spy_residual_forward.publish_directory", fail):
        rejects(
            fetch_forward_bundle, bundle, key="fake-secret",
            requester=requester(), requests_per_minute=0,
            current_time=datetime(2026, 8, 6, tzinfo=timezone.utc),
        )
    assert not bundle.exists()
    assert not tuple(directory.glob(".compose-mini-forward-*"))


def main() -> None:
    with tempfile.TemporaryDirectory(
        prefix="compose-mini-forward-fetch-", dir=ROOT,
    ) as name:
        directory = Path(name)
        test_import_boundary()
        test_fetches_one_exact_bundle(directory)
        test_rejects_early_or_incomplete_bundle(directory)
        test_default_transport_rejects_forged_future_time(directory)
        test_sanitizes_authenticated_request_failure(directory)
        test_validates_frozen_csv_before_reporting(directory)
        test_cleans_private_stage_when_publication_fails(directory)
    print("SPY-residual forward bundle fetcher tests passed")


if __name__ == "__main__":
    main()
