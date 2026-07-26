#!/usr/bin/env python3
"""Verify chronological, publication-gated SPY-residual forward inputs."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from math import isclose, log
from pathlib import Path
from unittest.mock import patch
import os
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.data_v1 import CSV_HEADER
from tools.files import freeze_inputs
from tools.float32 import f32
from tools.panel_contract import read_canonical_json, read_canonical_json_lines
from tools.relative_context_contract import SEEDS
from tools.session_calendar import (
    SessionBin, SessionCalendar, expected_bins,
)
from tools.spy_residual_forward_contract import FORWARD_UNIVERSE
from tools.spy_residual_forward_inputs import (
    ForwardPredictionSession, ForwardSeriesFiles, PredictionDraft,
    SpyResidualForwardInputs, _prepare_forward_inputs, derive_forward_grid,
)
from tools import spy_residual_forward_inputs as forward_inputs


def rejects(function: Callable[..., object], *args: object) -> BaseException:
    try:
        function(*args)
    except (OSError, TypeError, ValueError) as error:
        return error
    raise AssertionError("invalid forward input operation succeeded")


def calendar(start: date, end: date) -> SessionCalendar:
    return SessionCalendar(
        start, end, 570, 960, ("XNAS", "XNYS"), (), (),
    )


@dataclass(frozen=True, slots=True)
class Fixture:
    source_calendar: SessionCalendar
    future_calendar: SessionCalendar
    boundary: str
    source: tuple[tuple[str, Path], ...]
    future: tuple[tuple[str, Path], ...]
    source_spy: Path
    future_spy: Path
    ledger: Path
    receipt: Path
    source_bins: tuple[SessionBin, ...]
    future_bins: tuple[SessionBin, ...]


def parameters(series: str) -> tuple[float, float]:
    if series == "SPY":
        return 300.0, 0.00005
    index = FORWARD_UNIVERSE.index(series) + 1
    return 80.0 + 10.0 * index, 0.0001 * index


def bar(index: int, series: str) -> tuple[float, ...]:
    base, drift = parameters(series)
    open_ = base + index * 0.25
    close = open_ * (1.0 + drift * (index % 17 + 1))
    return tuple(map(f32, (
        open_, max(open_, close) + 1.0, min(open_, close) - 1.0,
        close, 1_000.0 + index,
    )))


def csv_text(
    bins: Sequence[SessionBin], series: str, *,
    invalid_after: str | None = None,
    missing: str | None = None,
) -> str:
    rows = [CSV_HEADER]
    for index, item in enumerate(bins, 1):
        if item.timestamp == missing:
            continue
        values: Sequence[object] = (
            ("x",) * 5 if invalid_after is not None and
            item.timestamp > invalid_after else bar(index, series)
        )
        rows.append(",".join((item.timestamp, *map(str, values))))
    return "\n".join(rows) + "\n"


def fixture(
    parent: Path, *,
    invalid_after: str | None = None,
    gap_series: str | None = None,
    gap_timestamp: str | None = None,
    boundary_bin: int = -1,
    poison_source_tail: bool = False,
    source_end: date = date(2026, 2, 20),
) -> Fixture:
    parent.mkdir(parents=True, exist_ok=True)
    source_calendar = calendar(date(2026, 2, 2), source_end)
    future_calendar = calendar(date(2026, 2, 21), date(2026, 6, 5))
    source_bins = tuple(expected_bins(
        source_calendar, source_calendar.start, source_calendar.end, 30,
    ))
    boundary = tuple(expected_bins(
        source_calendar, date(2026, 2, 20), date(2026, 2, 20), 30,
    ))[boundary_bin].timestamp
    future_bins = tuple(expected_bins(
        future_calendar, future_calendar.start, future_calendar.end, 30,
    ))
    source_root, future_root = parent / "source", parent / "future"
    source_root.mkdir()
    future_root.mkdir()

    source, future = [], []
    for series in FORWARD_UNIVERSE:
        name = f"{series.lower()}-30m.csv"
        source_path, future_path = source_root / name, future_root / name
        source_path.write_text(
            csv_text(
                source_bins, series,
                invalid_after=boundary if poison_source_tail else None,
            ),
            encoding="ascii",
        )
        future_path.write_text(csv_text(
            future_bins, series, invalid_after=invalid_after,
            missing=gap_timestamp if series == gap_series else None,
        ), encoding="ascii")
        source.append((series, source_path))
        future.append((series, future_path))

    source_spy, future_spy = (
        source_root / "spy-30m.csv", future_root / "spy-30m.csv",
    )
    source_spy.write_text(
        csv_text(
            source_bins, "SPY",
            invalid_after=boundary if poison_source_tail else None,
        ),
        encoding="ascii",
    )
    future_spy.write_text(
        csv_text(future_bins, "SPY", invalid_after=invalid_after),
        encoding="ascii",
    )
    return Fixture(
        source_calendar, future_calendar, boundary,
        tuple(source), tuple(future), source_spy, future_spy,
        parent / "predictions.jsonl", parent / "truth-access.json",
        source_bins, future_bins,
    )


@contextmanager
def frozen_files(
    value: Fixture,
) -> Iterator[tuple[
    tuple[ForwardSeriesFiles, ...], ForwardSeriesFiles, Callable[[], None],
]]:
    paths = tuple(
        path for pair in zip(
            (path for _, path in value.source),
            (path for _, path in value.future),
            strict=True,
        ) for path in pair
    ) + (value.source_spy, value.future_spy)
    with freeze_inputs(paths) as frozen:
        by_path = dict(zip(paths, frozen, strict=True))
        stocks = tuple(
            ForwardSeriesFiles(
                series, by_path[source], by_path[future],
            )
            for (series, source), (_, future) in zip(
                value.source, value.future, strict=True,
            )
        )
        spy = ForwardSeriesFiles(
            "SPY", by_path[value.source_spy], by_path[value.future_spy],
        )
        yield stocks, spy, lambda: None


def raw_predictions(index: int) -> dict[str, tuple[float, ...]]:
    return {
        series: tuple(
            (index + 1) * (member + 1) * (seed_index + 1) / 1_000_000
            for seed_index, _ in enumerate(SEEDS)
        )
        for member, series in enumerate(FORWARD_UNIVERSE)
    }


def publish(session: ForwardPredictionSession) -> PredictionDraft:
    result: SpyResidualForwardInputs | PredictionDraft = \
        session.current()
    while isinstance(result, SpyResidualForwardInputs):
        assert result.index < 780
        result = session.submit(result, raw_predictions(result.index))
    return result


def prepare(
    value: Fixture,
    stocks: Sequence[ForwardSeriesFiles],
    spy: ForwardSeriesFiles,
    verify: Callable[[], None],
) -> tuple[ForwardPredictionSession, Callable[..., Mapping[str, object]]]:
    grid = derive_forward_grid(
        value.source_calendar, value.future_calendar, value.boundary,
    )
    return _prepare_forward_inputs(
        grid, value.source_calendar, value.future_calendar,
        stocks, spy, value.ledger, value.receipt, verify,
    )


def test_selects_exactly_60_calendar_first_target_sessions() -> None:
    source = calendar(date(2026, 2, 2), date(2026, 2, 20))
    future = calendar(date(2026, 2, 21), date(2026, 6, 5))
    boundary = tuple(expected_bins(
        source, source.start, source.end, 30,
    ))[-1].timestamp
    grid = derive_forward_grid(source, future, boundary)
    targets = tuple(row[2][:10] for row in grid.triples)
    sessions = tuple(dict.fromkeys(targets))

    assert grid.boundary == boundary
    assert len(grid.target_sessions) == 60
    assert grid.target_sessions[0] == date(2026, 2, 24)
    assert tuple(map(str, grid.target_sessions)) == sessions
    assert targets.count(sessions[0]) == 13
    assert len(grid.triples) == 780
    assert all(entry > boundary for _, entry, _ in grid.triples)
    assert derive_forward_grid(source, future, boundary) == grid
    rejects(
        derive_forward_grid, source, future, "2026-02-10T20:30:00Z",
    )

    later = calendar(future.start, date(2026, 6, 30))
    extended = derive_forward_grid(source, later, boundary)
    assert extended.target_sessions == grid.target_sessions
    assert extended.triples == grid.triples

    source = SessionCalendar.read(
        ROOT / "universes/us-equities-core-2024-07-22_2026-07-21.json",
    )
    future = SessionCalendar.read(
        ROOT /
        "universes/us-equities-core-forward-2026-05-19_2026-08-18.json",
    )
    grid = derive_forward_grid(
        source, future, "2026-05-18T18:00:00Z",
    )
    assert (
        grid.target_sessions[0], grid.target_sessions[-1],
        len(grid.target_sessions),
    ) == (date(2026, 5, 22), date(2026, 8, 18), 60)
    assert len(grid.triples) == 780


def test_requires_complete_unique_ordered_common_grid() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-grid-", dir=ROOT,
    ) as directory:
        value = fixture(Path(directory))
        grid = derive_forward_grid(
            value.source_calendar, value.future_calendar, value.boundary,
        )
        with frozen_files(value) as (stocks, spy, verify):
            session, _ = prepare(value, stocks, spy, verify)
            current = session.current()
            assert tuple(item.series for item in current.stocks) == \
                FORWARD_UNIVERSE
            assert current.spy.series == "SPY"
            assert len(current.spy.values) == 17 * 5
            rejects(
                _prepare_forward_inputs, grid, value.source_calendar,
                value.future_calendar, stocks[::-1], spy, value.ledger,
                value.receipt, verify,
            )
            rejects(
                _prepare_forward_inputs, grid, value.source_calendar,
                value.future_calendar, stocks[:-1], spy, value.ledger,
                value.receipt, verify,
            )
            aliased = (
                stocks[0],
                ForwardSeriesFiles(
                    stocks[1].series, stocks[0].source, stocks[1].future,
                ),
                *stocks[2:],
            )
            rejects(
                _prepare_forward_inputs, grid, value.source_calendar,
                value.future_calendar, aliased, spy, value.ledger,
                value.receipt, verify,
            )

        missing = grid.triples[-1][2]
        damaged = fixture(
            Path(directory) / "damaged", gap_series=FORWARD_UNIVERSE[0],
            gap_timestamp=missing,
        )
        with frozen_files(damaged) as (stocks, spy, verify):
            rejects(prepare, damaged, stocks, spy, verify)


def test_rejects_links_and_replacements() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-identity-", dir=ROOT,
    ) as directory:
        root = Path(directory)
        linked = fixture(root / "linked")
        target = linked.source[0][1]
        alias = linked.source[1][1]
        alias.unlink()
        os.link(target, alias)
        with frozen_files(linked) as (stocks, spy, verify):
            rejects(prepare, linked, stocks, spy, verify)

        symlinked = fixture(root / "symlinked")
        target = symlinked.future[0][1]
        alias = symlinked.future[1][1]
        alias.unlink()
        alias.symlink_to(target)
        with frozen_files(symlinked) as (stocks, spy, verify):
            rejects(prepare, symlinked, stocks, spy, verify)

        replaced = fixture(root / "replaced")
        with frozen_files(replaced) as (stocks, spy, verify):
            session, _ = prepare(replaced, stocks, spy, verify)
            source = replaced.source[0][1]
            substitute = source.with_suffix(".replacement")
            substitute.write_bytes(source.read_bytes())
            os.replace(substitute, source)
            for _ in range(779):
                current = session.current()
                session.submit(current, raw_predictions(current.index))
            current = session.current()
            rejects(
                session.submit, current, raw_predictions(current.index),
            )


def test_decodes_only_the_current_as_of_window() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-prefix-", dir=ROOT,
    ) as directory:
        root = Path(directory)
        value = fixture(root / "valid")
        grid = derive_forward_grid(
            value.source_calendar, value.future_calendar, value.boundary,
        )
        poisoned = fixture(
            root / "poisoned", invalid_after=grid.triples[0][0],
        )
        observed = []
        for item in (value, poisoned):
            with frozen_files(item) as (stocks, spy, verify):
                session, _ = prepare(item, stocks, spy, verify)
                current = session.current()
                observed.append(tuple(
                    series.values
                    for series in (*current.stocks, current.spy)
                ))
                if item is poisoned:
                    rejects(
                        session.submit, current,
                        raw_predictions(current.index),
                    )
        assert observed[0] == observed[1]


def test_rejects_a_stale_batch_token() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-stale-", dir=ROOT,
    ) as directory:
        value = fixture(Path(directory))
        with frozen_files(value) as (stocks, spy, verify):
            session, _ = prepare(value, stocks, spy, verify)
            first = session.current()
            second = session.submit(
                first, raw_predictions(first.index),
            )
            assert isinstance(second, SpyResidualForwardInputs)
            rejects(
                session.submit, first, raw_predictions(first.index),
            )


def test_ignores_the_uninspected_source_tail() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-source-tail-", dir=ROOT,
    ) as directory:
        root = Path(directory)
        values = []
        for item in (
            fixture(
                root / "valid", boundary_bin=-4,
                source_end=date(2026, 3, 31),
            ),
            fixture(
                root / "poisoned", boundary_bin=-4,
                poison_source_tail=True,
                source_end=date(2026, 3, 31),
            ),
        ):
            grid = derive_forward_grid(
                item.source_calendar, item.future_calendar, item.boundary,
            )
            assert grid.target_sessions[0] == date(2026, 2, 26)
            with frozen_files(item) as (stocks, spy, verify):
                session, _ = prepare(item, stocks, spy, verify)
                current = session.current()
                values.append(tuple(
                    series.values
                    for series in (*current.stocks, current.spy)
                ))
        assert values[0] == values[1]


def test_truth_requires_the_complete_exclusive_ledger() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-truth-", dir=ROOT,
    ) as directory:
        value = fixture(Path(directory))
        grid = derive_forward_grid(
            value.source_calendar, value.future_calendar, value.boundary,
        )
        with frozen_files(value) as (stocks, spy, verify):
            session, read_truth = prepare(value, stocks, spy, verify)
            rejects(read_truth, object())
            claim = publish(session)
            rows = read_canonical_json_lines(claim.path)
            metadata = claim.path.stat(follow_symlinks=False)
            assert metadata.st_nlink == 1
            assert stat.S_IMODE(metadata.st_mode) == 0o600
            assert claim.identity == (metadata.st_dev, metadata.st_ino)
            assert claim.records == len(grid.triples) * len(FORWARD_UNIVERSE)
            assert len(rows) == claim.records + 1
            assert rows[0] == {
                "grid_sha256": rows[0]["grid_sha256"],
                "provenance": "unbound-task-3-draft",
                "records": claim.records,
                "schema": 1,
                "type": "spy-residual-forward-prediction-draft",
            }
            first = rows[1]
            assert first["series"] == FORWARD_UNIVERSE[0]
            assert (
                first["as_of"], first["entry"], first["target"],
            ) == grid.triples[0]
            assert first["mean_prediction"] == sum(
                item["prediction"] for item in first["raw_predictions"]
            ) / len(SEEDS)

            truth = read_truth(claim)
            receipt = read_canonical_json(value.receipt)
            assert receipt == {
                "draft": {
                    "identity": list(claim.identity),
                    "path": str(claim.path),
                    "records": claim.records,
                    "sha256": claim.sha256,
                },
                "grid_sha256": rows[0]["grid_sha256"],
                "schema": 1,
                "type": "spy-residual-forward-truth-access",
            }
            assert stat.S_IMODE(
                value.receipt.stat(follow_symlinks=False).st_mode,
            ) == 0o600
            assert tuple(truth) == FORWARD_UNIVERSE
            assert all(
                len(series) == len(grid.triples)
                for series in truth.values()
            )
            rejects(read_truth, claim)

        first = truth[FORWARD_UNIVERSE[0]][0]
        future_index = {
            item.timestamp: index
            for index, item in enumerate(value.future_bins, 1)
        }
        stock_open = bar(
            future_index[first.entry], FORWARD_UNIVERSE[0],
        )[0]
        stock_close = bar(
            future_index[first.target], FORWARD_UNIVERSE[0],
        )[3]
        spy_open = bar(future_index[first.entry], "SPY")[0]
        spy_close = bar(future_index[first.target], "SPY")[3]
        expected = log(stock_close / stock_open) - \
            log(spy_close / spy_open)
        assert first.as_of == grid.triples[0][0]
        assert isclose(first.value, expected, rel_tol=0.0, abs_tol=1e-15)

        value.ledger.unlink()
        with frozen_files(value) as (stocks, spy, verify):
            rejects(prepare, value, stocks, spy, verify)


def test_replaced_publication_fails_closed() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-ledger-replace-", dir=ROOT,
    ) as directory:
        value = fixture(Path(directory))
        with frozen_files(value) as (stocks, spy, verify):
            session, read_truth = prepare(value, stocks, spy, verify)
            claim = publish(session)
            replacement = claim.path.with_suffix(".replacement")
            replacement.write_bytes(claim.path.read_bytes())
            replacement.chmod(0o600)
            os.replace(replacement, claim.path)
            rejects(read_truth, claim)
            rejects(read_truth, claim)


def test_replaced_draft_before_truth_fails_closed() -> None:
    with tempfile.TemporaryDirectory(
        prefix="spy-forward-ledger-race-", dir=ROOT,
    ) as directory:
        value = fixture(Path(directory))
        with frozen_files(value) as (stocks, spy, verify):
            session, read_truth = prepare(value, stocks, spy, verify)
            claim = publish(session)
            publish_receipt = forward_inputs._publish_receipt

            def replace(*args: object) -> object:
                result = publish_receipt(*args)
                replacement = claim.path.with_suffix(".replacement")
                replacement.write_bytes(claim.path.read_bytes())
                replacement.chmod(0o600)
                os.replace(replacement, claim.path)
                return result

            def opened_truth(*args: object) -> object:
                raise AssertionError("truth opened after draft replacement")

            with patch.object(
                forward_inputs, "_publish_receipt", replace,
            ), patch.object(forward_inputs, "_truth", opened_truth):
                rejects(read_truth, claim)


def main() -> None:
    test_selects_exactly_60_calendar_first_target_sessions()
    test_requires_complete_unique_ordered_common_grid()
    test_rejects_links_and_replacements()
    test_decodes_only_the_current_as_of_window()
    test_rejects_a_stale_batch_token()
    test_ignores_the_uninspected_source_tail()
    test_truth_requires_the_complete_exclusive_ledger()
    test_replaced_publication_fails_closed()
    test_replaced_draft_before_truth_fails_closed()
    print("SPY residual forward input tests passed")


if __name__ == "__main__":
    main()
