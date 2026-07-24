# Point-in-Time Universe Selection Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the frozen 11/22/33/55-stock selection policy and the pure,
deterministic arithmetic that turns dated Massive reference and formation data
into one nested master cohort.

**Architecture:** Parse one strict tracked policy into immutable typed values.
Keep selection independent of files, credentials, clocks, network calls, model
results, and output publication so its eligibility, FIGI deduplication,
liquidity ranking, hash ordering, and round-robin behavior are exhaustively
testable offline.

**Tech Stack:** Python 3.12 standard library (`dataclasses`, `datetime`,
`decimal`, `hashlib`, `json`), existing `tools.fetch_massive.TICKER`,
procedural Python tests, GitButler.

## Global Constraints

- Follow `karpathy-guidelines` and `ponytail`: make the smallest direct change,
  reuse existing primitives, and add no dependency or speculative abstraction.
- Keep code clean, simple, DRY, scalable, and documented only where purpose or
  an invariant is not apparent from the code.
- Use only pre-model dates and values; no prediction, label, policy, return, or
  reserved-test input may influence membership.
- The exact research contract is
  `docs/superpowers/specs/2026-07-24-point-in-time-universe-selection.md`.
- Touch only `universes/liquid-common-ladder.example.json`,
  `tools/select_universe.py`, and `tests/python/test_massive.py`.
- Preserve unrelated `Makefile` and `docs/training.md` changes.
- Use GitButler for version-control inspection and writes. Create signed local
  checkpoints authored by `enkyuan <yuan.enkng@gmail.com>`; do not push or land.
- Do not fetch Massive, read credentials, create generated data, train, select
  a trading policy, or run the `$100` backtest in this plan.

---

### Task 1: Frozen Policy and Pure Nested-Cohort Selection

**Files:**

- Create: `universes/liquid-common-ladder.example.json`
- Create: `tools/select_universe.py`
- Modify: `tests/python/test_massive.py`

**Interfaces:**

- Consumes: `tools.fetch_massive.TICKER`.
- Produces:
  `SelectionPolicy.read(Path) -> SelectionPolicy`;
  `select_candidates(SelectionPolicy, Sequence[Reference],
  Sequence[tuple[date, Sequence[DailyRow]]]) -> Selection`.
- `Selection.candidates` is ticker-sorted and contains every dated reference,
  including rejected and FIGI-deduplicated records.
- `Selection.master` contains exactly `primary_cohort_size` representatives in
  final nested-prefix order.

- [ ] **Step 1: Add the exact offline policy test and stage the selection fixture**

In `tests/python/test_massive.py`, add these imports:

```python
from decimal import Decimal

from tools.select_universe import (
    Candidate, DailyRow, Reference, SelectionPolicy, select_candidates,
)
```

Add the following complete helpers and tests before `test_request_gate()`:

```python
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
    assert candidates["B4-02"].coverage == Decimal(59) / Decimal(60)
    assert candidates["B4-02"].median_close == Decimal("50")
    assert candidates["B4-02"].median_dollar_volume == Decimal("30000000")
    assert candidates["B4-03"].rejection_reasons == (
        "coverage-below-minimum",
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
        assert cohort == selection.master[:len(cohort)]
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
```

Call only the policy test from `main()` in this first red/green cycle:

```python
        test_selection_policy(directory)
```

- [ ] **Step 2: Run the focused test and record RED**

Run:

```sh
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tests/python/test_massive.py
```

Expected: FAIL because `tools.select_universe` does not exist.

- [ ] **Step 3: Create the exact tracked policy**

Create `universes/liquid-common-ladder.example.json`:

```json
{
  "adjusted": true,
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
  "purpose": "Select nested point-in-time U.S. common-stock cohorts before fetching model data.",
  "schema": 1,
  "selection_seed": "compose-mini-massive-universe-v1",
  "session": "regular",
  "start": "2024-11-01"
}
```

Create the parser boundary in `tools/select_universe.py`. This is the complete
first-cycle file; the explicit stub keeps the already-defined selection fixture
importable without implementing it before its red run:

```python
#!/usr/bin/env python3
"""Parse a frozen universe policy and select its liquid common stocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
import json
import os

from tools.fetch_massive import TICKER

POLICY_FIELDS = {
    "schema", "purpose", "declared_on", "anchor_date", "formation_start",
    "formation_end", "start", "end", "interval_minutes", "adjusted",
    "session", "cohort_sizes", "primary_cohort_size", "selection_seed",
    "liquidity_strata", "minimum_formation_sessions", "minimum_coverage",
    "minimum_median_close_usd", "minimum_median_dollar_volume_usd",
}


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("selection policy contains a duplicate field")
        value[name] = item
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"selection policy {name} must be nonempty text")
    return value


def _date(value: object, name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"selection policy {name} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"selection policy {name} must be an ISO date"
        ) from error
    if str(parsed) != value:
        raise ValueError(f"selection policy {name} must be an ISO date")
    return parsed


def _decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"selection policy {name} must be a decimal string"
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(
            f"selection policy {name} must be a decimal string"
        ) from error
    if not parsed.is_finite() or format(parsed.normalize(), "f") != value:
        raise ValueError(
            f"selection policy {name} must be a canonical decimal"
        )
    return parsed


def _integer(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"selection policy {name} is invalid")
    return value


@dataclass(frozen=True)
class Reference:
    ticker: str
    active: bool | None
    market: str | None
    locale: str | None
    type: str | None
    currency_name: str | None
    primary_exchange: str | None
    composite_figi: str | None
    share_class_figi: str | None


@dataclass(frozen=True)
class DailyRow:
    ticker: str
    close: Decimal
    volume: Decimal
    vwap: Decimal


@dataclass(frozen=True)
class Candidate:
    reference: Reference
    observed: int
    coverage: Decimal
    median_close: Decimal
    median_dollar_volume: Decimal
    rejection_reasons: tuple[str, ...]
    liquidity_rank: int | None = None
    stratum: int | None = None
    within_stratum_rank: int | None = None
    master_rank: int | None = None


@dataclass(frozen=True)
class Selection:
    candidates: tuple[Candidate, ...]
    master: tuple[Candidate, ...]


@dataclass(frozen=True)
class SelectionPolicy:
    schema: int
    purpose: str
    declared_on: date
    anchor_date: date
    formation_start: date
    formation_end: date
    start: date
    end: date
    interval_minutes: int
    adjusted: bool
    session: str
    cohort_sizes: tuple[int, ...]
    primary_cohort_size: int
    selection_seed: str
    liquidity_strata: int
    minimum_formation_sessions: int
    minimum_coverage: Decimal
    minimum_median_close_usd: Decimal
    minimum_median_dollar_volume_usd: Decimal

    @classmethod
    def read(cls, path: Path) -> SelectionPolicy:
        if not os.path.lexists(path) or path.is_symlink() or not path.is_file():
            raise ValueError("selection policy must be a regular file")
        try:
            value = json.loads(
                path.read_bytes().decode("utf-8"),
                object_pairs_hook=_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("selection policy is not valid JSON") from error
        if not isinstance(value, dict) or set(value) != POLICY_FIELDS:
            raise ValueError("selection policy fields are invalid")
        if type(value["schema"]) is not int or value["schema"] != 1:
            raise ValueError("selection policy schema must be 1")

        declared_on = _date(value["declared_on"], "declared_on")
        anchor = _date(value["anchor_date"], "anchor_date")
        formation_start = _date(value["formation_start"], "formation_start")
        formation_end = _date(value["formation_end"], "formation_end")
        start = _date(value["start"], "start")
        end = _date(value["end"], "end")
        interval = _integer(
            value["interval_minutes"], "interval_minutes", 1,
        )
        if interval > 59 or type(value["adjusted"]) is not bool or \
           value["session"] != "regular":
            raise ValueError("selection policy transport fields are invalid")

        raw_sizes = value["cohort_sizes"]
        if not isinstance(raw_sizes, list) or not raw_sizes:
            raise ValueError("selection policy cohort_sizes are invalid")
        sizes = tuple(
            _integer(item, "cohort_sizes", 1) for item in raw_sizes
        )
        primary = _integer(
            value["primary_cohort_size"], "primary_cohort_size", 1,
        )
        strata = _integer(
            value["liquidity_strata"], "liquidity_strata", 2,
        )
        minimum_sessions = _integer(
            value["minimum_formation_sessions"],
            "minimum_formation_sessions", 1,
        )
        coverage = _decimal(
            value["minimum_coverage"], "minimum_coverage",
        )
        minimum_close = _decimal(
            value["minimum_median_close_usd"],
            "minimum_median_close_usd",
        )
        minimum_dollar_volume = _decimal(
            value["minimum_median_dollar_volume_usd"],
            "minimum_median_dollar_volume_usd",
        )

        if declared_on < anchor or formation_start > formation_end or \
           formation_end != anchor or anchor >= start or start > end:
            raise ValueError("selection policy date relationship is invalid")
        if any(a >= b for a, b in zip(sizes, sizes[1:])) or \
           primary != sizes[-1] or not 2 <= strata <= primary:
            raise ValueError(
                "selection policy cohort configuration is invalid"
            )
        if not Decimal(0) < coverage <= Decimal(1) or \
           minimum_close <= 0 or minimum_dollar_volume <= 0:
            raise ValueError("selection policy thresholds are invalid")

        return cls(
            value["schema"], _text(value["purpose"], "purpose"),
            declared_on, anchor, formation_start, formation_end, start, end,
            interval, value["adjusted"], value["session"], sizes, primary,
            _text(value["selection_seed"], "selection_seed"), strata,
            minimum_sessions, coverage, minimum_close,
            minimum_dollar_volume,
        )


def select_candidates(
    policy: SelectionPolicy,
    references: Sequence[Reference],
    sessions: Sequence[tuple[date, Sequence[DailyRow]]],
) -> Selection:
    raise NotImplementedError
```

- [ ] **Step 4: Run the policy parser test and record GREEN**

Run:

```sh
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tests/python/test_massive.py
```

Expected:

```text
Massive downloader and universe tests passed
```

- [ ] **Step 5: Enable the pure-selection test and record RED**

Add the already-defined pure selector test to `main()`:

```python
        test_pure_selection()
```

Run:

```sh
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tests/python/test_massive.py
```

Expected: FAIL with `NotImplementedError` from `select_candidates()`.

- [ ] **Step 6: Add the complete minimal pure implementation**

Create `tools/select_universe.py` with:

```python
#!/usr/bin/env python3
"""Parse a frozen universe policy and select its liquid common stocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path
import hashlib
import json
import os

from tools.fetch_massive import TICKER

POLICY_FIELDS = {
    "schema", "purpose", "declared_on", "anchor_date", "formation_start",
    "formation_end", "start", "end", "interval_minutes", "adjusted",
    "session", "cohort_sizes", "primary_cohort_size", "selection_seed",
    "liquidity_strata", "minimum_formation_sessions", "minimum_coverage",
    "minimum_median_close_usd", "minimum_median_dollar_volume_usd",
}
EXCHANGES = {"XNAS", "XNYS", "XASE"}


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for name, item in pairs:
        if name in value:
            raise ValueError("selection policy contains a duplicate field")
        value[name] = item
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"selection policy {name} must be nonempty text")
    return value


def _date(value: object, name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"selection policy {name} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"selection policy {name} must be an ISO date"
        ) from error
    if str(parsed) != value:
        raise ValueError(f"selection policy {name} must be an ISO date")
    return parsed


def _decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"selection policy {name} must be a decimal string"
        )
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(
            f"selection policy {name} must be a decimal string"
        ) from error
    if not parsed.is_finite() or format(parsed.normalize(), "f") != value:
        raise ValueError(
            f"selection policy {name} must be a canonical decimal"
        )
    return parsed


def _integer(value: object, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"selection policy {name} is invalid")
    return value


@dataclass(frozen=True)
class Reference:
    ticker: str
    active: bool | None
    market: str | None
    locale: str | None
    type: str | None
    currency_name: str | None
    primary_exchange: str | None
    composite_figi: str | None
    share_class_figi: str | None


@dataclass(frozen=True)
class DailyRow:
    ticker: str
    close: Decimal
    volume: Decimal
    vwap: Decimal


@dataclass(frozen=True)
class Candidate:
    reference: Reference
    observed: int
    coverage: Decimal
    median_close: Decimal
    median_dollar_volume: Decimal
    rejection_reasons: tuple[str, ...]
    liquidity_rank: int | None = None
    stratum: int | None = None
    within_stratum_rank: int | None = None
    master_rank: int | None = None


@dataclass(frozen=True)
class Selection:
    candidates: tuple[Candidate, ...]
    master: tuple[Candidate, ...]


@dataclass(frozen=True)
class SelectionPolicy:
    schema: int
    purpose: str
    declared_on: date
    anchor_date: date
    formation_start: date
    formation_end: date
    start: date
    end: date
    interval_minutes: int
    adjusted: bool
    session: str
    cohort_sizes: tuple[int, ...]
    primary_cohort_size: int
    selection_seed: str
    liquidity_strata: int
    minimum_formation_sessions: int
    minimum_coverage: Decimal
    minimum_median_close_usd: Decimal
    minimum_median_dollar_volume_usd: Decimal

    @classmethod
    def read(cls, path: Path) -> SelectionPolicy:
        if not os.path.lexists(path) or path.is_symlink() or not path.is_file():
            raise ValueError("selection policy must be a regular file")
        try:
            value = json.loads(
                path.read_bytes().decode("utf-8"),
                object_pairs_hook=_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("selection policy is not valid JSON") from error
        if not isinstance(value, dict) or set(value) != POLICY_FIELDS:
            raise ValueError("selection policy fields are invalid")
        if type(value["schema"]) is not int or value["schema"] != 1:
            raise ValueError("selection policy schema must be 1")

        declared_on = _date(value["declared_on"], "declared_on")
        anchor = _date(value["anchor_date"], "anchor_date")
        formation_start = _date(value["formation_start"], "formation_start")
        formation_end = _date(value["formation_end"], "formation_end")
        start = _date(value["start"], "start")
        end = _date(value["end"], "end")
        interval = _integer(
            value["interval_minutes"], "interval_minutes", 1,
        )
        if interval > 59 or type(value["adjusted"]) is not bool or \
           value["session"] != "regular":
            raise ValueError("selection policy transport fields are invalid")

        raw_sizes = value["cohort_sizes"]
        if not isinstance(raw_sizes, list) or not raw_sizes:
            raise ValueError("selection policy cohort_sizes are invalid")
        sizes = tuple(
            _integer(item, "cohort_sizes", 1) for item in raw_sizes
        )
        primary = _integer(
            value["primary_cohort_size"], "primary_cohort_size", 1,
        )
        strata = _integer(
            value["liquidity_strata"], "liquidity_strata", 2,
        )
        minimum_sessions = _integer(
            value["minimum_formation_sessions"],
            "minimum_formation_sessions", 1,
        )
        coverage = _decimal(
            value["minimum_coverage"], "minimum_coverage",
        )
        minimum_close = _decimal(
            value["minimum_median_close_usd"],
            "minimum_median_close_usd",
        )
        minimum_dollar_volume = _decimal(
            value["minimum_median_dollar_volume_usd"],
            "minimum_median_dollar_volume_usd",
        )

        if declared_on < anchor or formation_start > formation_end or \
           formation_end != anchor or anchor >= start or start > end:
            raise ValueError("selection policy date relationship is invalid")
        if any(a >= b for a, b in zip(sizes, sizes[1:])) or \
           primary != sizes[-1] or not 2 <= strata <= primary:
            raise ValueError(
                "selection policy cohort configuration is invalid"
            )
        if not Decimal(0) < coverage <= Decimal(1) or \
           minimum_close <= 0 or minimum_dollar_volume <= 0:
            raise ValueError("selection policy thresholds are invalid")

        return cls(
            value["schema"], _text(value["purpose"], "purpose"),
            declared_on, anchor, formation_start, formation_end, start, end,
            interval, value["adjusted"], value["session"], sizes, primary,
            _text(value["selection_seed"], "selection_seed"), strata,
            minimum_sessions, coverage, minimum_close,
            minimum_dollar_volume,
        )


def _ticker_valid(value: object) -> bool:
    return isinstance(value, str) and TICKER.fullmatch(value) is not None and any(
        character.isascii() and character.isalnum() for character in value
    )


def _median(values: Sequence[Decimal]) -> Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )


def _metadata_reasons(reference: Reference) -> tuple[str, ...]:
    checks = (
        (reference.active is not True, "inactive"),
        (reference.market != "stocks", "market-not-stocks"),
        (reference.locale != "us", "locale-not-us"),
        (reference.type != "CS", "type-not-common-stock"),
        (reference.currency_name != "usd", "currency-not-usd"),
        (reference.primary_exchange not in EXCHANGES, "exchange-not-listed"),
        (not reference.composite_figi, "missing-composite-figi"),
        (not reference.share_class_figi, "missing-share-class-figi"),
    )
    return tuple(reason for failed, reason in checks if failed)


def _formation_rows(
    policy: SelectionPolicy,
    sessions: Sequence[tuple[date, Sequence[DailyRow]]],
) -> tuple[dict[str, DailyRow], ...]:
    ordered = sorted(sessions, key=lambda item: item[0])
    dates = [day for day, _ in ordered]
    if len(ordered) < policy.minimum_formation_sessions or \
       len(set(dates)) != len(dates) or any(
           not policy.formation_start <= day <= policy.formation_end
           for day in dates
       ):
        raise ValueError("formation sessions are invalid")

    normalized = []
    for _, rows in ordered:
        by_ticker: dict[str, DailyRow] = {}
        for row in rows:
            if not _ticker_valid(row.ticker) or row.ticker in by_ticker:
                raise ValueError("formation rows are invalid")
            by_ticker[row.ticker] = row
        if not by_ticker:
            raise ValueError("formation sessions must be nonempty")
        normalized.append(by_ticker)
    return tuple(normalized)


def _candidate(
    policy: SelectionPolicy,
    reference: Reference,
    sessions: Sequence[dict[str, DailyRow]],
) -> Candidate:
    rows = [
        row for session in sessions
        if (row := session.get(reference.ticker)) is not None and
        all(
            isinstance(value, Decimal) and value.is_finite() and value > 0
            for value in (row.close, row.volume, row.vwap)
        )
    ]
    coverage = Decimal(len(rows)) / Decimal(len(sessions))
    closes = [row.close for row in rows]
    dollar_volumes = [row.volume * row.vwap for row in rows]
    median_close = _median(closes) if closes else Decimal(0)
    median_dollar_volume = (
        _median(dollar_volumes) if dollar_volumes else Decimal(0)
    )
    numerical = (
        (
            coverage < policy.minimum_coverage,
            "coverage-below-minimum",
        ),
        (
            median_close < policy.minimum_median_close_usd,
            "median-close-below-minimum",
        ),
        (
            median_dollar_volume <
            policy.minimum_median_dollar_volume_usd,
            "median-dollar-volume-below-minimum",
        ),
    )
    reasons = _metadata_reasons(reference) + tuple(
        reason for failed, reason in numerical if failed
    )
    return Candidate(
        reference, len(rows), coverage, median_close,
        median_dollar_volume, reasons,
    )


def _hash_key(policy: SelectionPolicy, candidate: Candidate) -> tuple:
    reference = candidate.reference
    identity = reference.share_class_figi
    assert identity is not None
    return (
        hashlib.sha256(
            f"{policy.selection_seed}\0{identity}".encode()
        ).digest(),
        reference.composite_figi,
        reference.ticker,
    )


def select_candidates(
    policy: SelectionPolicy,
    references: Sequence[Reference],
    sessions: Sequence[tuple[date, Sequence[DailyRow]]],
) -> Selection:
    ordered = sorted(references, key=lambda item: item.ticker)
    tickers = [item.ticker for item in ordered]
    if len(set(tickers)) != len(tickers) or any(
        not _ticker_valid(ticker) for ticker in tickers
    ):
        raise ValueError("dated references are invalid")

    formation = _formation_rows(policy, sessions)
    candidates = {
        reference.ticker: _candidate(policy, reference, formation)
        for reference in ordered
    }
    eligible = [
        item for item in candidates.values() if not item.rejection_reasons
    ]

    by_identity: dict[str, list[Candidate]] = {}
    for item in eligible:
        identity = item.reference.share_class_figi
        assert identity is not None
        by_identity.setdefault(identity, []).append(item)
    for listings in by_identity.values():
        winner = min(
            listings,
            key=lambda item: (
                -item.median_dollar_volume,
                item.reference.composite_figi,
                item.reference.ticker,
            ),
        )
        for item in listings:
            if item is not winner:
                candidates[item.reference.ticker] = replace(
                    item,
                    rejection_reasons=("duplicate-share-class-figi",),
                )

    representatives = sorted(
        (
            item for item in candidates.values()
            if not item.rejection_reasons
        ),
        key=lambda item: (
            -item.median_dollar_volume,
            item.reference.share_class_figi,
            item.reference.composite_figi,
            item.reference.ticker,
        ),
    )
    count = len(representatives)
    strata: list[list[Candidate]] = [
        [] for _ in range(policy.liquidity_strata)
    ]
    for rank, item in enumerate(representatives):
        stratum = policy.liquidity_strata * rank // count + 1
        ranked = replace(item, liquidity_rank=rank, stratum=stratum)
        candidates[item.reference.ticker] = ranked
        strata[stratum - 1].append(ranked)

    ordered_strata = []
    for values in strata:
        ranked = []
        for within_rank, item in enumerate(
            sorted(values, key=lambda candidate: _hash_key(policy, candidate))
        ):
            item = replace(item, within_stratum_rank=within_rank)
            candidates[item.reference.ticker] = item
            ranked.append(item)
        ordered_strata.append(ranked)

    base, extra = divmod(
        policy.primary_cohort_size, policy.liquidity_strata,
    )
    if any(
        len(values) < base + (index < extra)
        for index, values in enumerate(ordered_strata)
    ):
        raise ValueError("not enough eligible candidates for every stratum")

    master = []
    for offset in range(max(map(len, ordered_strata))):
        for values in ordered_strata:
            if offset < len(values):
                item = replace(values[offset], master_rank=len(master))
                candidates[item.reference.ticker] = item
                master.append(item)
                if len(master) == policy.primary_cohort_size:
                    break
        if len(master) == policy.primary_cohort_size:
            break

    return Selection(
        tuple(candidates[ticker] for ticker in tickers),
        tuple(master),
    )
```

- [ ] **Step 7: Run the focused test and record GREEN**

Run:

```sh
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tests/python/test_massive.py
```

Expected:

```text
Massive downloader and universe tests passed
```

- [ ] **Step 8: Run task-scoped reviews and fix findings**

Review the Task 1 diff against this plan and the specification. Required review
checks:

- policy values and exact parser types match;
- all provider-input permutations are deterministic;
- invalid daily observations become missing observations, not zeros;
- FIGI deduplication occurs before liquidity rank;
- ranks use zero-based `floor(K*r/M)+1`;
- seeded ordering uses UTF-8
  `selection_seed + "\0" + share_class_figi`;
- master prefixes are balanced and contain no rejected candidate;
- no network, credential, model, report, or backtest path is reachable.

Expected: no Critical or Important findings after fixes.

- [ ] **Step 9: Create the signed local checkpoint**

Use `but diff`, copy the exact file IDs printed for the policy, selector, and
test, then pass those three IDs to the selected-change fast path on the existing
`enkyuan/universe-expansion-plan` branch:

```sh
but commit enkyuan/universe-expansion-plan -m \
  "feat(data): define point-in-time universe selection" --changes ID1,ID2,ID3
```

`ID1`, `ID2`, and `ID3` mean the three concrete IDs returned by the immediately
preceding `but diff`; GitButler generates them from the live workspace, so the
plan cannot predeclare their values.

Verify the resulting commit is authored and committed by
`enkyuan <yuan.enkng@gmail.com>` and has a valid ED25519 signature with
fingerprint:

```text
SHA256:mnKfNNJJUV5ELrMXZCwdu6v/PusXBeAi8tWUnRXmWMQ
```

Do not push or land.
