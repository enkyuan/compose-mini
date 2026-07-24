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
        raise ValueError(f"selection policy {name} must be an ISO date") from error
    if str(parsed) != value:
        raise ValueError(f"selection policy {name} must be an ISO date")
    return parsed


def _decimal(value: object, name: str) -> Decimal:
    if not isinstance(value, str) or not value:
        raise ValueError(f"selection policy {name} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as error:
        raise ValueError(f"selection policy {name} must be a decimal string") from error
    if not parsed.is_finite():
        raise ValueError(f"selection policy {name} must be a canonical decimal")
    canonical = format(parsed, "f")
    if "." in canonical:
        canonical = canonical.rstrip("0").rstrip(".")
    if canonical != value:
        raise ValueError(f"selection policy {name} must be a canonical decimal")
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
                path.read_bytes().decode("utf-8"), object_pairs_hook=_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("selection policy is not valid JSON") from error
        if not isinstance(value, dict) or set(value) != POLICY_FIELDS:
            raise ValueError("selection policy fields are invalid")
        if type(value["schema"]) is not int or value["schema"] != 1:
            raise ValueError("selection policy schema must be 1")
        declared_on = _date(value["declared_on"], "declared_on")
        anchor_date = _date(value["anchor_date"], "anchor_date")
        formation_start = _date(value["formation_start"], "formation_start")
        formation_end = _date(value["formation_end"], "formation_end")
        start = _date(value["start"], "start")
        end = _date(value["end"], "end")
        interval = _integer(value["interval_minutes"], "interval_minutes", 1)
        if interval > 59 or type(value["adjusted"]) is not bool or \
           value["session"] != "regular":
            raise ValueError("selection policy transport fields are invalid")
        sizes_value = value["cohort_sizes"]
        if not isinstance(sizes_value, list) or not sizes_value:
            raise ValueError("selection policy cohort_sizes are invalid")
        sizes = tuple(_integer(item, "cohort_sizes", 1) for item in sizes_value)
        primary = _integer(value["primary_cohort_size"], "primary_cohort_size", 1)
        strata = _integer(value["liquidity_strata"], "liquidity_strata", 2)
        sessions = _integer(
            value["minimum_formation_sessions"], "minimum_formation_sessions", 1,
        )
        coverage = _decimal(value["minimum_coverage"], "minimum_coverage")
        minimum_close = _decimal(
            value["minimum_median_close_usd"], "minimum_median_close_usd",
        )
        minimum_dollar_volume = _decimal(
            value["minimum_median_dollar_volume_usd"],
            "minimum_median_dollar_volume_usd",
        )
        if declared_on < anchor_date or formation_start > formation_end or \
           formation_end != anchor_date or anchor_date >= start or start > end:
            raise ValueError("selection policy date relationship is invalid")
        if any(left >= right for left, right in zip(sizes, sizes[1:])) or \
           primary != sizes[-1] or not 2 <= strata <= primary:
            raise ValueError("selection policy cohort configuration is invalid")
        if not Decimal(0) < coverage <= Decimal(1) or \
           minimum_close <= 0 or minimum_dollar_volume <= 0:
            raise ValueError("selection policy thresholds are invalid")
        return cls(
            value["schema"], _text(value["purpose"], "purpose"), declared_on,
            anchor_date, formation_start, formation_end, start, end, interval,
            value["adjusted"], value["session"], sizes, primary,
            _text(value["selection_seed"], "selection_seed"), strata, sessions,
            coverage, minimum_close, minimum_dollar_volume,
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
