#!/usr/bin/env python3
"""Parse a frozen universe policy and select its liquid common stocks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Context, Decimal, InvalidOperation, localcontext
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import argparse
import hashlib
import json
import os
import re
import secrets
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fetch_massive import (
    API_HOST, TICKER, Requester, api_key, authorized_url, request_gate,
    request_json,
)
from tools.files import (
    ExclusiveTemp, freeze_inputs, require_disjoint, verify_frozen,
    rename_may_have_committed, rename_noreplace, write_json_exclusive,
)

POLICY_FIELDS = {
    "schema", "purpose", "declared_on", "anchor_date", "formation_start",
    "formation_end", "start", "end", "interval_minutes", "adjusted",
    "session", "cohort_sizes", "primary_cohort_size", "selection_seed",
    "liquidity_strata", "minimum_formation_sessions", "minimum_coverage",
    "minimum_median_close_usd", "minimum_median_dollar_volume_usd",
}
EXCHANGES = {"XNAS", "XNYS", "XASE"}
SOURCE_PATHS = (
    ROOT / "tools/select_universe.py",
    ROOT / "tools/fetch_massive.py",
    ROOT / "tools/files.py",
)
PRIVATE_MARKER = re.compile(r"\.selection\.json\.[0-9a-f]{32}\.tmp")


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
class SourceArchive:
    name: str
    request: Mapping[str, object]
    records: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class SourceBundle:
    references: tuple[Reference, ...]
    sessions: tuple[tuple[date, tuple[DailyRow, ...]], ...]
    archives: tuple[SourceArchive, ...]


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


def _decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("selection value must be finite")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if value == 0 else text


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, allow_nan=False, indent=2, sort_keys=True,
        ).encode("utf-8") + b"\n"
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _member_value(candidate: Candidate) -> dict[str, object]:
    reference = candidate.reference
    return {
        "ticker": reference.ticker,
        "composite_figi": reference.composite_figi,
        "share_class_figi": reference.share_class_figi,
        "stratum": candidate.stratum,
    }


def _candidate_value(
    candidate: Candidate,
    representative_ticker: str | None,
) -> dict[str, object]:
    reference = candidate.reference
    return {
        "ticker": reference.ticker,
        "active": reference.active,
        "market": reference.market,
        "locale": reference.locale,
        "type": reference.type,
        "currency_name": reference.currency_name,
        "primary_exchange": reference.primary_exchange,
        "composite_figi": reference.composite_figi,
        "share_class_figi": reference.share_class_figi,
        "observed": candidate.observed,
        "coverage": _decimal_text(candidate.coverage),
        "median_close_usd": _decimal_text(candidate.median_close),
        "median_dollar_volume_usd": _decimal_text(
            candidate.median_dollar_volume,
        ),
        "rejection_reasons": list(candidate.rejection_reasons),
        "decision": (
            "rejected" if candidate.rejection_reasons else
            "selected" if candidate.master_rank is not None else
            "eligible-not-selected"
        ),
        "share_class_representative": representative_ticker,
        "liquidity_rank": candidate.liquidity_rank,
        "stratum": candidate.stratum,
        "within_stratum_rank": candidate.within_stratum_rank,
        "master_rank": candidate.master_rank,
    }


def _manifest_value(
    policy: SelectionPolicy,
    members: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema": policy.schema,
        "purpose": policy.purpose,
        "declared_on": str(policy.declared_on),
        "eligibility_date": str(policy.anchor_date),
        "start": str(policy.start),
        "end": str(policy.end),
        "interval_minutes": policy.interval_minutes,
        "adjusted": policy.adjusted,
        "session": policy.session,
        "series": [
            {
                "stratum": f"liquidity-{member['stratum']}",
                "ticker": member["ticker"],
            }
            for member in members
        ],
    }


def _source_archive_value(archive: SourceArchive) -> dict[str, object]:
    return {
        "schema": 1,
        "request": archive.request,
        "records": list(archive.records),
    }


def _source_binding_value(
    archive: SourceArchive,
    formation_sessions: frozenset[str],
    sha256: str,
) -> dict[str, object]:
    session = archive.name.removeprefix("daily-")
    return {
        "name": archive.name,
        "path": f"sources/{archive.name}.json",
        "sha256": sha256,
        "records": len(archive.records),
        "formation_session": (
            session in formation_sessions
            if session != archive.name else None
        ),
    }


def _ticker_valid(value: object) -> bool:
    return isinstance(value, str) and TICKER.fullmatch(value) is not None and any(
        character.isascii() and character.isalnum() for character in value
    )


def _provider_ticker_valid(value: object) -> bool:
    return isinstance(value, str) and value.isascii() and \
        _ticker_valid(value.upper())


REFERENCE_FIELDS = (
    "ticker", "active", "market", "locale", "type", "currency_name",
    "primary_exchange", "composite_figi", "share_class_figi",
)


def reference_universe_url(anchor: date) -> str:
    if type(anchor) is not date:
        raise ValueError("reference date is invalid")
    return urlunsplit((
        "https", API_HOST, "/v3/reference/tickers",
        urlencode({
            "active": "true",
            "date": str(anchor),
            "limit": 1000,
            "market": "stocks",
            "order": "asc",
            "sort": "ticker",
            "type": "CS",
        }),
        "",
    ))


def daily_summary_url(day: date) -> str:
    if type(day) is not date:
        raise ValueError("daily-summary date is invalid")
    return urlunsplit((
        "https", API_HOST,
        f"/v2/aggs/grouped/locale/us/market/stocks/{day}",
        urlencode({"adjusted": "false", "include_otc": "false"}),
        "",
    ))


def _public_request(url: str) -> tuple[str, dict[str, object]]:
    authorized_url(url, "validation-only")
    parts = urlsplit(url)
    pairs = sorted(
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if name != "apiKey"
    )
    if len({name for name, _ in pairs}) != len(pairs):
        raise ValueError("Massive request has duplicate query fields")
    public = urlunsplit((
        parts.scheme, parts.netloc, parts.path, urlencode(pairs), "",
    ))
    return public, {
        "path": parts.path,
        "query": dict(pairs),
    }


def _request(
    public: str,
    key: str,
    requester: Requester,
    before_request: Callable[[], None],
) -> Mapping[str, object]:
    before_request()
    try:
        payload = requester(authorized_url(public, key))
    except Exception:
        raise ValueError("Massive universe request failed") from None
    if not isinstance(payload, Mapping):
        raise ValueError("Massive returned a non-object universe response")
    return payload


def _results(
    payload: Mapping[str, object],
    name: str,
    *,
    allow_omitted_empty: bool = False,
) -> list[object]:
    if payload.get("status") != "OK":
        raise ValueError(f"Massive returned an unsuccessful {name}")
    results = payload.get("results")
    count = payload.get("resultsCount")
    if results is None and allow_omitted_empty and \
       type(count) is int and count == 0:
        return []
    if not isinstance(results, list) or (
        count is not None and (
            type(count) is not int or count != len(results)
        )
    ):
        raise ValueError(f"Massive returned an invalid {name}")
    return results


def _decimal_record(value: object, name: str) -> tuple[str, Decimal]:
    if type(value) not in (int, float):
        raise ValueError(f"Massive {name} is invalid")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(f"Massive {name} is invalid") from None
    if not parsed.is_finite():
        raise ValueError(f"Massive {name} is invalid")
    text = format(parsed, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if parsed == 0:
        text = "0"
    return text, parsed


def _reference_record(
    value: object,
) -> tuple[dict[str, object], Reference | None]:
    if not isinstance(value, Mapping) or not _provider_ticker_valid(
        value.get("ticker")
    ):
        raise ValueError("Massive returned an invalid reference record")
    ticker = value["ticker"]
    active = value.get("active")
    if active is not None and type(active) is not bool:
        raise ValueError("Massive returned an invalid reference record")
    strings = tuple(value.get(name) for name in REFERENCE_FIELDS[2:])
    if any(item is not None and not isinstance(item, str) for item in strings):
        raise ValueError("Massive returned an invalid reference record")
    record = {
        "ticker": ticker,
        "active": active,
        **dict(zip(REFERENCE_FIELDS[2:], strings, strict=True)),
    }
    return record, (
        Reference(*(record[name] for name in REFERENCE_FIELDS))
        if _ticker_valid(ticker) else None
    )


def _daily_record(
    value: object,
) -> tuple[dict[str, object], DailyRow]:
    if not isinstance(value, Mapping) or not _ticker_valid(value.get("T")):
        raise ValueError("Massive returned an invalid daily record")
    ticker = value["T"]
    try:
        values = tuple(
            _decimal_record(value[name], name)
            for name in ("c", "v", "vw")
        )
    except KeyError:
        raise ValueError("Massive returned an invalid daily record") from None
    record = {
        "ticker": ticker,
        **{
            name: item[0]
            for name, item in zip(("c", "v", "vw"), values, strict=True)
        },
    }
    return record, DailyRow(ticker, *(item[1] for item in values))


def fetch_sources(
    policy: SelectionPolicy,
    key: str,
    requester: Requester,
    before_request: Callable[[], None],
) -> SourceBundle:
    if not key or any(character.isspace() for character in key):
        raise ValueError("Massive API key is missing or invalid")

    references, archives, seen, previous = [], [], set(), ""
    url = reference_universe_url(policy.anchor_date)
    page = 1
    while url:
        public, contract = _public_request(url)
        if public in seen:
            raise ValueError("Massive reference pagination contains a cycle")
        seen.add(public)
        payload = _request(public, key, requester, before_request)
        results = _results(payload, "reference page")
        records = []
        for value in results:
            record, reference = _reference_record(value)
            ticker = record["ticker"]
            assert isinstance(ticker, str)
            if ticker <= previous:
                raise ValueError(
                    "Massive reference tickers are not strictly increasing"
                )
            previous = ticker
            records.append(record)
            if reference is not None:
                references.append(reference)
        archives.append(SourceArchive(
            f"tickers-{page:04d}", contract, tuple(records),
        ))
        next_url = payload.get("next_url", "")
        if not isinstance(next_url, str):
            raise ValueError("Massive returned an invalid reference next_url")
        url, page = next_url, page + 1
    if not references:
        raise ValueError("Massive returned no reference candidates")

    sessions = []
    reference_tickers = {item.ticker for item in references}
    day = policy.formation_start
    while day <= policy.formation_end:
        if day.weekday() < 5:
            public, contract = _public_request(daily_summary_url(day))
            payload = _request(public, key, requester, before_request)
            results = _results(
                payload, "daily summary", allow_omitted_empty=True,
            )
            raw_tickers = []
            retained = []
            for value in results:
                if not isinstance(value, Mapping) or not _provider_ticker_valid(
                    value.get("T")
                ):
                    raise ValueError("Massive returned an invalid daily record")
                ticker = value["T"]
                raw_tickers.append(ticker)
                if ticker in reference_tickers:
                    retained.append(_daily_record(value))
            if len(set(raw_tickers)) != len(raw_tickers):
                raise ValueError("Massive daily tickers are not unique")
            normalized = sorted(
                retained,
                key=lambda item: item[1].ticker,
            )
            archives.append(SourceArchive(
                f"daily-{day}", contract,
                tuple(record for record, _ in normalized),
            ))
            if results:
                sessions.append((
                    day, tuple(row for _, row in normalized),
                ))
        day += timedelta(days=1)
    if len(sessions) < policy.minimum_formation_sessions:
        raise ValueError("Massive returned too few formation sessions")
    return SourceBundle(
        tuple(references), tuple(sessions), tuple(archives),
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
    with localcontext(Context(prec=64)):
        return _select_candidates(policy, references, sessions)


def _select_candidates(
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


def _transport(
    requester: Requester | None,
    requests_per_minute: int,
) -> tuple[Requester, Callable[[], None]]:
    gate = request_gate(requests_per_minute)
    if requester is None:
        return (
            lambda url: request_json(url, before_request=gate),
            lambda: None,
        )
    return requester, gate


def _regular_identity(path: Path) -> tuple[int, int]:
    try:
        value = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise ValueError("universe source must be a regular file") from error
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("universe source must be a regular file")
    return value.st_dev, value.st_ino


def _input(path: Path) -> tuple[Path, tuple[int, int]]:
    if not os.path.lexists(path) or path.is_symlink():
        raise ValueError("universe source must be a regular file")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ValueError("universe source path is invalid") from error
    return resolved, _regular_identity(resolved)


def _directory_identity(value: os.stat_result) -> tuple[int, int, int]:
    if not stat.S_ISDIR(value.st_mode):
        raise ValueError("universe output directory changed")
    return value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode)


def _path_directory_identity(path: Path) -> tuple[int, int, int]:
    try:
        return _directory_identity(os.stat(path, follow_symlinks=False))
    except OSError as error:
        raise ValueError("universe output parent must be a directory") from error


def _open_directory(path: Path) -> tuple[int, tuple[int, int, int]]:
    expected = _path_directory_identity(path)
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
            getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as error:
        raise ValueError("universe output parent must be a directory") from error
    try:
        actual = _directory_identity(os.fstat(descriptor))
        if actual != expected:
            raise ValueError("universe output parent changed")
        return descriptor, actual
    except BaseException:
        os.close(descriptor)
        raise


def _open_child_directory(
    parent_fd: int,
    name: str,
) -> tuple[int, tuple[int, int, int]]:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) |
        getattr(os, "O_NOFOLLOW", 0),
        dir_fd=parent_fd,
    )
    try:
        actual = _directory_identity(os.fstat(descriptor))
        named = _directory_identity(os.stat(
            name, dir_fd=parent_fd, follow_symlinks=False,
        ))
        if actual != named:
            raise ValueError("universe output directory changed")
        return descriptor, actual
    except BaseException:
        os.close(descriptor)
        raise


def _member_identity(
    directory_fd: int,
    name: str,
) -> tuple[int, int, int, int]:
    return _file_identity(os.stat(
        name, dir_fd=directory_fd, follow_symlinks=False,
    ))


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ValueError("universe package member changed")
    return (
        value.st_dev, value.st_ino, stat.S_IFMT(value.st_mode),
        value.st_nlink,
    )


def _open_member(
    directory_fd: int,
    name: str,
    expected: tuple[int, int, int, int],
) -> int:
    descriptor = os.open(
        name,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_fd,
    )
    try:
        if _file_identity(os.fstat(descriptor)) != expected:
            raise ValueError("universe package member changed")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _hash_member(
    directory_fd: int,
    name: str,
    expected: tuple[int, int, int, int],
) -> str:
    descriptor = _open_member(directory_fd, name, expected)
    try:
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    if _member_identity(directory_fd, name) != expected:
        raise ValueError("universe package member changed")
    return digest.hexdigest()


def _member_bytes(
    directory_fd: int,
    name: str,
    expected: tuple[int, int, int, int],
) -> bytes:
    descriptor = _open_member(directory_fd, name, expected)
    try:
        chunks = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    if _member_identity(directory_fd, name) != expected:
        raise ValueError("universe package member changed")
    return b"".join(chunks)


def _check_directory(
    parent_fd: int,
    name: str,
    descriptor: int,
    expected: tuple[int, int, int],
) -> None:
    if _directory_identity(os.fstat(descriptor)) != expected or \
       _directory_identity(os.stat(
           name, dir_fd=parent_fd, follow_symlinks=False,
       )) != expected:
        raise ValueError("universe output directory changed")


def _exists_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _report_path(path: Path) -> str:
    root = ROOT.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _source_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("source closure must be under the project root") from error


def _output_target(
    output_dir: Path,
    inputs: Sequence[Path],
) -> tuple[Path, Path]:
    original = Path(output_dir)
    normalized = Path(os.path.abspath(original))
    try:
        resolved = normalized.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ValueError("universe output path is invalid") from error
    for path in dict.fromkeys((original, normalized, resolved)):
        if os.path.lexists(path):
            raise ValueError("universe output must not already exist")
    try:
        for path in (normalized, resolved):
            require_disjoint(inputs, (path, path / "selection.json"))
    except (OSError, RuntimeError) as error:
        raise ValueError("universe output path is invalid") from error
    _path_directory_identity(normalized.parent)
    return normalized, normalized.parent


def _marker_state(
    root_fd: int,
    name: str,
) -> tuple[tuple[int, int], int] | None:
    try:
        value = os.stat(name, dir_fd=root_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(value.st_mode):
        return None
    return (value.st_dev, value.st_ino), value.st_nlink


def _named_state(
    directory_fd: int,
    name: str,
) -> tuple[tuple[int, int], int, int] | None:
    try:
        value = os.stat(
            name, dir_fd=directory_fd, follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    return (
        (value.st_dev, value.st_ino),
        stat.S_IFMT(value.st_mode),
        value.st_nlink,
    )


def _owns_marker(
    root_fd: int,
    name: str,
    identity: tuple[int, int],
    links: tuple[int, ...],
) -> bool:
    state = _marker_state(root_fd, name)
    return state is not None and state[0] == identity and state[1] in links


def _quarantine_marker(
    root_fd: int,
    identity: tuple[int, int],
) -> bool:
    failure: OSError | None = None
    for _ in range(2):
        directory = f".selection-rollback.{secrets.token_hex(16)}"
        os.mkdir(directory, 0o700, dir_fd=root_fd)
        quarantine_fd, _ = _open_child_directory(root_fd, directory)
        try:
            attempt_failure: OSError | None = None
            try:
                rename_noreplace(
                    root_fd, "selection.json",
                    quarantine_fd, "selection.json",
                )
            except OSError as error:
                failure = error
                attempt_failure = error
            source = _named_state(root_fd, "selection.json")
            target = _named_state(quarantine_fd, "selection.json")
            if source is not None:
                os.fsync(quarantine_fd)
                os.fsync(root_fd)
                continue
            if target is None:
                os.fsync(quarantine_fd)
                os.fsync(root_fd)
                return False
            if not rename_may_have_committed(attempt_failure):
                os.fsync(quarantine_fd)
                os.fsync(root_fd)
                return False
            owned = target[0] == identity and \
                target[1] == stat.S_IFREG and target[2] in (1, 2)
            if not owned:
                restore_failure: OSError | None = None
                try:
                    rename_noreplace(
                        quarantine_fd, "selection.json",
                        root_fd, "selection.json",
                    )
                except OSError as error:
                    restore_failure = error
                if _named_state(quarantine_fd, "selection.json") is not None or \
                   _named_state(root_fd, "selection.json") != target:
                    raise OSError(
                        "universe marker rollback collided"
                    ) from restore_failure
            os.fsync(quarantine_fd)
            os.fsync(root_fd)
            return owned
        finally:
            os.close(quarantine_fd)
    raise OSError("universe marker rollback failed") from failure


def select_universe(
    policy_path: Path,
    output_dir: Path,
    *,
    key: str | None = None,
    requester: Requester | None = None,
    requests_per_minute: int = 0,
) -> dict[str, object]:
    resolved = tuple(_input(path) for path in (policy_path, *SOURCE_PATHS))
    inputs = tuple(path for path, _ in resolved)
    identities = tuple(identity for _, identity in resolved)
    if len(set(identities)) != len(identities):
        raise ValueError("universe sources must be distinct")
    output, parent = _output_target(output_dir, inputs)
    parent_fd, parent_identity = _open_directory(parent)
    descriptors = [parent_fd]
    try:
        with freeze_inputs(inputs) as frozen:
            policy = SelectionPolicy.read(frozen[0].snapshot)
            transport, before_request = _transport(
                requester, requests_per_minute,
            )
            bundle = fetch_sources(
                policy,
                api_key(ROOT / ".env") if key is None else key,
                transport,
                before_request,
            )
            selection = select_candidates(
                policy, bundle.references, bundle.sessions,
            )
            verify_frozen(frozen)
            if any(
                _regular_identity(path) != identity
                for path, identity in zip(inputs, identities, strict=True)
            ) or _path_directory_identity(parent) != parent_identity:
                raise ValueError("universe inputs changed")

            source_names = tuple(f"{archive.name}.json" for archive in bundle.archives)
            if len(set(source_names)) != len(source_names) or any(
                Path(name).name != name for name in source_names
            ):
                raise ValueError("universe source archive names are invalid")
            source_values = tuple(
                _source_archive_value(archive) for archive in bundle.archives
            )
            representatives = {
                candidate.reference.share_class_figi:
                candidate.reference.ticker
                for candidate in selection.candidates
                if not candidate.rejection_reasons
            }
            candidates = [
                _candidate_value(
                    candidate,
                    (
                        representatives.get(
                            candidate.reference.share_class_figi,
                        )
                        if not candidate.rejection_reasons or
                        candidate.rejection_reasons ==
                        ("duplicate-share-class-figi",)
                        else None
                    ),
                )
                for candidate in selection.candidates
            ]
            master = [_member_value(candidate) for candidate in selection.master]
            manifest_values = {
                str(size): _manifest_value(policy, master[:size])
                for size in policy.cohort_sizes
            }

            os.mkdir(output.name, 0o700, dir_fd=parent_fd)
            root_fd, root_identity = _open_child_directory(
                parent_fd, output.name,
            )
            descriptors.append(root_fd)
            os.fsync(parent_fd)
            child_fds: dict[str, int] = {}
            child_identities: dict[str, tuple[int, int, int]] = {}
            for name in ("sources", "manifests"):
                os.mkdir(name, 0o700, dir_fd=root_fd)
                descriptor, identity = _open_child_directory(root_fd, name)
                descriptors.append(descriptor)
                child_fds[name], child_identities[name] = descriptor, identity

            source_records: dict[
                str, tuple[tuple[int, int, int, int], str]
            ] = {}
            sources = []
            sessions = frozenset(str(day) for day, _ in bundle.sessions)
            for archive, name, value in zip(
                bundle.archives, source_names, source_values, strict=True,
            ):
                write_json_exclusive(
                    Path(name), value, child_fds["sources"],
                )
                identity = _member_identity(child_fds["sources"], name)
                digest = _hash_member(
                    child_fds["sources"], name, identity,
                )
                source_records[name] = identity, digest
                sources.append(
                    _source_binding_value(archive, sessions, digest)
                )

            manifest_records: dict[
                str, tuple[tuple[int, int, int, int], str]
            ] = {}
            cohorts: dict[str, object] = {}
            for size, size_text in zip(
                policy.cohort_sizes, manifest_values, strict=True,
            ):
                name = f"liquid-common-{size}.json"
                write_json_exclusive(
                    Path(name), manifest_values[size_text],
                    child_fds["manifests"],
                )
                identity = _member_identity(child_fds["manifests"], name)
                digest = _hash_member(
                    child_fds["manifests"], name, identity,
                )
                manifest_records[name] = identity, digest
                members = master[:size]
                cohorts[size_text] = {
                    "size": size,
                    "primary": size == policy.primary_cohort_size,
                    "members": members,
                    "members_sha256": _canonical_sha256(members),
                    "manifest": f"manifests/{name}",
                    "manifest_sha256": digest,
                }

            for descriptor in (
                child_fds["sources"], child_fds["manifests"], root_fd,
            ):
                os.fsync(descriptor)

            report: dict[str, object] = {
                "schema": policy.schema,
                "purpose": policy.purpose,
                "declared_on": str(policy.declared_on),
                "anchor_date": str(policy.anchor_date),
                "formation_start": str(policy.formation_start),
                "formation_end": str(policy.formation_end),
                "start": str(policy.start),
                "end": str(policy.end),
                "primary_cohort_size": policy.primary_cohort_size,
                "policy": {
                    "path": _report_path(inputs[0]),
                    "sha256": frozen[0].sha256,
                },
                "source_closure": [
                    {
                        "path": _source_path(item.source),
                        "sha256": item.sha256,
                    }
                    for item in frozen[1:]
                ],
                "sources": sources,
                "formation_sessions": [
                    str(day) for day, _ in bundle.sessions
                ],
                "candidates": candidates,
                "master": master,
                "master_sha256": _canonical_sha256(master),
                "cohorts": cohorts,
            }
            marker_identity: tuple[int, int] | None = None

            def validate(
                *, marker: bool, private: bool = False,
            ) -> set[str]:
                verify_frozen(frozen)
                if any(
                    _regular_identity(path) != identity
                    for path, identity in zip(
                        inputs, identities, strict=True,
                    )
                ) or _path_directory_identity(parent) != parent_identity:
                    raise ValueError("universe inputs changed")
                _check_directory(
                    parent_fd, output.name, root_fd, root_identity,
                )
                for name in ("sources", "manifests"):
                    _check_directory(
                        root_fd, name, child_fds[name],
                        child_identities[name],
                    )
                if set(os.listdir(child_fds["sources"])) != set(source_records) or \
                   set(os.listdir(child_fds["manifests"])) != set(
                       manifest_records
                   ):
                    raise ValueError("universe package membership changed")
                for directory_fd, records in (
                    (child_fds["sources"], source_records),
                    (child_fds["manifests"], manifest_records),
                ):
                    for name, (identity, digest) in records.items():
                        if _member_identity(directory_fd, name) != identity or \
                           _hash_member(
                               directory_fd, name, identity,
                           ) != digest:
                            raise ValueError("universe package member changed")
                entries = set(os.listdir(root_fd))
                temporary = {
                    name for name in entries
                    if PRIVATE_MARKER.fullmatch(name)
                }
                expected = {"sources", "manifests"}
                if marker:
                    expected.add("selection.json")
                if private:
                    if len(temporary) != 1:
                        raise ValueError(
                            "universe package membership changed"
                        )
                elif temporary:
                    raise ValueError("universe package membership changed")
                if entries - temporary != expected or \
                   _exists_at(root_fd, "selection.json") is not marker:
                    raise ValueError("universe package membership changed")
                if report["master"] != master or \
                   report["master_sha256"] != _canonical_sha256(master):
                    raise ValueError("universe master binding changed")
                for size, size_text in zip(
                    policy.cohort_sizes, cohorts, strict=True,
                ):
                    cohort = cohorts[size_text]
                    members = master[:size]
                    if not isinstance(cohort, Mapping) or \
                       cohort["members"] != members or \
                       cohort["members_sha256"] != _canonical_sha256(
                           members
                       ) or cohort["manifest_sha256"] != manifest_records[
                           f"liquid-common-{size}.json"
                       ][1]:
                        raise ValueError("universe cohort binding changed")
                if marker:
                    if marker_identity is None or not _owns_marker(
                        root_fd, "selection.json", marker_identity, (1,),
                    ):
                        raise ValueError(
                            "universe completion marker changed"
                        )
                    identity = _member_identity(root_fd, "selection.json")
                    if _member_bytes(
                        root_fd, "selection.json", identity,
                    ) != _canonical_bytes(report):
                        raise ValueError("universe completion marker changed")
                return temporary

            def before_link(binding: ExclusiveTemp) -> None:
                nonlocal marker_identity
                marker_identity = binding.identity
                temporary = validate(marker=False, private=True)
                if temporary != {binding.name} or \
                   not PRIVATE_MARKER.fullmatch(binding.name) or \
                   not _owns_marker(
                       root_fd, binding.name, binding.identity, (1,),
                   ):
                    raise ValueError("universe completion marker changed")

            def cleanup_marker() -> None:
                if marker_identity is None:
                    return
                _quarantine_marker(root_fd, marker_identity)

            try:
                write_json_exclusive(
                    Path("selection.json"), report, root_fd,
                    before_link_with_temp=before_link,
                )
            except BaseException:
                cleanup_marker()
                raise

            try:
                validate(marker=True)
                os.fsync(root_fd)
            except BaseException:
                cleanup_marker()
                raise
            return report
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and publish a point-in-time stock universe.",
    )
    parser.add_argument("policy", type=Path, metavar="POLICY")
    parser.add_argument("output_dir", type=Path, metavar="OUTPUT_DIR")
    parser.add_argument(
        "--requests-per-minute", type=int, default=0, metavar="N",
    )
    return parser.parse_args(argv)


def main() -> None:
    arguments = parse_args()
    try:
        report = select_universe(
            arguments.policy,
            arguments.output_dir,
            requests_per_minute=arguments.requests_per_minute,
        )
    except (OSError, ValueError):
        raise SystemExit("universe selection failed") from None
    print(json.dumps(report, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
