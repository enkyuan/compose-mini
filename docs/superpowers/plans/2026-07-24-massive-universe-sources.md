# Massive Universe Source Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fetch the frozen universe's dated Massive reference pages and
pre-model daily summaries into deterministic, secret-free in-memory source
archives.

**Architecture:** Keep URL construction, request sanitization, provider
normalization, and source collection as small functions in the existing
selector module. One `fetch_sources()` path owns pagination and weekday
iteration, calls one externally supplied callback before every logical request,
and returns immutable selection inputs plus canonical provenance records; it
does not write files or select stocks. Live orchestration must inject
`request_json(url, before_request=gate)` and pass a no-op logical callback, so
the same gate covers every physical attempt—including 429 retries—exactly once.

**Tech Stack:** Python 3.12 standard library (`dataclasses`, `datetime`,
`decimal`, `urllib.parse`), existing Massive request primitives, procedural
Python tests, GitButler.

## Global Constraints

- The exact research contract is
  `docs/superpowers/specs/2026-07-24-point-in-time-universe-selection.md`.
- Follow `karpathy-guidelines` and `ponytail`: make the smallest direct change,
  reuse existing primitives, and add no dependency or speculative client
  abstraction.
- Touch only `tools/select_universe.py` and
  `tests/python/test_massive.py`.
- Preserve unrelated `Makefile` and `docs/training.md` changes.
- Use `Decimal(str(value))` for provider numbers, reject nonfinite transport
  values, and emit normalized fixed-point strings without consulting the
  process-wide decimal precision.
- Store only public request paths and sorted public query parameters; never
  store, return, or interpolate `apiKey`.
- Request every formation weekday exactly once. Empty successful weekdays have
  archives but are excluded from the selection denominator; any failed weekday
  aborts the operation.
- Retain daily rows only for the dated reference tickers. Validate uniqueness
  across the full grouped response before filtering, so the smaller archive is
  lossless for selection without hiding provider corruption.
- Keep retrieval independent of files, output publication, model results,
  training, policy selection, and the reserved `$100` backtest.
- Preserve this exact Task 3 live composition instead of adding a second
  transport or key parser:

  ```python
  gate = request_gate(requests_per_minute)
  transport = lambda url: request_json(url, before_request=gate)
  sources = fetch_sources(
      policy, api_key(env_file), transport, lambda: None,
  )
  ```

  `api_key()` reads the secret once, `request_gate()` is shared across both
  endpoint types, `request_json()` gates every retry, and `fetch_sources()`
  still exposes a direct callback for deterministic offline tests. Passing
  `gate` as the final argument in this composition is forbidden because it
  would double-throttle first attempts.
- Use GitButler for version-control inspection and writes. Amend or commit only
  this plan's files on `enkyuan/universe-expansion-plan`, author and sign as
  `enkyuan <yuan.enkng@gmail.com>`, and do not push or land.

---

### Task 1: Retrieve and Normalize Massive Selection Sources

**Files:**

- Modify: `tools/select_universe.py`
- Modify: `tests/python/test_massive.py`

**Interfaces:**

- Consumes:
  `tools.fetch_massive.API_HOST`;
  `tools.fetch_massive.Requester`;
  `tools.fetch_massive.authorized_url`;
  `SelectionPolicy`;
  `Reference`;
  `DailyRow`.
- Produces:
  `reference_universe_url(date) -> str`;
  `daily_summary_url(date) -> str`;
  `SourceArchive`;
  `SourceBundle`;
  `fetch_sources(SelectionPolicy, str, Requester, Callable[[], None]) ->
  SourceBundle`.
- `SourceBundle.references` is strictly ticker-ordered across all reference
  pages.
- `SourceBundle.sessions` contains only successful nonempty weekday summaries,
  ordered by date, with ticker-ordered rows.
- `SourceBundle.archives` contains every reference page followed by every
  requested weekday, including empty successful weekdays.

- [ ] **Step 1: Add the complete offline success and rejection tests**

In `tests/python/test_massive.py`, add `replace` and `timedelta`, and replace
the existing `urllib.parse` import with:

```python
from dataclasses import replace
from datetime import date, datetime, timedelta
from urllib.parse import (
    parse_qsl, parse_qs, unquote, urlencode, urlsplit, urlunsplit,
)
```

Extend the selector import with the new public interfaces:

```python
from tools.select_universe import (
    Candidate, DailyRow, Reference, SelectionPolicy, SourceArchive,
    SourceBundle, daily_summary_url, fetch_sources, reference_universe_url,
    select_candidates,
)
```

Add these helpers and tests immediately after `test_pure_selection()`:

```python
def source_reference(ticker: str) -> dict[str, object]:
    return {
        "active": True,
        "composite_figi": f"BBG{ticker}",
        "currency_name": "usd",
        "locale": "us",
        "market": "stocks",
        "primary_exchange": "XNYS",
        "share_class_figi": f"SC{ticker}",
        "ticker": ticker,
        "type": "CS",
    }


def source_daily(ticker: str, close: object = 10.25) -> dict[str, object]:
    return {"T": ticker, "c": close, "v": 2_000_000, "vw": 10}


def weekdays(start: date, end: date) -> tuple[date, ...]:
    days = []
    while start <= end:
        if start.weekday() < 5:
            days.append(start)
        start += timedelta(days=1)
    return tuple(days)


def test_universe_sources() -> None:
    policy = SelectionPolicy.read(
        ROOT / "universes/liquid-common-ladder.example.json",
    )
    formation_days = weekdays(policy.formation_start, policy.formation_end)
    empty_day = formation_days[0]
    next_page = (
        "https://api.massive.com/v3/reference/tickers"
        "?apiKey=provider-value&cursor=next"
    )
    requested: list[str] = []
    gated: list[int] = []

    def before_request() -> None:
        gated.append(len(requested))

    def requester(url: str) -> dict[str, object]:
        gated_index = len(requested)
        requested.append(url)
        assert gated[-1] == gated_index
        parts = urlsplit(url)
        query = dict(parse_qsl(parts.query))
        assert query.pop("apiKey") == "source-secret"
        if parts.path == "/v3/reference/tickers":
            return (
                {
                    "status": "OK",
                    "results": [
                        source_reference("AAPL"),
                        source_reference("MSFT"),
                    ],
                    "next_url": next_page,
                }
                if "cursor" not in query
                else {
                    "status": "OK",
                    "results": [
                        source_reference("SPY") |
                        {"share_class_figi": None}
                    ],
                }
            )
        day = date.fromisoformat(parts.path.rsplit("/", 1)[-1])
        assert query == {"adjusted": "false", "include_otc": "false"}
        return (
            {"status": "OK", "resultsCount": 0}
            if day == empty_day
            else {
                "status": "OK",
                "resultsCount": 3,
                "results": [
                    {"T": "ZZZZ"},
                    source_daily("MSFT", 20.5),
                    source_daily("AAPL", 0.1 + 0.2),
                ],
            }
        )

    bundle = fetch_sources(
        policy, "source-secret", requester, before_request,
    )
    assert isinstance(bundle, SourceBundle)
    assert all(isinstance(item, SourceArchive) for item in bundle.archives)
    assert tuple(item.ticker for item in bundle.references) == (
        "AAPL", "MSFT", "SPY",
    )
    assert bundle.references[2].share_class_figi is None
    assert len(bundle.sessions) == len(formation_days) - 1
    assert all(
        tuple(row.ticker for row in rows) == ("AAPL", "MSFT")
        for _, rows in bundle.sessions
    )
    assert bundle.sessions[0][1][0].close == Decimal(
        "0.30000000000000004"
    )
    assert len(requested) == len(gated) == len(formation_days) + 2
    assert tuple(
        date.fromisoformat(urlsplit(url).path.rsplit("/", 1)[-1])
        for url in requested[2:]
    ) == formation_days

    first = bundle.archives[0]
    assert first.name == "tickers-0001"
    assert first.request == {
        "path": "/v3/reference/tickers",
        "query": {
            "active": "true",
            "date": "2024-10-31",
            "limit": "1000",
            "market": "stocks",
            "order": "asc",
            "sort": "ticker",
            "type": "CS",
        },
    }
    assert bundle.archives[1].name == "tickers-0002"
    assert bundle.archives[2].name == f"daily-{empty_day}"
    assert bundle.archives[2].records == ()
    populated = bundle.archives[3]
    assert populated.records == (
        {
            "ticker": "AAPL",
            "c": "0.30000000000000004",
            "v": "2000000",
            "vw": "10",
        },
        {"ticker": "MSFT", "c": "20.5", "v": "2000000", "vw": "10"},
    )
    for archive in bundle.archives:
        assert tuple(archive.request["query"]) == tuple(
            sorted(archive.request["query"])
        )
        value = {
            "name": archive.name,
            "records": archive.records,
            "request": archive.request,
            "schema": 1,
        }
        serialized = json.dumps(value, sort_keys=True)
        assert "apiKey" not in serialized
        assert "provider-value" not in serialized
        assert "source-secret" not in serialized


def test_universe_source_rejections() -> None:
    tracked = SelectionPolicy.read(
        ROOT / "universes/liquid-common-ladder.example.json",
    )
    policy = replace(
        tracked,
        anchor_date=tracked.formation_end,
        formation_start=tracked.formation_end,
        minimum_formation_sessions=1,
    )
    daily_ok = {
        "status": "OK",
        "results": [source_daily("AAPL")],
        "resultsCount": 1,
    }

    def attempt(
        reference: object,
        daily: object = daily_ok,
    ) -> BaseException:
        def requester(url: str) -> object:
            return (
                reference
                if urlsplit(url).path == "/v3/reference/tickers"
                else daily
            )

        return raises(
            ValueError, fetch_sources, policy, "source-secret",
            requester, lambda: None,
        )

    for payload in (
        [],
        {"status": "ERROR", "results": []},
        {"status": "OK", "results": {}},
        {"status": "OK", "results": []},
        {"status": "OK", "results": [{}]},
        {
            "status": "OK",
            "results": [source_reference("AAPL") | {"active": 1}],
        },
        {
            "status": "OK",
            "results": [
                source_reference("MSFT"),
                source_reference("AAPL"),
            ],
        },
        {
            "status": "OK",
            "results": [
                source_reference("AAPL"),
                source_reference("AAPL"),
            ],
        },
        {
            "status": "OK",
            "results": [source_reference("AAPL")],
            "next_url": 1,
        },
    ):
        attempt(payload)

    initial = reference_universe_url(policy.anchor_date)
    attempt({
        "status": "OK",
        "results": [source_reference("AAPL")],
        "next_url": "https://example.com/v3/reference/tickers",
    })
    parts = urlsplit(initial)
    cycle = urlunsplit((
        parts.scheme,
        parts.netloc,
        parts.path,
        urlencode(
            tuple(reversed(parse_qsl(parts.query))) +
            (("apiKey", "different-provider-value"),)
        ),
        "",
    ))
    attempt({
        "status": "OK",
        "results": [source_reference("AAPL")],
        "next_url": cycle,
    })

    reference_ok = {
        "status": "OK",
        "results": [source_reference("AAPL")],
    }
    for payload in (
        [],
        {"status": "ERROR", "results": []},
        {"status": "OK", "results": {}},
        {"status": "OK", "results": [{}]},
        {
            "status": "OK",
            "results": [source_daily("AAPL"), source_daily("AAPL")],
        },
        {
            "status": "OK",
            "results": [{"T": "ZZZZ"}, {"T": "ZZZZ"}],
        },
        {
            "status": "OK",
            "results": [{"T": "AAPL"}],
        },
        {
            "status": "OK",
            "results": [source_daily("AAPL", float("nan"))],
        },
        {
            "status": "OK",
            "results": [
                source_daily("AAPL") | {"c": True}
            ],
        },
        {"status": "OK", "results": [], "resultsCount": 1},
        {"status": "OK", "results": [], "resultsCount": False},
        {"status": "OK", "resultsCount": 0},
    ):
        attempt(reference_ok, payload)

    def leaking(url: str) -> object:
        raise OSError(url)

    error = raises(
        ValueError, fetch_sources, policy, "source-secret",
        leaking, lambda: None,
    )
    assert "source-secret" not in str(error)
    for invalid_key in ("", " ", "two words"):
        raises(
            ValueError, fetch_sources, policy, invalid_key,
            lambda _url: reference_ok, lambda: None,
        )

    assert reference_universe_url(policy.anchor_date) == (
        "https://api.massive.com/v3/reference/tickers"
        "?active=true&date=2024-10-31&limit=1000&market=stocks"
        "&order=asc&sort=ticker&type=CS"
    )
    assert daily_summary_url(policy.formation_end) == (
        "https://api.massive.com/v2/aggs/grouped/locale/us/market/stocks"
        "/2024-10-31?adjusted=false&include_otc=false"
    )
```

Add both tests to `main()` after `test_pure_selection()`:

```python
        test_universe_sources()
        test_universe_source_rejections()
```

These tests use three symbols because transport correctness does not require
the selector's 55 eligible members. The tracked three-month interval still
proves that all weekdays share one gate and that one empty day does not become
a zero-valued observation.

- [ ] **Step 2: Run the focused test and confirm the RED boundary**

Run:

```sh
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tests/python/test_massive.py
```

Expected: fail during import because `SourceArchive`, `SourceBundle`,
`reference_universe_url`, `daily_summary_url`, and `fetch_sources` do not yet
exist.

- [ ] **Step 3: Add the smallest complete source-retrieval implementation**

In `tools/select_universe.py`, extend the imports:

```python
from collections.abc import Callable, Mapping, Sequence
from datetime import date, timedelta
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from tools.fetch_massive import (
    API_HOST, TICKER, Requester, authorized_url,
)
```

Add these immutable boundaries after `Selection`:

```python
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
```

Add the URL, sanitization, normalization, and retrieval functions after
`_ticker_valid()`:

```python
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
) -> tuple[dict[str, object], Reference]:
    if not isinstance(value, Mapping) or not _ticker_valid(value.get("ticker")):
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
    return record, Reference(*(record[name] for name in REFERENCE_FIELDS))


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
            if reference.ticker <= previous:
                raise ValueError(
                    "Massive reference tickers are not strictly increasing"
                )
            previous = reference.ticker
            records.append(record)
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
                if not isinstance(value, Mapping) or not _ticker_valid(
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
            if normalized:
                sessions.append((
                    day, tuple(row for _, row in normalized),
                ))
        day += timedelta(days=1)
    if len(sessions) < policy.minimum_formation_sessions:
        raise ValueError("Massive returned too few nonempty formation sessions")
    return SourceBundle(
        tuple(references), tuple(sessions), tuple(archives),
    )
```

The two URL builders make the provider contract directly testable. The
sanitizer calls the existing trusted-host gate before constructing a canonical
public URL, removes any provider-supplied key, rejects ambiguous duplicate
query names, and never returns the authorized URL. `_request()` invokes the
single supplied callback immediately before transport and maps transport
failures to a fixed secret-free error. `_results()` distinguishes an explicitly
empty grouped response from a malformed missing result and verifies a supplied
count. The record functions normalize once at the boundary so both source
archives and selection inputs derive from the same values; noncandidate daily
rows are discarded only after whole-response ticker uniqueness is proved.

- [ ] **Step 4: Run the focused test and confirm GREEN**

Run:

```sh
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tests/python/test_massive.py
```

Expected:

```text
Massive downloader and universe tests passed
```

- [ ] **Step 5: Run an independent task review**

Run an independent specification and code-quality review over only the changes
to `tools/select_universe.py` and `tests/python/test_massive.py`.

Expected: no Critical or Important findings. If either appears, change only
these two files and repeat this step.

- [ ] **Step 6: Rerun the focused gate after review**

Run:

```sh
/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  tests/python/test_massive.py
```

Expected:

```text
Massive downloader and universe tests passed
```

- [ ] **Step 7: Obtain the exact implementation change IDs**

Run:

```sh
/usr/local/bin/but diff
```

Expected: the output includes one file ID for `tools/select_universe.py` and
one file ID or the task-owned hunks for `tests/python/test_massive.py`, while
the unrelated `Makefile` and `docs/training.md` IDs remain separate. Copy the
two task-owned IDs exactly; GitButler IDs are dynamic and must never be
invented.

- [ ] **Step 8: Create the signed local implementation checkpoint**

Run this command after replacing `TOOL_ID,TEST_ID` with the two literal IDs
printed by Step 7:

```sh
/usr/local/bin/but commit enkyuan/universe-expansion-plan \
  -m "feat(data): archive Massive universe sources" \
  --changes TOOL_ID,TEST_ID
```

Expected: GitButler prints one new top commit on
`enkyuan/universe-expansion-plan`; only the selector and its test belong to
that commit. Do not include this plan file in the implementation checkpoint.

- [ ] **Step 9: Verify identity and signature**

Copy the commit SHA printed by Step 8 and run:

```sh
/usr/local/bin/but show COMMIT_SHA
git -c gpg.ssh.allowedSignersFile=/tmp/compose-mini-allowed-signers \
  verify-commit COMMIT_SHA
```

Before the second command, create `/tmp/compose-mini-allowed-signers` with
`apply_patch` containing exactly:

```text
enkyuan ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIDBZVEfLEVJmB7ZzoIF+fqVA1sSKgmkLUKt1NjbTUBm2
```

Expected: `but show` reports author and committer
`enkyuan <yuan.enkng@gmail.com>`, and verification reports:

```text
Good "git" signature for enkyuan with ED25519 key SHA256:mnKfNNJJUV5ELrMXZCwdu6v/PusXBeAi8tWUnRXmWMQ
```

Delete the temporary allowed-signers file with `apply_patch`. Do not push or
land.
