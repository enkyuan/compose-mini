# Expected Session Grid Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every new calendar-aware universe fetch report leading,
internal, trailing, and whole-session missing aggregate bins without
synthesizing market data.

**Architecture:** Keep the existing scanner responsible for selecting ordered,
aligned, complete regular-session bars. Add one standard-library helper beside
it that compares those selected starts with the manifest's exact exchange
session grid and compresses missing starts into maximal, end-exclusive ranges.
Publish that stronger contract as fetch schema 4; continue validating
unversioned, schema-2, and schema-3 reports under their original semantics.

**Tech Stack:** Python 3.12 standard library, Massive REST API, Make,
GitButler.

## Global Constraints

- Do not synthesize, forward-fill, interpolate, or discard an observed OHLCV
  bar.
- Keep generated data, fetch reports, models, credentials, and caches
  untracked.
- Keep schema-3's calendar-aware internal-gap contract byte-compatible.
- Treat audit findings as evidence, not as an automatic stock rejection rule.
- Do not train or replay `$100` until the separate fixed-update forecast gates
  pass.
- Use no new dependency.

---

## Contract

For open session `d`, interval `m`, opening minute `o_d`, and closing minute
`c_d`, define the expected starts

\[
B_d=\{o_d+km\mid k\in\mathbb{N}_0,\ o_d+(k+1)m\le c_d\}.
\]

For observed regular starts `O_d`, the missing set is

\[
M_d=B_d\setminus O_d.
\]

The audit reports

\[
S=\sum_d 1[|B_d|>0],\quad
E=\sum_d |B_d|,\quad
A=\sum_d 1[|M_d|>0],\quad
M=\sum_d |M_d|.
\]

After rejecting observed starts outside the requested grid, these invariants
must hold:

\[
\texttt{csv.sessions}=S-|\{d:O_d=\varnothing\}|,
\]

\[
\texttt{csv.rows}=E-M
=E-\sum_r\texttt{r.absent_bins}.
\]

Schema-4 `session_audit` is exactly:

```json
{
  "scope": "all-expected-session-bins",
  "expected_sessions": 2,
  "affected_sessions": 2,
  "missing_sessions": ["2024-11-04"],
  "expected_bins": 26,
  "missing_bins": 24,
  "ranges": [
    {
      "session": "2024-11-01",
      "start_timestamp": "2024-11-01T13:30:00Z",
      "end_timestamp": "2024-11-01T14:00:00Z",
      "absent_bins": 1
    },
    {
      "session": "2024-11-01",
      "start_timestamp": "2024-11-01T14:30:00Z",
      "end_timestamp": "2024-11-01T15:00:00Z",
      "absent_bins": 1
    },
    {
      "session": "2024-11-01",
      "start_timestamp": "2024-11-01T15:30:00Z",
      "end_timestamp": "2024-11-01T20:00:00Z",
      "absent_bins": 9
    },
    {
      "session": "2024-11-04",
      "start_timestamp": "2024-11-04T14:30:00Z",
      "end_timestamp": "2024-11-04T21:00:00Z",
      "absent_bins": 13
    }
  ]
}
```

Range starts are missing bar starts. Range ends are exclusive session-grid
boundaries, not missing observations. Ranges are sorted and maximal within one
session. The outer CSV already records observed rows and sessions, so the audit
does not duplicate those values or serialize a floating coverage ratio.

Massive explicitly documents that an interval with no eligible trade has no
aggregate bar. Missing bins therefore remain observations about data
availability, not fabricated zero-volume bars:
`https://massive.com/docs/rest/stocks/aggregates/custom-bars`.

## Task 1: Add the shared full-grid audit

**Files:**

- Modify: `tools/fetch_massive.py`
- Modify: `tests/python/test_massive.py`

**Interfaces:**

- Consumes:
  `Sequence[Bar]`, interval minutes, `SessionCalendar`, inclusive start date,
  and inclusive end date.
- Produces:
  `session_grid_audit(...) -> dict[str, object]` with the exact schema above.

- [ ] **Step 1: Write the failing normal-session and DST test**

Add a two-session test spanning Friday, November 1, 2024 in EDT and Monday,
November 4 in EST. Observe only 10:00 and 11:00 ET on November 1 and no
November 4 bars.

```python
audit = session_grid_audit(
    observed, 30, SessionCalendar.read(DEFAULT_CALENDAR),
    date(2024, 11, 1), date(2024, 11, 4),
)
assert {
    name: audit[name] for name in (
        "expected_sessions", "affected_sessions", "missing_sessions",
        "expected_bins", "missing_bins",
    )
} == {
    "expected_sessions": 2,
    "affected_sessions": 2,
    "missing_sessions": ["2024-11-04"],
    "expected_bins": 26,
    "missing_bins": 24,
}
assert [item["absent_bins"] for item in audit["ranges"]] == [1, 1, 9, 13]
```

Assert all four exact UTC start/end pairs. This proves leading, internal,
trailing, whole-session, DST-aware, maximal, and end-exclusive behavior.

- [ ] **Step 2: Write the failing early-close and range-boundary tests**

Audit an empty November 29, 2024 early-close session:

```python
audit = session_grid_audit(
    (), 30, calendar, date(2024, 11, 29), date(2024, 11, 29),
)
assert audit["expected_bins"] == audit["missing_bins"] == 7
assert audit["ranges"] == [{
    "session": "2024-11-29",
    "start_timestamp": "2024-11-29T14:30:00Z",
    "end_timestamp": "2024-11-29T18:00:00Z",
    "absent_bins": 7,
}]
```

Also pass one valid regular-session bar outside `[start, end]` and require
`ValueError("observed bar is outside the requested session grid")`.

- [ ] **Step 3: Run the focused test and verify failure**

Run:

```sh
PRIMARY_PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3
"$PRIMARY_PYTHON" tests/python/test_massive.py
```

Expected: fail because `session_grid_audit` is not defined.

- [ ] **Step 4: Implement the minimum shared helper**

Add one timestamp conversion helper reused by the scanner and audit. Iterate
calendar dates once, generate complete interval starts with
`range(open_, close - minutes + 1, minutes)`, reject observed keys outside that
grid, and stream missing keys into maximal ranges. Convert each local boundary
through `America/New_York`; never assume a fixed UTC offset.

The function must be linear in scanned dates, expected bins, and observed bins:

\[
O(D+E+O).
\]

For the 55-stock benchmark, this is at most
`55 * 5,534 = 304,370` expected-grid membership checks plus
`55 * 628 = 34,540` calendar-date checks.

- [ ] **Step 5: Run the focused test and verify pass**

Run:

```sh
"$PRIMARY_PYTHON" tests/python/test_massive.py
```

Expected: `Massive downloader and universe tests passed`.

## Task 2: Publish schema 4 and preserve all older contracts

**Files:**

- Modify: `tools/fetch_universe.py`
- Modify: `tools/analyze_universe.py`
- Inspect: `tools/replay_calibration.py`
- Modify: `tests/python/test_massive.py`
- Modify: `tests/python/test_universe_analysis.py`

**Interfaces:**

- Consumes: the Task 1 `session_grid_audit` helper.
- Produces: schema-4 fetch reports containing `csv.session_audit`.
- Preserves:
  unversioned reports without an audit or calendar, schema 2 with fixed-clock
  `gap_audit`, and schema 3 with calendar-aware internal `gap_audit`.

- [ ] **Step 1: Write failing producer tests**

Require every new report to emit `fetch_schema: 4`. For the exact benchmark
range, assert `expected_sessions == 428` and `expected_bins == 5_534`.
For a fixture with only the November 1 opening bar, assert:

```python
assert audit["affected_sessions"] == 428
assert len(audit["missing_sessions"]) == 427
assert audit["missing_bins"] == 5_533
assert sum(item["absent_bins"] for item in audit["ranges"]) == 5_533
```

Assert representative first and last ranges instead of serializing the full
range list into the test.

- [ ] **Step 2: Write failing compatibility and mutation tests**

Construct four independent report shapes:

1. unversioned: no policy, calendar, or audit;
2. schema 2: fixed-clock internal `gap_audit`;
3. schema 3: frozen-calendar internal `gap_audit`;
4. schema 4: frozen-calendar `session_audit`.

Require all four valid forms to replay. For schema 4, reject mutations to every
scalar, missing-session order, range order, range start, exclusive end,
`absent_bins`, and any extra field. Use a multi-row, multi-session CSV fixture.
Remove a leading, trailing, or internal row from a still-nonempty session while
updating `rows` and `sha256`; stale audit metadata must fail. Remove every row
from one session while updating `rows`, `sha256`, and `sessions`; the unchanged
`session_audit` must still fail specifically because `missing_sessions` is
stale.

- [ ] **Step 3: Run tests and verify failure**

Run:

```sh
"$PRIMARY_PYTHON" tests/python/test_massive.py
"$PRIMARY_PYTHON" tests/python/test_universe_analysis.py
```

Expected: schema and audit assertions fail against schema 3.

- [ ] **Step 4: Implement schema dispatch**

Use explicit constants:

```python
FETCH_SCHEMA = 4
CALENDAR_FETCH_SCHEMA = 3
PREVIOUS_FETCH_SCHEMA = 2
INTERNAL_GAP_SCOPE = "internal-between-observed-bars"
SESSION_GAP_SCOPE = "all-expected-session-bins"
```

In the analyzer:

```python
calendar_aware = version in (CALENDAR_FETCH_SCHEMA, FETCH_SCHEMA)
session_aware = version == FETCH_SCHEMA
audit_field = (
    "session_audit" if session_aware
    else "gap_audit" if version is not None
    else None
)
```

Recompute schema-4 metadata from the frozen CSV, manifest bounds, and
independently frozen calendar. Keep the schema-2 and schema-3 code paths exact;
do not reinterpret old reports.

- [ ] **Step 5: Run focused tests and verify pass**

Run:

```sh
"$PRIMARY_PYTHON" tests/python/test_massive.py
"$PRIMARY_PYTHON" tests/python/test_universe_analysis.py
```

Expected: both suites pass.

## Task 3: Document, verify, and checkpoint

**Files:**

- Modify: `docs/forecasting-contract.md`
- Modify:
  `docs/superpowers/plans/2026-07-24-universe-scaling-benchmark.md`
- Leave unchanged: `docs/training.md` while its unrelated active hunk belongs
  to another session

**Interfaces:**

- Consumes: schema-4 fields and compatibility behavior from Task 2.
- Produces: explicit operator guidance that a full-grid audit still does not
  make row-offset model windows elapsed-time-safe.

- [ ] **Step 1: Update the operator contract**

State directly:

- schema 4 detects every expected missing bin and whole missing session;
- a missing aggregate may mean no qualifying trade, halt, provider omission,
  or pre-listing;
- no bar is filled or synthesized;
- training remains locked until a later sample-builder checkpoint either
  requires a complete lookback/target grid or explicitly masks affected
  windows.

Do not edit `docs/training.md` until its active owner checkpoints the unrelated
panel-Transformer guidance and exposes a clean dependency. Place this
checkpoint's guidance only in the two clean files listed above.

- [ ] **Step 2: Run syntax, focused, and aggregate checks**

Run:

```sh
"$PRIMARY_PYTHON" -m py_compile \
  tools/fetch_massive.py tools/fetch_universe.py \
  tools/analyze_universe.py tools/replay_calibration.py \
  tests/python/test_massive.py tests/python/test_universe_analysis.py
"$PRIMARY_PYTHON" tests/python/test_massive.py
"$PRIMARY_PYTHON" tests/python/test_universe_analysis.py
make -B PYTHON="$PRIMARY_PYTHON" check
```

Expected: all checks pass.

- [ ] **Step 3: Review for needless code**

Verify there is one calendar-grid implementation, no new dependency, no
floating coverage field, no duplicate observed counts, and no changes to
generated outputs or credentials.

- [ ] **Step 4: Create a signed local checkpoint**

Use `but diff`, then pass only the printed IDs for this plan's files or hunks
to `but commit enkyuan/session-grid-audit` with message
`feat(data): audit expected session grid`. Stack the branch above
`enkyuan/session-grid-audit-plan`. Do not push.

## Task 4: Resume the broader-universe work

- [ ] Fetch the exact ignored 55-stock manifest into the absent `-02` output
  paths with the new schema-4 report.
- [ ] Verify `55` ordered CSVs, hashes, audit invariants, elapsed-horizon
  distributions, split availability, and absence of credential bytes.
- [ ] Implement continuity-aware sample construction before interpreting
  row-offset horizon results.
- [ ] Implement the missing fixed-update `11/22/33/55` shared-model driver and
  validator.
- [ ] Open the `$100` backtest only if the predeclared forecast and cost-aware
  gates pass.

The invalid `data/liquid-common-55-20260724-01` remains untouched and must
never be reused.
