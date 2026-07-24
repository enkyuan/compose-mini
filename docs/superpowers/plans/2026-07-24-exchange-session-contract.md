# Exchange Session Contract Implementation Plan

> **For implementation:** complete each checkbox in order. Keep downloaded
> bars, reports, models, credentials, and caches untracked.

**Goal:** Make every new Massive universe fetch use a frozen, hash-bound U.S.
equities core-session calendar so holidays and 13:00 ET early closes cannot be
misclassified as regular-session training data.

**Architecture:** Store a 2024-07-22 through 2026-07-21 calendar that covers
the existing data flow and the benchmark's exact 2024-11-01 start as a small
tracked JSON input sourced from official ICE/NYSE and Nasdaq notices. Parse it
once with a standard-library `SessionCalendar`. Inject that value into the
existing bar scanner; keep the scanner's observed bar and internal-gap
semantics unchanged. Publish a versioned fetch report that binds both the
universe manifest and calendar by path and SHA-256. Require the same
independently frozen calendar when validating or replaying a new report, while
preserving legacy report replay.

**Tech stack:** Python standard library, Massive REST API, Make, GitButler.

---

## Contract and rationale

- Core session: `[09:30, close)` in `America/New_York`.
- Normal close: 16:00 ET.
- Early close: 13:00 ET on the exact dates frozen in the calendar.
- Closed dates include scheduled holidays and the January 9, 2025 National Day
  of Mourning.
- Bars on closed dates or at/after that session's close are excluded.
- A missing opening aggregate is retained as an observed-data fact, not treated
  as a fatal internal gap.
- Internal gaps remain measured only between adjacent observed bars. No OHLCV
  row is synthesized, forward-filled, or interpolated.
- New reports use `fetch_schema: 3`; schema-2 and unversioned reports remain
  replayable under their original fixed-clock semantics.

For session `d`, interval `m`, opening minute `o_d`, and closing minute `c_d`,
the expected regular-session bin starts are

\[
B_d=\{o_d+km\mid k\in\mathbb{N}_0,\ o_d+(k+1)m\le c_d\}.
\]

Filtering an observed timestamp at local minute `q` is therefore the predicate

\[
d\text{ is open}\ \land\ o_d\le q\ \land\ q+m\le c_d\ \land\
(q-o_d)\bmod m=0.
\]

This removes the early-close contamination mechanism: for a 13:00 close,
`q >= 780` is excluded even if Massive returns extended-hours aggregates.

## Task 1: Freeze and validate the calendar

**Files:**

- Create:
  `universes/us-equities-core-2024-07-22_2026-07-21.json`
- Create: `tools/session_calendar.py`
- Modify: `tests/python/test_massive.py`

- [ ] Add failing tests for exact fields, duplicate keys, canonical ISO dates,
  sorted unique closed dates, valid early closes, supported timezone, regular
  files only, and requested-date coverage.
- [ ] Add scanner-facing tests for a normal day, early close, closed day,
  daylight-saving transitions, and a date outside the frozen range.
- [ ] Implement one immutable `SessionCalendar` parser and a
  `session(date) -> (open_minute, close_minute) | None` lookup.
- [ ] Freeze the official source URLs in the JSON and enumerate only dates
  within its declared range.
- [ ] Run:

```sh
PRIMARY_PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
"$PRIMARY_PYTHON" tests/python/test_massive.py
```

Expected: pass.

## Task 2: Make filtering calendar-aware

**Files:**

- Modify: `tools/fetch_massive.py`
- Modify: `tests/python/test_massive.py`

- [ ] Add a failing early-close test containing observed 12:30, 13:00, and
  13:30 ET bars; retain only 12:30.
- [ ] Add a failing test showing a session whose first observed bar is later
  than 09:30 remains usable and its later internal gaps are audited.
- [ ] Pass `SessionCalendar` into `scan_regular_bars`; retain the current
  fixed-clock path only for legacy replay.
- [ ] Make the single-ticker CLI accept the same calendar input for regular
  sessions.
- [ ] Run:

```sh
"$PRIMARY_PYTHON" tests/python/test_massive.py
```

Expected: pass.

## Task 3: Bind the calendar into new fetch reports

**Files:**

- Modify: `tools/fetch_universe.py`
- Modify: `tools/analyze_universe.py`
- Modify: `tools/replay_calibration.py`
- Modify: `tests/python/test_massive.py`
- Modify: `tests/python/test_universe_analysis.py`

- [ ] Freeze the manifest and calendar together before parsing or requesting
  data.
- [ ] Require Massive's point-in-time `primary_exchange` to be one of the
  calendar's frozen `XNAS` or `XNYS` venues.
- [ ] Emit `fetch_schema: 3` and a top-level calendar path/SHA-256 binding.
- [ ] Require an independently frozen matching calendar for schema-3 analysis
  and replay; preserve schema-2 and unversioned validation.
- [ ] Recompute every schema-3 gap audit from the CSV timestamps and frozen
  calendar.
- [ ] Add producer-to-consumer tests that reject calendar path/hash changes,
  schedule changes, early-close after-hours rows, and missing calendar input.
- [ ] Run:

```sh
"$PRIMARY_PYTHON" tests/python/test_massive.py
"$PRIMARY_PYTHON" tests/python/test_universe_analysis.py
make -B PYTHON="$PRIMARY_PYTHON" check
```

Expected: all checks pass.

## Task 4: Fetch the clean 55-stock cohort

**Generated, ignored outputs:**

- Create: `data/liquid-common-55-20260724-02/`
- Create: `reports/liquid-common-55-20260724-02-fetch.json`

- [ ] Confirm `.env` is ignored and `MASSIVE_API_KEY` is available without
  printing it.
- [ ] Fetch the exact signed 55-stock manifest with the frozen calendar and
  Massive pagination/rate limiting.
- [ ] Validate every output CSV, calendar binding, reference contract, source
  row count, session count, and recomputed gap audit.
- [ ] Do not reuse or delete the invalid `-01` partial fetch.
- [ ] Record elapsed time and API request count without tracking generated
  outputs.

## Task 5: Resume the scaling experiment

- [ ] Run local breadth on all 55 stocks.
- [ ] Run shared-model curves at 11/22/33/55 with fixed optimizer updates.
- [ ] Run the separately labeled fixed-epoch curves.
- [ ] Evaluate unseen-stock transfer on fixed ranks 45 through 55.
- [ ] Report equal-stock macro MAE, direction accuracy, paired block-bootstrap
  intervals, and cross-sectional effective count.
- [ ] Open the `$100` backtest only if the predeclared forecast and cost-aware
  gates pass.

The remaining scaling math, fixed-update budget, paired comparison, and
backtest gates stay unchanged in
`docs/superpowers/plans/2026-07-24-universe-scaling-benchmark.md`.
