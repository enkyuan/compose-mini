# SPY Benchmark Fetch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` or execute the task inline with its stated
> checks. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one immutable, calendar-complete SPY data-and-provenance
bundle for residual calibration without changing authenticated source files,
reading model truth, or authorizing a backtest.

**Architecture:** Leave the existing Massive downloader, common-stock fetcher,
calendar parser, and XNAS/XNYS calendar byte-for-byte unchanged. Add one narrow
SPY acquisition tool that reuses those primitives against private staging
paths. It validates the point-in-time ETF identity, records the explicit
`ARCX -> XNYS operating MIC -> NYSE Group core session` applicability, and
publishes the filtered CSV and report with one no-replace directory rename.

**Tech Stack:** Python 3.12 standard library, Massive REST API, existing
`tools.fetch_massive`, `tools.files`, and `tools.session_calendar` primitives.

## Global Constraints

- Fetch only `SPY`, with reference date `2024-10-31`.
- Require exactly:
  `active=true`, `market=stocks`, `locale=us`, `type=ETF`,
  `currency_name=usd`, and `primary_exchange=ARCX`.
- Bind ARCX to operating MIC `XNYS` and the exact existing frozen calendar.
  Do not claim ARCX appears in that calendar's venue list.
- Fetch split-adjusted 30-minute price bars from `2024-11-01` through
  `2026-07-21`, ascending, with API limit `50000`.
- Treat the source as price-return data, not dividend-adjusted total-return
  data.
- Follow only opaque HTTPS pagination URLs on `api.massive.com`; never persist
  the API key.
- Filter extended-hours rows locally and require exactly `428` sessions and
  `5,534` bars with zero missing bins.
- Refuse existing, aliased, symlinked, or concurrently occupied bundles.
- Stage `spy.csv` and `fetch.json` privately, then make both visible with one
  atomic no-replace directory rename.
- Validate both final files after publication. Any failed post-publication
  validation burns the bundle name and does not return success.
- Keep generated CSVs, reports, credentials, models, and caches ignored.
- Do not modify `tools/fetch_massive.py`, `tools/fetch_universe.py`,
  `tools/session_calendar.py`, the default calendar, or `Makefile`.
- Do not import Torch, read labels or predictions, train, evaluate, or run the
  `$100` backtest in this checkpoint.

---

### Task 1: Write the Benchmark Acquisition Contract

**Files:**

- Create: `tools/fetch_benchmark.py`
- Modify: `tests/python/test_massive.py`

**Interfaces:**

- `fetch_benchmark(bundle, *, calendar_path=DEFAULT_CALENDAR,
  env_file=ROOT / ".env", key=None, requester=None) -> Mapping[str, object]`
- CLI:

  ```text
  tools/fetch_benchmark.py BUNDLE [--calendar PATH] [--env-file PATH]
  ```

- The canonical report contains:

  - schema, purpose, price-return basis, interval, and gap policy;
  - secret-free reference and aggregate request contracts;
  - the exact point-in-time SPY reference fields;
  - the explicit ARCX/XNYS calendar-applicability assertion;
  - calendar path and SHA-256;
  - CSV path, SHA-256, retained/source row counts, session count, and exact
    session-grid audit.

- [x] **Step 1: Add the failing happy-path test**

  Use a fake requester that returns the exact SPY reference and all `5,534`
  core bars plus post-early-close rows. Assert:

  - only the core grid is written;
  - the session audit is exactly clean;
  - request order and contracts are deterministic;
  - the report binds the final CSV and calendar bytes;
  - neither output contains the fake key;
  - Torch and training modules are not imported.

- [x] **Step 2: Run the focused test and verify the red state**

  ```sh
  $PYTHON tests/python/test_massive.py
  ```

  Expected: failure because `tools.fetch_benchmark` does not exist.

- [x] **Step 3: Implement only the pure contract helpers**

  Keep fixed constants and small local helpers for:

  - reference and aggregate request contracts without credentials;
  - exact reference-field validation;
  - exact calendar path/hash/applicability validation;
  - clean audit validation through `validate_spy_session_audit`.

  Reuse `aggregate_url`, `authorized_url`, `fetch_bars`,
  `scan_regular_bars`, `session_grid_audit`, and staging-only `write_csv`
  without modifying them.

- [x] **Step 4: Implement one-shot publication**

  Before network access, reject an existing or aliased bundle. Freeze the
  calendar, fetch and validate the source, write the private staged CSV with
  the existing formatter, verify its rows and hash, then:

  1. write the canonical report beside the staged CSV;
  2. freeze and revalidate both files and the calendar;
  3. rename the containing directory without replacement;
  4. fsync its parent and revalidate both final file identities and hashes.

  Never expose a successful one-file partial.

- [x] **Step 5: Re-run the focused happy path**

  ```sh
  $PYTHON tests/python/test_massive.py
  ```

  Expected: pass.

---

### Task 2: Harden the Fetch Boundary

**Files:**

- Modify: `tests/python/test_massive.py`
- Modify only as required: `tools/fetch_benchmark.py`

- [x] **Step 1: Add reference and calendar rejection tests**

  Reject every changed identity field, another ticker/type/exchange, a changed
  calendar hash, or a calendar lacking the residual interval. Verify that the
  applicability record is exactly `SPY/ETF/ARCX/XNYS/NYSE Group core session`.

- [x] **Step 2: Add grid and transport rejection tests**

  Reject missing, duplicate, misaligned, or out-of-grid core bars; malformed
  payloads; untrusted/cyclic pagination; and a non-canonical aggregate request.
  Confirm extended-hours rows are filtered rather than bound.

- [x] **Step 3: Add publication and mutation tests**

  Reject existing, symlinked, hard-linked, aliased, or concurrently occupied
  bundles before requests. Mutating the calendar or staged CSV across the
  rename must fail post-publication validation. Simulated rename or fsync
  failure must never return success.

- [x] **Step 4: Run focused and aggregate gates**

  ```sh
  $PYTHON tests/python/test_massive.py
  make -B PYTHON="$PYTHON" check
  ```

  Expected: all checks pass without Torch.

---

### Task 3: Create the Local Checkpoint and Fetch Generated Evidence

**Files:**

- Generated and ignored bundle:
  `data/spy-residual-20260725/{spy.csv,fetch.json}`

- [x] **Step 1: Review only this checkpoint's changes**

  ```sh
  but diff
  ```

- [x] **Step 2: Commit a signed local GitButler checkpoint**

  Create `enkyuan/spy-benchmark-fetch`, stack it directly above
  `enkyuan/spy-residual-protocol`, and commit only the plan, acquisition tool,
  and tests:

  ```sh
  but commit enkyuan/spy-benchmark-fetch -c \
    -m "feat(data): authenticate SPY benchmark bars" \
    --changes <selected-change-ids>
  but move enkyuan/spy-benchmark-fetch enkyuan/spy-residual-protocol
  ```

  Do not push.

- [x] **Step 3: Verify identity and signature**

  Verify the exact commit with the repository's allowed signer and require
  `enkyuan <yuan.enkng@gmail.com>` as both author and committer.

- [x] **Step 4: Fetch the ignored evidence**

  ```sh
  $PYTHON -I -B tools/fetch_benchmark.py \
    data/spy-residual-20260725
  ```

  Revalidate the report and CSV hashes locally. Do not add either generated
  file to GitButler.

- [x] **Step 5: Stop at the next authorization boundary**

  Report the CSV/report hashes and audit. Do not train until the separate
  residual armer binds these bytes in an exclusive attempt. Do not backtest
  because residual predictions are not executable returns.
