# Massive Rate-Limit Recovery and Universe Calibration Plan

> **For Codex:** Execute this plan with `subagent-driven-development`. Keep the
> failed attempt and every generated artifact ignored. Do not push or land.

**Goal:** Pace the manifest-driven Massive downloader to the account's declared
five-call-per-minute limit, then make one new fresh-directory attempt at the
unchanged 11-stock, 429-fit calibration.

**Architecture:** Add one shared monotonic request-attempt gate to the Massive
transport. `request_json` calls it immediately before every physical
`urlopen`, including 429 retries. The universe fetcher reuses that gate across
reference lookups and aggregate pages, so one budget covers the complete
universe. The model, target, universe, thresholds, policy grid, and analyzer
remain unchanged.

**Tech stack:** Python standard library, existing Massive/fetch tests,
GitButler, existing offline PyTorch environment.

---

## Failure Evidence and Locked Recovery Contract

- The first live directory was `reports/h13-universe.Ms6v2m`.
- Its single fetch process failed on 2026-07-23 with HTTP 429 after about
  150 seconds.
- It produced only ignored partial AAPL and JPM CSVs. It produced no final
  fetch report, training output, policy, replay, analysis, tracked diff, or
  credential leak.
- Massive currently advertises **5 API calls per minute** for Stocks Basic:
  <https://massive.com/pricing?product=stocks>.
- Massive custom bars paginate after at most 50,000 base aggregates:
  <https://massive.com/docs/rest/stocks/aggregates/custom-bars>.
- The recovery uses a conservative request-start interval:

```text
interval_seconds = 61 / requests_per_minute
```

  At five calls per minute this is 12.2 seconds. The extra second avoids an
  exact rolling-window boundary.
- `requests_per_minute == 0` means no client-side pacing. The live recovery
  must explicitly pass `5`; tests may pass `0`.
- The accepted rate domain is an integer, but not a boolean, from 0 through 60
  inclusive. Paid unlimited tiers use 0; positive values above 60 are
  unnecessary because 0 already preserves unpaced behavior.
- Pacing is local transport behavior. It must not appear in URLs, logs,
  credentials, response contracts, model inputs, or generated CSV content.
- Every physical HTTP attempt consumes its slot even if `urlopen` raises. A
  429 retry must satisfy both the existing `Retry-After` sleep and the shared
  monotonic gate before the next attempt.
- Do not change `request_json` retry counts or its `Retry-After` handling in
  this checkpoint. Backoff and the cross-request gate remain separate,
  cumulative protections.
- The second live attempt uses one new ignored `reports/h13-universe.*`
  directory. It never reuses or deletes the failed directory.
- The second live attempt is the only recovery attempt authorized by this
  plan. Provider/data failure stops again.
- The universe remains AAPL, JPM, XOM, JNJ, PG, AMZN, CAT, NEE, LIN, PLD,
  and GOOGL. No ticker may be replaced.
- The target remains executable horizon-13 `raw-17`; the five seeds, two folds,
  429 fits, forecast gates, policy grid, costs, and analyzer are unchanged.
- No reserved historical test label may be opened.

## Plan Checkpoint

After independent engineering and experiment-integrity reviews approve this
document, use `but diff` and commit only this plan on
`enkyuan/massive-rate-limit-plan`, stacked directly above
`enkyuan/universe-expansion`:

```text
docs(plan): pace Massive recovery attempt
```

Verify the enkyuan author, committer, and ED25519 signature. Do not push.

## Task 1: Add a Strict Request-Pacing Primitive

**Files:**

- Modify: `tools/fetch_massive.py`
- Modify: `tools/fetch_universe.py`
- Modify: `tests/python/test_massive.py`

### Step 1: Add failing unit tests

Add focused tests for a small public helper:

```python
request_gate(
    requests_per_minute,
    *,
    clock=None,
    sleeper=None,
) -> Callable[[], None]
```

`None` resolves `time.monotonic` and `time.sleep` at helper-call time, not
definition time. Use a fake monotonic clock whose sleeper advances time.
Require:

- the first gate call returns without sleeping;
- admitted starts at five calls/minute are `math.isclose` to `0.0`, `12.2`,
  `24.4`, and `36.6`;
- time spent after a gate call reduces, rather than duplicates, the next sleep;
- a gate slot remains consumed when the following transport call raises;
- the zero-rate gate is a no-op that never reads the clock or sleeps;
- booleans, negative values, non-integers, and integers above 60 are rejected;
- no URL, response, or credential is accepted or rendered by the helper.

Extend `request_json` with an optional keyword-only gate and add a deterministic
physical-attempt test:

```text
HTTP 429 -> successful retry -> next logical page
```

Every `urlopen` attempt must call the same gate. With a fake clock, assert the
three transport starts are at least 12.2 seconds apart. Preserve the existing
retry-count and `Retry-After` assertions.

Run:

```zsh
PRIMARY_PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3
"$PRIMARY_PYTHON" tests/python/test_massive.py
```

Expected: fail because `request_gate` does not exist.

### Step 2: Implement the minimum gate

Keep one closure with one `next_start` value. Validate the rate once. For each
gate call:

1. read the monotonic clock;
2. sleep only for positive remaining delay;
3. read the clock again after sleeping;
4. set the next start to the actual start plus `61 / rate`;
The gate has no URL or return value. Do not add a class, thread, queue,
persistence layer, or dependency. Rate zero returns a no-op closure without
reading the clock.

In `request_json`, call the optional gate immediately before each `urlopen`
attempt. Do not call it before local JSON validation or during response parsing.

### Step 3: Re-run the focused test

Run:

```zsh
"$PRIMARY_PYTHON" tests/python/test_massive.py
```

Expected: pass.

## Task 2: Wire One Budget Across Reference and Pagination Requests

**Files:**

- Modify: `tools/fetch_universe.py`
- Modify: `tools/fetch_massive.py`
- Modify: `tests/python/test_massive.py`
- Modify: `docs/training.md`

### Step 1: Add failing integration tests

Extend the offline universe fixture to call `fetch_universe` with
`requests_per_minute=5`, a fake monotonic clock/sleeper, and the existing fake
requester. Add `clock=None` and `sleeper=None` as keyword-only test seams on
`fetch_universe`; they are passed only to `request_gate`. Require:

- reference AAPL, aggregate AAPL, reference MSFT, and aggregate MSFT share one
  start sequence;
- pagination consumes additional slots from that same sequence;
- rate, manifest, and target validation happen before credential lookup and
  the first request;
- the report schema and URL sanitization remain byte-for-byte unchanged;
- no key reaches the sleeper, clock, report, or error text.

Add CLI parsing checks:

- default is `0`;
- `--requests-per-minute 5` parses as integer 5;
- values outside the exact 0-through-60 integer domain fail before credential
  lookup or any request.

Expected: the integration test fails before wiring.

### Step 2: Wire the decorator once

Change the `fetch_universe` requester default to `None` so production and test
paths are explicit. Add `requests_per_minute: int = 0`, `clock=None`, and
`sleeper=None`.

After local rate, manifest, and target checks but before credential lookup,
construct one gate. Then:

- production (`requester is None`) uses one callable that invokes
  `request_json(url, before_request=gate)`, allowing every internal retry to
  consume the shared budget;
- an injected offline requester is treated as one physical attempt and is
  preceded by the same gate;
- at rate zero, preserve the existing unpaced callable directly.

Pass the resulting single requester to both `_reference` and `fetch_bars`.

Add:

```text
--requests-per-minute INTEGER
```

to the CLI, defaulting to zero. The CLI passes it into `fetch_universe`.

Document the flag in the Massive training section. State that `5` matches the
current Stocks Basic limit and that paid unlimited tiers may use `0`. Do not
describe pacing as proof that a fetch will succeed.

### Step 3: Run focused and aggregate gates

Run:

```zsh
"$PRIMARY_PYTHON" tests/python/test_massive.py
make -B PYTHON="$PRIMARY_PYTHON" check
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch python \
  tests/python/test_experiment.py
```

Expected: Massive tests, all standard C/Python tests, and the experiment test
pass. The known missing-NumPy warning is nonfatal.

### Step 4: Create a signed local checkpoint

Use `but diff`, commit only these four files on
`enkyuan/massive-rate-limit`, stacked directly above
`enkyuan/massive-rate-limit-plan`:

```text
fix(data): pace Massive universe requests
```

Verify author, committer, and ED25519 signature for
`enkyuan <yuan.enkng@gmail.com>`. Do not push.

## Task 3: Independently Review the Recovery Boundary

Review the Task 1–2 diff read-only. Require evidence that:

- one monotonic budget covers reference and every aggregate page;
- `request_json` gates every physical `urlopen`, including a 429 retry;
- the first request is immediate and later starts are at least `61 / rate`
  apart;
- request work time is accounted for;
- failures consume a slot;
- invalid rates fail before network access;
- rate zero preserves the existing caller contract;
- secrets and URLs are never logged by pacing;
- no fetch/report/model semantics changed;
- all tests remain deterministic and perform no network I/O.

Any Important finding must be fixed by amending the same unpublished checkpoint
and re-reviewed. File length alone is not a finding.

## Task 4: Make One Fresh Recovery Fetch

Create exactly one new ignored directory:

```zsh
RUN_DIR=$(mktemp -d reports/h13-universe-recovery.XXXXXX)
CSV_DIR="$RUN_DIR/csv"
FETCH_REPORT="$RUN_DIR/fetch-report.json"
```

Before creating it, repeat the original ignore-boundary checks for `.env`,
`data/`, `models/`, `reports/`, and Python bytecode. Preserve
`reports/h13-universe.Ms6v2m` byte-for-byte.

Run one escalated process:

```zsh
"$PRIMARY_PYTHON" tools/fetch_universe.py \
  universes/liquid-common-11.json "$CSV_DIR" "$FETCH_REPORT" \
  --requests-per-minute 5
```

Capture the fetch exit without rerunning. Whether it succeeds or fails, always
run the non-rendering credential scan from the universe-expansion plan against
the recovery directory; it may print only `credential scan: pass` or
`credential scan: fail`. Stop if either the fetch or scan fails.

Expect about 11–13 minutes if the account is on the five-call tier. On success,
require:

- the report exists and is canonical;
- exactly 11 CSVs exist in manifest order;
- every reference identity, aggregate request, row count, session count, and
  SHA-256 passes the existing validator;
- the real-key no-leak scan prints only `credential scan: pass`.

On failure, preserve the recovery directory unmodified after the scan and stop
with no training or analyzer call.

## Task 5: Run the Unchanged Calibration Once

Only after a valid fetch report:

1. Build `SERIES_ARGS` from the manifest/fetch report in declared order.
2. Run the existing calibration-only experiment on CPU with `--max-runs 429`.
3. Do not pass `--predictions`, `--policy`, or any test authorization.
4. Select Transformer, MLP, and linear policies with the existing fixed grid,
   initial cash 100, spread 1 bp, slippage 1 bp, and fee 0.
5. Replay all three policies.
6. Invoke `tools/analyze_universe.py` exactly once.
7. Accept only exit 0/pass or exit 3/gate-failure; any other exit is an
   integrity failure and must not be rerun.

The recovery permits exactly:

- one experiment process;
- one policy-selection process and one replay process for each declared model,
  in Transformer, MLP, linear order;
- one analyzer process.

Any nonzero experiment, selector, or replay exit; missing output; run count
other than 429; or contract mismatch terminates the recovery. Do not rerun a
failed stage in this or another directory. After any output is observed, do not
change the universe, config, seeds, folds, gates, thresholds, policy grid,
costs, or command order.

Use the exact commands and postconditions from Task 4 of
`docs/superpowers/plans/2026-07-23-universe-expansion.md`, substituting only the
new `RUN_DIR`.

## Task 6: Record and Checkpoint Valid Evidence

Only after analyzer exit 0 or 3:

- append the exact validated evidence fields required by the universe-expansion
  plan to `docs/training.md`;
- explicitly label policy results calibration resubstitution;
- state that the first attempt was rate-limited and the recovery used five-call
  pacing;
- record the sanitized fetch command, pacing-code commit SHA, fetch start/end
  times, fetch exit, recovery-directory basename, and fetch-report SHA-256;
- retain the manual-universe selection/survivorship-bias warning;
- state that no reserved historical test labels were opened;
- never commit generated data, reports, ledgers, policies, models, credentials,
  or bytecode.

Run:

```zsh
"$PRIMARY_PYTHON" tests/python/test_massive.py
"$PRIMARY_PYTHON" tests/python/test_universe_analysis.py
/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch python \
  tests/python/test_experiment.py
make -B PYTHON="$PRIMARY_PYTHON" check
make PYTHON="/Users/Enkang.Yuan1/.local/bin/uv run --offline --with torch python" \
  check-training
```

Create a signed, docs-only local checkpoint on
`enkyuan/universe-recovery-evidence`, stacked directly above
`enkyuan/massive-rate-limit`:

```text
docs(training): record paced universe calibration
```

Do not push or land.

## Final Review Checklist

- [ ] The failed directory remains ignored and unmodified.
- [ ] The recovery used one fresh directory and one fetch process.
- [ ] Five-call pacing was explicit and tested.
- [ ] Every physical HTTP retry used the same gate as later pages.
- [ ] No credential or authorized URL was printed or committed.
- [ ] Exactly 11 manifest tickers were fetched; none was substituted.
- [ ] Exactly 429 calibration fits ran, or training did not start.
- [ ] Every experiment/selector/replay stage ran at most once.
- [ ] No historical test prediction or authorization was emitted.
- [ ] Analyzer ran exactly once, or did not run.
- [ ] Exit/status and all three forecast gates agree.
- [ ] `$100` policy figures remain descriptive calibration resubstitution.
- [ ] Generated artifacts remain ignored.
- [ ] Tracked checkpoints are signed by enkyuan and remain local.
