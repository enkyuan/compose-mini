# SPY Grid Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `subagent-driven-development` or execute each task inline with its stated
> checks. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pair each stock's retained training and evaluation rows with SPY
rows that represent the exact same causal timestamp triples.

**Architecture:** Add one Torch-free input helper between exchange-calendar
sampling and `spy_residual_data()`. The caller derives canonical SPY
`SessionSamples`; the helper selects rows by `as_of_ordinal`, proves exact
`(as_of, entry, target)` timestamp equality, and preserves the stock's split
counts. File hashes, calendar bindings, and experiment policy remain the
later armer's responsibility.

**Tech Stack:** Python 3.12+, existing session-sample and context-grid
primitives, GitButler.

## Why This Boundary Exists

The residual target is valid only when both returns use the same execution
interval:

\[
z_{i,t}
=
\log\!\left(\frac{C_{i,\mathrm{target}}}
                  {O_{i,\mathrm{entry}}}\right)
-
\log\!\left(\frac{C_{\mathrm{SPY},\mathrm{target}}}
                  {O_{\mathrm{SPY},\mathrm{entry}}}\right).
\]

Equal session ordinals alone do not prove equal source timestamps, and equal
array indices are wrong for sparse series. The helper therefore joins on the
calendar ordinal and then verifies the three resolved timestamps byte for
byte.

A local 2026-07-25 audit found that `data/spy-30m.csv` contains all 5,534
required bars for 2024-11-01 through 2026-07-21 plus five bars after early
closes. That file is not valid input to this boundary. Do not compact or
offset it in place; re-fetch SPY through the existing calendar-aware path
before a real run.

The authenticated coverage manifest also replaces unavailable `ENLC` with
`AAON`. Use that overlay, not the earlier selection manifest, when binding
dataset paths.

## Global Constraints

- Reuse `SessionSamples`, `PackedRows`, `timestamp_rows()`, and
  `timestamp_grid_sha256()`; do not create a parallel row or hash format.
- The helper accepts only SPY samples already derived by `session_samples()`.
  It is an aligner, not a calendar repair function.
- Preserve the stock's ordered rows and split counts exactly.
- Reject missing or duplicate SPY ordinals and any timestamp-triple mismatch.
- Keep the sealed context and universe runners unchanged.
- Do not read calibration labels, train models, expand the universe, or run
  the `$100` backtest in this checkpoint.
- Keep generated data, reports, attempts, models, credentials, and caches
  untracked.

---

### Task 1: Pure Stock-to-SPY Row Alignment

**Files:**

- Create: `tools/relative_context_inputs.py`
- Create: `tests/python/test_relative_context_inputs.py`

**Interface:**

```python
def align_spy_rows(
    stock_timestamps: Sequence[str],
    stock: PackedRows,
    spy_timestamps: Sequence[str],
    spy: SessionSamples,
) -> PackedRows:
    ...
```

- [x] **Step 1: Write the failing contract tests**

Cover these cases with small synthetic timestamp grids:

- Complete stock and SPY grids resolve to equal timestamp triples.
- A stock with missing earlier history maps to different SPY row indices
  while retaining equal timestamps and `as_of_ordinal` values.
- A missing or duplicate SPY ordinal fails.
- A shifted timestamp fails exact triple comparison. A forged nonadjacent row
  fails row validation; the canonical `session_samples()` path separately
  rejects an off-grid extra cell.
- Non-contract top-level values, list-backed payloads, malformed rows, and
  Boolean or negative row fields fail with `ValueError`.
- `PackedRows.counts` must contain exactly a nonempty training count and a
  nonnegative evaluation count whose sum equals the row count.
- The returned SPY counts equal the stock counts exactly.
- Direct timestamp triples and hashes computed from each stock/SPY split are
  equal.
- `session_samples()` rejects an off-grid post-early-close bar, documenting
  the required caller-side canonicalization.

- [x] **Step 2: Run the test and verify the red state**

Run:

```sh
$PYTHON tests/python/test_relative_context_inputs.py
```

Expected: failure because `tools.relative_context_inputs` does not exist.

- [x] **Step 3: Implement the minimal helper**

Validate one tuple-backed, nonempty `PackedRows` value with exactly two
counts: positive training, nonnegative evaluation, and a sum equal to the row
count. Validate the complete tuple-backed `SessionSamples` through
`timestamp_rows()`, including unselected rows, before building one ordinal
lookup. Select the SPY row for each ordered stock row and resolve both sides
through `timestamp_rows()`. Return
`PackedRows(selected_spy_rows, stock.counts)` only when the resolved triples
are identical.

The function intentionally does not return another contract object. The later
armer can hash each retained split with the existing
`timestamp_grid_sha256()` after freezing the stock CSV, SPY CSV, calendar, and
source attempt.

- [x] **Step 4: Run focused checks**

Run:

```sh
$PYTHON tests/python/test_relative_context_inputs.py
$PYTHON tests/python/test_relative_context.py
$PYTHON tests/python/test_context_diagnostic_inputs.py
```

Expected: all pass.

- [x] **Step 5: Run the aggregate gate**

Run:

```sh
make -B PYTHON=$PYTHON check
```

Expected: all mandatory C and Python suites pass.

- [ ] **Step 6: Create signed local checkpoints**

Commit this plan alone as:

```text
docs(training): plan SPY grid alignment
```

Then commit the helper and its test as:

```text
feat(training): align SPY context rows
```

Stack both checkpoints above `enkyuan/spy-residual-inputs`. Verify the final
author, committer, and ED25519 signatures. Do not push.

## Next Checkpoint Boundary

The following checkpoint must define a new additive calibration profile; it
must not extend either sealed runner. Before reading any calibration truth, it
must:

1. authenticate the stock source attempt without rewriting its existing
   terminal integrity-failure artifact;
2. freeze a newly calendar-clean SPY CSV and its audit;
3. bind every aligned training/evaluation grid and training-only scaler;
4. predeclare history `17`, horizon `13`, seeds `7/19/31/43/61`, transferred
   update budgets, and the residual-only comparison metrics; and
5. keep trading, universe expansion, and the `$100` backtest disabled.

The residual experiment becomes a trading candidate only after a separately
frozen SPY forecast composes an absolute executable return:

\[
\widehat r_{i,t}
=
\widehat r_{\mathrm{SPY},t}
+
\widehat z_{i,t}.
\]
