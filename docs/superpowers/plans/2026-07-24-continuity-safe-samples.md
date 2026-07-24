# Continuity-Safe Sample Construction Plan

> **For implementation:** follow this plan task by task. Keep generated market
> data, reports, models, credentials, and caches untracked.

**Goal:** Prevent a missing Massive aggregate from changing the forecast
horizon or joining nonadjacent bars into one model input.

**Architecture:** Assign every expected exchange-session bar start a stable
ordinal from the frozen calendar. Map observed CSV timestamps onto that grid,
then retain only samples with a complete consecutive input history and observed
entry and target endpoints. Keep the standalone artifact trainer's existing
row-offset behavior unchanged; the new mapping is opt-in until a frozen
universe execution contract binds it.

**Tech stack:** Python 3.12 standard library, PyTorch, Make, GitButler.

---

## Mathematical contract

Let the frozen expected starts be

\[
G=(g_0,\ldots,g_{E-1})
\]

and let \(\phi(j)\) map expected ordinal \(j\) to its observed CSV row when that
aggregate exists. Let \(H\) be the largest raw history required by any
candidate, \(A\) the common alignment horizon, and \(h\le A\) one candidate's
forecast horizon. Candidate target ordinals begin at

\[
j_{\min}=H+A-1.
\]

For target ordinal \(j\), decision ordinal \(a=j-h\), retain a sample exactly
when

\[
\{a-H+1,\ldots,a,a+1,a+h\}\subseteq\operatorname{dom}(\phi).
\]

The raw CSV rows are then

\[
\begin{aligned}
\text{maximum history} &= \phi(a-H+1),\ldots,\phi(a),\\
\text{entry} &= \phi(a+1),\\
\text{target} &= \phi(a+h).
\end{aligned}
\]

Candidate \(i\), whose raw history is \(H_i\le H\), consumes only the suffix
\(\phi(a-H_i+1),\ldots,\phi(a)\).

For the `raw-17`, horizon-13 executable target:

\[
y=\log\left(
\frac{\operatorname{close}_{\phi(a+13)}}
{\operatorname{open}_{\phi(a+1)}}
\right).
\]

Missing bins between entry and target do not change \(a+13\) and therefore do
not invalidate the endpoint label. They do prevent a complete mark-to-market
path, so `$100` replay remains locked until the backtester defines that
separate policy.

The implementation must be \(O(D+E+O)\): calendar days, expected bins, and
observed rows. It must not search one full history for every candidate.

## Non-goals

- Do not fill, interpolate, or synthesize OHLCV.
- Do not change Artifact V1 or standalone `prepare_data`.
- Do not weaken the existing three-stock panel's identical-grid contract.
- Do not choose training stocks, hyperparameters, or trading policy from these
  data.
- Do not run the reserved test or a `$100` replay.
- Do not let the current observed-row backtester accept a calendar-indexed
  prediction ledger.

---

## Task 1: Centralize the frozen expected grid

**Files:**

- Modify: `tools/session_calendar.py`
- Modify: `tools/fetch_massive.py`
- Modify: `tests/python/test_massive.py`

- [ ] Add one iterator that yields expected regular-session starts for an
  inclusive date range and interval. Index closed dates and early closes once;
  do not call the tuple-scanning `SessionCalendar.session()` for every bin.
- [ ] Test normal days, the November 2024 DST transition, an early close,
  calendar bounds, and a nondivisible interval remainder.
- [ ] Reuse the iterator in `session_grid_audit` without changing schema-4
  output. Build one expected-start map, then validate observed starts by
  membership so the audit is \(O(D+E+O)\).
- [ ] Run `tests/python/test_massive.py`.

## Task 2: Build a pure continuity-safe sample index

**Files:**

- Create: `tools/session_samples.py`
- Create: `tests/python/test_session_samples.py`

- [ ] Return compact raw CSV rows for decision, entry, and target plus the number of
  target opportunities beginning at \(H+A-1\). Derive each candidate's feature
  start as a suffix of the proven complete maximum history.
- [ ] Reject duplicate, out-of-grid, and noncanonical observed timestamps.
- [ ] Use one running observed-history length so construction is linear.
- [ ] Test a missing input bin, entry bin, target bin, and an irrelevant
  holding-period bin independently.
- [ ] Test stationary-feature history (`seq_len + 1`) and the DST/early-close
  grid.
- [ ] Prove a missing row never shifts the target to the next observed row.

## Task 3: Add opt-in indexed windows

**Files:**

- Modify: `tools/train.py`
- Modify: `tools/experiment.py`
- Modify: `tools/panel_contract.py`
- Modify: `tests/python/test_training.py`
- Modify: `tests/python/test_experiment.py`

- [ ] Let `prepare_rows` consume explicit sample rows only through a keyword
  argument; `None` must retain byte-for-byte legacy sample selection.
- [ ] Store feature-start, decision, entry, and target raw CSV rows in
  `Windows`; use the explicit entry for `open[a+1]`, the explicit target for
  `close[a+h]`, and both explicit times in prediction ledgers.
- [ ] Define the feature coordinate conversion once: for raw lookback
  \(L_i\), `feature_start = as_of_raw - (seq_len + L_i) + 1`. This is also the
  starting index in the transformed feature tensor because `stationary-v1`
  feature row zero represents raw row one.
- [ ] Fit feature scalers on the union of rows touched by retained training
  windows, weighting every unique feature row once, and fit target scalers on
  retained training labels only. Merge already-sorted intervals or use a
  difference mask in \(O(O+S)\), never expand \(S\) windows of length \(H\).
- [ ] Derive split boundaries from explicit target rows rather than
  `target_offset + count`.
- [ ] At every split boundary, prove the last retained training target ordinal
  is no later than the first following split's decision ordinal. Do not rely on
  a count-based embargo alone.
- [ ] Add the sample source to frozen implementation provenance.
- [ ] Prove legacy artifact output and legacy panel mismatch behavior are
  unchanged.

## Task 4: Bind a separate universe execution

**Files:**

- Create a frozen universe-run contract and focused tests.
- Modify the experiment entry point only through that contract.

- [ ] Bind the selection manifest, fetch report, session calendar, sweep,
  ordered series, source tree, and output paths before training.
- [ ] Include `tools/session_samples.py` and `tools/session_calendar.py` in the
  source-tree binding, and recompute the schema-4 session audit from each
  frozen CSV/calendar pair instead of trusting only the report hash.
- [ ] Split all stocks by common target-time intervals from the calendar while
  permitting unequal observed grids and sample counts.
- [ ] Record opportunities, retained samples, exclusions, elapsed horizon, and
  exact first/last target timestamps per stock and split.
- [ ] Keep the old panel execution unchanged.

## Verification

Run focused tests after each task, then:

```sh
make -B \
  PYTHON=/Users/Enkang.Yuan1/.cache/codex-runtimes/\
codex-primary-runtime/dependencies/python/bin/python3 check
```

Before any calibration, independently review:

- expected-grid and sample-index math;
- scaler and split leakage;
- frozen-input provenance;
- legacy compatibility;
- exact exclusion accounting.
