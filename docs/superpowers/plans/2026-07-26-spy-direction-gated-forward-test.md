# SPY-Direction-Gated Residual Forward Test Plan

> **For agentic workers:** Use the Karpathy Guidelines, Ponytail, and Writing
> Plans skills. Commit this preregistration before implementing it or fetching
> its forward data.

**Goal:** Test whether one fixed, causal SPY-direction gate turns the existing
five-seed residual Transformer ensemble into a positive out-of-sample
residual forecast.

**Architecture:** Reproduce the authenticated calibration-trained Transformer
states, verify their original fingerprints, and run them unchanged on one
later label-free panel. Average the five predictions, suppress them when the
completed 17-bar SPY window is negative, and apply one fold-1-fitted scale.
Freeze predictions before opening truth, then evaluate exactly one candidate.

**Tech Stack:** Python 3.12 standard library, existing optional PyTorch
runtime, Massive 30-minute adjusted aggregates, existing authentication and
block-bootstrap primitives, GitButler.

## Evidence and Decision

The fold-1 alignment report is:

- `reports/h13-spy-residual-20260725-01-alignment/alignment.json`
- SHA-256:
  `8aee02761357927097a74d13c3d62271fd57f629a4eeb6ac1bebc84a77fd3fa1`
- evidence role: `development-post-hoc-not-forward-clean`

Its authenticated residual source is:

- attempt:
  `experiments/h13-spy-residual-20260725-01-attempt.json`
- attempt SHA-256:
  `0fb90623c90b418dfff93d35dde1bb49024c25d3b2b27b799ce752b8deed9ea3`
- outcome:
  `experiments/h13-spy-residual-20260725-01-outcome.json`
- outcome SHA-256:
  `132c17cd7dde7abcdf625581d6b399d7e9b0011f4a9c8bc6d4d0f4065d3a0488`

For residual truth \(z\), five-seed mean prediction \(p\), and one cell \(g\),
the report retains:

\[
A_g=\sum z p,\qquad B_g=\sum p^2,\qquad C_g=\sum z^2.
\]

The global alignment is negative:

\[
A=-0.0038305157945735372,\qquad A/B=-0.09646697414378595.
\]

The causal SPY-direction cells differ:

\[
A_-=-0.011008419442602798,\qquad A_-/B_-=-0.5027907288686468,
\]

\[
A_+=0.007177903648029261,\qquad
\lambda=A_+/B_+=0.4029492434939931.
\]

The most optimistic same-sample gain is still only:

\[
\frac{A_+^2}{B_+C_{\mathrm{all}}}=0.00067595,
\]

where \(C_{\mathrm{all}}=4.278884963310273\) is global truth energy. The gain
is about `0.0676%` pooled residual \(R^2\). This warrants one cheap forward
test, not a learned gate, new architecture, stock lookup table, or trade.

## Frozen Candidate

Use the existing `panel_transformer` calibration-trained states for seeds:

```text
7, 19, 31, 43, 61
```

Their authenticated calibration fit ledger is:

- `reports/h13-spy-residual-20260725-01/calibration-fits.jsonl`
- SHA-256:
  `93237f9962a64665950252094e9119d1e1a806d7288af0d5c331cd724954203e`

Require these exact state fingerprints in seed order:

```text
fe12e7e77d81eb6761defa27f423739e634d55c608beb30d9125b2631fc1049b
3e2ccdc591baaf9d3b94efdeeddd86b6fed913bf7ea7fce29b6ec38f0c0fc2d2
3f09fcd8c5af5e17019ed4f907c4ade5d2a62c77a958718341efff40bdedb770
723554c420e19286431ce476d9f5ebfe4ed4941c3d5cbbd49c4bdae0b99f11e6
3a5309a9aa318eb2b4093a95a7185610f6cce5a1f15d3ba2856b8a8bdae7132e
```

For stock \(i\), completed as-of bar \(t\), and seed \(s\):

\[
\bar p_{i,t}=\frac{1}{5}\sum_s p_{i,t,s},
\qquad
m_t=\log\left(\frac{C^{SPY}_t}{C^{SPY}_{t-16}}\right),
\]

\[
\hat z_{i,t}
=
0.4029492434939931\,
\mathbf 1[m_t\ge 0]\,
\bar p_{i,t}.
\]

The unchanged target is:

\[
z_{i,t}
=
\log\left(\frac{C^{stock}_{target}}{O^{stock}_{entry}}\right)
-
\log\left(\frac{C^{SPY}_{target}}{O^{SPY}_{entry}}\right)
\]

at horizon `13` with history `17`. A tie is `nonnegative`. The gate reads
only completed SPY closes through `as_of`.

Freeze the source evaluation order:

```text
KRYS, TGT, STM, SSNC, NWL, AAON, GEV, SWKS, BMRN, ACI, HUN
```

Do not test another scale, threshold, lookback, seed weight, stock subset,
sign reversal, soft gate, learned regime state, target, or horizon on this
holdout.

## Forward Boundary

- Keep the original training data, scalers, model configuration, optimizer
  budgets, source implementation, and Torch package unchanged.
- Reproduce the original calibration phase and match every listed state
  fingerprint before using a fitted state on later inputs.
- Never train or refit on a later bar.
- Derive the final inspected calibration target from the authenticated source.
- Require every later `entry` timestamp to be strictly greater than that
  target.
- Require complete regular-session 30-minute grids for all 11 stocks and SPY.
  Missing data fails the attempt; it never changes the universe or dates.
- Generate rows with the existing `session_samples()` rule: require 17
  consecutive observed expected bins through `as_of`, set `entry` to the next
  expected bin, and set `target` to exactly 13 expected bins after `as_of`.
- Derive canonical `(as_of, entry, target)` triples from the bound calendar
  before reading market truth. Discard triples whose `entry` is not strictly
  after the source boundary.
- Define the first target session as the earliest session for which the
  remaining triples cover every expected 30-minute target bin. Exclude an
  earlier partial or empty session rather than retaining it.
- Freeze that full target session and the next 59 calendar sessions. Require
  every expected target bin in all 60 sessions; missing coverage fails rather
  than advancing the end date.
- Select every triple whose `target` is in that fixed set. Reject any stock
  whose ordered triple grid differs from the canonical grid, SPY, or another
  stock.
- Exclusively publish one final candidate ledger containing raw per-seed
  predictions, their arithmetic mean, the causal gate, and the gated
  prediction.
- Permit target-price access only after that ledger is closed, durably
  written, reopened from the same inode, and fully verified.
- Keep generated market data, ledgers, reports, model state, credentials, and
  caches ignored and untracked.

The first 60 sessions are a fixed sample, not a minimum followed by optional
stopping.

## Metrics and Gates

Use zero residual and the unchanged five-seed ensemble as references. For each
comparison, paired squared-error gain is:

\[
G=(z-\hat z_{reference})^2-(z-\hat z_{candidate})^2.
\]

Reuse shared circular target-date blocks of `5`, `10`, and `20` sessions,
`10,000` replicates, and seed `20260725`. The decision uses the predeclared
20-session interval.

The candidate passes only when all conditions hold:

1. pooled raw residual \(R^2\) against zero is positive;
2. the 20-session block lower bound for \(G\) against zero is positive;
3. the 20-session block lower bound for \(G\) against the unchanged ensemble
   is positive;
4. pooled raw residual \(R^2\) remains positive after omitting each stock; and
5. at least 6 of 11 stocks have positive mean squared-error gain against zero.

Report MAE, direction, regime cells, seed dispersion, and other block widths
as descriptive only. They cannot rescue a failed primary gate.

A pass retains one residual candidate for a separate absolute-return
experiment. It does not authorize reconstructed prices, policy selection,
the `$100` backtest, or trading. A failure retains zero residual and stops this
candidate family.

## Literature Boundary

- Giacomini and White permit conditional forecast comparison only with
  information observable at forecast time:
  <https://doi.org/10.1111/j.1468-0262.2006.00718.x>.
- White requires controlling specification search rather than reporting the
  best inspected rule:
  <https://doi.org/10.1111/1468-0262.00152>.
- Clark and West derive the zero-forecast nested-MSE adjustment from the same
  forecast covariance term:
  <https://doi.org/10.1016/S0304-4076(06)00096-0>.
- Nelson and Kim show why overlapping-return inference cannot use naive
  independent standard errors:
  <https://doi.org/10.1111/j.1540-6261.1993.tb04731.x>.

The fixed observable gate is deliberately smaller than a logistic transition,
latent Markov state, or learned mixture of experts. Those models add choices
that this diagnostic does not support.

## Task 1: Freeze a Torch-Free Forward Contract

**Files:**

- Create: `experiments/executable-h13-spy-direction-forward.example.json`
- Create: `tools/spy_residual_forward_contract.py`
- Create: `tests/python/test_spy_residual_forward_contract.py`

- [ ] Add one exact JSON profile containing the candidate, source bindings,
  fingerprints, ordered universe, 60-session rule, metrics, gates, and false
  execution locks.
- [ ] Add a pure expected-value function and exact validator.
- [ ] Reject reordered seeds or stocks, changed fingerprints, changed scale or
  threshold, alternative candidates, fewer or more sessions, optional
  stopping, and any enabled absolute/backtest/trading lock.
- [ ] Verify importing the contract does not import PyTorch or read data.

Run:

```sh
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -I -B \
  tests/python/test_spy_residual_forward_contract.py
```

## Task 2: Share the Causal Gate

**Files:**

- Create: `tools/spy_residual_gate.py`
- Modify: `tools/analyze_spy_residual_shrinkage.py`
- Create: `tests/python/test_spy_residual_gate.py`
- Modify: `tests/python/test_spy_residual_shrinkage.py`

- [ ] Promote the existing causal 17-bar SPY direction arithmetic into one
  pure helper without changing the completed alignment analyzer's behavior.
- [ ] Add one pure fixed-scale gate for ordered prediction vectors.
- [ ] Cover negative, zero, and positive direction; endpoint selection;
  shape preservation; finite values; and later-bar exclusion.
- [ ] Make both the historical analyzer and future runner call the shared
  helper. Do not duplicate gate arithmetic in either orchestrator.

## Task 3: Bind Future Inputs Without Truth

**Files:**

- Create: `tools/arm_spy_residual_forward.py`
- Create: `tools/spy_residual_forward_inputs.py`
- Create: `tests/python/test_spy_residual_forward_inputs.py`

- [ ] Authenticate the original attempt, outcome, alignment report,
  calibration fit ledger, source CSVs, source calendar, new calendar, and new
  Massive bundles.
- [ ] Derive exactly 60 expected target sessions and reject gaps, aliases,
  links, replacements, extra series, and timestamps at or before the source
  boundary.
- [ ] Build future stock/SPY feature windows through each `as_of` without
  decoding an entry or target price.
- [ ] Return one deferred truth reader that cannot run before predictions are
  exclusively published.

## Task 4: Reproduce States and Publish Predictions

**Files:**

- Create: `tools/run_spy_residual_forward.py`
- Create: `tools/spy_residual_forward_runtime.py`
- Create: `tests/python/test_run_spy_residual_forward.py`

- [ ] Keep every file in the historical residual source tree unchanged.
- [ ] Reuse the existing calibration preparation and unchanged
  `ResidualRuntime` fit order so the original combined fingerprints remain
  verifiable.
- [ ] Run the required ridge and MLP fits in their original order before
  retaining the five Transformer fit tokens; do not create a shortcut that
  changes random state or fit provenance.
- [ ] Retain only the five authenticated Transformer fit tokens for future
  inference; do not fit a new candidate.
- [ ] Put the narrow later-input inference adapter in the new forward runtime;
  do not change the historically fingerprinted runtime.
- [ ] Compute raw seeds, their mean, the fixed gate, and gated predictions in
  memory, then publish all four in one final candidate ledger.
- [ ] Bind prediction bytes, model fingerprints, source grids, future feature
  grids, gate inputs, implementation tree, and one-shot run identity.

## Task 5: Open Truth Once and Finalize

**Files:**

- Create: `tools/finalize_spy_residual_forward.py`
- Create: `tests/python/test_finalize_spy_residual_forward.py`

- [ ] Close, freeze, and re-read the final candidate ledger from the same
  inode before calling the deferred truth reader.
- [ ] Reuse pooled residual \(R^2\), paired squared-error block intervals, and
  no-refit leave-one-stock-out arithmetic.
- [ ] Enforce all five gates exactly; descriptive metrics cannot change the
  decision.
- [ ] Keep every absolute-return, policy, backtest, and trading lock false.
- [ ] Publish one ignored, mode-`0600`, single-link terminal report.

## Task 6: Verify and Wait for the Fixed Window

- [ ] Run focused tests and `make -B PYTHON="$PYTHON" check`.
- [ ] Create signed local GitButler checkpoints; do not push or land.
- [ ] Do not fetch or execute the forward attempt until all 60 fixed sessions
  can have completed targets.
- [ ] Execute the committed one-shot runner once, verify every binding and
  lock, and report the result without changing the candidate.
