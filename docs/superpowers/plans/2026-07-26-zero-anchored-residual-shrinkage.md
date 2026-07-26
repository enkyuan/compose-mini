# Zero-Anchored Residual Shrinkage Implementation Plan

> **For agentic workers:** Use the Karpathy Guidelines, Ponytail, and
> Writing Plans skills while implementing this plan.

**Goal:** Determine whether the panel Transformer's residual forecasts contain
useful covariance but excessive magnitude.

**Architecture:** Add one post-hoc analyzer around the completed authenticated
residual run. It averages the existing five Transformer seeds, fits one
zero-intercept scale on `fold-1`, publishes and freezes that fit, then opens
`calibration` and applies the scale unchanged. It does not retrain, alter an
authenticated residual source file, reconstruct absolute returns, or backtest.

**Files:**

- Create: `tools/analyze_spy_residual_shrinkage.py`
- Create: `tests/python/test_spy_residual_shrinkage.py`
- Modify: `.gitignore` only if the exact generated report names are not
  already ignored

## Frozen Contract

- Attempt:
  `experiments/h13-spy-residual-20260725-01-attempt.json`
- Attempt SHA-256:
  `0fb90623c90b418dfff93d35dde1bb49024c25d3b2b27b799ce752b8deed9ea3`
- Outcome:
  `experiments/h13-spy-residual-20260725-01-outcome.json`
- Outcome SHA-256:
  `132c17cd7dde7abcdf625581d6b399d7e9b0011f4a9c8bc6d4d0f4065d3a0488`
- Source implementation:
  `0bc33956ddbff9f706d1341b77f01e71a0b07496`
- Candidate:
  arithmetic mean of the existing five `panel_transformer` seed predictions
- Fit phase: `fold-1`
- Evaluation phase: `calibration`
- One global scale; no intercept, per-stock scale, per-seed scale, model
  switching, threshold search, or calibration refit

For fold-1 truth \(z\) and ensembled predictions \(p\):

\[
A=\sum zp,\qquad B=\sum p^2,\qquad
\lambda=
\begin{cases}
0,&B=0,\\
\operatorname{clip}_{[0,1]}(A/B),&B>0.
\end{cases}
\]

The shrunk forecast is \(\lambda p\). Equal observation weighting and stable
summation match the existing pooled residual-\(R^2\) contract. A zero
denominator selects the conservative zero forecast and is reported explicitly.

Because `calibration` was already inspected, the report is
`development-post-hoc-not-forward-clean`. A positive result may justify only a
new preregistered residual holdout. Every absolute-price, trading, universe
expansion, and backtest lock remains false.

## Literature Basis

- Campbell and Thompson show that economically motivated restrictions can
  stabilize noisy return forecasts:
  <https://doi.org/10.1093/rfs/hhm055>.
- Clark and West analyze MSE-minimizing combinations of restricted and
  unrestricted forecasts under weak incremental information:
  <https://www.federalreserve.gov/pubs/feds/2007/200743/index.html>.
- Gu, Kelly, and Xiu use disjoint tuning and testing periods and strong
  regularized baselines for nonlinear return prediction:
  <https://doi.org/10.1093/rfs/hhaa009>.

The scalar fit combines the unrestricted Transformer forecast with the
restricted zero forecast. Clipping prevents sign reversal or amplification
from being selected after seeing the tuning phase.

## Task 1: Implement and Test the Pure Math

- [ ] Keep the existing failing import as the red test.
- [ ] Implement `zero_anchored_scale`, `scale_predictions`, and `pooled_r2`
  in the analyzer.
- [ ] Require identical ordered stock mappings, exact nonempty lengths,
  finite values, valid `ResidualTruthRow` instances, and a duplicate-free
  excluded subset.
- [ ] Test the exact clipped minimizer, zero denominator, clipping at both
  boundaries, common-scaling invariance, duplication invariance,
  leave-one-stock-out fitting, shape preservation, and invalid inputs.
- [ ] Return the fit components needed for the report without recomputing
  row-level sums.

Run:

```sh
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -I -B \
  tests/python/test_spy_residual_shrinkage.py
```

Expected: `SPY residual shrinkage tests passed`.

## Task 2: Authenticate Inputs and Enforce the Phase Boundary

- [ ] Freeze the attempt, outcome, phase artifacts, and analyzer source.
- [ ] Validate both residual ledgers, receipts, run identities, truth-access
  bindings, and terminal outcome bindings with existing hardened helpers.
- [ ] Authenticate `fold-1` first and recover the existing five-seed ensemble
  with `_predictions`.
- [ ] Regenerate only fold-1 residual truth from `_collect_inputs` and
  `ResidualTruthRow`; do not persist raw truth.
- [ ] Verify the source lease before and after every truth read.
- [ ] Compute and exclusively publish `shrinkage-fit.json`, then freeze and
  re-read the same inode.
- [ ] Only after the fit file is frozen, authenticate and read calibration
  truth.
- [ ] Reject changed bytes, aliases, links, unexpected directory entries,
  replaced directories, forged bindings, and post-freeze mutations.
- [ ] Test explicitly that calibration truth cannot be opened before the fit
  file is durably published.

The fit report records:

- scale, unclipped scale, numerator, denominator, and observation count;
- fold-1 pooled raw residual \(R^2\);
- all 11 leave-one-stock-out scales, their deltas, and range;
- attempt, outcome, receipt, ledger, truth-access, and analyzer bindings; and
- `parameter_stability_only: true`.

## Task 3: Evaluate the Frozen Candidate

- [ ] Apply the bit-identical frozen scale to every calibration prediction.
- [ ] Report pooled raw and centered residual \(R^2\).
- [ ] Report paired MSE and MAE gains versus zero and the unshrunk
  Transformer; retain ridge and MLP as secondary comparisons.
- [ ] Use the existing 20-trading-day circular block bootstrap for the
  primary paired MSE gain versus zero.
- [ ] Report per-stock MSE gains and the count above zero.
- [ ] Compute leave-one-stock-out calibration \(R^2\) from retained
  per-stock sums without refitting or rebuilding a different common-date
  grid.
- [ ] Copy RankIC unchanged when scale is positive; report it unavailable
  when scale is zero. Shrinkage cannot improve direction accuracy or rank.
- [ ] Scale seed dispersion by the frozen nonnegative scale.

The adaptive decision is true only if:

1. `0 < scale <= 1`;
2. calibration pooled raw residual \(R^2 > 0\);
3. the 20-day bootstrap lower bound for paired MSE gain versus zero is
   positive;
4. every calibration leave-one-stock-out residual \(R^2\) is positive; and
5. at least 6 of 11 stocks have positive individual MSE gain.

The decision field is named
`later_residual_holdout_preregistration_warranted`; it does not authorize a
trade or backtest.

## Task 4: Publish, Verify, and Checkpoint

- [ ] Exclusively publish `shrinkage.json`, binding the frozen fit, attempt,
  outcome, analyzer source, and all authenticated inputs.
- [ ] Re-read the same inode and verify every live lease before success.
- [ ] Keep both reports ignored, mode `0600`, single-link, and uncommitted.
- [ ] Run focused tests, exact isolated CLI tests, and the aggregate gate.
- [ ] Create one signed local GitButler checkpoint stacked above
  `enkyuan/spy-residual-execution`.
- [ ] Do not push or land.
- [ ] Run the committed analyzer exactly once and report the decision.

Aggregate gate:

```sh
make -B PYTHON="$PYTHON" check
```

Checkpoint:

```text
feat(training): analyze zero-anchored residual shrinkage
```

## Stop Condition

If the decision is false, retain the zero-residual forecast and stop tuning
this residual model family. If true, write a separate preregistered plan for a
genuinely later residual-only holdout. Neither result permits absolute-return
reconstruction, a `$100` simulation, or trading.
