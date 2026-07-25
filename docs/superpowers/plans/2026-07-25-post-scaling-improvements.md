# Post-Scaling Forecast Improvement Plan

> **Purpose:** Pre-register the smallest evidence-driven experiments that may
> follow the active universe-scaling and forward-portfolio work. This file does
> not authorize changing, stopping, or interpreting a nonterminal attempt.

**Goal:** Improve chronology-safe horizon-13 development forecasts and costed
`$100` development returns without adding candidates after inspecting their
outcomes.

**Architecture:** Let the terminal scaling result screen whether breadth merits
a separate forward evaluation. Then test longer raw context against simple
controls under one shared update budget per model and seed. Calibrate the
existing five-seed ensemble before changing its output head. Add patching,
normalization, or probabilistic outputs only when the preceding diagnostic
identifies the limitation each technique addresses.

---

## Gate and invariants

Do not begin these experiments until:

1. the active scaling finalizer atomically publishes a terminal outcome;
2. a passing outcome binds the exact forward refits and prediction ledger;
3. the existing 13-trial portfolio family and its attempt are frozen without
   expansion; and
4. every attempt binds its candidate family and absent outcomes before its
   runner reads target labels.

Keep the current target, costs, chronological phases, five seeds, point-in-time
universe manifests, stock-balanced loss, and protected-data boundary fixed.
Generated data, reports, models, credentials, and caches remain untracked.

More epochs or model width are not default candidates. They are justified only
by a training diagnostic, such as continued validation improvement at the last
frozen update checkpoint without widening the train/validation gap.

## Checkpoint 1: Decide whether more stocks help

Use the terminal fixed-update core `11/22/33/55` and unseen-transfer
`11/22/33/44` curves only as same-phase development diagnostics. Their
checkpoints were selected on the phases they predict, so they cannot authorize
an expanded universe.

If the result passes, arm a distinct breadth-forward contract and attempt for
every exact registered comparison cell. Do not parameterize, widen, or reuse
the fixed cohort-55 portfolio contract for this family. For each candidate and
seed:

1. use only its fold-0-selected checkpoint to reinitialize and fit the fold-1
   training range for the exact target-phase update budget;
2. predict fold-1 once without reading fold-1 validation or outcome labels;
3. freeze its fold-1-selected checkpoint only after fold-1 prediction; and
4. reinitialize and fit the calibration training range, then predict
   calibration once without reading calibration labels.

For every aligned stock and target timestamp in that forward ledger, compute

\[
\Delta_{i,t}
=
|y_{i,t}-\hat y^{\mathrm{global}}_{i,t}|
-
|y_{i,t}-\hat y^{\mathrm{local}}_{i,t}|.
\]

Resample complete trading-day vectors so simultaneous stocks and overlapping
13-bar targets stay coupled. Reuse the existing circular moving-day bootstrap
at block lengths `5`, `10`, and `20`, with `10_000` replicates and seed
`20_260_725`. The applicable upper bound is the maximum 97.5th percentile over
those block lengths.

Expand the Massive universe only when all of these deterministic gates hold in
fold-1 and calibration separately:

- the largest-breadth unconditioned panel has an upper bound below zero for
  mean unseen-stock \(\Delta\) against the local Transformer;
- stock-macro return MAE is strictly decreasing over core breadths
  `11/22/33/55` and unseen-transfer breadths `11/22/33/44`;
- the largest-breadth unseen-transfer mean \(\Delta\) is nonpositive in each of
  the five manifest-bound liquidity strata;
- the largest-breadth unconditioned global Transformer passes the
  cross-sectional loss gate below; and
- no conditioned model is counted as unseen-stock transfer because its ticker
  identity is unavailable for a new stock.

If the curve plateaus or worsens, keep the smaller manifest. More correlated
rows are not independent evidence. Any later universe requires a new
point-in-time manifest and a fresh attempt; never append tickers to an existing
attempt.

Separate market timing from stock selection. Define an execution group as
\(g=(\text{phase},\text{as-of},\text{entry time},\text{target time})\). Before
revealing outcomes, bind each group's opportunity identities and require
exactly one opportunity for every series in the candidate's digest-bound
manifest. Reject the attempt rather than drop a missing series after outcomes
are available.

For each group, report the common-signal control
\(\hat y^{\mathrm{common}}_{i,g}=\bar{\hat y}_g\), mean per-group Spearman
correlation, and

\[
R^2_{\mathrm{XS}}
=
1-
\frac{\sum_{g,i}[(y_{i,g}-\bar y_g)-
(\hat y_{i,g}-\bar{\hat y}_g)]^2}
{\sum_{g,i}(y_{i,g}-\bar y_g)^2}.
\]

Also compute the paired cross-sectional absolute-loss difference

\[
\Delta^{\mathrm{XS}}_{i,g}
=
\left|(y_{i,g}-\bar y_g)-(\hat y_{i,g}-\bar{\hat y}_g)\right|
-|y_{i,g}-\bar y_g|.
\]

Bootstrap complete trading-day vectors with the same block lengths,
replicates, seed, and maximum-upper-bound rule used above. The gate passes only
when the maximum bootstrap 97.5th-percentile upper bound for
\(\operatorname{mean}(\Delta^{\mathrm{XS}})\) is below zero.
The \(R^2_{\mathrm{XS}}\) denominator must be positive or the gate fails.
Include a group in mean Spearman only when it has at least two cells and both
truth and prediction ranks are nonconstant. Report eligible and excluded
counts; if no group is eligible, report mean Spearman as `unavailable` without
changing the loss gate. Better aggregate MAE without this gate is only
consistent with a common market signal and cannot authorize universe expansion
or a ranked-winner policy.

## Checkpoint 2: Test context before architecture

Arm one equal-budget chronology-safe ablation:

- history lengths: `17`, `34`, and `68` completed bars;
- models: flattened ridge, the existing MLP, and the existing panel
  Transformer;
- target timestamps: the exact intersection shared by every history length;
- neural checkpoint grid, updates per checkpoint, optimizer, and five seeds:
  identical within each model's history comparison;
- ridge rows, train-only scaling, and penalty: identical and deterministic;
- phases, features, cohort, and target: otherwise unchanged.

For every neural model and seed, select the checkpoint on length `17` fold-0
only. Reinitialize each history and fit fold-1 training for that shared
checkpoint's target-phase update count, then publish every fold-1 prediction
before inspecting its labels. Select the later checkpoint on length `17`
fold-1 only, reinitialize each history, fit calibration training for that
second shared update count, and publish every calibration prediction before
inspecting calibration labels. Deterministic ridge instead fits once on each
target phase's training rows with its frozen scaling and penalty; it never
reads target-phase validation or outcome labels before prediction. Mutating
fold-1 validation or outcome labels must not change fold-1 state or
predictions; the same invariance applies within calibration. Mutating a phase's
training labels must change its fitted state.

The primary statistic is paired stock-macro return MAE. Direction accuracy and
costed portfolio results are secondary and cannot rescue a primary loss. Use
the same three-block bootstrap contract as Checkpoint 1. A longer context
advances only when its maximum upper bound against the same model at length
`17` is below zero in both fold-1 and calibration. If more than one length
passes, advance the shorter one. Patching additionally requires length `68` to
pass and to have lower point MAE than both length-68 simple controls in both
phases.

Do not add DLinear merely as another name. For a scalar output,

\[
\hat y=W_s(I-M)x+W_tMx
\]

is in the same affine function class as flattened ridge when \(M\) is fixed and
the component maps are unconstrained. Separate penalties, sharing, or
decomposition constraints can define a distinct estimator; add DLinear only
when one of those constraints is frozen in advance.

If length `68` passes, arm a separate no-padding, non-overlapping patch-token
candidate with patch length and stride `4`. It converts 68 temporal positions
to

\[
N_p=\left\lfloor\frac{68-4}{4}\right\rfloor+1=17,
\]

reducing per-head temporal scores from \(68^2=4624\) to \(17^2=289\). This is
a patch-only ablation, not full PatchTST: keep the existing feature-channel
treatment and normalization unchanged. Do not patch length `17`; it adds
machinery without adding history.

## Checkpoint 3: Calibrate the ensemble before changing its head

The current score

\[
\mu_t-\lambda\sigma_t
\]

uses seed disagreement as a ranking heuristic, not predictive uncertainty.
After the original 13-trial family is frozen, arm one separate one-sided split
conformal comparison with scaled and unscaled arms:

- miscoverage target `alpha = 0.10`;
- numerical floor `epsilon = 1e-6` log-return units;
- the exact calibration-phase ensemble held fixed across scoring and
  evaluation; and
- for \(D\) sorted distinct calibration target dates, require \(D\ge18\), use
  the first \(\lfloor D/2\rfloor\) as score dates, and reserve the remainder
  as evaluation candidates.

The runner must publish every calibration prediction before reading any
calibration label. It then reveals score-date outcomes, freezes \(q\), and only
then evaluates a candidate whose `as_of` is strictly later than every score
`target_time`. Candidates failing that embargo are excluded before policy
execution. Require at least `9` score dates and one post-embargo evaluation
date. Let \(G_d\) contain every opportunity from every complete, prebound
execution group whose `target_time` maps to registered exchange-session date
\(d\). Compute the two date-level maximum nonconformity scores

\[
A_d^{(0)}
=
\max_{t\in G_d}(\mu_t-y_t),
\qquad
A_d^{(\sigma)}
=
\max_{t\in G_d}
\frac{\mu_t-y_t}{\sigma_t+\epsilon}.
\]

For each arm \(m\), let \(q_m\) be the 1-based order statistic of its
\(n\) date scores at rank \(\lceil(n+1)(1-\alpha)\rceil\). The sample-size
requirement makes that rank at most \(n\). Each execution group must contain
exactly one opportunity for every series in the same digest-bound manifest; a
date may contain multiple groups and therefore multiple opportunities per
series. Bind the groups and their date mapping before labels and form them
identically across phases; reject the attempt if any execution group is
incomplete. Membership cannot depend on the arm, quantile, threshold, policy,
prediction, or outcome. The later lower bounds are

\[
L_t^{(0)}=\mu_t-q_0,
\qquad
L_t^{(\sigma)}=\mu_t-q_\sigma(\sigma_t+\epsilon).
\]

Date maxima account for within-date multiplicity only; chronological
dependence still prevents a finite-sample coverage guarantee. Evaluate both
frozen arms without choosing between them after results. For each arm, replace
the portfolio score with \(L_t\), enter only when \(L_t\) strictly exceeds the
cost-only round-trip break-even, and rank passers by descending \(L_t\), then
ascending manifest rank. Use `safety_bps = 0` and no additional disagreement
penalty. Reject incomplete seed sets, non-finite values, duplicate
opportunities, empty date groups, and any score fitted with an evaluation-date
label.

Report per-cell and simultaneous-date one-sided coverage, selected-opportunity
coverage, the mean/median/90th-percentile lower-bound offset \(\mu_t-L_t\),
trades, turnover, terminal cash, and realized-exit drawdown for both arms.
Smaller scaled offsets without lower coverage support seed disagreement as an
uncertainty signal; calibration alone does not. A one-sided set
\([L_t,\infty)\) has no finite interval width. Time dependence and reused
development history make all reported coverage empirical diagnostics, with no
finite-sample marginal, conditional, selected-opportunity, or
simultaneous-date 90% guarantee. Do not tune `alpha`, `epsilon`, the date
split, or the arm after inspection; adaptive calibration is outside this plan.

## Checkpoint 4: Change the Transformer only for an observed limitation

Run at most one architectural change per fresh attempt:

- **RevIN:** only if scale/regime diagnostics show input distribution shift
  remains after the existing train-only feature scaling;
- **variate tokens:** only if cross-feature ablations show the temporal-token
  model underuses OHLCV relationships;
- **patching:** only after longer raw context wins as defined above;
- **probabilistic head:** only if the fixed ensemble remains miscalibrated and
  the artifact/runtime contract is explicitly revised.

Those prerequisite outcomes are model-selection evidence. A triggered
architecture must be frozen with its complete comparison family before a later
untouched evaluation block begins. It cannot enter an SPA test computed on the
same phases that triggered it. Without a later block, report its replay as
descriptive only.

Train the probabilistic head in the existing standardized target units. For
member mean \(\mu_s^{\mathrm{scaled}}\) and raw variance output \(v_s\), freeze

\[
\tau_s^{\mathrm{scaled}}
=10^{-6}+\operatorname{softplus}(v_s)
\]

and use the proper Gaussian negative log-likelihood

\[
\ell_s
=
\frac12\left(
\log\tau_s^{\mathrm{scaled}}
+\frac{(y^{\mathrm{scaled}}-\mu_s^{\mathrm{scaled}})^2}
{\tau_s^{\mathrm{scaled}}}
\right).
\]

Before policy evaluation, use the prediction record's bound target mean \(m\)
and scale \(s\) to convert

\[
\mu_s^{\mathrm{raw}}
=s\mu_s^{\mathrm{scaled}}+m,
\qquad
\tau_s^{\mathrm{raw}}
=s^2\tau_s^{\mathrm{scaled}}.
\]

For \(\mu^{\mathrm{raw}}=M^{-1}\sum_s\mu_s^{\mathrm{raw}}\), aggregate total
ensemble variance with the numerically stable within-plus-between decomposition

\[
V^{\mathrm{raw}}
=
\frac1M\sum_s\tau_s^{\mathrm{raw}}
+\frac1M\sum_s(\mu_s^{\mathrm{raw}}-\mu^{\mathrm{raw}})^2.
\]

This changes training, serialization, C inference, and parity tests. It must
not be hidden inside a forecasting-only checkpoint. Parity tests must cover
the mean transform and the \(s^2\) variance transform for every bound scaler.

## Checkpoint 5: Correct for repeated inspection

Within one attempt, freeze two separate families before outcome access. Never
add a candidate chosen from that attempt's labels; a conditional candidate
requires the later-block boundary in Checkpoint 4.

- **Forecast loss:** the zero-return forecast is the benchmark. Alternatives
  are every inspected model-context candidate with predictions on the exact
  shared cells. \(L_{k,d}\) is that date's equal-stock mean absolute return
  error.
- **Portfolio return:** cash is the benchmark and is not duplicated as an
  alternative. Alternatives are every inspected, fully specified model-policy
  pair plus always-up. \(R_{k,d}\) is the sum of realized
  \(\log(C_{\mathrm{after}}/C_{\mathrm{before}})\) for trades exiting on date
  \(d\), or zero when none exit.

The return vector is realized-exit attribution over the exact shared date
range, not synthesized daily equity or mark-to-market risk. For each endpoint,
retain the complete simultaneous stock/policy vector when resampling a date.
Define

\[
d^{\mathrm{loss}}_{k,d}=L_{b,d}-L_{k,d},
\qquad
d^{\mathrm{return}}_{k,d}=R_{k,d}-R_{b,d}.
\]

For either differential, use

\[
T_{\mathrm{SPA}}
=
\max\left(0,\max_k
\frac{\sqrt D\,\bar d_k}{\hat\omega_k}\right).
\]

Freeze a Politis--Romano stationary bootstrap over complete days with mean
block length `10`, `10_000` replicates, and seed `20_260_725`. Require at least
`20` common days. Estimate \(\hat\omega_k\) as the bootstrap standard deviation
of \(\sqrt D(\bar d_k^*-\bar d_k)\). Exclude an exact all-zero differential as
a duplicate benchmark. A nonzero constant differential or any other
non-finite/zero-variance alternative fails the whole frozen attempt; never drop
it after outcome access.

Use Hansen's consistent sample-dependent recentering

\[
\hat\mu_k^c
=
\bar d_k\,
\mathbf1\left\{
\frac{\sqrt D\,\bar d_k}{\hat\omega_k}
\leq-\sqrt{2\log\log D}
\right\}.
\]

For bootstrap replicate \(r\), compute

\[
T_r^*
=
\max\left(0,\max_k
\frac{\sqrt D(\bar d_{k,r}^*-\bar d_k+\hat\mu_k^c)}
{\hat\omega_k}\right),
\]

and report the finite-sample p-value

\[
p=\frac{1+\sum_r\mathbf1\{T_r^*\geq T_{\mathrm{SPA}}\}}
{10\,001}.
\]

Use a `0.05` rejection threshold. Rejection means at least one alternative
beats its endpoint's benchmark; it does not prove the sample-best candidate is
the best member. Report forecast-loss and reference-cost portfolio tests
separately. A high terminal balance without family-wise evidence remains
exploratory.

Replay each frozen alternative's reference-cost trade schedule without
retuning at:

- zero costs, as an arithmetic diagnostic;
- the frozen `1 bp` spread plus `1 bp` per-side slippage reference; and
- `2`, `5`, and `10` times those reference costs.

Only spread and slippage scale; fees remain zero. Do not rerank, rethreshold,
or reselect trades during cost replay. The approximate round-trip break-even
levels are `3`, `6`, `15`, and `30` log basis points at `1`, `2`, `5`, and
`10` times reference costs. These multiples are sensitivity diagnostics; only
the frozen reference-cost path enters the return SPA family.

This plan does not claim quote-validated execution. Label every dollar result
assumption-bound and the spread assumption unvalidated; bar OHLC cannot
substitute for quotes. A future NBBO audit requires a separate registered
attempt, an access preflight for Massive Stocks Advanced or another bound
quote source, and a frozen sampling and quality contract. NBBO can describe
the prevailing spread and depth at or immediately before the registered
timestamps; slippage, latency, full fills, and open/close execution proxies
remain assumptions.

At `$100`, do not add nonlinear market-impact or optimal-execution machinery.
Quote/spread uncertainty dominates capacity impact at this notional. Freeze
participation, temporary/permanent impact, latency, and partial fills only
before testing materially larger principal.

## Implementation order

1. Complete the exact-update forward refits and predictions; implement, test,
   and freeze the existing 13-trial portfolio family and attempt, but do not
   execute the portfolio runner or read its outcome.
2. Add the breadth-forward attempt; reuse prior-phase checkpoint selection and
   the existing prediction-ledger validators for every registered comparison
   cell.
3. Add a pure context-attempt contract and its armer; bind all candidate axes,
   source hashes, inputs, and absent destinations.
4. Add one-shot context execution and a terminal finalizer; reuse existing
   training, metric, prediction-ledger, and complete-day bootstrap primitives.
5. Add and arm a small pure conformal module while the original portfolio
   outcome remains absent; keep score fitting separate from later-bound
   application. Only after this arm is frozen may the original portfolio
   runner execute and its terminal outcome be read.
6. Add family-wise analysis and cost replay after candidate ledgers are frozen.
7. Add one architecture candidate only when its prerequisite above passes.

Each checkpoint gets focused procedural tests, the relevant optional Torch
gate, and the aggregate `make -B check` gate. Keep tests with the behavior they
verify and do not duplicate execution, chronology, hashing, or metric math.

## Primary references

- [Deep Ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html)
- [Predictive Uncertainty under Dataset Shift](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html)
- [Distribution-Free Predictive Inference for Regression](https://www.stat.berkeley.edu/~ryantibs/papers/conformal.pdf)
- [Conformal Inference with Dependent Data](https://proceedings.mlr.press/v75/chernozhukov18a.html)
- [Conformal Prediction Beyond Exchangeability](https://arxiv.org/abs/2202.13415)
- [Selective Conformal Inference](https://arxiv.org/abs/2301.00584)
- [Principles and Algorithms for Forecasting Groups of Time Series](https://arxiv.org/abs/2008.00444)
- [Deep Factors for Forecasting](https://proceedings.mlr.press/v97/wang19k.html)
- [Empirical Asset Pricing via Machine Learning](https://academic.oup.com/rfs/article/33/5/2223/5758276)
- [Universal Features of Price Formation](https://arxiv.org/abs/1803.06917)
- [PatchTST](https://arxiv.org/abs/2211.14730)
- [DLinear](https://ojs.aaai.org/index.php/AAAI/article/view/26317)
- [RevIN](https://openreview.net/forum?id=cGDAkQo1C0p)
- [iTransformer](https://arxiv.org/abs/2310.06625)
- [Hansen's Superior Predictive Ability test](https://doi.org/10.1198/073500105000000063)
- [White's Reality Check](https://doi.org/10.1111/1468-0262.00152)
- [The Stationary Bootstrap](https://doi.org/10.1080/01621459.1994.10476870)
- [Almgren--Chriss](https://doi.org/10.21314/JOR.2001.041)
- [Massive historical NBBO quotes](https://massive.com/docs/rest/stocks/trades-quotes/quotes)
