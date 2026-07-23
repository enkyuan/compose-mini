# Architecture, math, and training

## End-to-end boundary

`compose-mini` predicts the next completed bar's log return, then reconstructs
its close. For a sequence length `S`, one inference window is:

```text
[S x 5] completed OHLCV rows
  -> training-fitted feature scaling
  -> input projection + fixed positional encoding
  -> L pre-norm Transformer encoder blocks
  -> final hidden row + scalar head
  -> inverse target scaling
  -> latest_close * exp(predicted_log_return)
```

Timestamps, instrument, and interval label the result but never enter the
network. The caller certifies that rows are completed, from one series, and
consecutive on the relevant exchange calendar.

## Exact schema-1 model

Let `X` have shape `[S, 5]`, hidden width be `D`, head count be `H`, and
feed-forward width be `F`. All arrays are float32 and row-major.

Training-fitted feature statistics scale each feature independently:

```text
X_scaled[t, j] = (X[t, j] - feature_mean[j]) / feature_scale[j]
```

The input projection has no bias:

```text
h_0 = X_scaled @ embed_W + PE                 [S x D]
PE[pos, 2i]     = sin(pos / 10000^(2i / D))
PE[pos, 2i + 1] = cos(pos / 10000^(2i / D))
```

Schema 1 computes positional frequencies with the same float32 recurrence as
the C runtime. This detail matters for reproducible cross-language output.

Each encoder layer is pre-normalized and has two residual branches:

```text
n_1 = LayerNorm(h)
Q = n_1 @ Wq; K = n_1 @ Wk; V = n_1 @ Wv
A_h = softmax(Q_h @ K_h^T / sqrt(D / H)) @ V_h
h = h + concat(A_1, ..., A_H) @ Wo

n_2 = LayerNorm(h)
h = h + GELU(n_2 @ W1 + b1) @ W2 + b2
```

LayerNorm uses population variance and epsilon `1e-5`:

```text
LayerNorm(x) = gamma * (x - mean(x)) / sqrt(var(x) + 1e-5) + beta
```

GELU is the exact error-function form:

```text
GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
```

Attention is full, not causal: every input row is already historical relative
to the one target after the window. There are no projection biases, dropout,
learned positions, or terminal LayerNorm in schema 1.

The scalar head reads only the final hidden row:

```text
z = h[S - 1] @ head_W + head_b
predicted_log_return = z * target_scale + target_mean
predicted_close = latest_close * exp(predicted_log_return)
```

Predicting log return instead of raw close makes the target relative to the
latest observed price and gives a meaningful zero-return baseline. It does not
guarantee better forecasts; chronological evaluation must establish that.

## Training samples without leakage

For `N` chronological rows and sequence length `S`, training has `N - S`
labeled samples. Sample `i` is:

```text
input  = rows[i : i + S]
target = log(close[i + S] / close[i + S - 1])
```

Inference has one additional unlabeled final window, so it emits `N - S + 1`
forecasts.

All three floor-derived splits must contain a sample. With the default 70/15/15
fractions, this requires at least seven labeled samples, or `N >= S + 7` rows.

Split samples by target time into contiguous train, validation, and test
segments. Never random-split time-series samples. Fit feature statistics only
on unique rows touched by training inputs, and fit target statistics only on
training targets. Reuse those values unchanged for validation, test, and C
inference. Constant columns or targets are rejected instead of hidden behind an
epsilon.

Training batches may shuffle samples inside the training segment. Validation
and test stay ordered. Select and restore the best validation checkpoint; use
the test segment once after selection.

## Start training

The Python tooling requires Python 3.10 or newer.

Fetch split-adjusted Massive aggregates after placing the API key in the
ignored `.env` file:

```sh
python tools/fetch_massive.py AAPL 2024-07-22 2026-07-21 \
  data/aapl-30m.csv --minutes 30
```

The downloader follows pagination, keeps regular-session bars, verifies that
each observed session starts at 09:30 US Eastern with no internal interval
gaps, converts timestamps to canonical UTC, and atomically writes the runtime's
six-column CSV. Missing sessions cannot be distinguished from exchange holidays
without an exchange calendar, so retain the provider and query parameters with
every experiment.

Create an isolated environment and install PyTorch for the machine's CPU or
accelerator:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch
```

Then train and export directly into the runtime's V1 format:

```sh
python tools/train.py bars.csv model.bin \
  --interval 1h \
  --model-version experiment-001 \
  --seq-len 32 \
  --model-dim 64 \
  --heads 4 \
  --ff-dim 256 \
  --layers 4
```

The trainer reads the runtime's literal six-column CSV grammar, constructs
zero-copy sliding views over one scaled feature tensor, trains on scaled log
returns, restores the best validation weights, evaluates chronologically, and
atomically writes a checksummed artifact. Its final JSON labels validation MSE
as scaled-target loss and reports raw-return MSE/MAE, predicted-close MAE, and
zero-return reference-price MAE and direction accuracy for the test segment.

## Compare configurations and baselines

Run the same experiment independently over one or more instruments:

```sh
python tools/experiment.py experiments/sweep.example.json \
  reports/market.json \
  AAPL=data/aapl-30m.csv \
  MSFT=data/msft-30m.csv \
  --calibration-only
```

The sweep compares the Transformer with ridge linear regression, a one-hidden-
layer MLP, a trailing-mean return, and a zero-return baseline. The legacy model
name `last_close` uses the target's reference price. The sweep uses expanding
walk-forward validation folds and repeated seeds. Candidates
with different sequence lengths are aligned to identical target timestamps.
Each model's configuration is selected only by mean validation scaled-return
MSE. A holdout is evaluated only through the frozen-policy workflow below.

Each CSV remains an independent model problem because artifact schema 1 stores
one feature scaler and must never form windows across instruments. The report
macro-averages return metrics so every instrument, fold, and seed has equal
weight. It keeps dollar close MAE separate by instrument, including each seed's
absolute and relative difference from zero return, and records sample counts,
dataset hashes, runtime versions, and exact target-time ranges. Reports are
atomic, strict JSON under the ignored `reports/` directory. Run and diagnostic
model-size limits reject accidental compute or memory explosions; raise
`--max-runs` only after estimating the required compute.

### Compare target horizons

For 30-minute bars, compare 30 minutes, 2 hours, and roughly one 6.5-hour
regular session with one raw-17 sweep:

```sh
for horizon in 1 4 13; do
  python tools/experiment.py experiments/horizons.example.json \
    "reports/horizon-${horizon}.json" \
    AAPL=data/aapl-30m.csv MSFT=data/msft-30m.csv SPY=data/spy-30m.csv \
    --horizon-bars "$horizon" --max-runs 117 --calibration-only
done
```

By default, horizon `H` predicts `log(close[t + H] / close[t])`. The shared 13-bar
alignment keeps target timestamps, folds, sample counts, and the final holdout
identical across reports. A fixed 12-bar embargo removes labels that would not
yet be known at the next split's earliest forecast origin. The trailing-mean
baseline scales its one-bar mean by `H`; `last_close` remains zero return.

Each horizon has a different target scaler, so do not rank horizons by scaled
MSE. Compare validation return MAE, direction accuracy, and variation across
seeds against horizon-matched baselines. These reports are diagnostic: artifact
V1 and the C runtime still forecast only the next bar.

To align an experiment with the prices available to the backtest, predict the
return from the next executable open instead:

```sh
python tools/experiment.py experiments/horizons.example.json \
  reports/executable-13-calibration.json \
  AAPL=data/aapl-30m.csv MSFT=data/msft-30m.csv SPY=data/spy-30m.csv \
  --horizon-bars 13 --target-kind executable-return-v1 --max-runs 117 \
  --calibration-predictions reports/executable-13-calibration.jsonl \
  --calibration-only
```

`executable-return-v1` is
`log(close[t + H] / open[t + 1])`: the same entry and exit prices simulated by
the backtest. The default `close-to-close-v1` target remains the Artifact V1
contract. The executable target is experiment-only and requires a future
artifact schema before deployment through the C runtime.

To isolate input representation from model size, run the paired feature sweep:

```sh
python tools/experiment.py experiments/features.example.json \
  reports/features.json \
  AAPL=data/aapl-30m.csv MSFT=data/msft-30m.csv SPY=data/spy-30m.csv \
  --max-runs 300 --calibration-only
```

The stationarity-oriented `stationary-v1` representation encodes each completed
candle as log gap, log body, upper and lower log wick, and log-volume change. It
uses only that candle and its previous completed bar; it does not claim that the
resulting series is statistically stationary. The raw 17-bar control matches
its total history, while the raw 16-bar control matches its token count. All
variants share target timestamps, folds, optimizer settings, and seeds without
opening the final holdout.

```text
[log(open[t] / close[t-1]), log(close[t] / open[t]),
 log(high[t] / max(open[t], close[t])),
 log(min(open[t], close[t]) / low[t]),
 log1p(volume[t]) - log1p(volume[t-1])]
```

This representation is diagnostic only. Artifact V1 and the C runtime still
require raw OHLCV; promote it only through an explicit artifact schema after it
shows consistent validation improvement across instruments, folds, and seeds.
The report includes per-candidate validation distributions and paired
candidate-minus-control deltas; negative error deltas favor the candidate.

### Calibrate horizon-13 feature candidates

```zsh
series=(
  AAPL=data/aapl-30m.csv
  MSFT=data/msft-30m.csv
  SPY=data/spy-30m.csv
)

python tools/experiment.py \
  experiments/executable-h13-features.example.json \
  reports/executable-h13-feature-calibration.json \
  "${series[@]}" \
  --max-runs 75 \
  --calibration-predictions \
    reports/executable-h13-feature-calibration.jsonl \
  --calibration-only
```

This ablation is calibration-only. Do not run a policy-authorized historical
test or inspect labels through 2026-07-21. Confirmatory evaluation requires a
pre-registered policy against later, previously unavailable labels.

#### Development result (2026-07-23)

This calibration-only run selected `raw-17`. For AAPL, MSFT, and SPY, each
selected final run used 633 targets from `2026-02-26T19:30:00Z` through
`2026-05-07T16:00:00Z`. Data through 2026-07-21 remains development-only. No
policy-authorized historical test was run.

- Macro validation return MAE was `0.00960146184341728` for `raw-17` and
  `0.009698446599541997` for `stationary-16`. Relative reduction was
  `-0.01010104062343471`: stationary was 1.0101% worse.
- Stationary won 1 of 6 series-fold seed-mean buckets. Promotion required at
  least 5% reduction and wins in at least 5 of 6 buckets, so it failed.
- The raw policy chose a 10 bps safety margin, objective
  `0.03636838431516431`, mean final equity `103.75624468627991`, and 52 trades.
  Execution coverage was `0.011058451816745656` for AAPL,
  `0.037914691943127965` for MSFT, and `0.03317535545023697` for SPY. These
  trading checks passed, but they do not rescue stationary.

The generated report, ledger, policy, and calibration backtest remain ignored
local evidence, not confirmation.

The one-shot seed-disagreement gate exited `0` and returned
`promote_seed_disagreement: true`. The selected schema-3 trial was
`long_above` with `disagreement_lambda: 0.5`, `safety_bps: 0.0`, objective
`0.0473046623458639`, mean final equity `105.03967733947705`, mean gross
turnover `44.96456738553214`, signal coverage `0.3070036861506056`, execution
coverage `0.03580832016850974`, and 68 trades. The best lambda-zero objective
was `0.03636838431516431`.

- AAPL executed 16 trades with coverage `0.02527646129541864`; MSFT and SPY
  each executed 26 trades with coverage `0.04107424960505529`.
- Every source and output binding passed: fresh paths, report and ledger
  provenance, calibration protocol and selection, prediction ledger, backtest
  protocol, exact unique series, and result contract.
- Every promotion check passed: long policy, nonzero lambda, objective strictly
  above lambda zero, at least 30 trades, matching trial and backtest counts,
  and positive trades and execution coverage for every series.
- The ignored policy SHA-256 is
  `a77313d7cf6aa92f8b55baf540ca2158030f68585aa844bd6ee5836bec949c41`;
  its diagnostic calibration backtest SHA-256 is
  `7ad959e20c55782da393626e652b8fc0e2f70668919d59d1d337c16e69859d2a`.

This is a calibration-selected member-disagreement heuristic, not calibrated
uncertainty, confidence, deployment evidence, or a confirmed trading result.
No retraining or historical test followed the gate. Before any evaluation,
externally pre-register the complete schema-3 policy hash above and a boundary
against previously unavailable labels strictly after `2026-07-21`.

## Backtest a frozen holdout

Define the three independent series once:

```zsh
series=(
  AAPL=data/aapl-30m.csv
  MSFT=data/msft-30m.csv
  SPY=data/spy-30m.csv
)
```

Run selection and calibration without opening the test interval:

```zsh
python tools/experiment.py experiments/horizons.example.json \
  reports/executable-h13-calibration.json \
  "${series[@]}" \
  --horizon-bars 13 --target-kind executable-return-v1 --max-runs 117 \
  --calibration-predictions reports/executable-h13-calibration.jsonl \
  --calibration-only
```

Freeze one cost-aware policy per model from that same report and schema-3
calibration ledger. Policy schema 3 adds a member-disagreement heuristic:

```text
decision_signal =
  mean(seed predicted_log_return)
  - disagreement_lambda
    * population_pstdev(seed predicted_log_return)
```

This signal is not calibrated uncertainty or confidence. Reports keep
`predicted_log_return` as the arithmetic seed mean; schema 3 freezes the
selected multiplier used only for scheduling.

```zsh
for model in transformer mlp linear; do
  disagreement=(0)
  [[ "$model" == linear ]] || disagreement+=(0.5 1)
  python tools/select_policy.py reports/executable-h13-calibration.json \
    reports/executable-h13-calibration.jsonl \
    "reports/executable-h13-${model}-policy-v3.json" \
    "${series[@]}" \
    --model "$model" --safety-bps 0 3 6 10 \
    --disagreement-lambda "${disagreement[@]}" \
    --initial-cash 100 --spread-bps 1 --slippage-bps 1 --fee-bps 0
done
```

The selector verifies every selected candidate, series, fold, seed, timestamp
boundary, and input hash. It averages the configured seeds, maximizes mean log
terminal growth across calibration accounts, deterministically breaks ties by
lower turnover and higher threshold, and selects cash when no trading rule wins.

All three schema-3 policies must exist and pass validation before one combined
test command opens holdout labels. No model gets an earlier look at the test
interval.
Existing `*-policy-v2.json` files remain exact schema-2 inputs and replay
unchanged with an implied disagreement multiplier of zero; they are not
rewritten or upgraded in place.

```zsh
python tools/experiment.py experiments/horizons.example.json \
  reports/executable-h13-test-v2.json \
  "${series[@]}" \
  --horizon-bars 13 --target-kind executable-return-v1 --max-runs 117 \
  --predictions reports/executable-h13-test-v2.jsonl \
  --policy reports/executable-h13-transformer-policy-v3.json \
  --policy reports/executable-h13-mlp-policy-v3.json \
  --policy reports/executable-h13-linear-policy-v3.json

for model in transformer mlp linear; do
  python tools/backtest.py reports/executable-h13-test-v2.jsonl \
    "reports/executable-h13-${model}-final-v2.json" \
    "${series[@]}" \
    --policy "reports/executable-h13-${model}-policy-v3.json" \
    --experiment-report reports/executable-h13-test-v2.json
done
```

Test mode rejects direct model, cost, seed, and threshold overrides. The frozen
policy must authorize the full experiment before it can evaluate its model.
No unauthorized model is evaluated on the holdout. The test report records each
authorized policy hash.
The frozen
calibration fingerprint must also match the test experiment's candidate
configuration, selection, folds, series, and validation results. The frozen
schema-3 policy schedules long only when its disagreement-adjusted decision
signal—the arithmetic seed mean minus `disagreement_lambda` times population
seed disagreement—exceeds exact round-trip break-even friction plus its frozen
safety margin. Trades and reports still record `predicted_log_return` as the
arithmetic seed mean; otherwise the policy remains cash. Each entry invests all
available equity in fractional shares without leverage, enters at the next
bar's open, exits at the target bar's close, and ignores signals made before
that exit.
The policy file is the trust root: archive it with the final report, which
records the exact policy SHA-256 used for the backtest.
Local files cannot make historical holdout access one-shot. For confirmatory
claims, pre-register the complete policy-hash set and test boundary in an
append-only external log; classify every later policy or rerun as exploratory.
Spread is the full quoted spread; slippage and proportional fees apply per side.
The report compares the forecast rule with cash, buy-and-hold, and an always-up
rule over identical bars. Equity is marked to cost-adjusted liquidation value at
each close and summarized by UTC day, ISO week, and month, with final return,
maximum drawdown, win rate, and gross notional turnover. Signal coverage counts
all forecasts above threshold, execution coverage counts completed trades over
all forecasts, and eligible-entry hit rate excludes forecasts blocked by an
open position.

With `close-to-close-v1`, opening gaps can still change a forecast's sign and
economics. `executable-return-v1` removes that target/execution mismatch. Cash
earns no yield, dividends are not credited separately, period endpoints may be
partial, and drawdown is sampled from bar-close equity rather than intrabar
lows.

These results are hypothetical. Earlier work already inspected the present
historical test interval, so label its P&L exploratory. Confirmation requires
later data whose labels were unavailable when the complete policy-hash set was
registered.

After choosing a Transformer candidate, pass its values to `tools/train.py` to
fit and export the deployable V1 artifact. Diagnostic linear and MLP models are
not serialized as Transformer artifacts.

Run the optional PyTorch integration tests after installing the framework:

```sh
make check-training
```

They perform a tiny deterministic optimization, restore and export the chosen
checkpoint, compare PyTorch forecasts with C, and verify fold alignment,
calibration-only selection, baselines, run limits, and strict experiment JSON.
A separate fixture compares exported operators with the scalar reference.

Run inference by supplying the next target timestamp for the newest window:

```sh
make
bin/transformer model.bin bars.csv AAPL 1h 2026-07-21T15:00:00Z
```

## Parity and scale

`make check` writes a nonzero two-layer artifact, runs the real C executable
twice, and compares every forecast with an independent float32 Python
implementation within 16 binary32 units in the last place (ULPs). It also locks
artifact order, checksum, determinism, failure atomicity, and the training CSV
grammar.
This path uses only the Python standard library; PyTorch is needed only for
training and `make check-training`.

Parameter count is:

```text
P = 5D + L * (4D^2 + 2DF + 5D + F) + D + 1
```

Model storage is `O(P)`. Runtime data storage is `O(N)`, overlapping windows
are borrowed views, and one caller-owned scratch allocation is reused across
all windows and layers. Encoder scratch is `5SD + S` floats; attention compute
is `O(L * (S * D^2 + S^2 * D))` for projections plus score/context products.
Optimize kernels only after parity and forecast quality are established.

Schema 1 fixes every operator described above. Changing an activation, adding
a bias or mask, changing normalization, or altering positional encoding
requires a new artifact schema rather than silently reinterpreting old weights.
