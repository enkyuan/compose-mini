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
last-close baseline MAE and direction accuracy for the test segment.

## Compare configurations and baselines

Run the same experiment independently over one or more instruments:

```sh
python tools/experiment.py experiments/sweep.example.json \
  reports/market.json \
  AAPL=data/aapl-30m.csv \
  MSFT=data/msft-30m.csv
```

The sweep compares the Transformer with ridge linear regression, a one-hidden-
layer MLP, a trailing-mean return, and the zero-return last-close baseline. It
uses expanding walk-forward validation folds and repeated seeds. Candidates
with different sequence lengths are aligned to identical target timestamps.
Each model's configuration is selected only by mean validation scaled-return
MSE, then evaluated once on a final untouched test holdout.

Each CSV remains an independent model problem because artifact schema 1 stores
one feature scaler and must never form windows across instruments. The report
macro-averages return metrics so every instrument, fold, and seed has equal
weight. It keeps dollar close MAE separate by instrument, including each seed's
absolute and relative difference from last-close, and records sample counts,
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
    --horizon-bars "$horizon" --max-runs 117
done
```

Horizon `H` predicts `log(close[t + H] / close[t])`. The shared 13-bar
alignment keeps target timestamps, folds, sample counts, and the final holdout
identical across reports. A fixed 12-bar embargo removes labels that would not
yet be known at the next split's earliest forecast origin. The trailing-mean
baseline scales its one-bar mean by `H`; last-close remains zero return.

Each horizon has a different target scaler, so do not rank horizons by scaled
MSE. Compare each model with its horizon-matched last-close and rolling-mean
baselines using return MAE, per-series close-MAE difference, direction accuracy,
and variation across seeds. These reports are diagnostic: artifact V1 and the C
runtime still forecast only the next bar.

To isolate input representation from model size, run the paired feature sweep:

```sh
python tools/experiment.py experiments/features.example.json \
  reports/features.json \
  AAPL=data/aapl-30m.csv MSFT=data/msft-30m.csv SPY=data/spy-30m.csv \
  --max-runs 300
```

The stationarity-oriented `stationary-v1` representation encodes each completed
candle as log gap, log body, upper and lower log wick, and log-volume change. It
uses only that candle and its previous completed bar; it does not claim that the
resulting series is statistically stationary. The raw 17-bar control matches
its total history, while the raw 16-bar control matches its token count. All
variants share target timestamps, folds, optimizer settings, seeds, and the
untouched final holdout.

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

## Backtest a frozen holdout

Export timestamped test predictions while running a frozen experiment:

```sh
python tools/experiment.py experiments/horizons.example.json \
  reports/horizon-13.json \
  AAPL=data/aapl-30m.csv MSFT=data/msft-30m.csv SPY=data/spy-30m.csv \
  --horizon-bars 13 --max-runs 117 \
  --predictions reports/horizon-13-predictions.jsonl
```

Then run each series, model, and seed independently from `$100`:

```sh
python tools/backtest.py reports/horizon-13-predictions.jsonl \
  reports/horizon-13-backtest.json \
  AAPL=data/aapl-30m.csv MSFT=data/msft-30m.csv SPY=data/spy-30m.csv \
  --spread-bps 1 --slippage-bps 1 --fee-bps 0
```

The frozen policy is long when predicted log return is positive and otherwise
cash. Each entry invests all available equity in fractional shares without
leverage, enters at the next bar's open, exits at the target bar's close, and
ignores signals made before that exit.
Spread is the full quoted spread; slippage and proportional fees apply per side.
The report compares the forecast rule with cash, buy-and-hold, and an always-up
rule over identical bars. Equity is marked to cost-adjusted liquidation value at
each close and summarized by UTC day, ISO week, and month, with final return,
maximum drawdown, win rate, and gross notional turnover.

The model target measures close-to-close return, while the tradable return starts
at the next open. Opening gaps can therefore change the sign and economics of a
forecast. Cash earns no yield, dividends are not credited separately, period
endpoints may be partial, and drawdown is sampled from bar-close equity rather
than intrabar lows.

These results are hypothetical. Because the current holdout has already informed
model and horizon discussion, label its P&L exploratory and freeze the complete
policy before evaluating a later untouched period.

After choosing a Transformer candidate, pass its values to `tools/train.py` to
fit and export the deployable V1 artifact. Diagnostic linear and MLP models are
not serialized as Transformer artifacts.

Run the optional PyTorch integration tests after installing the framework:

```sh
make check-training
```

They perform a tiny deterministic optimization, restore and export the chosen
checkpoint, compare PyTorch forecasts with C, and verify fold alignment,
validation-only selection, baselines, run limits, and strict experiment JSON.
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
