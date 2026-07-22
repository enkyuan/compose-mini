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
last-close baseline MAE for the test segment.

Run the optional PyTorch integration test after installing the framework:

```sh
make check-training
```

It performs a tiny deterministic optimization, restores and exports the chosen
checkpoint, and compares its PyTorch forecasts with C using the learned
scalers. A separate fixture compares exported operators with the scalar
reference.

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
