# Forecasting contract

## Purpose

`compose-mini` is a small C inference runtime for a pretrained, encoder-only
Transformer. It forecasts the close of the next completed bar from a fixed
window of completed OHLCV bars.

Training, market-data retrieval, exchange scheduling, trading decisions,
position sizing, and order execution are outside this runtime.

## Current status

The core utilities and LayerNorm are implemented and behaviorally tested. Data
loading, the remaining encoder components, artifact loading, the prediction
head, and the CLI remain stubs or planned interfaces. The runtime does not yet
produce forecasts.

## Runtime input

The process loads one versioned model artifact. A single-window request supplies:

- Exactly `seq_len` completed bars for one instrument and interval.
- Bars ordered oldest to newest with strictly increasing timestamps.
- Features in the fixed order `open, high, low, close, volume`.
- The instrument and interval as metadata. `as_of` is the final bar's timestamp;
  the scheduling layer supplies `target_time`.

A batch CSV may contain `N >= seq_len` bars. It produces
`N - seq_len + 1` overlapping windows and the same number of forecast records.

Only the scaled `[seq_len x 5]` feature tensor enters the model. Metadata names
and timestamps the result. Reject incomplete bars, mixed instruments or
intervals, invalid timestamps, non-finite values, non-positive closes, and
requests whose interval, feature order, or dimensions do not match the
artifact. The artifact fixes the forecast horizon; callers do not change it.

The current CSV and `DataSet` scaffold do not yet carry all required metadata.
Their interfaces must be extended before artifact-backed inference is complete.

## Runtime output

V1 returns one structured record per valid window:

```json
{
  "instrument": "AAPL",
  "interval": "1h",
  "as_of": "2026-07-11T15:00:00Z",
  "target_time": "2026-07-11T16:00:00Z",
  "horizon_bars": 1,
  "predicted_log_return": 0.0031,
  "predicted_close": 231.42,
  "model_version": "v1"
}
```

V1 uses next-bar log return as its training target and reconstructs the public
price from the predicted value:

```text
target_log_return = log(next_close / latest_close)
predicted_close = latest_close * exp(predicted_log_return)
```

Do not report confidence intervals unless the artifact was trained and
validated to produce calibrated uncertainty. This runtime emits forecasts, not
buy or sell decisions.

## Model artifact

An offline training pipeline produces the artifact. It contains:

- Artifact schema version and model version.
- Feature order, interval, horizon, and target definition.
- `seq_len`, `in_dim`, `model_dim`, `num_heads`, `ff_dim`, and `num_layers`.
- Feature and target scaling formulas with their training-fitted parameters.
- Input projection, distinct per-layer encoder weights, and prediction-head
  weights and bias.
- Numeric format and integrity metadata used to reject incompatible files.

Inference always uses the artifact's training-fitted statistics. It never fits
normalization from an inference file or window.

## Internal boundary

```text
validate completed bars
  -> scale with artifact statistics
  -> project features and add positional encoding once
  -> run every encoder block
  -> select the final timestep representation
  -> apply the scalar prediction head
  -> invert target scaling
  -> reconstruct and emit the predicted close
```

Full attention over the input window is safe because every input bar precedes
the forecast target. A per-position forecasting objective must instead mask
later bars.

## Delivery stages

1. Freeze this input, output, target, and artifact contract.
2. Implement and test the core math utilities.
3. Implement embeddings, encoder blocks, per-layer weights, and prediction head.
4. Parse and validate chronological windows without fitting a runtime scaler.
5. Train externally, export a versioned artifact, and load it in C.
6. Implement the CLI inference loop and structured output.
7. Compare C output with the training framework and evaluate chronologically
   against a last-close baseline.
8. Optimize only after correctness and parity are established.
