# Model artifact V1

## Purpose

The artifact is the immutable boundary between external training and this C11
inference runtime. It stores model dimensions, fitted scalers, metadata, and
weights in one checksummed little-endian file. V1 fixes features to
`open, high, low, close, volume`, values to IEEE-754 binary32, and the target to
horizon log return.

Schema 1 also fixes the operator graph: bias-free input and attention
projections, float32 sinusoidal positions, full attention, pre-norm residual
blocks, LayerNorm epsilon `1e-5`, exact-erf GELU, no dropout, and no terminal
normalization. See [training.md](training.md) for the equations. Any operator
change requires a new schema version.

## Header

The 32-byte header is decoded byte by byte; it is never read into a C struct.

| Offset | Bytes | Value |
| ---: | ---: | --- |
| 0 | 8 | Magic `CMPMINI\0` |
| 8 | 4 | Schema version `1` |
| 12 | 4 | Header size `32` |
| 16 | 8 | Body size |
| 24 | 8 | CRC-32 of the complete body, zero-extended to 64 bits |

CRC-32 uses zlib's `crc32(0, body, body_size)` ISO-HDLC semantics.

## Body

| Offset | Bytes | Value |
| ---: | ---: | --- |
| 0 | 24 | `model_dim`, `num_heads`, `ff_dim`, `num_layers`, `seq_len`, `in_dim` |
| 24 | 4 | Forecast horizon in bars; must be `1` in V1 |
| 28 | 4 | Reserved; zero in V1 |
| 32 | 64 | NUL-terminated, zero-padded model version |
| 96 | 16 | NUL-terminated, zero-padded interval |
| 112 | 20 | Five feature means in OHLCV order |
| 132 | 20 | Five positive feature scales in OHLCV order |
| 152 | 4 | Target mean |
| 156 | 4 | Positive target scale |
| 160 | `4 * P` | Contiguous model weights |

The exact body size is `160 + 4 * P`, where:

```text
P = in_dim * D
  + L * (4 * D * D + 2 * D * F + 5 * D + F)
  + D + 1
```

`D` is `model_dim`, `F` is `ff_dim`, and `L` is `num_layers`.

All six dimensions are positive signed 32-bit values. V1 requires five input
features and `model_dim % num_heads == 0`. Model version and interval are
nonempty visible ASCII tokens with maxima of 63 and 15 bytes, respectively.

## Weight order

Weights use the same field-major order as `TransformerWeights`:

```text
embed_W,
Wq, Wk, Wv, Wo,
norm1_g, norm1_b,
W1, b1, W2, b2,
norm2_g, norm2_b,
head_W, head_b
```

Each per-layer field contains all `L` slices consecutively. The loader rejects
wrong dimensions, lengths, padding, checksums, non-finite values, non-positive
scales, truncation, and trailing bytes before exposing the artifact. The
CPU-only V1 loader caps artifacts at `67,108,864` parameters, or 256 MiB of
float weights. It also caps peak internal forward scratch at `67,108,864`
floats, using `5 * seq_len * model_dim + seq_len`, so sequence length cannot
bypass the memory bound.
