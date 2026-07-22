# compose-mini

`compose-mini` is a staged C11 inference runtime for a pretrained,
encoder-only Transformer. Its target is the next completed bar's close from a
fixed window of completed OHLCV history.

The core encoder, scalar prediction head, and versioned artifact loader are
implemented and tested. Market-data preparation and forecast emission remain
to be wired; training and trading decisions are outside this runtime. See
[docs/forecasting-contract.md](docs/forecasting-contract.md) for the system
boundary and delivery stages.
