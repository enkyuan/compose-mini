# compose-mini

`compose-mini` is a staged C11 inference runtime for a pretrained,
encoder-only Transformer. Its target is the next completed bar's close from a
fixed window of completed OHLCV history.

The core encoder, scalar head, versioned artifact loader, and chronological CSV
windowing are implemented and tested. Forecast emission remains to be wired;
training and trading decisions are outside this runtime. See
[docs/forecasting-contract.md](docs/forecasting-contract.md) for the system
boundary and delivery stages.
