"""Build label-free inputs for causal universe-forward inference."""

from array import array
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from tools.data_v1 import FEATURE_COUNT
from tools.session_samples import SampleRows
from tools.train import feature_lookback, feature_values


class ForwardFeatureWindows(Dataset):
    """Return normalized histories ending at each completed as-of bar."""

    def __init__(
        self, rows: array, samples: Sequence[SampleRows], seq_len: int,
        feature_set: str, feature_mean: torch.Tensor,
        feature_scale: torch.Tensor,
    ) -> None:
        values = (feature_mean, feature_scale)
        if not isinstance(rows, array) or rows.typecode != "f" or \
           len(rows) % FEATURE_COUNT or not isinstance(samples, Sequence) or \
           type(seq_len) is not int or seq_len < 1 or any(
               not isinstance(value, torch.Tensor) or
               value.shape != (FEATURE_COUNT,) or
               value.dtype != torch.float32 or value.device.type != "cpu" or
               not torch.isfinite(value).all()
               for value in values
           ) or not torch.all(feature_scale > 0):
            raise ValueError("forward feature inputs are invalid")
        items = tuple(samples)
        if not items or any(
            type(item) is not SampleRows or any(
                type(value) is not int or value < 0 for value in (
                    item.as_of, item.entry, item.target, item.as_of_ordinal,
                )
            ) or item.as_of >= item.entry or item.entry > item.target
            for item in items
        ):
            raise ValueError("forward sample rows are invalid")
        lookback = feature_lookback(feature_set)
        starts = tuple(
            item.as_of - seq_len - lookback + 1 for item in items
        )
        row_count = len(rows) // FEATURE_COUNT
        if row_count != max(item.as_of for item in items) + 1 or \
           any(start < 0 for start in starts):
            raise ValueError("forward sample rows are invalid")
        raw = torch.frombuffer(rows, dtype=torch.float32).view(
            row_count, FEATURE_COUNT,
        ).clone()
        if not torch.isfinite(raw).all() or not torch.all(raw[:, 3] > 0):
            raise ValueError("forward bars are invalid")
        features = feature_values(raw, feature_set).sub_(
            feature_mean,
        ).div_(feature_scale)
        if not torch.isfinite(features).all():
            raise ValueError("normalized forward features are invalid")
        self.features = features
        self.starts, self.seq_len = starts, seq_len

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> torch.Tensor:
        start = self.starts[index]
        return self.features[start:start + self.seq_len]
