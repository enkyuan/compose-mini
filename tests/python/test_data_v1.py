#!/usr/bin/env python3
"""Verify the training reader enforces the runtime's literal CSV contract."""

from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.data_v1 import (
    CSV_HEADER,
    FEATURE_COUNT,
    LINE_CAP,
    read_bars,
    read_bars_through,
    read_csv,
    read_timestamps,
    read_timestamps_through,
)
from tools.float32 import f32

VALID_ROWS = (
    "2026-07-21T10:00:00Z,0x1,0x.8,0x1p-149,1e2,1000",
    "2026-07-21T11:00:00Z,+2,3.,.25,101,1100",
)


def main() -> None:
    invalid = (
        '2026-07-21T10:00:00Z,"1",2,0.5,100,1000',
        "2026-07-21T10:00:00Z,1_0,2,0.5,100,1000",
        "2026-07-21T10:00:00Z, 1,2,0.5,100,1000",
        "2026-02-29T10:00:00Z,1,2,0.5,100,1000",
        "2026-07-21T10:00:00Z,1,2,0.5,nan,1000",
        "2026-07-21T10:00:00Z,1e-999,2,0.5,100,1000",
        "2026-07-21T10:00:00Z,1,2,0.5,100,1000\n\n",
        "x" * 512,
    )
    with tempfile.TemporaryDirectory(prefix="compose-mini-data-") as directory:
        path = Path(directory) / "bars.csv"
        path.write_bytes((CSV_HEADER + "\r\n" + "\r\n".join(VALID_ROWS)).encode("ascii"))
        assert list(read_csv(path)) == [
            f32(value) for value in
            (1.0, 0.5, 2.0 ** -149, 100.0, 1000.0,
             2.0, 3.0, 0.25, 101.0, 1100.0)
        ]
        timestamps, values = read_bars(path)
        assert timestamps == tuple(row.partition(",")[0] for row in VALID_ROWS)
        assert list(values) == list(read_csv(path))
        through_last = read_bars_through(path, timestamps[-1])
        assert through_last[0] == timestamps
        assert list(through_last[1]) == list(values)
        for reader in (read_bars_through, read_timestamps_through):
            for missing in (
                "2026-07-21T09:00:00Z",
                "2026-07-21T10:30:00Z",
                "2026-07-21T12:00:00Z",
            ):
                try:
                    reader(path, missing)
                except ValueError:
                    continue
                raise AssertionError(
                    f"missing cutoff {missing!r} was accepted"
                )
            try:
                reader(path, "invalid")
            except ValueError:
                pass
            else:
                raise AssertionError("noncanonical cutoff was accepted")
        with patch(
            "tools.data_v1._number",
            side_effect=AssertionError("numeric parser was called"),
        ):
            assert read_timestamps(path) == timestamps
            assert read_timestamps_through(path, timestamps[0]) == \
                timestamps[:1]
        prefix = f"{CSV_HEADER}\n{VALID_ROWS[0]}\n{timestamps[1]},".encode(
            "ascii"
        )
        for payload in (b"broken", b"\0", b"\xff", b"x" * (LINE_CAP + 1)):
            path.write_bytes(prefix + payload)
            bounded_timestamps, bounded_values = read_bars_through(
                path, timestamps[0]
            )
            assert bounded_timestamps == timestamps[:1]
            assert list(bounded_values) == list(values[:FEATURE_COUNT])
            assert read_timestamps_through(path, timestamps[0]) == \
                timestamps[:1]
            for stop in (timestamps[1], "2026-07-21T12:00:00Z"):
                for reader in (read_bars_through, read_timestamps_through):
                    try:
                        reader(path, stop)
                    except ValueError:
                        continue
                    raise AssertionError(
                        f"malformed row passed cutoff {stop!r}"
                    )
        for row in invalid:
            path.write_bytes((CSV_HEADER + "\n" + row).encode("ascii"))
            try:
                read_csv(path)
            except ValueError:
                continue
            raise AssertionError(f"invalid CSV was accepted: {row[:40]!r}")
        for value in (
            "wrong,header",
            f"{VALID_ROWS[0]}\n{VALID_ROWS[0]}",
            "2026-07-21T10:00:00Z,1,2,3,4",
            "2026-02-29T10:00:00Z,1,2,3,4,5",
        ):
            path.write_text(
                value if value == "wrong,header" else f"{CSV_HEADER}\n{value}",
                encoding="ascii",
            )
            try:
                read_timestamps(path)
            except ValueError:
                continue
            raise AssertionError("invalid timestamp CSV was accepted")
    print("training CSV tests passed")


if __name__ == "__main__":
    main()
