#!/usr/bin/env python3
"""Verify the training reader enforces the runtime's literal CSV contract."""

from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.data_v1 import CSV_HEADER, read_csv
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
        for row in invalid:
            path.write_bytes((CSV_HEADER + "\n" + row).encode("ascii"))
            try:
                read_csv(path)
            except ValueError:
                continue
            raise AssertionError(f"invalid CSV was accepted: {row[:40]!r}")
    print("training CSV tests passed")


if __name__ == "__main__":
    main()
