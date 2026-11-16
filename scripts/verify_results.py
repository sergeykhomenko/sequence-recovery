"""Verify archived file integrity and summary means against per-window scores."""

from __future__ import annotations

import csv
import gzip
import hashlib
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "results/base"
METRICS = ("epa", "oar", "uar", "pvr", "vwr")


def main() -> None:
    for line in (ARCHIVE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        actual = hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"Checksum mismatch: {name}")

    groups = defaultdict(list)
    seen = set()
    with gzip.open(ARCHIVE / "metrics.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            identity = (row["method"], row["challenge"])
            if identity in seen:
                raise ValueError(f"Duplicate score: {identity}")
            seen.add(identity)
            values = [float(row[m]) for m in METRICS]
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                raise ValueError(f"Invalid score: {identity}")
            size = int(row["size"])
            low = 1 << (size.bit_length() - 1)
            for bucket in (f"{low}-{2 * low - 1}", "all"):
                for level in (row["level"], "all"):
                    groups[row["method"], bucket, level].append(values)

    summaries = 0
    with (ARCHIVE / "summary.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["method"], row["bucket"], row["level"])
            values = groups.pop(key)
            if len(values) != int(row["n"]):
                raise ValueError(f"Wrong observation count: {key}")
            for i, metric in enumerate(METRICS):
                mean = math.fsum(value[i] for value in values) / len(values)
                # Both the input scores and published summary have six decimals.
                if abs(mean - float(row[f"{metric}_mean"])) > 1e-6:
                    raise ValueError(f"Summary mean mismatch: {key}, {metric}")
            summaries += 1
    if groups:
        raise ValueError("The summary omits groups from the per-window metrics")
    print(f"Verified checksums, {len(seen)} window scores, and {summaries} summary rows.")


if __name__ == "__main__":
    main()
