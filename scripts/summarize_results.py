"""Reanalyse archived window scores at the course level, without retraining.

Run from the repository root with `make analysis`. This is a post hoc analysis:
paired observations and bootstrap units are courses, not individual windows.
"""

from __future__ import annotations

import argparse
import csv
import gzip
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("epa", "oar", "uar", "pvr", "vwr")
OTHERS = ("markov-5-k10", "markov-5-k100", "markov-5-k1000")


def course_means(path: Path) -> dict[tuple[str, str], np.ndarray]:
    """Return means grouped by (method, graph), rejecting duplicate scores."""
    opener = gzip.open if path.suffix == ".gz" else open
    groups = defaultdict(list)
    seen = set()
    with opener(path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["method"], row["challenge"])
            if key in seen:
                raise ValueError(f"Duplicate method/challenge score: {key}")
            seen.add(key)
            values = np.array([float(row[name]) for name in METRICS])
            if not np.all(np.isfinite(values)) or np.any((values < 0) | (values > 1)):
                raise ValueError(f"Invalid score: {key}")
            groups[row["method"], row["graph"]].append(values)
    if not groups:
        raise ValueError("No scores found")
    return {key: np.mean(values, axis=0) for key, values in groups.items()}


def holm(pvalues: list[float]) -> np.ndarray:
    """Holm adjustment across all method-by-metric comparisons together."""
    values = np.asarray(pvalues)
    order = np.argsort(values)
    adjusted = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def compare(means: dict, resamples: int, seed: int) -> list[dict]:
    graphs = sorted(graph for method, graph in means if method == "ae")
    if not graphs:
        raise ValueError("No autoencoder scores found")
    reference = np.array([means["ae", graph] for graph in graphs])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(graphs), size=(resamples, len(graphs)))
    rows = []
    for other in OTHERS:
        other_graphs = sorted(graph for method, graph in means if method == other)
        if other_graphs != graphs:
            raise ValueError(f"Course IDs differ between ae and {other}")
        difference = reference - np.array([means[other, graph] for graph in graphs])
        for i, metric in enumerate(METRICS):
            delta = difference[:, i]
            low, high = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
            pvalue = (
                float(wilcoxon(delta, zero_method="wilcox", correction=False, method="auto").pvalue)
                if np.any(delta)
                else 1.0
            )
            oriented = -delta if metric == "pvr" else delta
            rows.append(
                {
                    "reference": "ae",
                    "other": other,
                    "metric": metric,
                    "courses": len(graphs),
                    "mean_difference": float(delta.mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "better": int(np.sum(oriented > 0)),
                    "worse": int(np.sum(oriented < 0)),
                    "tied": int(np.sum(oriented == 0)),
                    "p": pvalue,
                }
            )
    for row, adjusted in zip(rows, holm([row["p"] for row in rows])):
        row["p_holm"] = float(adjusted)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=ROOT / "results/base/metrics.csv.gz")
    parser.add_argument("--output", type=Path, default=ROOT / "out/analysis")
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.resamples < 1:
        parser.error("--resamples must be positive")
    means = course_means(args.metrics)
    comparisons = compare(means, args.resamples, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        {"method": method, "graph": graph, **dict(zip(METRICS, values.tolist()))}
        for (method, graph), values in sorted(means.items())
    ]
    write_csv(args.output / "course-summary.csv", rows)
    write_csv(args.output / "course-comparisons.csv", comparisons)
    print(f"Wrote {len(rows)} course summaries and {len(comparisons)} comparisons to {args.output}")


if __name__ == "__main__":
    main()
