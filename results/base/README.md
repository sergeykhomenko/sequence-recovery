# Base experiment archive

This archive contains the recorded scores underlying [RESULTS.md](../../RESULTS.md)
and the accompanying research manuscript. It supports inspection and
statistical reanalysis without rerunning the model training.

## Original run artifacts

| File | Contents |
|---|---|
| `config.json` | Configuration saved by the pipeline; matches `experiments/base.json` |
| `summary.csv` | Means and window-level 95% bootstrap intervals for every method and group |
| `comparisons.csv` | Original paired window-level Wilcoxon comparisons with Holm adjustment |
| `metrics.csv.gz` | All 136,000 per-window scores: 17 methods × 8000 challenges |
| `ae/train-log.jsonl` | Training loss and validation EPA through step 30,000 |

These files were supplied from the original `out/base/` run dated September 26,
2026 in the experiment materials and original filesystem timestamps. The log
itself contains step numbers, not timestamps. Compression of `metrics.csv` is
lossless; decompression yields the original bytes.

Hardware and elapsed-time descriptions are author-reported. No complete
runtime environment manifest was saved. The full base model was not retrained
during preparation of this publication.

## Reproducible post hoc tables

| File | Contents |
|---|---|
| `course-summary.csv` | 680 course-level rows: 17 methods × 40 courses |
| `course-comparisons.csv` | 15 course-level comparisons with paired bootstrap intervals |

These two files are generated from the original scores by
[`scripts/summarize_results.py`](../../scripts/summarize_results.py), using
10,000 bootstrap resamples and seed 1. The analysis is post hoc and does not
replace the original window-level reports.

From the repository root:

```bash
make verify-results   # verify SHA-256 and recompute the original summary means
make analysis         # write reanalysis to out/analysis/ without changing this archive
```

`SHA256SUMS` covers all seven data files listed above. CSV values in the original
reports have six decimal places; `0.000000` for a p-value is a rounded value,
not a claim of zero probability. In the original `comparisons.csv`, `n` counts
pairs with nonzero differences. In `course-comparisons.csv`, `courses` is the
full paired course count, including ties.

The generated course-level CSV tables use UTF-8 and LF line endings, matching
Git's text normalization. Checksums cover the bytes distributed in the repository.

Full generated graphs, trajectories, predictions, and trained model checkpoints
are not distributed. `make pipeline EXP=base` regenerates them in `out/base/`.
A new GPU training run is not guaranteed to match the archived model scores.
