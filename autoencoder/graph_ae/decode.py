"""Joint assignment of symbols to the masks of one window: every node at most once."""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import log_softmax


def assign(logits: np.ndarray) -> list[int]:
    """One node per row of logits [m, N] (-inf = forbidden), all distinct,
    maximising the total log-probability."""
    logp = log_softmax(logits, axis=1)
    cost = np.where(np.isfinite(logp), -logp, np.inf)
    m, n = cost.shape
    # Some optimal assignment gives every mask one of its m cheapest nodes:
    # the other masks can take at most m-1 of them.
    cand = np.unique(np.argpartition(cost, m - 1, axis=1)[:, :m]) if n > m else np.arange(n)
    sub = cost[:, cand]
    rows, cols = linear_sum_assignment(np.where(np.isfinite(sub), sub, 1e9))
    out = [0] * m
    for r, c in zip(rows, cols):
        out[r] = int(cand[c])
    return out
