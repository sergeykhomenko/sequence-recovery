import numpy as np

from graph_ae.decode import assign

NEG = -np.inf


def test_assign_resolves_conflicts_by_total_probability():
    # Both masks prefer node 1; giving it to mask 0 and node 2 to mask 1 costs
    # 0.0247 + 1.3266 in -log p, less than any other distinct pair. Node 3 is forbidden.
    logits = np.array([[0.0, 5.0, 1.0, NEG], [0.0, 4.0, 3.0, NEG]])
    assert assign(logits) == [1, 2]


def test_assign_is_distinct_and_avoids_forbidden_nodes():
    rng = np.random.default_rng(0)
    for _ in range(50):
        logits = rng.normal(size=(6, 40))
        logits[:, rng.choice(40, size=10, replace=False)] = NEG
        picked = assign(logits)
        assert len(set(picked)) == 6
        assert all(np.isfinite(logits[i, j]) for i, j in enumerate(picked))
