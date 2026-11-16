import json

import numpy as np
import pytest
import torch

from graph_ae.data import Experiment, challenge_window, load_graph, make_batch, node_features


def test_node_features_on_a_chain():
    # A -> B -> C: in/out degrees, entry flag, depth and height normalised by the longest path.
    f = node_features(3, np.array([0, 1]), np.array([1, 2]))
    want = np.array(
        [
            [0, np.log(2), 1, 0.0, 1.0],
            [np.log(2), np.log(2), 0, 0.5, 0.5],
            [np.log(2), 0, 0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(f, want, rtol=1e-6)


def test_features_follow_nodes_not_names_or_file_order(tmp_path):
    a = {
        "params": {"id": "a", "split": "train"},
        "nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}, {"id": "D"}],
        "edges": [{"from": "A", "to": "B"}, {"from": "A", "to": "C"}, {"from": "C", "to": "D"}],
    }
    rename = {"A": "q1", "B": "z9", "C": "k4", "D": "m0"}
    b = {
        "params": {"id": "b", "split": "train"},
        "nodes": [{"id": rename[n]} for n in ("D", "B", "C", "A")],
        "edges": [{"from": rename[e["from"]], "to": rename[e["to"]]} for e in reversed(a["edges"])],
    }
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(b))
    ga, gb = load_graph(tmp_path / "a.json"), load_graph(tmp_path / "b.json")
    for node in "ABCD":
        np.testing.assert_allclose(ga.features[ga.index[node]], gb.features[gb.index[rename[node]]])


def test_make_batch_orders_targets_like_masks():
    batch = make_batch([[3, 1, 4], [1, 5]], [[1], [0, 1]])
    assert batch.tokens.tolist() == [[3, 0, 4], [0, 0, 0]]
    assert batch.masked.tolist() == [[False, True, False], [True, True, False]]
    assert batch.padding.tolist() == [[False, False, False], [False, False, True]]
    assert batch.targets.tolist() == [1, 1, 5]


def test_challenge_window_marks_unknown_tokens(tiny):
    exp = Experiment(tiny)
    c = exp.challenges()[0]
    tokens, masked = challenge_window(c, exp.graph("g0003"))
    assert len(tokens) == len(c["window"])
    assert [tokens[i] for i in masked] == [-1] * len(masked)
    assert masked == [i for i, t in enumerate(c["window"]) if t is None]


def test_orders_of_test_graphs_are_off_limits(tiny):
    exp = Experiment(tiny)
    assert exp.orders("g0001").shape == (1, 12)
    with pytest.raises(ValueError):
        exp.orders("g0003")


def test_batch_moves_to_device():
    batch = make_batch([[1, 2]], [[0]]).to(torch.device("cpu"))
    assert batch.targets.tolist() == [1]
