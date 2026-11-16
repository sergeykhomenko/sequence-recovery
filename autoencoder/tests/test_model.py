import numpy as np
import torch

from graph_ae.data import Graph, graph_tensors, make_batch, node_features
from graph_ae.model import GraphEncoder, build

AE = {"dim": 16, "gnn_layers": 2, "layers": 1, "heads": 2, "ffn": 32, "random_features": 4}


def graph(n, edges) -> Graph:
    src = np.array([a for a, _ in edges], dtype=np.int64)
    dst = np.array([b for _, b in edges], dtype=np.int64)
    nodes = [f"n{i}" for i in range(n)]
    return Graph(
        "g",
        "train",
        nodes,
        {v: i for i, v in enumerate(nodes)},
        src,
        dst,
        node_features(n, src, dst),
    )


def test_logits_cover_every_mask_and_forbid_kept_nodes():
    torch.manual_seed(0)
    model = build(AE, max_len=8).eval()
    g = graph_tensors(graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]))
    batch = make_batch([[0, 1, 2, 3], [2, 3, 4]], [[1, 2], [0]])
    logits = model(g, batch)
    assert logits.shape == (3, 6)
    # Window 0 keeps nodes 0 and 3, window 1 keeps 3 and 4.
    assert torch.isinf(logits[0, [0, 3]]).all() and torch.isinf(logits[1, [0, 3]]).all()
    assert torch.isinf(logits[2, [3, 4]]).all()
    assert torch.isfinite(logits[0, [1, 2, 4, 5]]).all()


def test_graph_encoder_ignores_node_order():
    # Without random features the encoder must be permutation-equivariant:
    # it may not read anything from the position of a node in the file.
    torch.manual_seed(0)
    enc = GraphEncoder(dim=8, layers=2, random_features=0).eval()
    edges = [(0, 1), (0, 2), (2, 3), (1, 3)]
    perm = [3, 0, 2, 1]  # new position of old node i
    moved = [(perm[a], perm[b]) for a, b in edges]
    h = enc(graph_tensors(graph(4, edges)))
    h_moved = enc(graph_tensors(graph(4, moved)))
    torch.testing.assert_close(h_moved[perm], h)
