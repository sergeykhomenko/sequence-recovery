import json
import random
from pathlib import Path

import pytest


def chain(n: int, prefix: str) -> tuple[list[str], list[tuple[str, str]]]:
    nodes = [f"{prefix}{i:02d}" for i in range(n)]
    return nodes, list(zip(nodes, nodes[1:]))


def write_graph(
    root: Path, gid: str, split: str, nodes: list[str], edges: list[tuple[str, str]]
) -> None:
    data = {
        "params": {"id": gid, "split": split},
        "nodes": [{"id": n, "entry": i == 0} for i, n in enumerate(nodes)],
        "edges": [{"from": a, "to": b} for a, b in edges],
    }
    (root / "graphs" / f"{gid}.json").write_text(json.dumps(data))


def windows(rng: random.Random, gid: str, n: int, count: int) -> list[dict]:
    out = []
    for _ in range(count):
        length = rng.randint(8, n)
        masked = sorted(rng.sample(range(length), rng.randint(1, 3)))
        out.append(
            {
                "graph": gid,
                "order": 0,
                "start": rng.randint(0, n - length),
                "length": length,
                "level": "hard",
                "masked": masked,
            }
        )
    return out


def make_experiment(root: Path, steps: int = 5) -> Path:
    """Chains: train g0001 (12 nodes), val g0002 (10), test g0003 (9).
    A chain has a single trajectory, the chain itself."""
    rng = random.Random(0)
    for d in ("graphs", "orders", "predictions"):
        (root / d).mkdir(parents=True, exist_ok=True)
    ae = {
        "dim": 32,
        "gnn_layers": 2,
        "layers": 2,
        "heads": 4,
        "ffn": 64,
        "random_features": 8,
        "batch": 16,
        "steps": steps,
        "lr": 0.003,
        "warmup": 2,
        "eval_every": max(1, steps // 5),
    }
    (root / "config.json").write_text(
        json.dumps({"seed": 3, "windows": {"min_length": 8, "max_length": 12}, "ae": ae})
    )
    for gid, split, n, prefix in (
        ("g0001", "train", 12, "t"),
        ("g0002", "val", 10, "v"),
        ("g0003", "test", 9, "x"),
    ):
        nodes, edges = chain(n, prefix)
        write_graph(root, gid, split, nodes, edges)
        if split != "test":
            (root / "orders" / f"{gid}.txt").write_text(" ".join(nodes) + "\n")
    with open(root / "ae-train.jsonl", "w") as f:
        f.writelines(json.dumps(w) + "\n" for w in windows(rng, "g0001", 12, 200))
    with open(root / "ae-val.jsonl", "w") as f:
        f.writelines(json.dumps(w) + "\n" for w in windows(rng, "g0002", 10, 40))
    test_nodes, _ = chain(9, "x")
    with open(root / "challenges.jsonl", "w") as f:
        for k in range(6):
            window = list(test_nodes[k % 2 : k % 2 + 8])
            for p in sorted(rng.sample(range(8), 2)):
                window[p] = None
            f.write(
                json.dumps(
                    {
                        "id": f"g0003-hard-{k:03d}",
                        "graph": "g0003",
                        "level": "hard",
                        "window": window,
                    }
                )
                + "\n"
            )
    return root


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    return make_experiment(tmp_path)
