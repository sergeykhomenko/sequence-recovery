"""Reads an experiment directory written by the Go pipeline (see internal/store)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

STATIC_FEATURES = 5  # log(1+in-degree), log(1+out-degree), entry flag, depth, height


@dataclass
class Graph:
    id: str
    split: str
    nodes: list[str]  # node IDs in file order
    index: dict[str, int]
    src: np.ndarray  # prerequisite end of every edge
    dst: np.ndarray  # dependant end of every edge
    features: np.ndarray  # [N, STATIC_FEATURES]


def load_graph(path: Path) -> Graph:
    raw = json.loads(Path(path).read_text())
    nodes = [n["id"] for n in raw["nodes"] or []]
    index = {n: i for i, n in enumerate(nodes)}
    edges = raw["edges"] or []
    src = np.array([index[e["from"]] for e in edges], dtype=np.int64)
    dst = np.array([index[e["to"]] for e in edges], dtype=np.int64)
    params = raw["params"]
    return Graph(
        params["id"], params["split"], nodes, index, src, dst, node_features(len(nodes), src, dst)
    )


def node_features(n: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    indeg = np.bincount(dst, minlength=n)
    outdeg = np.bincount(src, minlength=n)
    depth = longest_paths(n, src, dst)
    height = longest_paths(n, dst, src)
    return np.stack(
        [
            np.log1p(indeg),
            np.log1p(outdeg),
            (indeg == 0).astype(np.float64),
            depth / max(depth.max(initial=0), 1),
            height / max(height.max(initial=0), 1),
        ],
        axis=1,
    ).astype(np.float32)


def longest_paths(n: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Length of the longest src -> dst path ending at every node."""
    out: list[list[int]] = [[] for _ in range(n)]
    left = np.zeros(n, dtype=np.int64)
    for s, d in zip(src.tolist(), dst.tolist()):
        out[s].append(d)
        left[d] += 1
    dist = np.zeros(n)
    queue = [v for v in range(n) if left[v] == 0]
    for v in queue:  # grows while we walk it: Kahn's order
        for w in out[v]:
            dist[w] = max(dist[w], dist[v] + 1)
            left[w] -= 1
            if left[w] == 0:
                queue.append(w)
    return dist


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class Experiment:
    """File layout of one experiment directory, mirroring the Go store package."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._graphs: dict[str, Graph] = {}

    def config(self) -> dict:
        return json.loads((self.root / "config.json").read_text())

    def graph(self, gid: str) -> Graph:
        if gid not in self._graphs:
            self._graphs[gid] = load_graph(self.root / "graphs" / f"{gid}.json")
        return self._graphs[gid]

    def orders(self, gid: str) -> np.ndarray:
        """Stored trajectories of a train or val graph as node indices [K, N]."""
        g = self.graph(gid)
        if g.split not in ("train", "val"):
            raise ValueError(
                f"{gid}: trajectories of {g.split} graphs are off limits for the autoencoder"
            )
        lines = (self.root / "orders" / f"{gid}.txt").read_text().splitlines()
        return np.array(
            [[g.index[t] for t in line.split()] for line in lines if line], dtype=np.int64
        )

    def windows(self, split: str) -> list[dict]:
        return read_jsonl(self.root / f"ae-{split}.jsonl")

    def challenges(self) -> list[dict]:
        return read_jsonl(self.root / "challenges.jsonl")

    @property
    def model_dir(self) -> Path:
        return self.root / "ae"

    @property
    def prediction_path(self) -> Path:
        return self.root / "predictions" / "ae.jsonl"


@dataclass
class GraphTensors:
    features: torch.Tensor  # [N, STATIC_FEATURES]
    src: torch.Tensor  # [E]
    dst: torch.Tensor  # [E]
    in_norm: torch.Tensor  # [N, 1] = 1 / max(1, prerequisites)
    out_norm: torch.Tensor  # [N, 1] = 1 / max(1, dependants)

    @property
    def size(self) -> int:
        return self.features.shape[0]


def graph_tensors(g: Graph, device: torch.device | str = "cpu") -> GraphTensors:
    n = len(g.nodes)
    indeg = np.maximum(np.bincount(g.dst, minlength=n), 1)
    outdeg = np.maximum(np.bincount(g.src, minlength=n), 1)

    def t(a, dtype):
        return torch.as_tensor(a, dtype=dtype, device=device)

    return GraphTensors(
        t(g.features, torch.float32),
        t(g.src, torch.long),
        t(g.dst, torch.long),
        t(1 / indeg, torch.float32).unsqueeze(1),
        t(1 / outdeg, torch.float32).unsqueeze(1),
    )


@dataclass
class Batch:
    tokens: torch.Tensor  # [B, L] node index of kept tokens, 0 at masks and padding
    masked: torch.Tensor  # [B, L] True at masks
    padding: torch.Tensor  # [B, L] True past the end of a window
    targets: torch.Tensor  # [M] true node of every mask in row-major order, -1 if unknown

    def to(self, device: torch.device | str) -> Batch:
        return Batch(
            self.tokens.to(device),
            self.masked.to(device),
            self.padding.to(device),
            self.targets.to(device),
        )


def make_batch(windows, masked) -> Batch:
    """windows hold node indices (-1 where the truth is unknown); masked lists mask positions."""
    size, length = len(windows), max(len(w) for w in windows)
    tokens = torch.zeros(size, length, dtype=torch.long)
    mask = torch.zeros(size, length, dtype=torch.bool)
    padding = torch.ones(size, length, dtype=torch.bool)
    targets: list[int] = []
    for b, (w, m) in enumerate(zip(windows, masked)):
        tokens[b, : len(w)] = torch.as_tensor(list(w)).clamp(min=0)
        padding[b, : len(w)] = False
        mask[b, list(m)] = True
        targets.extend(w[p] for p in sorted(m))
    tokens[mask] = 0
    return Batch(tokens, mask, padding, torch.as_tensor(targets, dtype=torch.long))


def train_window(w: dict, orders: np.ndarray) -> tuple[list[int], list[int]]:
    return orders[w["order"], w["start"] : w["start"] + w["length"]].tolist(), w["masked"]


def challenge_window(c: dict, g: Graph) -> tuple[list[int], list[int]]:
    window = c["window"]
    tokens = [-1 if t is None else g.index[t] for t in window]
    return tokens, [i for i, t in enumerate(window) if t is None]
