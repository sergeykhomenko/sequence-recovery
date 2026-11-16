"""Fill the masks of every challenge with the trained model."""

from __future__ import annotations

import hashlib
import json
from itertools import groupby
from pathlib import Path

import torch

from .data import Experiment, challenge_window, graph_tensors, make_batch
from .decode import assign
from .model import GraphAE, build


def graph_seed(seed: int, gid: str) -> int:
    return int(hashlib.sha256(f"{seed}:{gid}".encode()).hexdigest()[:15], 16)


def load_model(exp: Experiment, device: torch.device) -> GraphAE:
    ckpt = torch.load(exp.model_dir / "model.pt", map_location=device, weights_only=False)
    model = build(ckpt["ae"], ckpt["max_len"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval()


@torch.no_grad()
def predict(exp: Experiment, device: torch.device) -> Path:
    cfg = exp.config()
    model = load_model(exp, device)
    challenges = exp.challenges()
    fills: dict[str, list[str]] = {}
    for gid, group in groupby(
        sorted(challenges, key=lambda c: c["graph"]), key=lambda c: c["graph"]
    ):
        graph = exp.graph(gid)
        nodes = model.encode_graph(
            graph_tensors(graph, device),
            torch.Generator().manual_seed(graph_seed(cfg["seed"], gid)),
        )
        group = list(group)
        for i in range(0, len(group), cfg["ae"]["batch"]):
            chunk = group[i : i + cfg["ae"]["batch"]]
            windows, masks = zip(*(challenge_window(c, graph) for c in chunk))
            logits = (
                model.decode(nodes, make_batch(windows, masks).to(device)).float().cpu().numpy()
            )
            at = 0
            for c, m in zip(chunk, masks):
                fills[c["id"]] = [graph.nodes[j] for j in assign(logits[at : at + len(m)])]
                at += len(m)
    exp.prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with open(exp.prediction_path, "w") as f:
        for c in challenges:
            f.write(json.dumps({"id": c["id"], "fill": fills[c["id"]]}) + "\n")
    return exp.prediction_path
