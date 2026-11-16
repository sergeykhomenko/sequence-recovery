"""Time training steps on the largest training graph to size the step budget."""

from __future__ import annotations

import time

import torch
from torch.nn import functional as F

from .data import Experiment, make_batch, train_window
from .model import build
from .train import Split


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def bench(exp: Experiment, device: torch.device, steps: int = 20) -> dict:
    cfg = exp.config()
    ae = cfg["ae"]
    split = Split(exp, "train", device)
    gid = max(split.graphs, key=lambda k: split.graphs[k][0].size)
    g, orders, ws = split.graphs[gid]
    longest = sorted(ws, key=lambda w: -w["length"])[: ae["batch"]]
    batch = make_batch(*zip(*(train_window(w, orders) for w in longest))).to(device)
    model = build(ae, cfg["windows"]["max_length"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=ae["lr"])

    def step() -> None:
        loss = F.cross_entropy(model(g, batch), batch.targets)
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(3):
        step()
    sync(device)
    start = time.perf_counter()
    for _ in range(steps):
        step()
    sync(device)
    ms = (time.perf_counter() - start) / steps * 1000
    return {
        "graph": gid,
        "nodes": g.size,
        "batch": len(longest),
        "ms_per_step": round(ms, 1),
        "minutes_for_budget_at_this_size": round(ms * ae["steps"] / 60000, 1),
    }
