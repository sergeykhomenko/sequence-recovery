"""Training loop (see docs/architecture.md): batches of windows from one graph at a time."""

from __future__ import annotations

import json
import math

import numpy as np
import torch
from torch.nn import functional as F

from .data import Experiment, graph_tensors, make_batch, train_window
from .model import GraphAE, build
from .provenance import check_resume_identity, training_identity


class Split:
    """Autoencoder windows of one split, grouped by graph."""

    def __init__(self, exp: Experiment, split: str, device: torch.device):
        by_graph: dict[str, list[dict]] = {}
        for w in exp.windows(split):
            by_graph.setdefault(w["graph"], []).append(w)
        self.graphs = {
            gid: (graph_tensors(exp.graph(gid), device), exp.orders(gid), ws)
            for gid, ws in by_graph.items()
        }

    def batches(self, gid: str, size: int):
        _, orders, ws = self.graphs[gid]
        for i in range(0, len(ws), size):
            yield make_batch(*zip(*(train_window(w, orders) for w in ws[i : i + size])))


def schedule(warmup: int, steps: int):
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, steps - warmup))))

    return factor


@torch.no_grad()
def evaluate(model: GraphAE, split: Split, batch_size: int, seed: int) -> float:
    """Share of masks whose most likely allowed node is the true one."""
    model.eval()
    hits = total = 0
    for gid in sorted(split.graphs):
        nodes = model.encode_graph(split.graphs[gid][0], torch.Generator().manual_seed(seed))
        for batch in split.batches(gid, batch_size):
            batch = batch.to(nodes.device)
            hits += (model.decode(nodes, batch).argmax(dim=1) == batch.targets).sum().item()
            total += batch.targets.numel()
    return hits / total


def device_rng(device: torch.device):
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None


def set_device_rng(device: torch.device, state) -> None:
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state)


def save(obj: dict, path) -> None:
    """Write through a temporary file, so a stop during saving keeps the old file."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def train(exp: Experiment, device: torch.device, log=print, resume: bool = False) -> dict:
    """Train on the train split, keep the best model on the val split in
    ae/model.pt. The full state goes to ae/state.pt at every evaluation;
    resume=True continues from it exactly where the stopped run left off."""
    cfg = exp.config()
    ae = cfg["ae"]
    state_path, log_path = exp.model_dir / "state.pt", exp.model_dir / "train-log.jsonl"
    identity = training_identity(exp, cfg, device.type)
    if resume:
        # Validate before constructing the model or modifying any existing output.
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        check_resume_identity(state.get("training_identity"), identity)
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = build(ae, cfg["windows"]["max_length"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=ae["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule(ae["warmup"], ae["steps"]))
    train_split, val_split = Split(exp, "train", device), Split(exp, "val", device)
    if not val_split.graphs:
        raise ValueError("the autoencoder needs val graphs to select its model")
    gids = sorted(train_split.graphs)

    exp.model_dir.mkdir(parents=True, exist_ok=True)
    best, start = {"val_epa": -1.0}, 1
    if resume:
        # Generator states must stay on the CPU; load_state_dict moves the rest to the device.
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        set_device_rng(device, state["device_rng"])
        best, start = state["best"], state["step"] + 1
        kept = [
            line
            for line in log_path.read_text().splitlines()
            if json.loads(line)["step"] <= state["step"]
        ]
        log_path.write_text("".join(line + "\n" for line in kept))

    with open(log_path, "a" if resume else "w") as logf:
        for step in range(start, ae["steps"] + 1):
            model.train()
            g, orders, ws = train_split.graphs[gids[rng.integers(len(gids))]]
            pick = rng.integers(len(ws), size=ae["batch"])
            batch = make_batch(*zip(*(train_window(ws[i], orders) for i in pick))).to(device)
            loss = F.cross_entropy(model(g, batch), batch.targets)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if step % ae["eval_every"] == 0 or step == ae["steps"]:
                rec = {
                    "step": step,
                    "loss": round(loss.item(), 5),
                    "val_epa": round(evaluate(model, val_split, ae["batch"], cfg["seed"]), 5),
                }
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                if rec["val_epa"] > best["val_epa"]:
                    best = rec
                    save(
                        {
                            "model": model.state_dict(),
                            "ae": ae,
                            "max_len": cfg["windows"]["max_length"],
                            "training_identity": identity,
                            **rec,
                        },
                        exp.model_dir / "model.pt",
                    )
                save(
                    {
                        "step": step,
                        "training_identity": identity,
                        "ae": ae,
                        "best": best,
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(),
                        "numpy_rng": rng.bit_generator.state,
                        "torch_rng": torch.get_rng_state(),
                        "device_rng": device_rng(device),
                    },
                    state_path,
                )
                log(rec)
    return best
