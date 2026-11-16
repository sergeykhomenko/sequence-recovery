import json
import shutil

import pytest
import torch
from conftest import make_experiment

from graph_ae.data import Experiment
from graph_ae.train import train


class Stop(Exception):
    pass


def test_resume_continues_as_if_never_stopped(tmp_path):
    full = Experiment(make_experiment(tmp_path / "full", steps=40))
    train(full, torch.device("cpu"), log=lambda _: None)

    cut = Experiment(make_experiment(tmp_path / "cut", steps=40))

    def stop_at_24(rec):  # evaluations run every 8 steps; stop right after the third
        if rec["step"] == 24:
            raise Stop

    with pytest.raises(Stop):
        train(cut, torch.device("cpu"), log=stop_at_24)
    # Relocation and JSON formatting must not change the experiment's identity.
    shutil.copytree(cut.root, tmp_path / "relocated")
    cut = Experiment(tmp_path / "relocated")
    (cut.root / "config.json").write_text(json.dumps(cut.config(), indent=2, sort_keys=True))
    train(cut, torch.device("cpu"), log=lambda _: None, resume=True)

    a = torch.load(full.model_dir / "state.pt", weights_only=False)["model"]
    b = torch.load(cut.model_dir / "state.pt", weights_only=False)["model"]
    for name in a:
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    assert (cut.model_dir / "train-log.jsonl").read_text() == (
        full.model_dir / "train-log.jsonl"
    ).read_text()


@pytest.mark.parametrize(
    "change, message",
    [
        ("seed", "configuration differs"),
        ("max_length", "configuration differs"),
        ("ae-train.jsonl", "ae-train.jsonl"),
        ("ae-val.jsonl", "ae-val.jsonl"),
        ("orders/g0001.txt", "orders/g0001.txt"),
        ("orders/g0002.txt", "orders/g0002.txt"),
        ("graphs/g0001.json", "graphs/g0001.json"),
        ("graphs/g0002.json", "graphs/g0002.json"),
        ("legacy checkpoint", "no supported training identity"),
        ("device", "device type differs"),
    ],
)
def test_resume_rejects_incompatible_inputs_without_changing_output(tmp_path, change, message):
    exp = Experiment(make_experiment(tmp_path))

    def stop(rec):
        if rec["step"] == 2:
            raise Stop

    with pytest.raises(Stop):
        train(exp, torch.device("cpu"), log=stop)
    if change in ("seed", "max_length"):
        cfg = exp.config()
        if change == "seed":
            cfg["seed"] += 1
        else:
            cfg["windows"]["max_length"] += 1
        (exp.root / "config.json").write_text(json.dumps(cfg))
    elif change in ("legacy checkpoint", "device"):
        path = exp.model_dir / "state.pt"
        state = torch.load(path, weights_only=False)
        if change == "device":
            state["training_identity"]["device_type"] = "mps"
        else:
            del state["training_identity"]
        torch.save(state, path)
    else:
        path = exp.root / change
        path.write_bytes(path.read_bytes() + b"\n")
    before = {p.name: p.read_bytes() for p in exp.model_dir.iterdir()}
    with pytest.raises(ValueError, match=message):
        train(Experiment(exp.root), torch.device("cpu"), resume=True)
    assert {p.name: p.read_bytes() for p in exp.model_dir.iterdir()} == before


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_resume_on_mps(tmp_path):
    # Random generator states must come back on the CPU, whatever the training device.
    exp = Experiment(make_experiment(tmp_path, steps=20))

    def stop_at_8(rec):
        if rec["step"] == 8:
            raise Stop

    with pytest.raises(Stop):
        train(exp, torch.device("mps"), log=stop_at_8)
    train(exp, torch.device("mps"), log=lambda _: None, resume=True)
    steps = [
        json.loads(line)["step"]
        for line in (exp.model_dir / "train-log.jsonl").read_text().splitlines()
    ]
    assert steps == [4, 8, 12, 16, 20]


def test_learns_chains_and_transfers_to_an_unseen_one(tmp_path):
    # Train on one chain, select on another: in a chain the masked node is the one
    # whose prerequisite stands to its left, which the model can only learn
    # through the graph, never from the names.
    exp = Experiment(make_experiment(tmp_path, steps=400))
    best = train(exp, torch.device("cpu"), log=lambda _: None)
    assert best["val_epa"] >= 0.8
    log = [
        json.loads(line) for line in (exp.model_dir / "train-log.jsonl").read_text().splitlines()
    ]
    assert log[-1]["step"] == 400
    assert (exp.model_dir / "model.pt").exists()
