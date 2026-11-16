import json

import torch

from graph_ae.data import Experiment
from graph_ae.predict import predict
from graph_ae.train import train


def test_predict_fills_every_mask_with_distinct_unused_symbols(tiny):
    exp = Experiment(tiny)
    train(exp, torch.device("cpu"), log=lambda _: None)
    path = predict(exp, torch.device("cpu"))
    preds = [json.loads(line) for line in path.read_text().splitlines()]
    challenges = exp.challenges()
    assert [p["id"] for p in preds] == [c["id"] for c in challenges]
    nodes = set(exp.graph("g0003").nodes)
    for p, c in zip(preds, challenges):
        kept = {t for t in c["window"] if t is not None}
        assert len(p["fill"]) == c["window"].count(None)
        assert len(set(p["fill"])) == len(p["fill"])
        assert set(p["fill"]) <= nodes - kept


def test_predict_is_deterministic(tiny):
    exp = Experiment(tiny)
    train(exp, torch.device("cpu"), log=lambda _: None)
    first = predict(exp, torch.device("cpu")).read_text()
    assert predict(exp, torch.device("cpu")).read_text() == first
