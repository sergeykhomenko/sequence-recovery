"""python -m graph_ae {train,predict,bench} --experiment DIR [--device auto|cpu|mps|cuda]"""

import argparse
import json

import torch

from .bench import bench
from .data import Experiment
from .predict import predict
from .train import train


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="graph_ae")
    parser.add_argument("command", choices=["train", "predict", "bench"])
    parser.add_argument(
        "--experiment", required=True, help="experiment directory written by the Go pipeline"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true", help="train: continue from ae/state.pt")
    args = parser.parse_args(argv)
    exp, device = Experiment(args.experiment), pick_device(args.device)
    print(f"device: {device}", flush=True)
    if args.command == "train":
        best = train(
            exp, device, log=lambda rec: print(json.dumps(rec), flush=True), resume=args.resume
        )
        print(json.dumps(best))
    elif args.command == "predict":
        print(f"wrote {predict(exp, device)}")
    else:
        print(json.dumps(bench(exp, device)))


if __name__ == "__main__":
    main()
