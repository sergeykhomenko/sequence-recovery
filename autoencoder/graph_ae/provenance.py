"""Identify the configuration and input bytes used by a training checkpoint."""

from __future__ import annotations

import hashlib
import json

from .data import Experiment


def training_identity(exp: Experiment, config: dict, device_type: str) -> dict:
    """Hash all inputs read by training and validation, independent of root path.

    Configuration formatting is ignored; data files are compared byte for byte.
    Test trajectories and answers are not read or included in this identity.
    """
    paths = {"ae-train.jsonl", "ae-val.jsonl"}
    for split in ("train", "val"):
        for gid in {window["graph"] for window in exp.windows(split)}:
            paths.add(f"graphs/{gid}.json")
            paths.add(f"orders/{gid}.txt")
    files = {}
    for name in sorted(paths):
        with (exp.root / name).open("rb") as stream:
            files[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        "version": 1,
        "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "files": files,
        "device_type": device_type,
    }


def check_resume_identity(saved: dict | None, current: dict) -> None:
    """Refuse unverified resumes before modifying logs or checkpoints."""
    if not saved or saved.get("version") != current["version"]:
        raise ValueError(
            "Cannot resume: checkpoint has no supported training identity. "
            "Keep the old checkpoint; start a new run in a separate experiment directory."
        )
    if saved.get("config_sha256") != current["config_sha256"]:
        raise ValueError("Cannot resume: experiment configuration differs from the checkpoint.")
    if saved.get("files") != current["files"]:
        previous = saved.get("files", {})
        changed = sorted(
            name
            for name in previous.keys() | current["files"].keys()
            if previous.get(name) != current["files"].get(name)
        )
        raise ValueError("Cannot resume: training/validation inputs changed: " + ", ".join(changed))
    if saved.get("device_type") != current["device_type"]:
        raise ValueError("Cannot resume: device type differs from the checkpoint.")
