# Reproducing the experiment

[README](../README.md) · [Configuration](configuration.md) · [Results](../RESULTS.md)

## Three levels of reproduction

1. **Inspect and reanalyse the recorded run.** `make verify-results` checks the
   report checksums and aggregated means. `make analysis` recomputes course
   summaries and paired comparisons from the compressed per-window metrics,
   without training a model. `make figures` recreates the publication figures.
2. **Check the software end to end.** `make check`, `make test`, and
   `make smoke DEVICE=cpu` exercise the current code on a small experiment.
3. **Rerun the base experiment.** `make pipeline EXP=base` regenerates data,
   trains the models, reconstructs challenges, and evaluates the predictions.
   This is a new training run, not a bit-exact replay of the archived GPU run.

All commands run from the repository root. Python and its dependencies are
managed by uv; the optional `docs` dependency group is needed only for figures.

## Outputs and resources

Each experiment writes to `out/<name>/`. `pipeline` executes `data`, `markov`,
`recover`, `ae`, and `evaluate` in order, including when Make runs with `-j`.
Individual targets expect their input files to exist. Use a new experiment
name for a changed configuration and avoid concurrent writers to one directory.
Dataset generation checks an existing directory before writing: it accepts the
same configuration, but refuses a different configuration or a nonempty directory
without a readable experiment configuration. Existing results are left intact on
refusal. Configuration values are compared, ignoring JSON whitespace and key order.

For the Make pipeline, select a separate configuration with `EXP=<name>`. The
dataset command also accepts an explicit destination, for example:

```bash
go run ./cmd/dataset -config experiments/base.json -out out/base-repeat
```

That command generates data only; subsequent commands must use the same output
directory. Do not edit the saved `out/<name>/config.json` to repurpose a run.

The base run uses 240 graphs with requested sizes from 16 to 4096 nodes,
8000 challenges, and 30,000 training steps. The original author-reported MPS
run took approximately three hours and occupied about 1.5 GB. CPU execution is
supported but can be substantially slower. Run `make data EXP=base` followed
by `make ae-bench EXP=base` to estimate the training cost on your device.

`DEVICE=auto` selects CUDA, then Apple MPS, then CPU. Set `DEVICE=cpu`,
`DEVICE=cuda`, or `DEVICE=mps` explicitly when required.

## Resume

Training saves model, optimizer, learning-rate schedule, and random-generator
state at every evaluation. After interruption:

```bash
make ae-resume EXP=base
make evaluate EXP=base
```

Keep the original experiment data, configuration, device type, and environment.
New checkpoints store a versioned training identity: a SHA-256 of the complete
configuration, SHA-256 hashes of both window files and the graph/trajectory files
used by training and validation, and the device type. Resume checks this identity
before modifying logs or checkpoints and names changed inputs when it refuses.
JSON configuration whitespace and key order do not matter; data file bytes do.
Moving an unchanged experiment directory is supported. Test trajectories and
answers are never read for this check.

Older checkpoints without this identity cannot be verified and are refused for
resume. Preserve them; start a new run in a separate directory if needed. They
remain usable for inference. These checks do not verify software versions or
make GPU execution deterministic; keep the same locked environment.
The CPU regression test compares resumed and uninterrupted model states bit for
bit. GPU kernels are not guaranteed deterministic.

## Provenance and limits

The archived configuration, window-level reports, training log, and compressed
metrics come from the original base run described in the research manuscript.
Course-level tables are reproducible post hoc analyses of those scores. See
[the archive manifest](../results/base/README.md) for exact file provenance.

The original full runtime manifest and model checkpoint are not distributed.
Thus the archive supports score reanalysis and rerunning the protocol, but does
not reproduce the exact original trained weights. The full base model was not
retrained during repository preparation.

Dataset generation uses seeds derived from stable stream labels. Go modules
are declared in `go.mod` and verified with `go.sum`. Python dependencies are
locked in `autoencoder/uv.lock`; all Make commands use `uv run --locked`.
This release retains the experiment's algorithm, base configuration, and
original reported numbers; publication changes concern packaging, formatting,
documentation, checks, and standalone result analysis.
