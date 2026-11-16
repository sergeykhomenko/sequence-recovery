# Contributing

Use a focused branch and explain changes to the experiment's assumptions,
configuration, or output format. Keep generated datasets, environments, and
model checkpoints out of Git. Small research summaries belong in `results/`
with their configuration and provenance.

## Checks

Install the tools listed in [README.md](README.md), then run from the root:

```bash
make check
make test
make smoke DEVICE=cpu
```

Use `gofmt -w cmd internal` for Go formatting. For Python, run
`uv run --project autoencoder ruff format autoencoder scripts`. `make test-go` and
`make test-python` run the test suites independently. Add regression coverage
when changing algorithms, file formats, or recovery behavior.

Python dependencies are managed with `uv`. When deliberately changing them,
update `autoencoder/pyproject.toml` and `autoencoder/uv.lock` together, then
rerun the checks. Normal test and experiment commands use the existing lock.

## Commits

Write commit subjects in English using this form:

```text
<type>[optional scope]: <short imperative description>
```

Examples:

```text
feat: add graph-based sequence recovery experiments
fix: run experiment stages in dependency order
docs: explain experiment assumptions and reproduction
chore: exclude generated experiment artifacts
test: cover recovery with consecutive masks
ci: run unit tests and the smoke experiment
```

Use `refactor:` for restructuring without behavior changes and `perf:` for
performance improvements. Keep unrelated changes in separate commits. Do not rewrite a published release tag; archive a new version for changes
after publication.

## Reporting results

Include the configuration, seed, method budgets, training log, software
versions, and hardware when recording a new experiment. Distinguish original
reports from reruns, and state whether intervals are computed over windows or
graphs. Do not update result tables without checking the underlying reports.
