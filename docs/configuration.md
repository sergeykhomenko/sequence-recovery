# Configuration

[Back to README](../README.md)


An experiment is one JSON file in `experiments/`; unknown fields are rejected.

| Field | Default (`base.json`) | Meaning |
|---|---|---|
| `seed` | 1 | root of every random stream |
| `graphs.train / val / test` | 180 / 20 / 40 | graphs per split |
| `graphs.size` | 16–4096 | alphabet size range, drawn log-uniformly |
| `graphs.entries_ratio` | 0.02–0.05 | share of entry points |
| `graphs.min_prereq`, `graphs.max_prereq` | 1, 2–8 | prerequisites per symbol |
| `graphs.decay` | 0.7–0.9 | locality of prerequisites |
| `graphs.symbols` | `A–Z0–9`, length 3 | alphabet of symbol names (`^`, `$`, `_` are reserved) |
| `orders.markov`, `orders.ae` | 1000, 200 | stored trajectories per test / training graph |
| `orders.markov_budgets` | 10, 100, 1000 | Markov cold-start budgets K |
| `windows` | 8–256 | window length |
| `challenges_per_level` | 50 | challenges per level and test graph |
| `ae_windows` | 2000 / 200 | autoencoder windows per training / validation graph |
| `ae` | see [base.json](../experiments/base.json) | autoencoder size and training schedule |

`smoke.json` is a tiny configuration: once the dependencies are installed, it
is intended for a short end-to-end check; runtime depends on the machine.

To add an experiment, copy a configuration to `experiments/<name>.json`, set
its `name` field to `<name>`, and run `make pipeline EXP=<name>`. Use a new name
for a changed configuration to keep its files separate from earlier runs.

Individual Make targets assume that their input files already exist. Run the
full `pipeline` target to execute all stages in order, including under `make -j`.
Avoid starting two runs that write to the same output directory.
