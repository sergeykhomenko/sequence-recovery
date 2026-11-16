# Experiment design

[Back to README](../README.md) · [Results](../RESULTS.md)

The current experiment compares random recovery, Markov chains, and a
graph-conditioned masked autoencoder on synthetic prerequisite graphs. All
methods receive the same damaged windows, but their information differs:
Markov sees trajectories of each test graph; the autoencoder sees its structure
and is trained on separate graphs.

Test challenge trajectories are sampled from independent random streams. They
are not appended to the stored training trajectories, but the generator does
not reject duplicate trajectories or windows. In graphs with few possible
topological orders, an independently sampled test trajectory can coincide with
a training trajectory. The train/validation/test split for the autoencoder is
by graph ID, without a graph-isomorphism deduplication check.


## Predicate alphabets

A graph over N symbols is generated as follows:

1. The first `entries` symbols are entry points with no prerequisites.
2. Every next symbol `i` gets `k ~ U[min_prereq, min(max_prereq, i)]`
   prerequisites.
3. The prerequisites are drawn without replacement among the earlier symbols
   `j < i`, with weight `decay^(i-1-j)`. Recent symbols are picked more often;
   `decay = 1` is a uniform choice.
4. Edges only go from earlier to later symbols, so the graph is acyclic.

Size and content of the alphabet float within one experiment. Every graph draws
its own parameters:

- size N, log-uniformly from 16 to 4096;
- the share of entry points, 2–5 % of N;
- `max_prereq`, from 2 to 8;
- `decay`, from 0.7 to 0.9.

The nodes are then renamed to random three-character codes over `A–Z0–9`, and
nodes and edges are shuffled in the file. Neither a symbol's name nor its
position in the file carries information about the generation order.

The default experiment has 180 training graphs and 20 validation graphs, both
used only by the autoencoder, and 40 test graphs.

## Trajectories

A trajectory is a random topological order of the graph: at each step the next
topic is drawn uniformly among the topics whose prerequisites are all done.
The experiment stores:

- 1000 trajectories of every test graph, for Markov;
- 200 trajectories of every training and validation graph, for autoencoder windows.

## Damage

A window of length `L ~ U{8 .. min(256, N)}` is cut from a trajectory at a
random offset. Its masked positions are known to every method. A gap is a run
of 1–4 masks, and neighbouring gaps are separated by at least one kept symbol.

| Level | Gaps | Masks `m` |
|---|---|---|
| easy | 1 | `m ~ U{1 .. max(1, min(4, ⌊0.05·L⌋))}` |
| medium | 2 | `m ~ U{2 .. max(2, min(8, ⌊0.05·L⌋))}` |
| hard | at least 2 when `m ≥ 2` | `ρ ~ U(0.05, 0.20]`, `m = max(1, round(ρ·L))` |
| extreme | at least 2 | `ρ ~ U(0.20, 0.50]`, `m = round(ρ·L)` |

Every test graph gets 50 challenges per level, so the default experiment has
8000 challenges.

## Methods

Every method follows the same rules:

- it fills only the masks and never changes a kept symbol;
- a symbol appears in a window at most once;
- every mask gets exactly one symbol.

**Random.** Fills the masks with distinct random symbols that are not in the
window yet.

**Markov, `markov-<n>-k<K>`.** Chains of orders 1..5 are trained with
[gomarkov](https://github.com/mb-14/gomarkov) on the first K stored trajectories
of the test graph, which models a cold start with little history. Masks are
filled left to right:

- each mask gets the most frequent unused successor of the last `n` symbols of
  the window;
- if the context is too short, unseen, or all of its successors are used, the
  model backs off to shorter contexts;
- if no order helps, it picks a random unused symbol;
- ties are broken with a seeded draw.

Only the left context is used, as in the classical approach.

**Autoencoder, `ae`.** A graph-conditioned masked autoencoder
(`autoencoder/graph_ae`):

- **Graph encoder.** Four message-passing layers that aggregate prerequisites
  and dependants separately. Node inputs are:
  - log in-degree and log out-degree;
  - an entry-point flag;
  - normalised depth and height in the graph;
  - 16 random features, redrawn on every pass so that structurally identical
    nodes stay distinguishable.

  Names and file order are never used.
- **Window encoder.** A transformer encoder (6 layers, d = 256, 8 heads) over
  the window. A kept symbol is represented by its node vector, a mask by a
  learned vector.
- **Output.** Every mask scores every node of the graph with a dot product.
  Nodes already in the window are excluded.
- **Training.** Cross-entropy on masks, using windows of the training graphs
  only. AdamW, lr 3·10⁻⁴, 1000 warm-up steps, cosine decay, 30 000 steps,
  batches of 64 windows of one graph. The checkpoint with the best accuracy on
  the validation graphs is kept.
- **Decoding.** A joint assignment of symbols to the masks of a window (Hungarian
  algorithm on log-probabilities), so symbols never repeat.

The autoencoder never reads trajectories of test graphs; `graph_ae` refuses to.
It sees a test graph only through its structure and the challenge window.

## Metrics

For a window with masks `M` (`m = |M|`), true symbols `t`, predicted symbols `p`
and the restored window `R`:

| Metric | Definition |
|---|---|
| EPA, exact position accuracy | share of masks with `p_i = t_i` |
| OAR, ordered accuracy | `LCS(t_M, p_M) / m` |
| UAR, unordered accuracy | size of the multiset intersection of `t_M` and `p_M`, divided by `m` |
| PVR, prerequisite violation rate | share of inserted symbols that have a transitive prerequisite to their right, or a transitive dependant to their left, in `R` |
| VWR, valid window rate | 1 if no inserted symbol violates a prerequisite, else 0 |

`EPA ≤ OAR ≤ UAR` always holds. A true window has `PVR = 0` and `VWR = 1`.
Markov rows also report `backoff`, the share of masks filled from a lower order
than asked, and `fallback`, the share of random fallbacks.

The evaluation writes three files:

- `metrics.csv`: one row per challenge and method.
- `summary.csv`: the mean and 95 % bootstrap interval of every metric, per
  method × alphabet size bucket (`[2^k, 2^(k+1)−1]`) × damage level, with
  `all` rows over buckets and levels.
- `comparisons.csv`: a two-sided paired Wilcoxon signed-rank test of the
  reference method (`-reference`, `ae` by default) against every other method
  on the same challenges, for EPA, OAR, UAR and PVR. It reports the median
  difference, z, p, the Holm-adjusted p within a metric, and the effect size
  `r = z/√n`, where n counts pairs with a nonzero difference.

## Statistical interpretation

Means and bootstrap intervals are calculated over challenge windows. Windows
from the same graph share structure, so these intervals are not graph-clustered
estimates of generalization to new courses. The paired tests also use windows
as their observations. Interpret their p-values within that scope; additional
training seeds are needed for broader claims. A post hoc course-level analysis
is included in [RESULTS.md](../RESULTS.md#course-level-robustness-analysis).

Validation EPA selects the checkpoint using independent per-mask argmax
predictions. Final test predictions use joint Hungarian decoding, so validation
and test EPA are not calculated with identical decoding procedures.
