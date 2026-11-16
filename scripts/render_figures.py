"""Render publication figures. Rebuild with `make figures` from the repository root."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
INK = "#172f43"
MUTED = "#526878"
BLUE = "#3169a0"
TEAL = "#087e83"
LINE = "#b9c8d1"

plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "svg.fonttype": "none",
        "svg.hashsalt": "sequence-recovery-v1",
        "figure.facecolor": "white",
    }
)


def canvas(width, height, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=0.025, right=0.975, top=0.97, bottom=0.025)
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.axis("off")
    return fig, ax


def text(ax, x, y, label, size=12, **kwargs):
    ax.text(x, y, label, ha="center", va="center", fontsize=size, **kwargs)


def block(ax, x, y, w, h, label, fill="white", size=12, edge=INK):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, lw=0.85, zorder=3))
    text(ax, x + w / 2, y + h / 2, label, size=size, linespacing=1.4, zorder=4)


def arrow(ax, points, color=INK, lw=0.9):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=2)
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=11,
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def graph(ax, x, y, scale=1):
    pos = {"A": (0, 8), "B": (0, 0), "C": (7, 4), "D": (14, 8), "E": (14, 0), "F": (21, 4)}
    edges = [("A", "C"), ("B", "C"), ("C", "D"), ("C", "E"), ("D", "F"), ("E", "F")]
    r = 1.65 * scale
    for u, v in edges:
        a = np.array(pos[u]) * scale + np.array([x, y])
        b = np.array(pos[v]) * scale + np.array([x, y])
        unit = (b - a) / np.linalg.norm(b - a)
        arrow(ax, [a + r * unit, b - r * unit], color=MUTED)
    for label, (px, py) in pos.items():
        center = (x + px * scale, y + py * scale)
        ax.add_patch(
            Circle(
                center,
                r,
                facecolor="#edf5f4" if label in "AB" else "white",
                edgecolor=INK,
                lw=0.85,
                zorder=4,
            )
        )
        text(ax, *center, label, size=11, zorder=5)


def sequence(ax, x, y, labels, cell=3.8):
    for i, label in enumerate(labels):
        mask = label == "?"
        block(ax, x + i * cell, y, cell, 4.5, label, fill="#e4efed" if mask else "white", size=12)


def matrix(ax, x, y, rows, cols, w, h, masked=False):
    values = np.random.default_rng(4).uniform(0.12, 0.8, (rows, cols))
    for r in range(rows):
        for c in range(cols):
            forbidden = masked and c in (0, 2, 4, 5)
            color = "#e4e6e8" if forbidden else plt.cm.GnBu(0.15 + values[r, c] * 0.55)
            ax.add_patch(
                Rectangle(
                    (x + c * w / cols, y + (rows - 1 - r) * h / rows),
                    w / cols,
                    h / rows,
                    facecolor=color,
                    edgecolor="white",
                    lw=0.5,
                    zorder=3,
                    hatch="//" if forbidden else None,
                )
            )
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=INK, lw=0.75, zorder=4))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, dpi=180, metadata={"Date": None} if ext == "svg" else {})
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)


def pipeline():
    fig, ax = canvas(120, 67, (14, 7.6))
    text(ax, 60, 63, "Experimental design", size=22)
    text(ax, 20, 56, "Prerequisite graphs", size=15)
    graph(ax, 7, 40, 1.15)
    text(ax, 20, 34, "180 training · 20 validation · 40 test", size=11)
    arrow(ax, [(35, 46), (42, 46)])
    text(ax, 58, 56, "Topological trajectories", size=15)
    sequence(ax, 45, 44, list("ABCDEF"), 4)
    text(ax, 57, 38, "Random choice among available topics", size=11)
    arrow(ax, [(71, 46), (78, 46)])
    text(ax, 96, 56, "Masked windows", size=15)
    sequence(ax, 83, 44, ["A", "?", "C", "?", "E", "F"], 4)
    text(ax, 95, 38, "8000 challenges · four damage levels", size=11)
    # Shared challenges are explicitly routed to each recovery method.
    arrow(ax, [(95, 34), (95, 29), (20, 29), (20, 25)])
    arrow(ax, [(60, 29), (60, 25)])
    arrow(ax, [(100, 29), (100, 25)])
    ax.plot([95, 100], [29, 29], color=INK, lw=0.9)
    text(ax, 60, 31.5, "Identical recovery challenges", size=11, color=MUTED)
    block(ax, 4, 15, 32, 10, "Random baseline\nAlphabet only", fill="#f7f8f8", size=13)
    block(
        ax,
        44,
        15,
        32,
        10,
        "Markov chains, orders 1–5\nK = 10, 100, 1000 trajectories",
        fill="#f4f7fa",
        size=13,
    )
    block(
        ax,
        84,
        15,
        32,
        10,
        "Graph-conditioned autoencoder\nGraph + training on other courses",
        fill="#edf5f4",
        size=13,
    )
    for x in (20, 60, 100):
        arrow(ax, [(x, 15), (x, 10)])
    block(
        ax,
        4,
        3,
        112,
        7,
        "Shared evaluation: position, order, and set accuracy · prerequisite violations · valid windows",
        size=13,
    )
    text(
        ax,
        60,
        0.6,
        "Synthetic courses; independently sampled trajectories are not explicitly deduplicated.",
        size=10,
        color=MUTED,
    )
    save(fig, "experiment-overview")


def architecture():
    fig, ax = canvas(120, 82, (15, 10.2))
    text(ax, 60, 79, "Graph-conditioned masked autoencoder", size=23)
    ax.text(2, 74, "(a)  Direction-aware graph encoder", fontsize=15, weight="bold", va="center")
    graph(ax, 5, 60, 1)
    text(ax, 15.5, 55, r"Prerequisite graph $G=(V,E)$", size=12)
    arrow(ax, [(29, 64), (35, 64)])
    matrix(ax, 36, 58, 7, 5, 13, 13)
    text(ax, 42.5, 73, r"$X\in\mathbb{R}^{N\times21}$", size=13)
    text(ax, 42.5, 54, "5 structural + 16 random features", size=10)
    arrow(ax, [(49, 64), (56, 64)])
    # One repeated message-passing layer with distinct directional means.
    ax.add_patch(Rectangle((56, 56), 39, 16, facecolor="none", edgecolor=MUTED, lw=0.8))
    text(ax, 75.5, 73.5, "Linear projection + 4 graph layers", size=12)
    block(ax, 58, 65, 16, 5, "Prerequisite mean", fill="#f2f5f8", size=11)
    block(ax, 77, 65, 16, 5, "Dependant mean", fill="#edf5f4", size=11)
    arrow(ax, [(66, 65), (66, 62)])
    arrow(ax, [(85, 65), (85, 62)])
    block(ax, 59, 57, 33, 5, "Concatenate → MLP → residual + norm", size=11)
    arrow(ax, [(95, 64), (102, 64)])
    matrix(ax, 103, 58, 7, 7, 12, 13)
    text(ax, 109, 73, r"$H\in\mathbb{R}^{N\times256}$", size=13)
    text(ax, 109, 55, "Node embeddings", size=11)
    text(
        ax,
        67,
        49.6,
        r"$h_v^{(\ell+1)}=\mathrm{LN}\!\left(h_v^{(\ell)}+\mathrm{MLP}\,[h_v^{(\ell)}\,\Vert\,\overline{h}_{\mathrm{pre}(v)}\,\Vert\,\overline{h}_{\mathrm{dep}(v)}]\right)$",
        size=16,
    )
    # Shared node embeddings supply input symbol vectors and output keys.
    arrow(ax, [(109, 53), (109, 44.5), (39, 44.5), (39, 38)], color=TEAL)
    arrow(ax, [(91, 44.5), (91, 35)], color=TEAL)
    text(ax, 64, 46.1, "Shared node embeddings", size=11, color=TEAL)
    ax.text(2, 43, "(b)  Window reconstruction", fontsize=15, weight="bold", va="center")
    sequence(ax, 2, 29, ["A", "?", "C", "?", "E", "F"], 3.4)
    text(ax, 12.2, 25, "Observed window", size=12)
    text(ax, 12.2, 21.5, "? denotes a masked position", size=10, color=MUTED)
    arrow(ax, [(23, 31.25), (30, 31.25)])
    block(
        ax, 30, 25, 18, 13, "Node lookup\nLearned mask vector\n+ position", fill="#f3f7f6", size=12
    )
    text(ax, 47, 40.5, r"$H$ lookup", size=11, color=TEAL)
    arrow(ax, [(48, 31.25), (55, 31.25)])
    # Offsets denote repeated transformer layers; no decorative flow arrows.
    for offset in (2, 1):
        ax.add_patch(
            Rectangle(
                (55 + offset, 25 + offset),
                20,
                13,
                facecolor="white",
                edgecolor=LINE,
                lw=0.75,
                zorder=2,
            )
        )
    block(
        ax, 55, 25, 20, 13, "Self-attention\nFeed-forward\nResidual + norm", fill="#f3f6f9", size=12
    )
    text(ax, 66, 41.3, "Transformer × 6", size=12)
    text(ax, 65, 22.4, "8 heads · dimension 256", size=11)
    arrow(ax, [(75, 31.25), (83, 31.25)])
    text(ax, 79, 34.2, r"$Q_M$", size=13)
    matrix(ax, 84, 27, 2, 6, 14, 8, masked=True)
    text(ax, 99, 40.5, r"$K=HW_K$", size=12, color=TEAL)
    text(ax, 91, 23.2, r"$Q_M K^{\mathsf{T}}/\sqrt{256}$", size=14)
    text(ax, 91, 19.6, "Exclude observed symbols", size=10)
    arrow(ax, [(98, 31.25), (103, 31.25)])
    block(ax, 103, 26, 15, 11, "Hungarian\nassignment", fill="#edf5f4", size=13)
    arrow(ax, [(110.5, 26), (110.5, 18)])
    sequence(ax, 100, 13, list("ABCDEF"), 3)
    text(ax, 109, 10.3, "Reconstructed window", size=11)
    # Training objective and decoding constraints are deliberately distinguished.
    ax.plot([2, 118], [7.7, 7.7], color=LINE, lw=0.7)
    text(
        ax,
        60,
        5,
        r"Training: masked-position cross-entropy $\mathcal{L}=-\frac{1}{|M|}\sum_{i\in M}\log p(s_i\mid\widetilde{W},G)$",
        size=15,
    )
    text(
        ax,
        60,
        1.3,
        "Joint assignment enforces distinct symbols; prerequisite consistency is evaluated, not imposed.",
        size=11,
        color=MUTED,
    )
    save(fig, "autoencoder-architecture")


def results():
    with (ROOT / "results/base/summary.csv").open() as stream:
        rows = {
            r["method"]: r
            for r in csv.DictReader(stream)
            if r["bucket"] == "all" and r["level"] == "all"
        }
    methods = ["random", "markov-5-k10", "markov-5-k100", "markov-5-k1000", "ae"]
    labels = [
        "Random",
        "Markov 5 · K = 10",
        "Markov 5 · K = 100",
        "Markov 5 · K = 1000",
        "Graph autoencoder",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.subplots_adjust(left=0.21, right=0.97, top=0.77, bottom=0.21, wspace=0.18)
    fig.suptitle(
        "Base experiment: accuracy and prerequisite validity",
        x=0.04,
        ha="left",
        y=0.97,
        fontsize=19,
        weight="bold",
    )
    fig.text(
        0.04,
        0.87,
        "8000 challenges · 40 test courses · one training run · seed 1",
        color=MUTED,
        fontsize=11,
    )
    for ax, metric, title in zip(
        axes, ["epa", "pvr"], ["Exact position accuracy  ↑", "Prerequisite violation rate  ↓"]
    ):
        for i, method in enumerate(methods):
            row = rows[method]
            mean, lo, hi = [float(row[f"{metric}_{suffix}"]) for suffix in ["mean", "lo", "hi"]]
            color = TEAL if method == "ae" else ("#8899a5" if method == "random" else BLUE)
            ax.errorbar(
                mean,
                i,
                xerr=np.array([[mean - lo], [hi - mean]]),
                fmt="o",
                color=color,
                capsize=4,
                ms=7,
                lw=1.7,
            )
            ax.annotate(
                f"{mean:.3f}",
                (mean, i),
                xytext=(0, 11),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color=color,
            )
        ax.set(xlim=(-0.025, 1.025), ylim=(4.6, -0.65), yticks=range(5), yticklabels=labels)
        ax.set_title(title, loc="left", fontsize=12, pad=13, weight="bold")
        ax.set_xticks(np.linspace(0, 1, 6))
        ax.grid(axis="x", color="#e4ebef", lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, colors=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(
        0.04,
        0.075,
        "Points: means. Whiskers: 95% window-level bootstrap intervals (not graph-clustered).",
        color=MUTED,
        fontsize=10,
    )
    fig.text(
        0.04,
        0.025,
        "Source: results/base/summary.csv. See RESULTS.md for course-level comparisons and limitations.",
        color=MUTED,
        fontsize=9,
    )
    save(fig, "base-results")


if __name__ == "__main__":
    pipeline()
    architecture()
    results()
    print(f"Wrote PNG and SVG figures to {OUT}")
