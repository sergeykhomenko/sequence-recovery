"""Graph-conditioned masked autoencoder (see docs/architecture.md)."""

from __future__ import annotations

import math

import torch
from torch import nn

from .data import STATIC_FEATURES, Batch, GraphTensors


class MessageLayer(nn.Module):
    """h' = LayerNorm(h + MLP([h, mean over prerequisites, mean over dependants]))."""

    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, h: torch.Tensor, g: GraphTensors) -> torch.Tensor:
        prereqs = torch.zeros_like(h).index_add_(0, g.dst, h[g.src]) * g.in_norm
        dependants = torch.zeros_like(h).index_add_(0, g.src, h[g.dst]) * g.out_norm
        return self.norm(h + self.mlp(torch.cat([h, prereqs, dependants], dim=1)))


class GraphEncoder(nn.Module):
    """Node vectors from the prerequisite graph alone: structural features plus
    random ones, drawn anew on every call to tell structurally equal nodes apart.
    Node names and their order in the file are never used."""

    def __init__(self, dim: int, layers: int, random_features: int):
        super().__init__()
        self.random_features = random_features
        self.input = nn.Linear(STATIC_FEATURES + random_features, dim)
        self.layers = nn.ModuleList(MessageLayer(dim) for _ in range(layers))

    def forward(self, g: GraphTensors, generator: torch.Generator | None = None) -> torch.Tensor:
        noise = torch.randn(g.size, self.random_features, generator=generator).to(g.features.device)
        h = self.input(torch.cat([g.features, noise], dim=1))
        for layer in self.layers:
            h = layer(h, g)
        return h


class GraphAE(nn.Module):
    def __init__(
        self,
        dim: int,
        gnn_layers: int,
        layers: int,
        heads: int,
        ffn: int,
        random_features: int,
        max_len: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.graph = GraphEncoder(dim, gnn_layers, random_features)
        self.token = nn.Linear(dim, dim)
        self.mask = nn.Parameter(torch.randn(dim) * 0.02)
        self.position = nn.Embedding(max_len, dim)
        layer = nn.TransformerEncoderLayer(
            dim, heads, ffn, dropout=dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)

    def encode_graph(
        self, g: GraphTensors, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return self.graph(g, generator)

    def decode(self, nodes: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Logits [M, N] for every mask of the batch in row-major order; nodes
        already standing in the same window get -inf."""
        size, length = batch.tokens.shape
        x = self.token(nodes[batch.tokens])
        x = torch.where(batch.masked.unsqueeze(-1), self.mask.expand_as(x), x)
        x = x + self.position(torch.arange(length, device=x.device)).unsqueeze(0)
        h = self.norm(self.encoder(x, src_key_padding_mask=batch.padding))
        q = self.query(h[batch.masked])
        logits = q @ self.key(nodes).T / math.sqrt(q.shape[1])
        kept = ~batch.masked & ~batch.padding
        rows, cols = kept.nonzero(as_tuple=True)
        present = torch.zeros(size, nodes.shape[0], dtype=torch.bool, device=x.device)
        present[rows, batch.tokens[rows, cols]] = True
        window_of_mask = batch.masked.nonzero(as_tuple=True)[0]
        return logits.masked_fill(present[window_of_mask], float("-inf"))

    def forward(
        self, g: GraphTensors, batch: Batch, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        return self.decode(self.encode_graph(g, generator), batch)


def build(ae: dict, max_len: int) -> GraphAE:
    return GraphAE(
        ae["dim"],
        ae["gnn_layers"],
        ae["layers"],
        ae["heads"],
        ae["ffn"],
        ae["random_features"],
        max_len,
    )
