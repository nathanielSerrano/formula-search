"""
Dual-branch graph encoder: one formula (SLT graph + OPT graph) → one L2-normalised vector.

Per branch (SLT and OPT have separate weights):
  node embedding   tag embedding + symbol embedding (hidden size; <pad>/<unk>/<none>
                   each have their own vector)
  message passing  `layers` × GATv2 with edge-type embeddings as edge features,
                   residual connection, LayerNorm, GELU, dropout
  readout          attention pooling ‖ mean pooling → linear → branch vector
Fusion:
  the two branch vectors are concatenated (a learned vector stands in for a missing
  graph) and an MLP maps them to the output dimension, followed by L2 normalisation.

Input is the dict produced by src.pretrain.dataset.collate_views.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.nn.aggr import AttentionalAggregation

REPS = ("slt", "opt")


@dataclass
class EncoderConfig:
    hidden: int = 256
    layers: int = 4
    heads: int = 4
    dropout: float = 0.1
    out_dim: int = 256
    # Vocabulary sizes per representation, filled from the graph vocabularies.
    n_tags: Dict[str, int] = None
    n_symbols: Dict[str, int] = None
    n_edge_types: Dict[str, int] = None  # including reverse types

    @classmethod
    def from_vocabs(cls, vocabs, **kwargs) -> "EncoderConfig":
        return cls(n_tags={r: len(vocabs[r].tags) for r in REPS},
                   n_symbols={r: len(vocabs[r].symbols) for r in REPS},
                   n_edge_types={r: vocabs[r].num_edge_types for r in REPS}, **kwargs)


class BranchEncoder(nn.Module):
    def __init__(self, n_tags: int, n_symbols: int, n_edge_types: int, cfg: EncoderConfig):
        super().__init__()
        if cfg.hidden % cfg.heads:
            raise ValueError("hidden must be divisible by heads")
        h = cfg.hidden
        self.tag_emb = nn.Embedding(n_tags, h, padding_idx=0)
        self.symbol_emb = nn.Embedding(n_symbols, h, padding_idx=0)
        self.edge_emb = nn.Embedding(n_edge_types, h)
        self.convs = nn.ModuleList([
            GATv2Conv(h, h // cfg.heads, heads=cfg.heads, edge_dim=h, dropout=cfg.dropout, add_self_loops=True)
            for _ in range(cfg.layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(cfg.layers)])
        self.dropout = nn.Dropout(cfg.dropout)
        self.pool_att = AttentionalAggregation(gate_nn=nn.Linear(h, 1))
        self.readout = nn.Linear(2 * h, h)

    def forward(self, graphs) -> torch.Tensor:
        x = self.tag_emb(graphs.x[:, 0]) + self.symbol_emb(graphs.x[:, 1])
        edge_attr = self.edge_emb(graphs.edge_type)
        for conv, norm in zip(self.convs, self.norms):
            x = norm(x + self.dropout(F.gelu(conv(x, graphs.edge_index, edge_attr))))
        pooled = torch.cat([self.pool_att(x, graphs.batch, dim_size=graphs.num_graphs),
                            global_mean_pool(x, graphs.batch, size=graphs.num_graphs)], dim=-1)
        return self.readout(pooled)


class FormulaEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.branches = nn.ModuleDict({
            r: BranchEncoder(cfg.n_tags[r], cfg.n_symbols[r], cfg.n_edge_types[r], cfg) for r in REPS})
        self.missing = nn.ParameterDict({r: nn.Parameter(torch.randn(cfg.hidden) * 0.02) for r in REPS})
        self.fusion = nn.Sequential(
            nn.Linear(2 * cfg.hidden, 2 * cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(2 * cfg.hidden, cfg.out_dim))

    def forward(self, batch: dict) -> torch.Tensor:
        """batch: {"size": B, "slt": {"graphs", "present"} or None, "opt": …} → [B, out_dim], unit length."""
        size = batch["size"]
        parts = []
        for r in REPS:
            vec = self.missing[r].unsqueeze(0).expand(size, -1)
            if batch[r] is not None:
                encoded = self.branches[r](batch[r]["graphs"])
                vec = vec.index_copy(0, batch[r]["present"], encoded.to(vec.dtype))
            parts.append(vec)
        return F.normalize(self.fusion(torch.cat(parts, dim=-1)), dim=-1)

    # -- checkpoints -----------------------------------------------------------

    def config_dict(self) -> dict:
        return asdict(self.cfg)

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "FormulaEncoder":
        return cls(EncoderConfig(**cfg))


def batch_to(batch: dict, device) -> dict:
    """Move a collated batch (see collate_views) to a device."""
    out = {"size": batch["size"]}
    for r in REPS:
        part = batch[r]
        out[r] = None if part is None else {"graphs": part["graphs"].to(device), "present": part["present"].to(device)}
    return out
