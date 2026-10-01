"""
Symmetric InfoNCE with in-batch negatives and a learnable temperature.

For a batch of B pairs (a_i, b_i) of unit vectors, row i of a·bᵀ / τ should peak at
column i and vice versa; every other formula in the batch is a negative. τ is learned
through log(1/τ) and capped (τ ≥ 0.01), as in CLIP.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCE(nn.Module):
    def __init__(self, init_temperature: float = 0.07, min_temperature: float = 0.01):
        super().__init__()
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))
        self.max_log_scale = math.log(1.0 / min_temperature)

    @property
    def temperature(self) -> float:
        return float(1.0 / self.log_scale.detach().clamp(max=self.max_log_scale).exp())

    def query_to_docs(self, queries: torch.Tensor, docs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        One-way InfoNCE for fine-tuning: queries [B, d] against docs [N ≥ B, d], where
        docs[i] is the positive of query i and the rest (other positives, hard
        negatives) are negatives.
        """
        logits = self.log_scale.clamp(max=self.max_log_scale).exp() * (queries.float() @ docs.float().t())
        labels = torch.arange(len(queries), device=queries.device)
        loss = F.cross_entropy(logits, labels)
        with torch.no_grad():
            accuracy = (logits.argmax(dim=1) == labels).float().mean()
        return {"loss": loss, "accuracy": accuracy}

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.log_scale.clamp(max=self.max_log_scale).exp() * (a.float() @ b.float().t())
        labels = torch.arange(len(a), device=a.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        with torch.no_grad():
            accuracy = (logits.argmax(dim=1) == labels).float().mean()
        return {"loss": loss, "accuracy": accuracy}
