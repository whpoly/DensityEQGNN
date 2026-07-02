"""Prediction heads."""

from __future__ import annotations

import torch
from torch import nn

from .layers import MLP, masked_mean


class PropertyHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_dim: int = 1,
        hidden_dim: int = 256,
        pooling: str = "mean",
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "sum"}:
            raise ValueError("pooling must be 'mean' or 'sum'")
        self.pooling = pooling
        self.mlp = MLP(latent_dim, hidden_dim, out_dim, num_layers=3)

    def forward(self, atom_latent: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        mask_f = atom_mask.to(atom_latent.dtype)
        if self.pooling == "mean":
            pooled = masked_mean(atom_latent, atom_mask, dim=1)
        else:
            pooled = torch.sum(atom_latent * mask_f.unsqueeze(-1), dim=1)
        return self.mlp(pooled)

