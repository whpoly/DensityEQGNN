"""Atom latent to probe/grid density decoder."""

from __future__ import annotations

import torch
from torch import nn

from .geometry import atom_probe_displacements, safe_norm
from .layers import MLP
from .radial import GaussianRadialBasis, cosine_cutoff


class AtomToGridDecoder(nn.Module):
    """Decode atom latents into density values at probe positions."""

    def __init__(
        self,
        latent_dim: int,
        num_basis: int = 20,
        cutoff: float = 5.0,
        hidden_dim: int = 256,
        positive_density: bool = True,
        pbc: bool = True,
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.pbc = pbc
        self.positive_density = positive_density
        self.rbf = GaussianRadialBasis(num_basis=num_basis, cutoff=cutoff)
        self.message = MLP(latent_dim + num_basis + 4, hidden_dim, 1, num_layers=3)
        self.softplus = nn.Softplus()

    def forward(
        self,
        atom_latent: torch.Tensor,
        atom_positions: torch.Tensor,
        probe_positions: torch.Tensor,
        atom_mask: torch.Tensor,
        probe_mask: torch.Tensor,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        disp = atom_probe_displacements(atom_positions, probe_positions, cell, pbc=self.pbc)
        distances = safe_norm(disp)
        directions = disp / distances.unsqueeze(-1).clamp_min(1e-8)
        rbf = self.rbf(distances)

        h = atom_latent[:, :, None, :].expand(-1, -1, probe_positions.shape[1], -1)
        edge_input = torch.cat([h, rbf, distances.unsqueeze(-1), directions], dim=-1)
        contrib = self.message(edge_input).squeeze(-1)

        valid = (atom_mask[:, :, None] & probe_mask[:, None, :]).to(contrib.dtype)
        contrib = contrib * valid * cosine_cutoff(distances, self.cutoff)
        density = contrib.sum(dim=1)
        if self.positive_density:
            density = self.softplus(density)
        return density * probe_mask.to(density.dtype)

