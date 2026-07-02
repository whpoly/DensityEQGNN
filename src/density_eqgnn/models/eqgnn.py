"""ChargeE3Net-inspired atom message passing.

This is a lightweight pure-PyTorch scaffold. It keeps the same separation of
atom representation and atom-to-probe decoding used by ChargE3Net, but does not
yet implement full e3nn irreducible representations.
"""

from __future__ import annotations

import torch
from torch import nn

from .geometry import atom_pair_displacements, safe_norm
from .layers import MLP
from .radial import GaussianRadialBasis, cosine_cutoff


class AtomInteractionBlock(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_basis: int,
        cutoff: float,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        self.edge_mlp = MLP(2 * latent_dim + num_basis + 1, hidden_dim, latent_dim, 3)
        self.update = MLP(2 * latent_dim, hidden_dim, latent_dim, 3)
        self.gate = nn.Sequential(nn.Linear(2 * latent_dim, latent_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(latent_dim)

    def forward(
        self,
        h: torch.Tensor,
        rbf: torch.Tensor,
        distances: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, natoms, _, _ = rbf.shape
        h_i = h[:, :, None, :].expand(bsz, natoms, natoms, h.shape[-1])
        h_j = h[:, None, :, :].expand(bsz, natoms, natoms, h.shape[-1])
        edge_input = torch.cat([h_i, h_j, rbf, distances.unsqueeze(-1)], dim=-1)
        msg = self.edge_mlp(edge_input)

        pair_mask = atom_mask[:, :, None] & atom_mask[:, None, :]
        eye = torch.eye(natoms, dtype=torch.bool, device=h.device).unsqueeze(0)
        pair_mask = pair_mask & ~eye
        msg = msg * pair_mask.unsqueeze(-1).to(msg.dtype)
        msg = msg * cosine_cutoff(distances, self.cutoff).unsqueeze(-1)
        agg = msg.sum(dim=2) / pair_mask.sum(dim=2).clamp_min(1).unsqueeze(-1)
        update = self.update(torch.cat([h, agg], dim=-1))
        gate = self.gate(torch.cat([h, agg], dim=-1))
        h = h + gate * update
        return self.norm(h) * atom_mask.unsqueeze(-1).to(h.dtype)


class ChargeE3LikeAtomNetwork(nn.Module):
    """Structure-only atom latent predictor."""

    def __init__(
        self,
        latent_dim: int,
        num_interactions: int = 4,
        num_species: int = 119,
        num_basis: int = 20,
        cutoff: float = 5.0,
        hidden_dim: int = 256,
        pbc: bool = True,
    ) -> None:
        super().__init__()
        self.pbc = pbc
        self.embedding = nn.Embedding(num_species, latent_dim, padding_idx=0)
        self.rbf = GaussianRadialBasis(num_basis=num_basis, cutoff=cutoff)
        self.blocks = nn.ModuleList(
            [
                AtomInteractionBlock(latent_dim, num_basis, cutoff, hidden_dim)
                for _ in range(num_interactions)
            ]
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        atom_mask: torch.Tensor,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        disp = atom_pair_displacements(positions, cell, pbc=self.pbc)
        distances = safe_norm(disp)
        rbf = self.rbf(distances)
        h = self.embedding(atomic_numbers.clamp_min(0))
        h = h * atom_mask.unsqueeze(-1).to(h.dtype)
        for block in self.blocks:
            h = block(h, rbf, distances, atom_mask)
        return h

