"""Grid/probe density to atom latent teacher encoder."""

from __future__ import annotations

import torch
from torch import nn

from .geometry import atom_probe_displacements, safe_norm
from .layers import MLP
from .radial import GaussianRadialBasis


class DensityToAtomEncoder(nn.Module):
    """Encode sampled density values into atom-centered latent vectors.

    The encoder collects radial density moments and first directional moments
    around every atom. This is intentionally learnable but still constrained
    enough to behave like an atom-centered density projection.
    """

    def __init__(
        self,
        latent_dim: int,
        num_species: int = 119,
        num_basis: int = 16,
        cutoff: float = 5.0,
        hidden_dim: int = 256,
        pbc: bool = True,
    ) -> None:
        super().__init__()
        self.pbc = pbc
        self.embedding = nn.Embedding(num_species, latent_dim, padding_idx=0)
        self.rbf = GaussianRadialBasis(num_basis=num_basis, cutoff=cutoff)
        feature_dim = num_basis + 3 * num_basis + latent_dim
        self.mlp = MLP(feature_dim, hidden_dim, latent_dim, num_layers=3)

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        atom_positions: torch.Tensor,
        probe_positions: torch.Tensor,
        density: torch.Tensor,
        atom_mask: torch.Tensor,
        probe_mask: torch.Tensor,
        cell: torch.Tensor | None = None,
    ) -> torch.Tensor:
        disp = atom_probe_displacements(atom_positions, probe_positions, cell, pbc=self.pbc)
        distances = safe_norm(disp)
        directions = disp / distances.unsqueeze(-1).clamp_min(1e-8)
        basis = self.rbf(distances)

        valid = (atom_mask[:, :, None] & probe_mask[:, None, :]).to(basis.dtype)
        rho = density[:, None, :, None]
        weighted_basis = basis * valid.unsqueeze(-1)

        denom = weighted_basis.sum(dim=2).clamp_min(1e-8)
        scalar_moments = (weighted_basis * rho).sum(dim=2) / denom
        vector_moments = (
            weighted_basis.unsqueeze(-1)
            * rho.unsqueeze(-1)
            * directions.unsqueeze(-2)
        ).sum(dim=2) / denom.unsqueeze(-1)

        atom_embed = self.embedding(atomic_numbers.clamp_min(0))
        features = torch.cat(
            [atom_embed, scalar_moments, vector_moments.flatten(start_dim=-2)],
            dim=-1,
        )
        latent = self.mlp(features)
        return latent * atom_mask.unsqueeze(-1).to(latent.dtype)

