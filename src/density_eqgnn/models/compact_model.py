"""Coefficient-supervised compact spherical-harmonic GNN."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from e3nn import o3
from torch import nn

from density_eqgnn.periodic import atom_graph_geometry, gather_neighbors

from .model import GaussianEdgeBasis, InvariantInteraction, _mlp


@dataclass
class CompactModelConfig:
    num_radial: int = 12
    lmax: int = 3
    hidden_dim: int = 128
    num_interactions: int = 4
    edge_basis: int = 24
    cutoff: float = 4.0
    num_species: int = 119
    pbc: bool = False

    @property
    def angular_channels(self) -> int:
        return (self.lmax + 1) ** 2

    @property
    def coefficient_size(self) -> int:
        return self.num_radial * self.angular_channels


class CompactCoefficientGNN(nn.Module):
    """Predict one equivariant coefficient block for each ``(n, l)``."""

    def __init__(self, config: CompactModelConfig | dict) -> None:
        super().__init__()
        if isinstance(config, dict):
            config = CompactModelConfig(**config)
        if config.lmax < 0 or config.lmax > 3:
            raise ValueError("compact model supports 0 <= lmax <= 3")
        self.config = config
        self.embedding = nn.Embedding(
            config.num_species, config.hidden_dim, padding_idx=0
        )
        self.edge_basis = GaussianEdgeBasis(config.edge_basis, config.cutoff)
        self.interactions = nn.ModuleList(
            [
                InvariantInteraction(config.hidden_dim, config.edge_basis)
                for _ in range(config.num_interactions)
            ]
        )
        self.scalar_head = _mlp(
            config.hidden_dim, config.hidden_dim, config.num_radial
        )
        self.equivariant_heads = nn.ModuleDict(
            {
                str(degree): _mlp(
                    2 * config.hidden_dim + config.edge_basis,
                    config.hidden_dim,
                    config.num_radial,
                )
                for degree in range(1, config.lmax + 1)
            }
        )
        self.register_buffer(
            "atomic_scalar_baseline",
            torch.zeros(config.num_species, config.num_radial),
        )
        for head in [self.scalar_head, *self.equivariant_heads.values()]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def set_scalar_baseline(self, values: dict[int, list[float]]) -> None:
        baseline = torch.zeros_like(self.atomic_scalar_baseline)
        for atomic_number, coefficients in values.items():
            baseline[int(atomic_number)] = torch.as_tensor(
                coefficients, dtype=baseline.dtype, device=baseline.device
            )
        self.atomic_scalar_baseline.copy_(baseline)

    def _geometry(
        self,
        positions: torch.Tensor,
        atom_mask: torch.Tensor,
        cell: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return atom_graph_geometry(
            positions, atom_mask, cell, self.config.cutoff, self.config.pbc
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        atom_mask: torch.Tensor,
        cell: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        distances, directions, pair_mask, source_indices = self._geometry(
            positions, atom_mask, cell
        )
        radial_edges = self.edge_basis(distances)
        features = self.embedding(atomic_numbers)
        for interaction in self.interactions:
            features = interaction(features, radial_edges, pair_mask, source_indices)
            features *= atom_mask[..., None].to(features.dtype)

        batch, atoms = atomic_numbers.shape
        coefficients = features.new_zeros(
            batch,
            atoms,
            self.config.num_radial,
            self.config.angular_channels,
        )
        scalar = (
            self.atomic_scalar_baseline[atomic_numbers]
            + self.scalar_head(features)
        )
        coefficients[..., 0] = scalar

        if self.config.lmax:
            left = features[:, :, None, :].expand(-1, -1, pair_mask.shape[2], -1)
            right = gather_neighbors(features, source_indices)
            edge_features = torch.cat([left, right, radial_edges], dim=-1)
            count = (
                pair_mask.sum(dim=2, keepdim=True)
                .clamp_min(1)
                .to(features.dtype)
            )
            all_harmonics = o3.spherical_harmonics(
                list(range(self.config.lmax + 1)),
                directions,
                normalize=True,
                normalization="integral",
            )
            for degree in range(1, self.config.lmax + 1):
                weights = self.equivariant_heads[str(degree)](edge_features)
                weights *= pair_mask[..., None].to(weights.dtype)
                angular = all_harmonics[
                    ..., degree**2 : (degree + 1) ** 2
                ]
                block = torch.einsum("bijn,bijm->binm", weights, angular)
                block /= torch.sqrt(count[..., None])
                coefficients[
                    ..., degree**2 : (degree + 1) ** 2
                ] = block

        coefficients *= atom_mask[..., None, None].to(coefficients.dtype)
        return {
            "coefficient_blocks": coefficients,
            "coefficients": coefficients.reshape(batch, atoms, -1),
        }
