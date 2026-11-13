"""A compact E(3)-equivariant predictor of atom-centred density coefficients."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from density_eqgnn.periodic import atom_graph_geometry, gather_neighbors


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


@dataclass
class ModelConfig:
    num_radial: int = 4
    lmax: int = 2
    hidden_dim: int = 64
    num_interactions: int = 3
    edge_basis: int = 12
    cutoff: float = 4.0
    num_species: int = 119
    pbc: bool = False

    @property
    def coefficient_size(self) -> int:
        return self.num_radial * (
            1 + (3 if self.lmax >= 1 else 0) + (9 if self.lmax >= 2 else 0)
        )


class GaussianEdgeBasis(nn.Module):
    def __init__(self, count: int, cutoff: float) -> None:
        super().__init__()
        self.cutoff = float(cutoff)
        centers = torch.linspace(0.0, cutoff, count)
        self.register_buffer("centers", centers)
        self.width = cutoff / max(count - 1, 1)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        radial = torch.exp(-0.5 * ((distances[..., None] - self.centers) / self.width).square())
        x = (distances / self.cutoff).clamp(0.0, 1.0)
        envelope = 0.5 * (torch.cos(torch.pi * x) + 1.0)
        envelope = envelope * (distances < self.cutoff).to(envelope.dtype)
        return radial * envelope[..., None]


class InvariantInteraction(nn.Module):
    """Distance-only message passing used to construct invariant atom features."""

    def __init__(self, hidden_dim: int, edge_basis: int) -> None:
        super().__init__()
        self.message = _mlp(2 * hidden_dim + edge_basis, hidden_dim, hidden_dim)
        self.update = _mlp(2 * hidden_dim, hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        radial: torch.Tensor,
        pair_mask: torch.Tensor,
        source_indices: torch.Tensor,
    ) -> torch.Tensor:
        left = features[:, :, None, :].expand(-1, -1, pair_mask.shape[2], -1)
        right = gather_neighbors(features, source_indices)
        messages = self.message(torch.cat([left, right, radial], dim=-1))
        messages = messages * pair_mask[..., None].to(messages.dtype)
        count = pair_mask.sum(dim=2, keepdim=True).clamp_min(1)
        aggregate = messages.sum(dim=2) / torch.sqrt(count.to(messages.dtype))
        update = self.update(torch.cat([features, aggregate], dim=-1))
        return self.norm(features + update)


class EquivariantCoefficientGNN(nn.Module):
    """Predict l=0 scalars and l=1 vectors for every atom.

    The scalar backbone sees only atomic numbers and pair distances.  The l=1
    output is a learned scalar edge weight times the unit bond vector, summed
    over neighbours.  Consequently a rigid rotation leaves scalar outputs
    unchanged and rotates every vector output by the same rotation.
    """

    def __init__(self, config: ModelConfig | dict | None = None) -> None:
        super().__init__()
        if config is None:
            config = ModelConfig()
        if isinstance(config, dict):
            config = ModelConfig(**config)
        if config.lmax not in {0, 1, 2}:
            raise ValueError("The compact model supports lmax=0, 1, or 2")
        self.config = config
        self.embedding = nn.Embedding(config.num_species, config.hidden_dim, padding_idx=0)
        self.register_buffer(
            "atomic_baseline",
            torch.zeros(config.num_species, config.num_radial),
        )
        self.edge_basis = GaussianEdgeBasis(config.edge_basis, config.cutoff)
        self.interactions = nn.ModuleList(
            [
                InvariantInteraction(config.hidden_dim, config.edge_basis)
                for _ in range(config.num_interactions)
            ]
        )
        self.scalar_head = _mlp(config.hidden_dim, config.hidden_dim, config.num_radial)
        if config.lmax >= 1:
            self.vector_head = _mlp(
                2 * config.hidden_dim + config.edge_basis,
                config.hidden_dim,
                config.num_radial,
            )
        if config.lmax >= 2:
            self.tensor_head = _mlp(
                2 * config.hidden_dim + config.edge_basis,
                config.hidden_dim,
                config.num_radial,
            )
        # With an atomic baseline loaded, the untrained network represents
        # exactly that baseline and learns only environment-dependent changes.
        for head in (
            self.scalar_head,
            getattr(self, "vector_head", None),
            getattr(self, "tensor_head", None),
        ):
            if head is not None:
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)

    def set_atomic_baseline(self, values: dict[int, list[float]]) -> None:
        baseline = torch.zeros_like(self.atomic_baseline)
        for atomic_number, coefficients in values.items():
            tensor = torch.as_tensor(
                coefficients,
                dtype=baseline.dtype,
                device=baseline.device,
            )
            if tensor.shape != (self.config.num_radial,):
                raise ValueError(
                    f"baseline Z={atomic_number} must contain "
                    f"{self.config.num_radial} values"
                )
            baseline[int(atomic_number)] = tensor
        self.atomic_baseline.copy_(baseline)

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
        radial = self.edge_basis(distances)
        features = self.embedding(atomic_numbers)
        for interaction in self.interactions:
            features = interaction(features, radial, pair_mask, source_indices)
            features = features * atom_mask[..., None].to(features.dtype)

        scalar = self.atomic_baseline[atomic_numbers] + self.scalar_head(features)
        scalar = scalar * atom_mask[..., None].to(features.dtype)
        output = {"scalar_coefficients": scalar}
        flattened = [scalar]

        edge_features = None
        if self.config.lmax >= 1:
            left = features[:, :, None, :].expand(-1, -1, pair_mask.shape[2], -1)
            right = gather_neighbors(features, source_indices)
            edge_features = torch.cat([left, right, radial], dim=-1)
            weights = self.vector_head(edge_features)
            weights = weights * pair_mask[..., None].to(weights.dtype)
            count = pair_mask.sum(dim=2, keepdim=True).clamp_min(1).to(weights.dtype)
            vector = torch.einsum("bijn,bijd->bind", weights, directions)
            vector = vector / torch.sqrt(count[..., None])
            vector = vector * atom_mask[..., None, None].to(vector.dtype)
            output["vector_coefficients"] = vector
            flattened.append(vector.reshape(*vector.shape[:2], -1))
        if self.config.lmax >= 2:
            if edge_features is None:
                raise RuntimeError("l=2 requires edge features")
            weights = self.tensor_head(edge_features)
            weights = weights * pair_mask[..., None].to(weights.dtype)
            count = pair_mask.sum(dim=2, keepdim=True).clamp_min(1).to(weights.dtype)
            identity = torch.eye(3, dtype=directions.dtype, device=directions.device) / 3.0
            quadrupole = (
                directions[..., :, None] * directions[..., None, :] - identity
            )
            tensor = torch.einsum("bijn,bijde->bined", weights, quadrupole)
            tensor = tensor / torch.sqrt(count[..., None, None])
            tensor = tensor * atom_mask[..., None, None, None].to(tensor.dtype)
            output["tensor_coefficients"] = tensor
            flattened.append(tensor.reshape(*tensor.shape[:2], -1))

        output["coefficients"] = torch.cat(flattened, dim=-1)
        return output
