"""Radial basis and cutoff functions."""

from __future__ import annotations

import torch
from torch import nn


class GaussianRadialBasis(nn.Module):
    """Gaussian radial basis with a smooth cosine cutoff."""

    def __init__(self, num_basis: int, cutoff: float, start: float = 0.0) -> None:
        super().__init__()
        centers = torch.linspace(start, cutoff, num_basis)
        spacing = centers[1] - centers[0] if num_basis > 1 else torch.tensor(cutoff)
        self.register_buffer("centers", centers)
        self.gamma = nn.Parameter(torch.tensor(1.0 / float(spacing**2)), requires_grad=False)
        self.cutoff = float(cutoff)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        basis = torch.exp(-self.gamma * diff.square())
        return basis * cosine_cutoff(distances, self.cutoff).unsqueeze(-1)


def cosine_cutoff(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    values = 0.5 * (torch.cos(torch.clamp(distances, max=cutoff) * torch.pi / cutoff) + 1.0)
    return torch.where(distances < cutoff, values, torch.zeros_like(values))

