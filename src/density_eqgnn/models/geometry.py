"""Geometry helpers for batched atom/probe tensors."""

from __future__ import annotations

import torch


def minimum_image(displacements: torch.Tensor, cell: torch.Tensor | None) -> torch.Tensor:
    """Apply a batched minimum-image convention when a cell is available."""
    if cell is None:
        return displacements
    try:
        inv_cell = torch.linalg.inv(cell)
    except RuntimeError:
        return displacements
    frac = torch.einsum("b...c,bcd->b...d", displacements, inv_cell)
    frac = frac - torch.round(frac)
    return torch.einsum("b...d,bdc->b...c", frac, cell)


def atom_pair_displacements(
    positions: torch.Tensor,
    cell: torch.Tensor | None = None,
    pbc: bool = True,
) -> torch.Tensor:
    """Return r_ij = r_j - r_i with shape [B, N, N, 3]."""
    disp = positions[:, None, :, :] - positions[:, :, None, :]
    if pbc:
        disp = minimum_image(disp, cell)
    return disp


def atom_probe_displacements(
    atom_positions: torch.Tensor,
    probe_positions: torch.Tensor,
    cell: torch.Tensor | None = None,
    pbc: bool = True,
) -> torch.Tensor:
    """Return r_g - r_i with shape [B, N, K, 3]."""
    disp = probe_positions[:, None, :, :] - atom_positions[:, :, None, :]
    if pbc:
        disp = minimum_image(disp, cell)
    return disp


def safe_norm(vectors: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return torch.sqrt(torch.sum(vectors.square(), dim=-1) + eps)
