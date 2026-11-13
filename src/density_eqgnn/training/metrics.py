"""Compact coefficient and reconstructed-density metrics."""

from __future__ import annotations

import torch


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < prediction.ndim:
        mask = mask.unsqueeze(-1)
    weights = mask.to(prediction.dtype)
    denominator = weights.sum() * prediction.shape[-1]
    return (torch.abs(prediction - target) * weights).sum() / denominator.clamp_min(1.0)


def density_nmae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    probe_mask: torch.Tensor,
) -> torch.Tensor:
    mask = probe_mask.to(prediction.dtype)
    absolute_error = (torch.abs(prediction - target) * mask).sum()
    reference = (torch.abs(target) * mask).sum()
    return absolute_error / reference.clamp_min(1e-12)
