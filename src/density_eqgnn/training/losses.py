"""Losses for per-atom coefficient supervision."""

from __future__ import annotations

import torch


def masked_coefficient_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    atom_mask: torch.Tensor,
) -> torch.Tensor:
    mask = atom_mask[..., None].to(prediction.dtype)
    squared = (prediction - target).square() * mask
    return squared.sum() / (mask.sum() * prediction.shape[-1]).clamp_min(1.0)


def masked_density_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    probe_mask: torch.Tensor,
) -> torch.Tensor:
    mask = probe_mask.to(prediction.dtype)
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1.0)


def masked_density_nmae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    probe_mask: torch.Tensor,
) -> torch.Tensor:
    """ChargE3Net-style normalized absolute density error."""

    mask = probe_mask.to(prediction.dtype)
    numerator = (torch.abs(prediction - target) * mask).sum()
    denominator = (torch.abs(target) * mask).sum().clamp_min(1e-12)
    return numerator / denominator
