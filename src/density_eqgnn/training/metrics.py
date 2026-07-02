"""Evaluation metrics."""

from __future__ import annotations

import torch


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(pred.dtype)
    return (torch.abs(pred - target) * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(pred.dtype)
    mse = ((pred - target).square() * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    return torch.sqrt(mse)


def normalized_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(pred.dtype)
    mae = masked_mae(pred, target, mask)
    scale = (torch.abs(target) * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    return mae / scale.clamp_min(1e-12)

