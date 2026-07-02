"""Loss functions for density-supervised training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    density: float = 1.0
    teacher_recon: float = 0.5
    latent: float = 0.1
    property: float = 0.0
    charge: float = 0.01


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(pred.dtype)
    loss = (pred - target).square() * mask_f
    return loss.sum() / mask_f.sum().clamp_min(1.0)


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(pred.dtype)
    loss = torch.abs(pred - target) * mask_f
    return loss.sum() / mask_f.sum().clamp_min(1.0)


def charge_loss(
    density_pred: torch.Tensor,
    probe_mask: torch.Tensor,
    probe_volume: torch.Tensor | None,
    electron_count: torch.Tensor | None,
) -> torch.Tensor:
    if probe_volume is None or electron_count is None:
        return density_pred.new_zeros(())
    pred_charge = (density_pred * probe_volume * probe_mask.to(density_pred.dtype)).sum(dim=1)
    return F.mse_loss(pred_charge, electron_count)


def compute_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    weights: LossWeights,
    density_loss: str = "mse",
) -> dict[str, torch.Tensor]:
    mask = batch["probe_mask"]
    target_density = batch["density"]
    density_metric = masked_l1 if density_loss == "l1" else masked_mse

    terms: dict[str, torch.Tensor] = {}
    terms["density"] = density_metric(output["density"], target_density, mask)

    if "density_teacher_recon" in output:
        terms["teacher_recon"] = density_metric(
            output["density_teacher_recon"], target_density, mask
        )
    else:
        terms["teacher_recon"] = target_density.new_zeros(())

    if "z_teacher" in output:
        atom_mask = batch["atom_mask"].unsqueeze(-1).to(output["z_structure"].dtype)
        latent_err = (output["z_structure"] - output["z_teacher"].detach()).square() * atom_mask
        terms["latent"] = latent_err.sum() / atom_mask.sum().clamp_min(1.0)
    else:
        terms["latent"] = target_density.new_zeros(())

    if "property" in batch:
        terms["property"] = F.mse_loss(output["property"], batch["property"])
    else:
        terms["property"] = target_density.new_zeros(())

    terms["charge"] = charge_loss(
        output["density"],
        batch["probe_mask"],
        batch.get("probe_volume"),
        batch.get("electron_count"),
    )

    total = (
        weights.density * terms["density"]
        + weights.teacher_recon * terms["teacher_recon"]
        + weights.latent * terms["latent"]
        + weights.property * terms["property"]
        + weights.charge * terms["charge"]
    )
    terms["total"] = total
    return terms

