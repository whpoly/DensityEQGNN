"""Train the minimal equivariant coefficient predictor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from density_eqgnn.basis import DensityBasisConfig, reconstruct_density_torch
from density_eqgnn.data import CoefficientDataset, collate_coefficients
from density_eqgnn.models import EquivariantCoefficientGNN, ModelConfig
from density_eqgnn.periodic import PERIODIC_GEOMETRY_VERSION

from .config import load_config
from .losses import masked_coefficient_mse, masked_density_mse, masked_density_nmae


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_loader(config: dict, split: str) -> DataLoader:
    external_root = os.environ.get("DENSITY_EQGNN_DATA_ROOT")
    if external_root:
        root = Path(external_root) / config["data"].get("dataset_name", "nmc_subset")
    else:
        root = config["data"]["root"]
    split_file = config["data"].get(f"{split}_split")
    dataset = CoefficientDataset(
        root,
        split_file,
        augment_rotations=bool(
            split == "train" and config["data"].get("augment_rotations", True)
        ),
        num_radial=int(config["basis"]["num_radial"]),
        seed=int(config["training"].get("seed", 0)),
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"].get("batch_size", 8)),
        shuffle=split == "train",
        num_workers=int(config["data"].get("num_workers", 0)),
        collate_fn=collate_coefficients,
        persistent_workers=int(config["data"].get("num_workers", 0)) > 0,
    )


def forward_batch(
    model: EquivariantCoefficientGNN,
    batch: dict,
    basis_config: DensityBasisConfig,
    coefficient_weight: float,
    density_weight: float,
    density_loss_name: str = "mse",
    max_density_probes: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model(
        batch["atomic_numbers"],
        batch["positions"],
        batch["atom_mask"],
        batch["cell"],
    )
    coefficient_loss = masked_coefficient_mse(
        output["coefficients"], batch["coefficients"], batch["atom_mask"]
    )
    density_loss = coefficient_loss.new_zeros(())
    if density_weight > 0 and "density" in batch:
        probe_positions = batch["probe_positions"]
        probe_mask = batch["probe_mask"]
        density_target = batch["density"]
        if max_density_probes and probe_positions.shape[1] > max_density_probes:
            indices = torch.randperm(
                probe_positions.shape[1], device=probe_positions.device
            )[:max_density_probes]
            probe_positions = probe_positions[:, indices]
            probe_mask = probe_mask[:, indices]
            density_target = density_target[:, indices]
        density_prediction = reconstruct_density_torch(
            output["coefficients"],
            batch["positions"],
            probe_positions,
            batch["atom_mask"],
            probe_mask,
            batch["cell"],
            basis_config,
        )
        output["density"] = density_prediction
        if density_loss_name == "mse":
            density_loss = masked_density_mse(
                density_prediction, density_target, probe_mask
            )
        elif density_loss_name == "nmae":
            density_loss = masked_density_nmae(
                density_prediction, density_target, probe_mask
            )
        else:
            raise ValueError(f"Unknown density loss: {density_loss_name}")
    total = coefficient_weight * coefficient_loss + density_weight * density_loss
    return total, {
        "coefficient_loss": coefficient_loss,
        "density_loss": density_loss,
        **output,
    }


@torch.no_grad()
def evaluate(
    model: EquivariantCoefficientGNN,
    loader: DataLoader,
    basis_config: DensityBasisConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    coefficient_error = 0.0
    coefficient_count = 0
    density_error = 0.0
    density_reference = 0.0
    oracle_error = 0.0
    sample_nmae: list[float] = []
    oracle_sample_nmae: list[float] = []
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        output = model(
            batch["atomic_numbers"],
            batch["positions"],
            batch["atom_mask"],
            batch["cell"],
        )
        coefficient_mask = batch["atom_mask"][..., None].expand_as(output["coefficients"])
        coefficient_error += float(
            (
                torch.abs(output["coefficients"] - batch["coefficients"])
                * coefficient_mask
            ).sum()
        )
        coefficient_count += int(coefficient_mask.sum())
        if "density" in batch:
            density = reconstruct_density_torch(
                output["coefficients"],
                batch["positions"],
                batch["probe_positions"],
                batch["atom_mask"],
                batch["probe_mask"],
                batch["cell"],
                basis_config,
            )
            oracle_density = reconstruct_density_torch(
                batch["coefficients"],
                batch["positions"],
                batch["probe_positions"],
                batch["atom_mask"],
                batch["probe_mask"],
                batch["cell"],
                basis_config,
            )
            mask = batch["probe_mask"].to(density.dtype)
            target = batch["density"]
            error_by_sample = (torch.abs(density - target) * mask).sum(dim=1)
            oracle_by_sample = (torch.abs(oracle_density - target) * mask).sum(dim=1)
            reference_by_sample = (torch.abs(target) * mask).sum(dim=1).clamp_min(1e-12)
            density_error += float(error_by_sample.sum())
            oracle_error += float(oracle_by_sample.sum())
            density_reference += float(reference_by_sample.sum())
            sample_nmae.extend((error_by_sample / reference_by_sample).cpu().tolist())
            oracle_sample_nmae.extend(
                (oracle_by_sample / reference_by_sample).cpu().tolist()
            )

    import numpy as np

    return {
        "coefficient_mae": coefficient_error / max(coefficient_count, 1),
        # This is Eq. (2) in ChargE3Net, evaluated on the stored random probes.
        "density_nmae": density_error / max(density_reference, 1e-12),
        "oracle_density_nmae": oracle_error / max(density_reference, 1e-12),
        "sample_density_nmae_mean": float(np.mean(sample_nmae)) if sample_nmae else 0.0,
        "sample_density_nmae_median": float(np.median(sample_nmae)) if sample_nmae else 0.0,
        "sample_density_nmae_p90": float(np.quantile(sample_nmae, 0.9))
        if sample_nmae
        else 0.0,
        "sample_oracle_nmae_mean": float(np.mean(oracle_sample_nmae))
        if oracle_sample_nmae
        else 0.0,
    }


def run_training(config: dict) -> dict[str, float]:
    torch.manual_seed(int(config["training"].get("seed", 0)))
    device = resolve_device(config["training"].get("device", "auto"))
    model_config = ModelConfig(**config["model"])
    basis_config = DensityBasisConfig(**config["basis"])
    if model_config.num_radial != basis_config.num_radial:
        raise ValueError("model.num_radial must match basis.num_radial")
    if model_config.lmax != basis_config.lmax:
        raise ValueError("model.lmax must match basis.lmax")

    model = EquivariantCoefficientGNN(model_config).to(device)
    baseline_path = config["data"].get("atomic_baseline")
    if baseline_path:
        baseline_data = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        model.set_atomic_baseline(
            {
                int(atomic_number): values
                for atomic_number, values in baseline_data[
                    "atomic_coefficients"
                ].items()
            }
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 2e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-6)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["training"].get("lr_factor", 0.5)),
        patience=int(config["training"].get("lr_patience", 8)),
        min_lr=float(config["training"].get("min_lr", 1e-6)),
    )
    train_loader = build_loader(config, "train")
    val_loader = build_loader(config, "val")
    output_dir = Path(config["training"].get("output_dir", "runs/toy"))
    output_dir.mkdir(parents=True, exist_ok=True)
    coefficient_weight = float(config["training"].get("coefficient_weight", 1.0))
    density_weight = float(config["training"].get("density_weight", 0.1))
    density_loss_name = str(config["training"].get("density_loss", "mse"))
    train_probes = config["training"].get("train_probes")
    train_probes = int(train_probes) if train_probes else None

    best = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    early_stopping = int(config["training"].get("early_stopping_patience", 25))
    epochs = int(config["training"].get("epochs", 30))
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for raw_batch in train_loader:
            batch = move_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = forward_batch(
                model,
                batch,
                basis_config,
                coefficient_weight,
                density_weight,
                density_loss_name,
                max_density_probes=train_probes,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"].get("grad_clip", 5.0))
            )
            optimizer.step()
            total += float(loss.detach())
        train_loss = total / max(len(train_loader), 1)
        val_metrics = evaluate(model, val_loader, basis_config, device)
        scheduler.step(val_metrics["density_nmae"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": optimizer.param_groups[0]["lr"],
                **val_metrics,
            }
        )
        if val_metrics["density_nmae"] < best:
            best = val_metrics["density_nmae"]
            best_epoch = epoch
            torch.save(
                {
                    "periodic_geometry_version": PERIODIC_GEOMETRY_VERSION,
                    "model": model.state_dict(),
                    "model_config": config["model"],
                    "basis_config": config["basis"],
                    "metrics": val_metrics,
                },
                output_dir / "best.pt",
            )
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"val_coeff_mae={val_metrics['coefficient_mae']:.6f} "
                f"val_density_nmae={val_metrics['density_nmae']:.6f} "
                f"oracle_nmae={val_metrics['oracle_density_nmae']:.6f}"
            )
        if epoch - best_epoch >= early_stopping:
            print(f"early stopping at epoch {epoch}; best epoch={best_epoch}")
            break
    return val_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nmc_small.yaml")
    args = parser.parse_args()
    run_training(load_config(args.config))


if __name__ == "__main__":
    main()
