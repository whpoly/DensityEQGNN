"""Train the l<=3 compact model using coefficient supervision only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from density_eqgnn.compact_basis import (
    CompactBasisConfig,
    reconstruct_compact_density,
    check_basis_checkpoint,
    basis_checkpoint_metadata,
)
from density_eqgnn.models import build_compact_coefficient_model
from density_eqgnn.periodic import PERIODIC_GEOMETRY_VERSION, check_periodic_checkpoint

from .config import load_config
from .train import build_loader, move_to_device, resolve_device


def coefficient_statistics(
    dataset, basis: CompactBasisConfig
) -> tuple[dict[int, list[float]], torch.Tensor]:
    scalar_values: dict[int, list[np.ndarray]] = defaultdict(list)
    cached: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        atomic_numbers = sample["atomic_numbers"]
        blocks = sample["coefficients"].reshape(
            len(atomic_numbers), basis.num_radial, basis.angular_channels
        )
        cached.append((atomic_numbers, blocks))
        for atomic_number, scalar in zip(
            atomic_numbers, blocks[..., 0], strict=True
        ):
            scalar_values[int(atomic_number)].append(scalar)
    baseline = {
        atomic_number: np.mean(np.stack(values), axis=0).tolist()
        for atomic_number, values in scalar_values.items()
    }

    sum_squares = np.zeros((basis.num_radial, basis.lmax + 1))
    counts = np.zeros((basis.num_radial, basis.lmax + 1))
    for atomic_numbers, blocks in cached:
        residual = blocks.copy()
        for atom_index, atomic_number in enumerate(atomic_numbers):
            residual[atom_index, :, 0] -= np.asarray(
                baseline[int(atomic_number)]
            )
        for degree in range(basis.lmax + 1):
            values = residual[
                ..., degree**2 : (degree + 1) ** 2
            ]
            sum_squares[:, degree] += np.square(values).sum(axis=(0, 2))
            counts[:, degree] += values.shape[0] * values.shape[2]
    rms = np.sqrt(sum_squares / np.maximum(counts, 1.0))
    rms = np.maximum(rms, 1e-4)
    scale = np.zeros((basis.num_radial, basis.angular_channels))
    for degree in range(basis.lmax + 1):
        scale[:, degree**2 : (degree + 1) ** 2] = rms[:, degree, None]
    return baseline, torch.from_numpy(scale.astype(np.float32))


def coefficient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    atom_mask: torch.Tensor,
    scale: torch.Tensor,
    basis: CompactBasisConfig,
) -> torch.Tensor:
    pred = prediction.reshape(
        *prediction.shape[:2], basis.num_radial, basis.angular_channels
    )
    ref = target.reshape_as(pred)
    mask = atom_mask[..., None, None].to(pred.dtype)
    normalized = (pred - ref) / scale[None, None]
    return (
        normalized.square() * mask
    ).sum() / (mask.sum() * basis.coefficient_size).clamp_min(1.0)


def density_nmae_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weights = mask.to(prediction.dtype)
    numerator = (torch.abs(prediction - target) * weights).sum()
    denominator = (torch.abs(target) * weights).sum().clamp_min(1e-12)
    return numerator / denominator


def density_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weights = mask.to(prediction.dtype)
    return (
        torch.abs(prediction - target) * weights
    ).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def evaluate(
    model,
    loader,
    basis: CompactBasisConfig,
    scale: torch.Tensor,
    device: torch.device,
) -> dict[str, float | None]:
    model.eval()
    density_error = 0.0
    reference = 0.0
    ls_error = 0.0
    normalized_squared = 0.0
    normalized_count = 0
    has_reference_coefficients = True
    sample_nmae: list[float] = []
    for raw_batch in loader:
        batch = move_to_device(raw_batch, device)
        output = model(
            batch["atomic_numbers"],
            batch["positions"],
            batch["atom_mask"],
            batch["cell"],
        )
        predicted_density = reconstruct_compact_density(
            output["coefficients"],
            batch["positions"],
            batch["probe_positions"],
            batch["atom_mask"],
            batch["probe_mask"],
            batch["cell"],
            basis,
        )
        reference_coefficients = batch.get("coefficients")
        compatible_reference = (
            reference_coefficients is not None
            and reference_coefficients.shape[-1] == basis.coefficient_size
            and not (basis.regular_radial or basis.radial_whitening)
        )
        has_reference_coefficients &= compatible_reference
        if compatible_reference:
            ls_density = reconstruct_compact_density(
                reference_coefficients,
                batch["positions"],
                batch["probe_positions"],
                batch["atom_mask"],
                batch["probe_mask"],
                batch["cell"],
                basis,
            )
        mask = batch["probe_mask"].to(predicted_density.dtype)
        target_density = batch["density"]
        error_by_sample = (
            torch.abs(predicted_density - target_density) * mask
        ).sum(dim=1)
        reference_by_sample = (
            torch.abs(target_density) * mask
        ).sum(dim=1).clamp_min(1e-12)
        density_error += float(error_by_sample.sum())
        if compatible_reference:
            ls_by_sample = (
                torch.abs(ls_density - target_density) * mask
            ).sum(dim=1)
            ls_error += float(ls_by_sample.sum())
        reference += float(reference_by_sample.sum())
        sample_nmae.extend(
            (error_by_sample / reference_by_sample).cpu().tolist()
        )

        if compatible_reference:
            pred = output["coefficients"].reshape(
                *output["coefficients"].shape[:2],
                basis.num_radial,
                basis.angular_channels,
            )
            ref = reference_coefficients.reshape_as(pred)
            atom_mask = batch["atom_mask"][..., None, None].to(pred.dtype)
            normalized_squared += float(
                (((pred - ref) / scale[None, None]).square() * atom_mask).sum()
            )
            normalized_count += int(atom_mask.sum()) * basis.coefficient_size

    return {
        "density_nmae": density_error / max(reference, 1e-12),
        "ls_reference_density_nmae": (
            ls_error / max(reference, 1e-12)
            if has_reference_coefficients
            else None
        ),
        "sample_density_nmae_mean": float(np.mean(sample_nmae)),
        "sample_density_nmae_median": float(np.median(sample_nmae)),
        "sample_density_nmae_p90": float(np.quantile(sample_nmae, 0.9)),
        "normalized_coefficient_rmse": (
            float(np.sqrt(normalized_squared / max(normalized_count, 1)))
            if has_reference_coefficients
            else None
        ),
    }


def run(config: dict) -> dict[str, float | str | int | None]:
    seed = int(config["training"].get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(config["training"].get("device", "auto"))
    basis = CompactBasisConfig(**config["basis"])
    model = build_compact_coefficient_model(config["model"]).to(device)
    train_loader = build_loader(config, "train")
    val_loader = build_loader(config, "val")
    loss_mode = str(config["training"].get("loss_mode", "coefficient"))
    density_loss_name = str(
        config["training"].get("density_loss", "nmae")
    )
    coefficient_weight = float(
        config["training"].get("coefficient_weight", 0.0)
    )
    if (basis.regular_radial or basis.radial_whitening) and (loss_mode == "coefficient" or coefficient_weight):
        raise ValueError("Transformed bases require density-only training until coefficient label basis metadata is supported")
    if loss_mode == "coefficient" or coefficient_weight:
        baseline, scale = coefficient_statistics(
            train_loader.dataset, basis
        )
        model.set_scalar_baseline(baseline)
        scale = scale.to(device)
    else:
        # Pure end-to-end density training does not require LS coefficient
        # labels.  Coefficients are latent variables and the atomic scalar
        # baseline, when enabled, is learned from density loss.
        scale = torch.ones(
            basis.num_radial,
            basis.angular_channels,
            device=device,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("lr", 1e-3)),
        weight_decay=float(config["training"].get("weight_decay", 1e-6)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=int(config["training"].get("lr_patience", 8)),
        min_lr=1e-6,
    )
    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best = float("inf")
    best_epoch = 0
    start_epoch = 1
    epochs = int(config["training"].get("epochs", 200))
    patience = int(config["training"].get("early_stopping_patience", 30))
    train_probes = int(config["training"].get("train_probes", 0))
    last_path = output_dir / "last.pt"
    if bool(config["training"].get("resume", False)) and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state.get("linear_readout_dtype"):
            raise ValueError("A frozen-readout calibration checkpoint is not a resumable joint-training state")
        check_basis_checkpoint(state, basis)
        check_periodic_checkpoint(
            state, pbc=basis.pbc or bool(config["model"].get("pbc", False)), resume=True
        )
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best = float(state["best"])
        best_epoch = int(state["best_epoch"])
        start_epoch = int(state["epoch"]) + 1
        torch.set_rng_state(state["torch_rng_state"].cpu())
        np.random.set_state(state["numpy_rng_state"])
        print(f"resuming from epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        losses = []
        for raw_batch in train_loader:
            batch = move_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                batch["atomic_numbers"],
                batch["positions"],
                batch["atom_mask"],
                batch["cell"],
            )
            coeff_loss = output["coefficients"].new_zeros(())
            if loss_mode == "coefficient" or coefficient_weight:
                coeff_loss = coefficient_loss(
                    output["coefficients"],
                    batch["coefficients"],
                    batch["atom_mask"],
                    scale,
                    basis,
                )
            if loss_mode == "coefficient":
                loss = coeff_loss
            elif loss_mode == "density":
                probe_positions = batch["probe_positions"]
                probe_mask = batch["probe_mask"]
                target_density = batch["density"]
                if train_probes and probe_positions.shape[1] > train_probes:
                    indices = torch.randperm(
                        probe_positions.shape[1],
                        device=probe_positions.device,
                    )[:train_probes]
                    probe_positions = probe_positions[:, indices]
                    probe_mask = probe_mask[:, indices]
                    target_density = target_density[:, indices]
                predicted_density = reconstruct_compact_density(
                    output["coefficients"],
                    batch["positions"],
                    probe_positions,
                    batch["atom_mask"],
                    probe_mask,
                    batch["cell"],
                    basis,
                )
                if density_loss_name == "l1":
                    loss = density_l1_loss(
                        predicted_density, target_density, probe_mask
                    )
                elif density_loss_name == "nmae":
                    loss = density_nmae_loss(
                        predicted_density, target_density, probe_mask
                    )
                else:
                    raise ValueError(
                        f"Unknown density_loss: {density_loss_name}"
                    )
                if coefficient_weight:
                    loss = loss + coefficient_weight * coeff_loss
            else:
                raise ValueError(f"Unknown loss_mode: {loss_mode}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = evaluate(model, val_loader, basis, scale, device)
        scheduler.step(metrics["density_nmae"])
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "loss_mode": loss_mode,
            "lr": optimizer.param_groups[0]["lr"],
            **metrics,
        }
        history.append(record)
        if metrics["density_nmae"] < best:
            best = metrics["density_nmae"]
            best_epoch = epoch
            torch.save(
                {
                    "periodic_geometry_version": PERIODIC_GEOMETRY_VERSION,
                    "model": model.state_dict(),
                    "model_config": config["model"],
                    **basis_checkpoint_metadata(basis),
                    "coefficient_scale": scale.cpu(),
                    "metrics": metrics,
                },
                output_dir / "best.pt",
            )
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        torch.save(
            {
                "periodic_geometry_version": PERIODIC_GEOMETRY_VERSION,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "history": history,
                "best": best,
                "best_epoch": best_epoch,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
                "model_config": config["model"],
                **basis_checkpoint_metadata(basis),
            },
            last_path,
        )
        if epoch == 1 or epoch % 5 == 0:
            print(
                f"epoch={epoch:03d} {loss_mode}_loss={record['train_loss']:.5f} "
                f"val_nmae={metrics['density_nmae']:.5f} "
                + (
                    f"ls_ref={metrics['ls_reference_density_nmae']:.5f}"
                    if metrics["ls_reference_density_nmae"] is not None
                    else "ls_ref=n/a"
                ),
                flush=True,
            )
        if epoch - best_epoch >= patience:
            print(f"early stopping at {epoch}; best={best_epoch}", flush=True)
            break
    return history[best_epoch - 1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run(load_config(args.config)), indent=2))


if __name__ == "__main__":
    main()
