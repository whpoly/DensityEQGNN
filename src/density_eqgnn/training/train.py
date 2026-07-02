"""Minimal training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from density_eqgnn.data import DensityNPZDataset, collate_density_samples
from density_eqgnn.models import DensitySupervisedEQGNN
from density_eqgnn.models.model import ModelConfig

from .config import load_config
from .losses import LossWeights, compute_losses
from .metrics import masked_mae, normalized_mae


def move_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def resolve_device(value: str | None) -> torch.device:
    if value in {None, "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def build_dataloader(cfg: dict, split: str) -> DataLoader:
    data_cfg = cfg["data"]
    split_file = data_cfg.get(f"{split}_split")
    max_probes = data_cfg.get(f"{split}_probes", data_cfg.get("max_probes"))
    dataset = DensityNPZDataset(
        root=data_cfg["root"],
        split_file=split_file,
        max_probes=max_probes,
        seed=int(data_cfg.get("seed", 0)),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("batch_size", 1)),
        shuffle=(split == "train"),
        num_workers=int(data_cfg.get("num_workers", 0)),
        collate_fn=collate_density_samples,
        pin_memory=bool(data_cfg.get("pin_memory", False)),
    )


def train_one_epoch(model, loader, optimizer, weights, cfg, device):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        losses = compute_losses(
            output,
            batch,
            weights,
            density_loss=cfg["training"].get("density_loss", "mse"),
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("grad_clip", 5.0))
        optimizer.step()
        total += float(losses["total"].detach().cpu())
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, weights, cfg, device):
    model.eval()
    totals = {"loss": 0.0, "mae": 0.0, "nmae": 0.0}
    for batch in tqdm(loader, desc="val", leave=False):
        batch = move_to_device(batch, device)
        output = model(batch)
        losses = compute_losses(
            output,
            batch,
            weights,
            density_loss=cfg["training"].get("density_loss", "mse"),
        )
        totals["loss"] += float(losses["total"].cpu())
        totals["mae"] += float(masked_mae(output["density"], batch["density"], batch["probe_mask"]).cpu())
        totals["nmae"] += float(
            normalized_mae(output["density"], batch["density"], batch["probe_mask"]).cpu()
        )
    denom = max(len(loader), 1)
    return {key: value / denom for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["training"].get("device", "auto"))

    model = DensitySupervisedEQGNN(ModelConfig(**cfg["model"])).to(device)
    weights = LossWeights(**cfg.get("loss_weights", {}))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("lr", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-6)),
    )

    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val") if cfg["data"].get("val_split") else None
    output_dir = Path(cfg["training"].get("output_dir", "runs/density_eqgnn"))
    output_dir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    epochs = int(cfg["training"].get("epochs", 1))
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, weights, cfg, device)
        print(f"epoch={epoch} train_loss={train_loss:.6f}")
        metric = train_loss
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, weights, cfg, device)
            metric = val_metrics["loss"]
            print(
                f"epoch={epoch} val_loss={val_metrics['loss']:.6f} "
                f"val_mae={val_metrics['mae']:.6f} val_nmae={val_metrics['nmae']:.6f}"
            )
        if metric < best:
            best = metric
            torch.save(
                {"model": model.state_dict(), "config": cfg, "epoch": epoch, "metric": metric},
                output_dir / "best.pt",
            )
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": epochs}, output_dir / "last.pt")


if __name__ == "__main__":
    main()
