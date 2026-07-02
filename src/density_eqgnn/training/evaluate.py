"""Checkpoint evaluation entry point."""

from __future__ import annotations

import argparse

import torch

from density_eqgnn.models import DensitySupervisedEQGNN
from density_eqgnn.models.model import ModelConfig
from density_eqgnn.training.losses import LossWeights
from density_eqgnn.training.train import build_dataloader, evaluate, resolve_device

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["training"].get("device", "auto"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DensitySupervisedEQGNN(ModelConfig(**cfg["model"])).to(device)
    model.load_state_dict(checkpoint["model"])

    loader = build_dataloader(cfg, args.split)
    metrics = evaluate(model, loader, LossWeights(**cfg.get("loss_weights", {})), cfg, device)
    print(metrics)


if __name__ == "__main__":
    main()
