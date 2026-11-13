"""Evaluate a saved coefficient model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from density_eqgnn.basis import DensityBasisConfig
from density_eqgnn.models import EquivariantCoefficientGNN, ModelConfig
from density_eqgnn.periodic import check_periodic_checkpoint

from .config import load_config
from .train import build_loader, evaluate, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nmc_small.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["training"].get("device", "auto"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    check_periodic_checkpoint(
        checkpoint,
        pbc=bool(checkpoint["basis_config"].get("pbc", False))
        or bool(checkpoint["model_config"].get("pbc", False)),
    )
    model = EquivariantCoefficientGNN(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate(
        model,
        build_loader(config, args.split),
        DensityBasisConfig(**checkpoint["basis_config"]),
        device,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
