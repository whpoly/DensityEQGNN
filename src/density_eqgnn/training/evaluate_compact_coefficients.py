"""Evaluate a coefficient-supervised compact model checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from density_eqgnn.compact_basis import CompactBasisConfig, check_basis_checkpoint
from density_eqgnn.models import build_compact_coefficient_model, load_compact_model_state
from density_eqgnn.periodic import check_periodic_checkpoint

from .config import load_config
from .train import build_loader, resolve_device
from .train_compact_coefficients import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
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
    model = build_compact_coefficient_model(
        checkpoint["model_config"]
    ).to(device)
    load_compact_model_state(model,checkpoint)
    basis = CompactBasisConfig(**checkpoint["basis_config"])
    check_basis_checkpoint(checkpoint,basis)
    metrics = evaluate(
        model,
        build_loader(config, args.split),
        basis,
        checkpoint["coefficient_scale"].to(device),
        device,
    )
    text = json.dumps(metrics, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
