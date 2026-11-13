"""Predict per-atom coefficients and reconstruct a density grid for a structure."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from ase.io import read

from density_eqgnn.basis import DensityBasisConfig, reconstruct_density_numpy
from density_eqgnn.chgcar import vasp_grid
from density_eqgnn.models import EquivariantCoefficientGNN, ModelConfig


def parse_grid(text: str) -> tuple[int, int, int]:
    values = tuple(int(value) for value in text.lower().replace("x", ",").split(","))
    if len(values) != 3 or min(values) < 1:
        raise argparse.ArgumentTypeError("grid must look like 64x64x64")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure", required=True, help="Any ASE-readable POSCAR/CIF file")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Output .npz file")
    parser.add_argument("--grid", type=parse_grid, default=(32, 32, 32))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if args.device != "auto"
        else "cpu"
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = EquivariantCoefficientGNN(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    atoms = read(args.structure)
    numbers = torch.as_tensor(atoms.get_atomic_numbers(), dtype=torch.long, device=device)[None]
    positions = torch.as_tensor(atoms.get_positions(), dtype=torch.float32, device=device)[None]
    cell = torch.as_tensor(atoms.cell.array, dtype=torch.float32, device=device)[None]
    mask = torch.ones_like(numbers, dtype=torch.bool)
    with torch.no_grad():
        coefficients = model(numbers, positions, mask, cell)["coefficients"][0].cpu().numpy()

    basis = DensityBasisConfig(**checkpoint["basis_config"])
    probes = vasp_grid(atoms.cell.array, args.grid)
    density = reconstruct_density_numpy(
        coefficients,
        atoms.get_positions(),
        probes,
        atoms.cell.array,
        basis,
    ).reshape(args.grid)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        atomic_numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        cell=atoms.cell.array,
        coefficients=coefficients,
        density=density.astype(np.float32),
        grid=np.asarray(args.grid, dtype=np.int64),
    )
    print(f"wrote coefficients and {args.grid} density grid to {output}")


if __name__ == "__main__":
    main()
