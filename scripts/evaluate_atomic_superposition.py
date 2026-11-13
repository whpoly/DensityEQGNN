"""Evaluate a fixed spherical atomic-density superposition baseline.

The public QM9/VASP density archive contains only the converged CHGCAR, not
the POTCAR-derived density that VASP used before its first electronic step.
Consequently the exact VASP SAD NMAE cannot be recovered from this archive.

This script constructs the closest reproducible analogue available from the
benchmark itself: for every chemical element, average the fitted l=0
coefficients over training atoms, set all l>0 coefficients to zero, and
superpose these fixed spherical atomic densities on test structures.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import yaml

from density_eqgnn.basis import (
    DensityBasisConfig,
    design_matrix_numpy,
    reconstruct_density_numpy,
)


def _read_split(root: Path, name: str) -> list[Path]:
    return [
        root / line.strip()
        for line in (root / name).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _element_design(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    probes: np.ndarray,
    cell: np.ndarray,
    elements: list[int],
    basis: DensityBasisConfig,
) -> np.ndarray:
    radial = design_matrix_numpy(
        positions, probes, cell, replace(basis, lmax=0)
    ).reshape(len(probes), len(positions), basis.num_radial)
    design = np.zeros(
        (len(probes), len(elements) * basis.num_radial), dtype=np.float64
    )
    for element_index, atomic_number in enumerate(elements):
        atom_mask = atomic_numbers == atomic_number
        if np.any(atom_mask):
            start = element_index * basis.num_radial
            design[:, start : start + basis.num_radial] = radial[
                :, atom_mask
            ].sum(axis=1)
    return design


def fit_fixed_atomic_coefficients(
    paths: list[Path], basis: DensityBasisConfig, ridge: float = 1e-8
) -> tuple[list[int], np.ndarray]:
    element_set: set[int] = set()
    for path in paths:
        with np.load(path, allow_pickle=False) as sample:
            element_set.update(int(value) for value in sample["atomic_numbers"])
    elements = sorted(element_set)
    size = len(elements) * basis.num_radial
    gram = np.zeros((size, size), dtype=np.float64)
    rhs = np.zeros(size, dtype=np.float64)
    for path in paths:
        with np.load(path, allow_pickle=False) as sample:
            design = _element_design(
                sample["atomic_numbers"].astype(int),
                sample["positions"].astype(np.float64),
                sample["probe_positions"].astype(np.float64),
                sample["cell"].astype(np.float64),
                elements,
                basis,
            )
            density = sample["density"].astype(np.float64)
        gram += design.T @ design
        rhs += design.T @ density
    scale = float(np.trace(gram) / max(size, 1))
    coefficients = np.linalg.solve(
        gram + ridge * max(scale, 1.0) * np.eye(size), rhs
    )
    return elements, coefficients.reshape(len(elements), basis.num_radial)


def evaluate(
    paths: list[Path],
    elements: list[int],
    atomic_coefficients: np.ndarray,
    basis: DensityBasisConfig,
) -> dict[str, float | int | list[int]]:
    absolute_error = 0.0
    oracle_absolute_error = 0.0
    reference = 0.0
    sample_nmae: list[float] = []
    oracle_sample_nmae: list[float] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as sample:
            atomic_numbers = sample["atomic_numbers"].astype(int)
            positions = sample["positions"].astype(np.float64)
            cell = sample["cell"].astype(np.float64)
            probes = sample["probe_positions"].astype(np.float64)
            target = sample["density"].astype(np.float64)
            oracle_coefficients = sample["coefficients"].astype(np.float64)
        design = _element_design(
            atomic_numbers, positions, probes, cell, elements, basis
        )
        prediction = design @ atomic_coefficients.reshape(-1)
        oracle_prediction = reconstruct_density_numpy(
            oracle_coefficients, positions, probes, cell, basis
        )
        error = float(np.abs(prediction - target).sum())
        oracle_error = float(np.abs(oracle_prediction - target).sum())
        scale = max(float(np.abs(target).sum()), 1e-12)
        absolute_error += error
        oracle_absolute_error += oracle_error
        reference += scale
        sample_nmae.append(error / scale)
        oracle_sample_nmae.append(oracle_error / scale)
    return {
        "samples": len(paths),
        "density_nmae": absolute_error / max(reference, 1e-12),
        "oracle_density_nmae": oracle_absolute_error / max(reference, 1e-12),
        "sample_density_nmae_mean": float(np.mean(sample_nmae)),
        "sample_density_nmae_median": float(np.median(sample_nmae)),
        "sample_density_nmae_p90": float(np.quantile(sample_nmae, 0.9)),
        "sample_oracle_nmae_mean": float(np.mean(oracle_sample_nmae)),
        "sample_oracle_nmae_median": float(np.median(oracle_sample_nmae)),
        "elements": elements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    basis = DensityBasisConfig(**config["basis"])
    root = args.data_root
    train_paths = _read_split(root, config["data"]["train_split"])
    test_paths = _read_split(root, config["data"]["test_split"])
    elements, atomic_coefficients = fit_fixed_atomic_coefficients(train_paths, basis)
    metrics = evaluate(test_paths, elements, atomic_coefficients, basis)
    metrics["baseline"] = "train-fitted fixed spherical atomic superposition"
    metrics["atomic_coefficients"] = {
        str(atomic_number): coefficients.tolist()
        for atomic_number, coefficients in zip(
            elements, atomic_coefficients, strict=True
        )
    }
    text = json.dumps(metrics, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
