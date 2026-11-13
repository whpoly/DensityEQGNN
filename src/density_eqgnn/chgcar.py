"""VASP CHGCAR to fitted atom-coefficient sample conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from pymatgen.io.vasp.outputs import Chgcar

from .basis import (
    DensityBasisConfig,
    fit_density_coefficients,
    fit_density_coefficients_blockwise,
    fit_density_coefficients_stockholder,
    reconstruct_density_numpy,
)


def vasp_grid(cell: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Return VASP grid coordinates at fractional indices i/N."""

    axes = [np.arange(size, dtype=np.float64) / size for size in shape]
    fractional = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return fractional @ cell


def extract_chgcar_sample(
    chgcar_path: str | Path,
    output_path: str | Path,
    config: DensityBasisConfig,
    ridge: float = 1e-8,
    fit_probes: int = 20000,
    eval_probes: int = 4096,
    seed: int = 0,
    fit_mode: str = "global",
) -> dict[str, float]:
    """Fit a CHGCAR and save structure, coefficients, and a small eval grid."""

    chgcar = Chgcar.from_file(str(chgcar_path))
    structure = chgcar.structure
    atoms = Atoms(
        numbers=structure.atomic_numbers,
        positions=structure.cart_coords,
        cell=structure.lattice.matrix,
        pbc=True,
    )
    cell = atoms.cell.array
    raw_grid = np.asarray(chgcar.data["total"], dtype=np.float64)
    # VASP CHGCAR stores grid density multiplied by cell volume.
    density = raw_grid.reshape(-1) / atoms.get_volume()
    probes = vasp_grid(cell, raw_grid.shape)
    probe_volume = atoms.get_volume() / len(probes)
    rng = np.random.default_rng(seed)
    fit_count = min(fit_probes, len(density))
    high_count = fit_count // 4
    high_density = np.argpartition(np.abs(density), -high_count)[-high_count:]
    remaining = np.setdiff1d(np.arange(len(density)), high_density, assume_unique=False)
    random_count = fit_count - high_count
    random_indices = rng.choice(remaining, random_count, replace=False)
    fit_indices = np.sort(np.concatenate([high_density, random_indices]))

    if fit_mode == "global":
        coefficients, _ = fit_density_coefficients(
            density[fit_indices],
            atoms.get_positions(),
            probes[fit_indices],
            cell,
            config,
            ridge=ridge,
            conserve_charge=False,
            probe_volume=1.0,
        )
    elif fit_mode == "stockholder":
        coefficients, _ = fit_density_coefficients_stockholder(
            density[fit_indices],
            atoms.get_positions(),
            probes[fit_indices],
            cell,
            config,
            ridge=ridge,
        )
    elif fit_mode == "blockwise":
        coefficients, _ = fit_density_coefficients_blockwise(
            density[fit_indices],
            atoms.get_positions(),
            probes[fit_indices],
            cell,
            config,
            ridge=ridge,
        )
    else:
        raise ValueError(f"Unknown fit_mode={fit_mode!r}")

    count = min(eval_probes, len(density))
    indices = np.sort(rng.choice(len(density), count, replace=False))
    reconstructed = reconstruct_density_numpy(
        coefficients,
        atoms.get_positions(),
        probes[indices],
        cell,
        config,
    )
    residual = reconstructed - density[indices]
    mae = float(np.mean(np.abs(residual)))
    target_charge = float(np.mean(density[indices]) * atoms.get_volume())
    fitted_charge = float(np.mean(reconstructed) * atoms.get_volume())
    diagnostics = {
        "mae": mae,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "nmae": mae / max(float(np.mean(np.abs(density[indices]))), 1e-12),
        "target_charge": target_charge,
        "fitted_charge": fitted_charge,
        "charge_error": fitted_charge - target_charge,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        atomic_numbers=atoms.get_atomic_numbers().astype(np.int64),
        positions=atoms.get_positions().astype(np.float32),
        cell=cell.astype(np.float32),
        coefficients=coefficients,
        probe_positions=probes[indices].astype(np.float32),
        density=density[indices].astype(np.float32),
        probe_volume=np.float32(probe_volume),
        fit_nmae=np.float32(diagnostics["nmae"]),
        fit_mode=np.asarray(fit_mode),
    )
    return diagnostics
