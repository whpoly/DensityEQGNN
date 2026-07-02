"""Probe/grid sampling helpers."""

from __future__ import annotations

import numpy as np


def fractional_grid(shape: tuple[int, int, int]) -> np.ndarray:
    """Return fractional coordinates at voxel centers for a regular 3D grid."""
    nx, ny, nz = shape
    axes = [
        (np.arange(n, dtype=np.float64) + 0.5) / float(n)
        for n in (nx, ny, nz)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1).reshape(-1, 3)


def cartesian_grid(cell: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Return Cartesian probe coordinates for a regular grid in a periodic cell."""
    frac = fractional_grid(shape)
    return frac @ np.asarray(cell, dtype=np.float64)


def random_probe_indices(
    num_grid_points: int,
    num_probes: int | None,
    seed: int | None = None,
) -> np.ndarray:
    """Sample grid indices without replacement; return all indices when num_probes is None."""
    if num_probes is None or num_probes >= num_grid_points:
        return np.arange(num_grid_points, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(num_grid_points, size=num_probes, replace=False)).astype(np.int64)


def normalize_density(density: np.ndarray, mode: str, cell_volume: float) -> np.ndarray:
    """Normalize CHGCAR-like density arrays.

    mode="raw" leaves values untouched. mode="vasp_divide_volume" divides by the
    cell volume, which is commonly needed when values are stored as rho * volume.
    """
    density = np.asarray(density, dtype=np.float64)
    if mode == "raw":
        return density
    if mode == "vasp_divide_volume":
        return density / float(cell_volume)
    raise ValueError(f"Unknown density normalization mode: {mode}")

