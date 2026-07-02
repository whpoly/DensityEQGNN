"""VASP CHGCAR conversion utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .sampling import cartesian_grid, normalize_density, random_probe_indices


def chgcar_to_npz(
    chgcar_path: str | Path,
    output_path: str | Path,
    num_probes: int | None = None,
    seed: int | None = None,
    density_normalization: str = "vasp_divide_volume",
) -> Path:
    """Convert one VASP CHGCAR into the project NPZ format.

    This uses pymatgen lazily so the rest of the package can be imported without it.
    """
    try:
        from pymatgen.io.vasp.outputs import Chgcar
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pymatgen is required to convert CHGCAR files") from exc

    chgcar_path = Path(chgcar_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chg = Chgcar.from_file(str(chgcar_path))
    structure = chg.structure
    cell = np.asarray(structure.lattice.matrix, dtype=np.float64)
    cell_volume = abs(float(np.linalg.det(cell)))
    grid = np.asarray(chg.data["total"], dtype=np.float64)
    full_density = normalize_density(grid.reshape(-1), density_normalization, cell_volume)
    probe_positions = cartesian_grid(cell, grid.shape)

    full_probe_volume = cell_volume / grid.size
    electron_count = np.float32(float(np.sum(full_density) * full_probe_volume))

    indices = random_probe_indices(full_density.shape[0], num_probes, seed)
    density = full_density[indices]
    probe_positions = probe_positions[indices]
    probe_volume = np.float32(cell_volume / len(indices))

    np.savez_compressed(
        output_path,
        atomic_numbers=np.asarray(structure.atomic_numbers, dtype=np.int64),
        positions=np.asarray(structure.cart_coords, dtype=np.float32),
        cell=cell.astype(np.float32),
        probe_positions=probe_positions.astype(np.float32),
        density=density.astype(np.float32),
        probe_volume=probe_volume,
        electron_count=electron_count,
    )
    return output_path
