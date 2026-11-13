"""POTCAR SACD alternative and an independent check against published native VASP grids."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile

import numpy as np
from pymatgen.io.vasp.outputs import Chgcar
from scipy.interpolate import CubicSpline

from scripts.benchmark_public_density import save
from scripts.evaluate_vasp_sad_nmc import TABLE_SIZE, _floats_after, reciprocal_grid, structure_factor


@lru_cache(maxsize=32)
def atomic_table(path):
    lines = Path(path).read_text(encoding="latin1").splitlines()
    # The local potential and atomic charge share the 1000-point reciprocal mesh.
    # The scalar AFTER the charge table belongs to the nonlocal projectors.
    header = next(i for i, line in enumerate(lines) if line.strip() == "local part")
    maximum = float(lines[header + 1])
    start = next(i for i, line in enumerate(lines) if line.strip() == "atomic pseudo charge-density")
    values, _ = _floats_after(lines, start, TABLE_SIZE)
    q = np.arange(len(values)) * (maximum / len(values))
    return CubicSpline(q, values, bc_type="natural", extrapolate=False), values[0]


def reconstruct(symbols, pos, cell, shape, pp):
    shape = tuple(map(int, shape))
    freq, g = reciprocal_grid(cell, shape)
    frac = pos @ np.linalg.inv(cell)
    coeff = np.zeros(shape, dtype=np.complex128)
    nelec = 0.0
    for element in np.unique(symbols):
        ff, zval = atomic_table(pp[element]["path_pp"])
        atoms = frac[symbols == element]
        coeff += np.nan_to_num(ff(g), nan=0.0) * structure_factor(freq, atoms)
        nelec += zval * len(atoms)
    # The Nyquist planes of even grids are excluded in the native real FFT.
    for axis, length in enumerate(shape):
        if length % 2 == 0:
            slices = [slice(None)] * 3
            slices[axis] = length // 2
            coeff[tuple(slices)] = 0.0
    volume = abs(np.linalg.det(cell))
    density = np.fft.ifftn(coeff).real * (np.prod(shape) / volume)
    if not np.isclose(density.mean() * volume, nelec, rtol=0, atol=2e-7):
        raise ValueError("Atomic-density reconstruction changed the electron count")
    return density


def cached_sacd(root, index, z, pos, cell, shape, pp):
    from ase.data import chemical_symbols

    geometry = hashlib.sha256()
    for array in (z, pos, cell, np.asarray(shape, dtype="<i8")):
        geometry.update(np.ascontiguousarray(array).tobytes())
    path = root / f"{index}_{geometry.hexdigest()[:16]}.npy"
    if not path.is_file():
        root.mkdir(exist_ok=True, parents=True)
        symbols = np.array([chemical_symbols[number] for number in z])
        density = reconstruct(symbols, pos, cell, shape, pp)
        with tempfile.NamedTemporaryFile(dir=root, delete=False, suffix=".tmp") as tmp:
            np.save(tmp, density.astype(np.float32))
        Path(tmp.name).replace(path)
    return np.load(path)


def validate_native_examples():
    from scripts.nmc_eden_fullgrid import ROOT, PUBLIC, potcar_files

    records, unavailable = [], []
    with ZipFile(PUBLIC / "data.zip") as release, tempfile.TemporaryDirectory() as tmp:
        for member in sorted(n for n in release.namelist() if "/sad/" in n and n.endswith("/CHGCAR")):
            system = member.split("/")[-2]
            path = Path(tmp) / "CHGCAR"
            path.write_bytes(release.read(member))
            native = Chgcar.from_file(path)
            structure = native.structure
            symbols = np.array([a.symbol for a in structure.species])
            try:
                pp = potcar_files(set(symbols), "data/vasp/pseudo.json")
            except (KeyError, ValueError) as exc:
                unavailable.append(dict(native_file=member, reason=str(exc)))
                continue
            target = native.data["total"] / structure.volume
            pred = reconstruct(symbols, structure.cart_coords, structure.lattice.matrix, target.shape, pp)
            delta = pred - target
            records.append(dict(system=system, native_file=member, elements=sorted(set(symbols.tolist())), shape=list(target.shape),
                                points=target.size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                nelec_native=float(target.mean() * structure.volume),
                                electron_count_difference=float(delta.mean() * structure.volume),
                                relative_l1_pct=100 * np.abs(delta).sum()/np.abs(target).sum(),
                                max_abs_e_per_A3=float(np.abs(delta).max())))
    result = dict(source="Author VASP zero-step CHGCAR, matched POTCAR hashes",
                  records=records, unmatched_potcars=unavailable)
    save(ROOT / "sacd_native_examples.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    validate_native_examples()
