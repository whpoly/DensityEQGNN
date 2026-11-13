"""Evaluate VASP's POTCAR-derived SAD initialization on the NMC test set.

VASP's default ICHARG=2 initialization superposes the spherical atomic
pseudo-charge densities stored in POTCAR.  The tabulated form factor is
sampled in reciprocal space at 200 intervals per cutoff value; the table
contains 1,000 entries so that its oscillatory tail is also represented.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import lz4.frame
import numpy as np
from ase.io.vasp import read_vasp_configuration
from pymatgen.io.vasp.outputs import Chgcar
from scipy.interpolate import CubicSpline


POTCAR_ELEMENTS = ("Li", "Ni", "Co", "Mn", "O")
POTCAR_PREFIX = "potpaw54/potpaw_PBE/PAW_GGA_PBE"
TABLE_SIZE = 1000
INTERVALS_PER_CUTOFF = 200.0


@dataclass(frozen=True)
class AtomicFormFactor:
    symbol: str
    title: str
    zval: float
    reciprocal_cutoff: float
    values: np.ndarray
    spline: CubicSpline

    def evaluate(self, reciprocal_norm: np.ndarray) -> np.ndarray:
        table_coordinate = (
            reciprocal_norm * INTERVALS_PER_CUTOFF / self.reciprocal_cutoff
        )
        clipped = np.minimum(table_coordinate, TABLE_SIZE - 1)
        result = self.spline(clipped)
        return np.where(table_coordinate <= TABLE_SIZE - 1, result, 0.0)


def _floats_after(lines: list[str], marker: int, count: int) -> tuple[np.ndarray, int]:
    values: list[float] = []
    line_index = marker + 1
    while len(values) < count:
        values.extend(
            float(token.replace("D", "E")) for token in lines[line_index].split()
        )
        line_index += 1
    if len(values) != count:
        raise ValueError(
            f"Expected exactly {count} table values, found {len(values)}"
        )
    return np.asarray(values, dtype=np.float64), line_index


def read_form_factors(potcar_zip: Path) -> dict[str, AtomicFormFactor]:
    factors: dict[str, AtomicFormFactor] = {}
    with zipfile.ZipFile(potcar_zip) as archive:
        for symbol in POTCAR_ELEMENTS:
            member = f"{POTCAR_PREFIX}/{symbol}/POTCAR"
            lines = archive.read(member).decode("latin1").splitlines()
            marker = next(
                index
                for index, line in enumerate(lines)
                if line.strip() == "atomic pseudo charge-density"
            )
            values, next_line = _floats_after(lines, marker, TABLE_SIZE)
            reciprocal_cutoff = float(lines[next_line].split()[0])
            zval_line = next(line for line in lines if "ZVAL" in line)
            zval = float(zval_line.split("ZVAL", 1)[1].split("=", 1)[1].split()[0])
            if not np.isclose(values[0], zval, atol=1e-7, rtol=0.0):
                raise ValueError(
                    f"{symbol}: form factor at G=0 is {values[0]}, ZVAL is {zval}"
                )
            factors[symbol] = AtomicFormFactor(
                symbol=symbol,
                title=lines[0].strip(),
                zval=zval,
                reciprocal_cutoff=reciprocal_cutoff,
                values=values,
                spline=CubicSpline(
                    np.arange(TABLE_SIZE, dtype=np.float64),
                    values,
                    bc_type="natural",
                    extrapolate=False,
                ),
            )
    return factors


def reciprocal_grid(cell: np.ndarray, shape: tuple[int, int, int]) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray
]:
    frequencies = tuple(np.fft.fftfreq(size) * size for size in shape)
    h_x, h_y, h_z = np.meshgrid(*frequencies, indexing="ij", sparse=True)
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell).T
    g_x = (
        h_x * reciprocal[0, 0]
        + h_y * reciprocal[1, 0]
        + h_z * reciprocal[2, 0]
    )
    g_y = (
        h_x * reciprocal[0, 1]
        + h_y * reciprocal[1, 1]
        + h_z * reciprocal[2, 1]
    )
    g_z = (
        h_x * reciprocal[0, 2]
        + h_y * reciprocal[1, 2]
        + h_z * reciprocal[2, 2]
    )
    norm = np.sqrt(g_x * g_x + g_y * g_y + g_z * g_z)
    return frequencies, norm


def structure_factor(
    frequencies: tuple[np.ndarray, np.ndarray, np.ndarray],
    fractional_positions: np.ndarray,
) -> np.ndarray:
    n_x, n_y, n_z = (len(axis) for axis in frequencies)
    if not len(fractional_positions):
        return np.zeros((n_x, n_y, n_z), dtype=np.complex128)
    phase_x = np.exp(
        -2j * np.pi * np.outer(frequencies[0], fractional_positions[:, 0])
    )
    phase_y = np.exp(
        -2j * np.pi * np.outer(frequencies[1], fractional_positions[:, 1])
    )
    phase_z = np.exp(
        -2j * np.pi * np.outer(frequencies[2], fractional_positions[:, 2])
    )
    return np.einsum(
        "ia,ja,ka->ijk", phase_x, phase_y, phase_z, optimize=True
    )


def sad_density(
    symbols: np.ndarray,
    fractional_positions: np.ndarray,
    cell: np.ndarray,
    shape: tuple[int, int, int],
    form_factors: dict[str, AtomicFormFactor],
) -> np.ndarray:
    frequencies, reciprocal_norm = reciprocal_grid(cell, shape)
    coefficients = np.zeros(shape, dtype=np.complex128)
    for symbol in np.unique(symbols):
        factor = form_factors[str(symbol)]
        positions = fractional_positions[symbols == symbol]
        coefficients += factor.evaluate(reciprocal_norm) * structure_factor(
            frequencies, positions
        )
    # NumPy's inverse FFT includes 1/N. CHGCAR's main data block has a
    # grid mean equal to NELECT, so its Fourier coefficients require N.
    return np.fft.ifftn(coefficients * np.prod(shape)).real


def validate_structure(ase_atoms, chgcar: Chgcar, member_name: str) -> None:
    structure = chgcar.structure
    if ase_atoms.get_atomic_numbers().tolist() != list(structure.atomic_numbers):
        raise ValueError(f"ASE/pymatgen species mismatch in {member_name}")
    np.testing.assert_allclose(
        ase_atoms.cell.array,
        structure.lattice.matrix,
        atol=2e-5,
        rtol=0.0,
        err_msg=f"ASE/pymatgen cell mismatch in {member_name}",
    )
    np.testing.assert_allclose(
        ase_atoms.get_positions(),
        structure.cart_coords,
        atol=2e-4,
        rtol=0.0,
        err_msg=f"ASE/pymatgen positions mismatch in {member_name}",
    )


def evaluate_member(
    compressed,
    member_name: str,
    temporary_chgcar: Path,
    form_factors: dict[str, AtomicFormFactor],
) -> dict[str, float | int | str | list[int]]:
    with lz4.frame.open(compressed, "rb") as source, temporary_chgcar.open(
        "wb"
    ) as target_file:
        shutil.copyfileobj(source, target_file, length=8 * 1024 * 1024)

    with temporary_chgcar.open("r", encoding="utf-8") as handle:
        atoms = read_vasp_configuration(handle)
    atoms.set_pbc(True)
    chgcar = Chgcar.from_file(temporary_chgcar)
    validate_structure(atoms, chgcar, member_name)

    target = np.asarray(chgcar.data["total"], dtype=np.float64)
    shape = tuple(int(value) for value in target.shape)
    symbols = np.asarray(atoms.get_chemical_symbols())
    fractional = atoms.get_scaled_positions(wrap=True)
    prediction = sad_density(
        symbols,
        fractional,
        atoms.cell.array,
        shape,
        form_factors,
    )

    expected_electrons = sum(form_factors[symbol].zval for symbol in symbols)
    target_electrons = float(target.mean())
    predicted_electrons = float(prediction.mean())
    if not np.isclose(target_electrons, expected_electrons, atol=2e-4, rtol=0.0):
        raise ValueError(
            f"{member_name}: target integrates to {target_electrons}, "
            f"but POTCAR ZVAL gives {expected_electrons}"
        )
    if not np.isclose(
        predicted_electrons, expected_electrons, atol=2e-8, rtol=0.0
    ):
        raise ValueError(
            f"{member_name}: SAD integrates to {predicted_electrons}, "
            f"expected {expected_electrons}"
        )

    absolute_error = float(np.abs(prediction - target).sum())
    reference = float(np.abs(target).sum())
    return {
        "sample_id": int(Path(member_name).name.split(".", 1)[0]),
        "member": member_name,
        "formula": atoms.get_chemical_formula(),
        "grid_shape": list(shape),
        "grid_points": int(target.size),
        "electrons": expected_electrons,
        "target_electrons": target_electrons,
        "sad_electrons": predicted_electrons,
        "absolute_error": absolute_error,
        "absolute_reference": reference,
        "nmae": absolute_error / reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--potcar-zip", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--archives", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    test_ids = {int(sample_id) for sample_id in split["test"]}
    if args.limit is not None:
        test_ids = set(sorted(test_ids)[: args.limit])
    form_factors = read_form_factors(args.potcar_zip)
    records: list[dict[str, float | int | str | list[int]]] = []
    started = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix="nmc_sad_") as temporary:
        temporary_chgcar = Path(temporary) / "CHGCAR"
        for archive_path in args.archives:
            with tarfile.open(archive_path, "r:") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(
                        ".CHGCAR.lz4"
                    ):
                        continue
                    sample_id = int(Path(member.name).name.split(".", 1)[0])
                    if sample_id not in test_ids:
                        continue
                    compressed = archive.extractfile(member)
                    if compressed is None:
                        raise IOError(f"Could not read {member.name}")
                    record = evaluate_member(
                        compressed,
                        member.name,
                        temporary_chgcar,
                        form_factors,
                    )
                    records.append(record)
                    count = len(records)
                    if count == 1 or count % 10 == 0 or count == len(test_ids):
                        elapsed = time.perf_counter() - started
                        print(
                            f"{count}/{len(test_ids)} {member.name}: "
                            f"NMAE={100 * float(record['nmae']):.4f}% "
                            f"({elapsed:.1f}s)",
                            flush=True,
                        )

    found = {int(record["sample_id"]) for record in records}
    if found != test_ids:
        missing = sorted(test_ids - found)
        raise FileNotFoundError(f"Missing test members: {missing[:20]}")
    records.sort(key=lambda record: int(record["sample_id"]))
    sample_nmae = np.asarray(
        [float(record["nmae"]) for record in records], dtype=np.float64
    )
    total_error = sum(float(record["absolute_error"]) for record in records)
    total_reference = sum(
        float(record["absolute_reference"]) for record in records
    )
    result = {
        "baseline": "VASP ICHARG=2 superposition of atomic densities (SAD)",
        "dataset": "NMC official test split",
        "samples": len(records),
        "nmae_definition": "sum_grid(abs(SAD-SC))/sum_grid(abs(SC))",
        "macro_nmae": float(sample_nmae.mean()),
        "macro_nmae_percent": float(100.0 * sample_nmae.mean()),
        "macro_standard_error_percent": float(
            100.0 * sample_nmae.std(ddof=1) / np.sqrt(len(sample_nmae))
        ),
        "median_nmae_percent": float(100.0 * np.median(sample_nmae)),
        "p90_nmae_percent": float(100.0 * np.quantile(sample_nmae, 0.9)),
        "micro_nmae": float(total_error / total_reference),
        "micro_nmae_percent": float(100.0 * total_error / total_reference),
        "potcars": {
            symbol: {
                "title": factor.title,
                "zval": factor.zval,
                "reciprocal_cutoff": factor.reciprocal_cutoff,
            }
            for symbol, factor in form_factors.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
