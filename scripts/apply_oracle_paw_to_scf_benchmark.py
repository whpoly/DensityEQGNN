"""Replace smooth-density cases' PAW data with oracle_full occupancies.

This creates the paper-style, oracle-PAW-assisted SCF protocol used to
compare smooth charge-density predictions.  It deliberately leaks the
converged PAW one-center data and must not be reported as end-to-end
deployable acceleration.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.vasp.outputs import Chgcar


ASSISTED_CASES = ("ml", "charge3net", "oracle_smooth")


def normalized_augmentation(chgcar: Chgcar) -> dict[str, object]:
    if not chgcar.data_aug or "total" not in chgcar.data_aug:
        raise ValueError("oracle_full CHGCAR has no PAW augmentation data")
    total = chgcar.data_aug["total"]
    if isinstance(total, dict):
        copied: object = {
            int(atom_index): np.asarray(values).copy()
            for atom_index, values in total.items()
        }
    else:
        # Newer pymatgen versions preserve the complete augmentation block as
        # raw CHGCAR lines.  Keeping those lines verbatim is the most faithful
        # representation and is accepted by Chgcar.write_file.
        copied = copy.deepcopy(total)
    return {"total": copied}


def augmentation_equal(left: Chgcar, right: Chgcar) -> bool:
    left_total = left.data_aug.get("total", {})
    right_total = right.data_aug.get("total", {})
    if not isinstance(left_total, dict) or not isinstance(right_total, dict):
        return left_total == right_total
    if set(left_total) != set(right_total):
        return False
    return all(
        np.array_equal(
            np.asarray(left_total[atom_index]),
            np.asarray(right_total[atom_index]),
        )
        for atom_index in left_total
    )


def rewrite_case(case_path: Path, reference: Chgcar) -> None:
    original = Chgcar.from_file(case_path)
    original_grid = np.asarray(original.data["total"]).copy()
    temporary = case_path.with_suffix(".oracle-paw.tmp")
    Chgcar(
        Poscar(original.structure),
        {"total": original_grid},
        data_aug=copy.deepcopy(normalized_augmentation(reference)),
    ).write_file(temporary)
    content = temporary.read_bytes().replace(b"\r\n", b"\n")
    case_path.write_bytes(content)
    temporary.unlink()

    rewritten = Chgcar.from_file(case_path)
    if not np.array_equal(original_grid, np.asarray(rewritten.data["total"])):
        raise RuntimeError(f"Smooth density changed while rewriting {case_path}")
    if not augmentation_equal(rewritten, reference):
        raise RuntimeError(f"PAW occupancies do not match oracle_full: {case_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    args = parser.parse_args()

    root = args.benchmark_dir.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_ids = [int(value) for value in manifest["sample_ids"]]

    for sample_id in sample_ids:
        sample_root = root / f"sample_{sample_id:04d}"
        reference = Chgcar.from_file(sample_root / "oracle_full" / "CHGCAR")
        for case_name in ASSISTED_CASES:
            case_path = sample_root / case_name / "CHGCAR"
            if not case_path.exists():
                raise FileNotFoundError(case_path)
            rewrite_case(case_path, reference)
        print(f"sample_{sample_id:04d}: shared oracle PAW installed")

    manifest["cases"]["ml"] = (
        "ML smooth density with converged oracle PAW occupancies "
        "(oracle-assisted)"
    )
    manifest["cases"]["charge3net"] = (
        "Official ChargE3Net smooth density with converged oracle PAW "
        "occupancies (oracle-assisted)"
    )
    manifest["cases"]["oracle_smooth"] = (
        "DFT smooth grid with converged oracle PAW occupancies"
    )
    manifest["paw_protocol"] = (
        "Converged oracle_full PAW one-center occupancies are shared by ml, "
        "oracle_smooth, charge3net, and oracle_full. This isolates the smooth "
        "FFT-grid density and is not end-to-end deployable."
    )
    manifest["incar_common"]["LCHARG"] = False
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
