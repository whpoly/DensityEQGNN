"""Prepare a paired VASP benchmark for charge-density SCF initialization.

The benchmark creates four cases for every selected NMC test structure:

* ``sad``: VASP's native ICHARG=2 superposition of atomic densities.
* ``ml``: the model-predicted smooth density with oracle PAW occupancies.
* ``oracle_smooth``: the DFT smooth density with oracle PAW occupancies.
* ``oracle_full``: the complete converged DFT CHGCAR, including PAW data.

The three externally supplied smooth-density cases use the same converged
one-center PAW occupancies.  This isolates the effect of the smooth FFT-grid
density and follows the ChargE3Net paper's oracle-PAW-assisted SCF protocol.
It is not an end-to-end deployable benchmark because the PAW data leak the
converged DFT result.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import tarfile
import zipfile
from pathlib import Path

import lz4.frame
import numpy as np
import torch
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar
from pymatgen.io.vasp.outputs import Chgcar

from density_eqgnn.chgcar import vasp_grid
from density_eqgnn.compact_basis import CompactBasisConfig, reconstruct_compact_density, check_basis_checkpoint
from density_eqgnn.models import build_compact_coefficient_model, load_compact_model_state


POTCAR_PREFIX = "potpaw54/potpaw_PBE/PAW_GGA_PBE"
DEFAULT_SAMPLE_IDS = (1, 3, 6)


def archive_members(raw_dir: Path) -> dict[int, tuple[Path, str]]:
    """Index NMC CHGCAR members by integer sample ID."""

    result: dict[int, tuple[Path, str]] = {}
    archives = sorted(raw_dir.glob("cellrelax_*.tar"))
    if not archives:
        raise FileNotFoundError(f"No cellrelax_*.tar files under {raw_dir}")
    for archive_path in archives:
        with tarfile.open(archive_path, "r:") as archive:
            for member in archive:
                if member.isfile() and member.name.endswith(".CHGCAR.lz4"):
                    sample_id = int(Path(member.name).name.split(".", 1)[0])
                    result[sample_id] = (archive_path, member.name)
    return result


def extract_chgcar(archive_path: Path, member_name: str, destination: Path) -> None:
    """Extract one lz4-compressed CHGCAR from a non-compressed tar."""

    with tarfile.open(archive_path, "r:") as archive:
        compressed = archive.extractfile(member_name)
        if compressed is None:
            raise OSError(f"Could not read {member_name} from {archive_path}")
        with lz4.frame.open(compressed, "rb") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def oracle_augmentation(chgcar: Chgcar) -> dict[str, object]:
    """Return the converged per-atom PAW occupancies for paper-style SCF tests."""

    if not chgcar.data_aug or "total" not in chgcar.data_aug:
        raise ValueError("Reference CHGCAR has no total PAW augmentation data")
    total = chgcar.data_aug["total"]
    if isinstance(total, dict):
        copied: object = {
            int(atom_index): np.asarray(values).copy()
            for atom_index, values in total.items()
        }
    else:
        copied = copy.deepcopy(total)
    return {"total": copied}


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_compact_coefficient_model(checkpoint["model_config"]).to(device)
    load_compact_model_state(model,checkpoint)
    model.eval()
    basis = CompactBasisConfig(**checkpoint["basis_config"])
    check_basis_checkpoint(checkpoint,basis)
    return model, basis


@torch.no_grad()
def predict_density(
    chgcar: Chgcar,
    model,
    basis: CompactBasisConfig,
    device: torch.device,
    chunk_size: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Predict, positivity-project, and charge-normalize a CHGCAR grid."""

    structure = chgcar.structure
    positions_np = np.asarray(structure.cart_coords, dtype=np.float32)
    cell_np = np.asarray(structure.lattice.matrix, dtype=np.float32)
    numbers_np = np.asarray(structure.atomic_numbers, dtype=np.int64)
    grid_shape = tuple(int(value) for value in chgcar.data["total"].shape)
    probes_np = vasp_grid(cell_np, grid_shape).astype(np.float32)

    atomic_numbers = torch.from_numpy(numbers_np)[None].to(device)
    positions = torch.from_numpy(positions_np)[None].to(device)
    cell = torch.from_numpy(cell_np)[None].to(device)
    atom_mask = torch.ones((1, len(numbers_np)), dtype=torch.bool, device=device)
    coefficients = model(atomic_numbers, positions, atom_mask, cell)["coefficients"]

    chunks: list[np.ndarray] = []
    for start in range(0, len(probes_np), chunk_size):
        stop = min(start + chunk_size, len(probes_np))
        probe_positions = torch.from_numpy(probes_np[start:stop])[None].to(device)
        probe_mask = torch.ones((1, stop - start), dtype=torch.bool, device=device)
        density = reconstruct_compact_density(
            coefficients,
            positions,
            probe_positions,
            atom_mask,
            probe_mask,
            cell,
            basis,
        )[0]
        chunks.append(density.detach().cpu().numpy())

    density_per_volume = np.concatenate(chunks).astype(np.float64)
    volume = float(structure.volume)
    raw_prediction = density_per_volume * volume
    expected_electrons = float(np.asarray(chgcar.data["total"]).mean())
    negative_mask = raw_prediction < 0.0
    negative_l1 = float(np.abs(raw_prediction[negative_mask]).sum())
    raw_l1 = float(np.abs(raw_prediction).sum())

    # A VASP initial density should be nonnegative and integrate to NELECT.
    # Projection is deterministic and is applied only to the deployable ML case.
    projected = np.maximum(raw_prediction, 0.0)
    projected_mean = float(projected.mean())
    if not np.isfinite(projected_mean) or projected_mean <= 0.0:
        raise ValueError("Predicted density cannot be normalized")
    projected *= expected_electrons / projected_mean
    projected = projected.reshape(grid_shape)

    target = np.asarray(chgcar.data["total"], dtype=np.float64)
    diagnostics = {
        "expected_electrons": expected_electrons,
        "raw_predicted_electrons": float(raw_prediction.mean()),
        "projected_electrons": float(projected.mean()),
        "negative_grid_fraction": float(negative_mask.mean()),
        "negative_l1_fraction": negative_l1 / max(raw_l1, 1e-30),
        "postprocessed_nmae": float(np.abs(projected - target).sum() / np.abs(target).sum()),
    }
    return projected, diagnostics


def combined_potcar(
    archive: zipfile.ZipFile,
    site_symbols: list[str],
    destination: Path,
) -> list[str]:
    """Write standard PBE POTCARs in POSCAR species order."""

    titles: list[str] = []
    with destination.open("wb") as output:
        for symbol in site_symbols:
            member = f"{POTCAR_PREFIX}/{symbol}/POTCAR"
            try:
                content = archive.read(member)
            except KeyError as error:
                raise FileNotFoundError(f"{member} is absent from the POTCAR archive") from error
            output.write(content)
            if not content.endswith(b"\n"):
                output.write(b"\n")
            titles.append(content.decode("latin1").splitlines()[0].strip())
    return titles


def common_incar(ediff: float, nelm: int) -> dict[str, object]:
    """Static settings shared by all initialization cases."""

    return {
        "SYSTEM": "NMC charge-density SCF initialization benchmark",
        "GGA": "PS",
        "ISTART": 0,
        "ALGO": "Normal",
        "NELM": nelm,
        "ISPIN": 1,
        "ISYM": 0,
        "LREAL": "Auto",
        "PREC": "Normal",
        "EDIFF": ediff,
        "NELMIN": 2,
        "ISMEAR": 0,
        "SIGMA": 0.1,
        "NSW": 0,
        "IBRION": -1,
        "LWAVE": False,
        # Keep the supplied initial CHGCAR intact for reproducibility.  The
        # benchmark only needs OSZICAR/OUTCAR convergence data.
        "LCHARG": False,
        # Required to preserve d-channel PAW information on restart.
        "LMAXMIX": 4,
    }


def write_case(
    case_dir: Path,
    poscar: Poscar,
    potcar_bytes: bytes,
    incar_settings: dict[str, object],
    initial_chgcar: Path | None,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    poscar.write_file(case_dir / "POSCAR")
    (case_dir / "POTCAR").write_bytes(potcar_bytes)
    Kpoints.monkhorst_automatic((3, 3, 1)).write_file(case_dir / "KPOINTS")
    Incar(incar_settings).write_file(case_dir / "INCAR")
    if initial_chgcar is not None:
        shutil.copy2(initial_chgcar, case_dir / "CHGCAR")
    for file_name in ("INCAR", "POSCAR", "KPOINTS", "CHGCAR"):
        path = case_dir / file_name
        if path.exists():
            content = path.read_bytes().replace(b"\r\n", b"\n")
            if file_name == "INCAR":
                content = content.replace(b" = True\n", b" = .TRUE.\n")
                content = content.replace(b" = False\n", b" = .FALSE.\n")
            path.write_bytes(content)


def write_runner(output_dir: Path, case_count: int) -> None:
    """Write portable local/cluster runners without assuming a scheduler."""

    runner = """#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CASE_DIRECTORY" >&2
  exit 2
fi

case_dir="$(cd "$1" && pwd)"
: "${VASP_COMMAND:?Set VASP_COMMAND, e.g. 'srun vasp_std' or 'mpirun -np 32 vasp_std'}"
cd "$case_dir"
if [[ -f CHGCAR && ! -e CHGCAR.initial ]]; then
  cp -p CHGCAR CHGCAR.initial
fi
start_epoch="$(date +%s)"
printf '[%s] START case=%s host=%s command=%s\\n' \
  "$(date -Is)" "$case_dir" "$(hostname)" "$VASP_COMMAND"
set +e
eval "$VASP_COMMAND" > vasp.stdout 2> vasp.stderr
status=$?
set -e
end_epoch="$(date +%s)"
printf '%s\\n' "$status" > exit_status.txt
printf '%s\\n' "$((end_epoch-start_epoch))" > wall_seconds.txt
printf '[%s] END case=%s status=%s wall_seconds=%s\\n' \
  "$(date -Is)" "$case_dir" "$status" "$((end_epoch-start_epoch))"
if [[ "$status" -ne 0 ]]; then
  echo "VASP failed; last 20 lines of $case_dir/vasp.stderr:" >&2
  tail -n 20 vasp.stderr >&2 || true
fi
exit "$status"
"""
    run_all = """#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
while IFS= read -r case_dir; do
  [[ -z "$case_dir" ]] && continue
  bash "$root/run_case.sh" "$root/$case_dir"
done < "$root/cases.txt"
"""
    slurm = """#!/usr/bin/env bash
#SBATCH --job-name=nmc-scf
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --time=02:00:00
#SBATCH --array=1-__CASE_COUNT__
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail
if [[ -n "${BENCHMARK_ROOT:-}" ]]; then
  root="$(cd "$BENCHMARK_ROOT" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/cases.txt" ]]; then
  root="$(cd "$SLURM_SUBMIT_DIR" && pwd)"
else
  script_dir="$(cd "$(dirname "$0")" && pwd)"
  if [[ -f "$script_dir/cases.txt" ]]; then
    root="$script_dir"
  else
    echo "Cannot locate cases.txt." >&2
    echo "Submit from the benchmark directory or export BENCHMARK_ROOT=/absolute/path." >&2
    exit 2
  fi
fi
case_rel="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$root/cases.txt")"
if [[ -z "$case_rel" ]]; then
  echo "No case for array index ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi

# Load the site-specific VASP module before this line if needed.
if [[ -z "${VASP_COMMAND:-}" ]]; then
  vasp_exe="$(command -v vasp_std || true)"
  if [[ -z "$vasp_exe" ]]; then
    echo "vasp_std is not in PATH. Load the cluster VASP module or export" >&2
    echo "VASP_COMMAND='mpirun -np 32 /absolute/path/to/vasp_std' before sbatch." >&2
    exit 127
  fi
  if command -v mpirun >/dev/null 2>&1; then
    VASP_COMMAND="mpirun -np ${SLURM_NTASKS:-1} $vasp_exe"
  else
    VASP_COMMAND="srun $vasp_exe"
  fi
fi
export VASP_COMMAND
printf '[%s] array_task=%s root=%s case=%s command=%s\\n' \
  "$(date -Is)" "$SLURM_ARRAY_TASK_ID" "$root" "$case_rel" "$VASP_COMMAND"
bash "$root/run_case.sh" "$root/$case_rel"
""".replace("__CASE_COUNT__", str(case_count))
    (output_dir / "run_case.sh").write_text(runner, encoding="utf-8", newline="\n")
    (output_dir / "run_all.sh").write_text(run_all, encoding="utf-8", newline="\n")
    (output_dir / "submit_slurm.sh").write_text(slurm, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--potcar-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-ids", type=int, nargs="+", default=DEFAULT_SAMPLE_IDS)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--ediff", type=float, default=1e-6)
    parser.add_argument("--nelm", type=int, default=180)
    args = parser.parse_args()

    split = json.loads(args.split.read_text(encoding="utf-8"))
    test_ids = {int(value) for value in split["test"]}
    invalid = sorted(set(args.sample_ids) - test_ids)
    if invalid:
        raise ValueError(f"Requested IDs are not in the official test split: {invalid}")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )
    model, basis = load_model(args.checkpoint, device)
    members = archive_members(args.raw_dir)
    missing = sorted(set(args.sample_ids) - set(members))
    if missing:
        raise FileNotFoundError(f"Samples are absent from raw archives: {missing}")

    output_dir = args.output_dir.resolve()
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(
            f"{output_dir} already contains a benchmark manifest; "
            "choose a new output directory to avoid mixing VASP results"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir / "_source"
    temporary_dir.mkdir(exist_ok=True)
    records: list[dict[str, object]] = []
    case_paths: list[str] = []

    with zipfile.ZipFile(args.potcar_zip) as potcar_archive:
        for sample_id in args.sample_ids:
            archive_path, member_name = members[sample_id]
            source_chgcar = temporary_dir / f"{sample_id}.CHGCAR"
            extract_chgcar(archive_path, member_name, source_chgcar)
            reference = Chgcar.from_file(source_chgcar)
            poscar = Poscar(reference.structure)

            potcar_path = temporary_dir / f"{sample_id}.POTCAR"
            potcar_titles = combined_potcar(
                potcar_archive, poscar.site_symbols, potcar_path
            )
            potcar_bytes = potcar_path.read_bytes()
            potcar_path.unlink()

            prediction, prediction_diagnostics = predict_density(
                reference, model, basis, device, args.chunk_size
            )
            true_aug = oracle_augmentation(reference)
            ml_chgcar = temporary_dir / f"{sample_id}.ml.CHGCAR"
            Chgcar(
                poscar,
                {"total": prediction},
                data_aug=copy.deepcopy(true_aug),
            ).write_file(ml_chgcar)
            oracle_smooth_chgcar = temporary_dir / f"{sample_id}.oracle_smooth.CHGCAR"
            Chgcar(
                poscar,
                {"total": np.asarray(reference.data["total"], dtype=np.float64)},
                data_aug=copy.deepcopy(true_aug),
            ).write_file(oracle_smooth_chgcar)

            settings = common_incar(args.ediff, args.nelm)
            cases = {
                "sad": (2, None),
                "ml": (1, ml_chgcar),
                "oracle_smooth": (1, oracle_smooth_chgcar),
                "oracle_full": (1, source_chgcar),
            }
            for case_name, (icharg, initial_chgcar) in cases.items():
                case_settings = dict(settings)
                case_settings["ICHARG"] = icharg
                case_dir = output_dir / f"sample_{sample_id:04d}" / case_name
                write_case(
                    case_dir,
                    poscar,
                    potcar_bytes,
                    case_settings,
                    initial_chgcar,
                )
                case_paths.append(case_dir.relative_to(output_dir).as_posix())

            record = {
                "sample_id": sample_id,
                "formula": reference.structure.composition.reduced_formula,
                "num_atoms": len(reference.structure),
                "grid_shape": list(reference.data["total"].shape),
                "site_symbols": poscar.site_symbols,
                "potcar_titles": potcar_titles,
                "prediction": prediction_diagnostics,
            }
            records.append(record)
            print(
                f"id={sample_id} {record['formula']}: "
                f"postprocessed NMAE="
                f"{100 * prediction_diagnostics['postprocessed_nmae']:.4f}%",
                flush=True,
            )
            source_chgcar.unlink()
            ml_chgcar.unlink()
            oracle_smooth_chgcar.unlink()

    (output_dir / "cases.txt").write_text(
        "\n".join(case_paths) + "\n", encoding="utf-8", newline="\n"
    )
    write_runner(output_dir, len(case_paths))
    analyzer_source = Path(__file__).with_name("analyze_nmc_scf_benchmark.py")
    if analyzer_source.exists():
        shutil.copy2(analyzer_source, output_dir / "analyze.py")
    manifest = {
        "purpose": "Paired NMC SCF initialization benchmark",
        "checkpoint": str(args.checkpoint.resolve()),
        "raw_dir": str(args.raw_dir.resolve()),
        "split": str(args.split.resolve()),
        "sample_ids": list(args.sample_ids),
        "cases": {
            "sad": "Native VASP ICHARG=2 atomic-density superposition",
            "ml": "ML smooth density with converged oracle PAW occupancies (oracle-assisted)",
            "oracle_smooth": "DFT smooth grid with converged oracle PAW occupancies",
            "oracle_full": "Complete converged DFT CHGCAR (sanity check/data leakage)",
        },
        "paw_protocol": (
            "Converged oracle_full PAW one-center occupancies are shared by "
            "ml, oracle_smooth, charge3net, and oracle_full. This isolates "
            "smooth-grid initialization and is not end-to-end deployable."
        ),
        "incar_common": common_incar(args.ediff, args.nelm),
        "kpoints": "Monkhorst-Pack 3x3x1",
        "basis_config": vars(basis),
        "records": records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    temporary_dir.rmdir()
    print(f"Prepared {len(case_paths)} VASP cases under {output_dir}")
    print("Set VASP_COMMAND and run run_all.sh, or submit cases.txt as a job array.")


if __name__ == "__main__":
    main()
