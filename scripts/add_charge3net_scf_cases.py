"""Add official ChargE3Net initial densities to an NMC SCF benchmark.

This is intentionally a separate, resumable step because full-grid
ChargE3Net inference takes roughly ten minutes per NMC structure on the local
16 GB GPU.  Each completed density is written immediately, and rerunning the
script skips already completed samples.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar
from pymatgen.io.vasp.outputs import Chgcar


def load_official_modules(repository: Path):
    sys.path.insert(0, str(repository))
    from src.charge3net.data.graph_construction import KdTreeGraphConstructor
    from src.charge3net.models.e3 import E3DensityModel
    from src.utils.data import calculate_grid_pos, read_vasp

    return KdTreeGraphConstructor, E3DensityModel, calculate_grid_pos, read_vasp


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def atom_graph_batch(atoms, constructor, device: torch.device):
    atom_edges, atom_displacements, neighborlist, inv_cell_t = (
        constructor.atoms_to_graph(atoms)
    )
    atom_edges = np.concatenate(atom_edges, axis=0)
    atom_displacements = np.concatenate(atom_displacements, axis=0)
    batch = {
        "nodes": torch.as_tensor(
            atoms.get_atomic_numbers(), dtype=torch.long, device=device
        )[None],
        "atom_edges": torch.as_tensor(
            atom_edges, dtype=torch.long, device=device
        )[None],
        "atom_edges_displacement": torch.as_tensor(
            atom_displacements, dtype=torch.float32, device=device
        )[None],
        "num_nodes": torch.tensor([len(atoms)], dtype=torch.long, device=device),
        "num_atom_edges": torch.tensor(
            [len(atom_edges)], dtype=torch.long, device=device
        ),
        "atom_xyz": torch.as_tensor(
            atoms.get_positions(), dtype=torch.float32, device=device
        )[None],
        "cell": torch.as_tensor(
            np.asarray(atoms.get_cell()), dtype=torch.float32, device=device
        )[None],
    }
    return batch, neighborlist, inv_cell_t


def probe_graph_batch(
    base_batch,
    atoms,
    probe_positions,
    constructor,
    neighborlist,
    inv_cell_t,
    device,
):
    probe_edges, probe_displacements = constructor.probes_to_graph(
        atoms,
        probe_positions,
        neighborlist=neighborlist,
        inv_cell_T=inv_cell_t,
    )
    batch = dict(base_batch)
    batch.update(
        probe_edges=torch.as_tensor(
            probe_edges, dtype=torch.long, device=device
        )[None],
        probe_edges_displacement=torch.as_tensor(
            probe_displacements, dtype=torch.float32, device=device
        )[None],
        num_probes=torch.tensor(
            [len(probe_positions)], dtype=torch.long, device=device
        ),
        num_probe_edges=torch.tensor(
            [len(probe_edges)], dtype=torch.long, device=device
        ),
        probe_xyz=torch.as_tensor(
            probe_positions, dtype=torch.float32, device=device
        )[None],
    )
    return batch


@torch.inference_mode()
def predict_full_grid(
    model,
    constructor,
    atoms,
    grid_positions: np.ndarray,
    device: torch.device,
    probe_chunk: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    started = time.perf_counter()
    graph_started = time.perf_counter()
    base_batch, neighborlist, inv_cell_t = atom_graph_batch(
        atoms, constructor, device
    )
    atom_graph_seconds = time.perf_counter() - graph_started

    synchronize(device)
    atom_started = time.perf_counter()
    atom_representation = model.atom_model(base_batch)
    synchronize(device)
    atom_forward_seconds = time.perf_counter() - atom_started

    probes = grid_positions.reshape(-1, 3).astype(np.float32)
    prediction_chunks: list[np.ndarray] = []
    probe_graph_seconds = 0.0
    probe_forward_seconds = 0.0
    num_probe_edges = 0
    for start in range(0, len(probes), probe_chunk):
        stop = min(start + probe_chunk, len(probes))
        probe_graph_started = time.perf_counter()
        batch = probe_graph_batch(
            base_batch,
            atoms,
            probes[start:stop],
            constructor,
            neighborlist,
            inv_cell_t,
            device,
        )
        probe_graph_seconds += time.perf_counter() - probe_graph_started
        num_probe_edges += int(batch["num_probe_edges"].item())

        synchronize(device)
        forward_started = time.perf_counter()
        prediction = model.probe_model(batch, atom_representation)[0]
        synchronize(device)
        probe_forward_seconds += time.perf_counter() - forward_started
        prediction_chunks.append(prediction.detach().cpu().numpy())

        completed = stop / len(probes)
        if stop == len(probes) or (start // probe_chunk + 1) % 20 == 0:
            print(
                f"  probes {stop}/{len(probes)} ({100 * completed:.1f}%)",
                flush=True,
            )

    prediction = np.concatenate(prediction_chunks)
    return prediction, {
        "num_probe_edges": num_probe_edges,
        "atom_graph_seconds": atom_graph_seconds,
        "atom_forward_seconds": atom_forward_seconds,
        "probe_graph_seconds": probe_graph_seconds,
        "probe_forward_seconds": probe_forward_seconds,
        "generation_seconds": time.perf_counter() - started,
    }


def oracle_augmentation(chgcar: Chgcar) -> dict[str, object]:
    total = chgcar.data_aug["total"]
    if isinstance(total, dict):
        copied: object = {
            int(atom_index): np.asarray(values).copy()
            for atom_index, values in total.items()
        }
    else:
        copied = copy.deepcopy(total)
    return {"total": copied}


def postprocess_density(
    density_per_volume: np.ndarray,
    reference: Chgcar,
) -> tuple[np.ndarray, dict[str, float]]:
    volume = float(reference.structure.volume)
    raw = density_per_volume.astype(np.float64) * volume
    target = np.asarray(reference.data["total"], dtype=np.float64)
    expected_electrons = float(target.mean())
    negative = raw < 0.0
    negative_l1 = float(np.abs(raw[negative]).sum())
    raw_l1 = float(np.abs(raw).sum())
    projected = np.maximum(raw, 0.0)
    projected *= expected_electrons / float(projected.mean())
    projected = projected.reshape(target.shape)
    return projected, {
        "expected_electrons": expected_electrons,
        "raw_predicted_electrons": float(raw.mean()),
        "projected_electrons": float(projected.mean()),
        "negative_grid_fraction": float(negative.mean()),
        "negative_l1_fraction": negative_l1 / max(raw_l1, 1e-30),
        "postprocessed_nmae": float(
            np.abs(projected - target).sum() / np.abs(target).sum()
        ),
    }


def update_array_size(path: Path, case_count: int) -> None:
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [
        f"#SBATCH --array=1-{case_count}"
        if line.startswith("#SBATCH --array=")
        else line
        for line in lines
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-chunk", type=int, default=2500)
    parser.add_argument("--ours-timing-json", type=Path)
    args = parser.parse_args()

    root = args.benchmark_dir.resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    (
        KdTreeGraphConstructor,
        E3DensityModel,
        calculate_grid_pos,
        read_vasp,
    ) = load_official_modules(args.repository.resolve())
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model = E3DensityModel(
        num_interactions=3,
        num_neighbors=20,
        mul=500,
        lmax=4,
        cutoff=4.0,
        basis="gaussian",
        num_basis=20,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    constructor = KdTreeGraphConstructor(
        cutoff=4.0, disable_pbc=False, sorted_edges=False
    )

    records = {int(record["sample_id"]): record for record in manifest["records"]}
    if args.ours_timing_json and args.ours_timing_json.exists():
        ours_result = json.loads(
            args.ours_timing_json.read_text(encoding="utf-8")
        )
        ours_records = {
            int(record["sample_id"]): record for record in ours_result["records"]
        }
        for sample_id, record in records.items():
            if sample_id in ours_records:
                timing = ours_records[sample_id]
                record["prediction"]["generation_seconds"] = float(
                    timing["coefficient_seconds"]
                    + timing["reconstruction_seconds"]
                )

    for sample_id in manifest["sample_ids"]:
        sample_id = int(sample_id)
        case_dir = root / f"sample_{sample_id:04d}" / "charge3net"
        completion_marker = case_dir / "density_generation.json"
        if completion_marker.exists() and (case_dir / "CHGCAR").exists():
            print(f"id={sample_id}: ChargE3Net density already exists; skipping")
            records[sample_id]["charge3net_prediction"] = json.loads(
                completion_marker.read_text(encoding="utf-8")
            )
            continue

        oracle_dir = root / f"sample_{sample_id:04d}" / "oracle_full"
        source_chgcar = oracle_dir / "CHGCAR"
        reference = Chgcar.from_file(source_chgcar)
        file_content = source_chgcar.read_bytes()
        target_density, atoms, origin = read_vasp(file_content)
        atoms.set_pbc(True)
        grid_positions = calculate_grid_pos(
            target_density, origin, atoms.get_cell()
        )
        print(
            f"id={sample_id} {atoms.get_chemical_formula()}: "
            f"{target_density.size} probes",
            flush=True,
        )
        prediction, timing = predict_full_grid(
            model,
            constructor,
            atoms,
            grid_positions,
            device,
            args.probe_chunk,
        )
        projected, diagnostics = postprocess_density(prediction, reference)
        charge_record = {
            **timing,
            **diagnostics,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_step": checkpoint.get("step"),
            "probe_chunk": args.probe_chunk,
        }

        case_dir.mkdir(parents=True, exist_ok=True)
        Poscar(reference.structure).write_file(case_dir / "POSCAR")
        shutil.copy2(oracle_dir / "POTCAR", case_dir / "POTCAR")
        Kpoints.from_file(oracle_dir / "KPOINTS").write_file(
            case_dir / "KPOINTS"
        )
        incar = Incar.from_file(oracle_dir / "INCAR")
        incar["ICHARG"] = 1
        incar.write_file(case_dir / "INCAR")
        Chgcar(
            Poscar(reference.structure),
            {"total": projected},
            data_aug=copy.deepcopy(oracle_augmentation(reference)),
        ).write_file(case_dir / "CHGCAR")
        for file_name in ("INCAR", "POSCAR", "KPOINTS", "CHGCAR"):
            path = case_dir / file_name
            content = path.read_bytes().replace(b"\r\n", b"\n")
            if file_name == "INCAR":
                content = content.replace(b" = True\n", b" = .TRUE.\n")
                content = content.replace(b" = False\n", b" = .FALSE.\n")
            path.write_bytes(content)
        completion_marker.write_text(
            json.dumps(charge_record, indent=2), encoding="utf-8"
        )
        records[sample_id]["charge3net_prediction"] = charge_record
        print(
            f"id={sample_id}: NMAE={100 * diagnostics['postprocessed_nmae']:.4f}% "
            f"time={timing['generation_seconds']:.1f}s",
            flush=True,
        )

    manifest["cases"]["charge3net"] = (
        "Official ChargE3Net smooth density, nonnegative and charge normalized; "
        "converged oracle PAW occupancies (oracle-assisted)"
    )
    manifest["paw_protocol"] = (
        "Converged oracle_full PAW one-center occupancies are shared by ml, "
        "oracle_smooth, charge3net, and oracle_full. This follows the "
        "ChargE3Net paper-style smooth-density SCF test and is not end-to-end "
        "deployable."
    )
    manifest["records"] = [records[int(value)] for value in manifest["sample_ids"]]
    manifest["charge3net_checkpoint"] = str(args.checkpoint.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    case_lines = [
        line.strip()
        for line in (root / "cases.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().endswith("/charge3net")
    ]
    case_lines.extend(
        f"sample_{int(sample_id):04d}/charge3net"
        for sample_id in manifest["sample_ids"]
    )
    (root / "cases.txt").write_text(
        "\n".join(case_lines) + "\n", encoding="utf-8", newline="\n"
    )
    update_array_size(root / "submit_slurm.sh", len(case_lines))
    print(f"Added {len(manifest['sample_ids'])} ChargE3Net SCF cases")


if __name__ == "__main__":
    main()
