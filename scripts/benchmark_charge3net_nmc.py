"""Benchmark the official pretrained ChargE3Net model on NMC CHGCAR grids.

This intentionally uses the model and graph-construction implementation from
the official repository.  Probe points are processed in chunks, while the
atomic representation is computed once per structure, matching the official
test-time optimization.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch


def load_official_modules(repository: Path):
    sys.path.insert(0, str(repository))
    from src.charge3net.data.graph_construction import KdTreeGraphConstructor
    from src.charge3net.models.e3 import E3DensityModel
    from src.utils.data import calculate_grid_pos, decompress_tarmember, read_vasp

    return (
        KdTreeGraphConstructor,
        E3DensityModel,
        calculate_grid_pos,
        decompress_tarmember,
        read_vasp,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def archive_index(
    archives: list[Path], wanted: set[int]
) -> dict[int, tuple[Path, tarfile.TarInfo]]:
    result: dict[int, tuple[Path, tarfile.TarInfo]] = {}
    for archive_path in archives:
        with tarfile.open(archive_path, "r:") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(
                    ".CHGCAR.lz4"
                ):
                    continue
                sample_id = int(Path(member.name).name.split(".", 1)[0])
                if sample_id in wanted:
                    result[sample_id] = (archive_path, member)
    return result


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
        "num_nodes": torch.tensor(
            [len(atoms)], dtype=torch.long, device=device
        ),
        "num_atom_edges": torch.tensor(
            [len(atom_edges)], dtype=torch.long, device=device
        ),
        "atom_xyz": torch.as_tensor(
            atoms.get_positions(), dtype=torch.float32, device=device
        )[None],
        "cell": torch.as_tensor(
            np.asarray(atoms.get_cell()),
            dtype=torch.float32,
            device=device,
        )[None],
    }
    return batch, neighborlist, inv_cell_t


def probe_graph_batch(
    base_batch: dict[str, torch.Tensor],
    atoms,
    probe_positions: np.ndarray,
    constructor,
    neighborlist,
    inv_cell_t,
    device: torch.device,
) -> dict[str, torch.Tensor]:
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
def evaluate_structure(
    model,
    constructor,
    atoms,
    density: np.ndarray,
    grid_positions: np.ndarray,
    device: torch.device,
    probe_chunk: int,
) -> dict[str, float | int | str | list[int]]:
    started = time.perf_counter()
    atom_graph_started = time.perf_counter()
    base_batch, neighborlist, inv_cell_t = atom_graph_batch(
        atoms, constructor, device
    )
    atom_graph_seconds = time.perf_counter() - atom_graph_started

    synchronize(device)
    atom_forward_started = time.perf_counter()
    atom_representation = model.atom_model(base_batch)
    synchronize(device)
    atom_forward_seconds = time.perf_counter() - atom_forward_started

    targets = density.reshape(-1).astype(np.float32)
    probes = grid_positions.reshape(-1, 3).astype(np.float32)
    absolute_error = 0.0
    absolute_reference = 0.0
    probe_graph_seconds = 0.0
    probe_forward_seconds = 0.0
    num_probe_edges = 0
    for start in range(0, len(targets), probe_chunk):
        stop = min(start + probe_chunk, len(targets))
        graph_started = time.perf_counter()
        batch = probe_graph_batch(
            base_batch,
            atoms,
            probes[start:stop],
            constructor,
            neighborlist,
            inv_cell_t,
            device,
        )
        probe_graph_seconds += time.perf_counter() - graph_started
        num_probe_edges += int(batch["num_probe_edges"].item())

        synchronize(device)
        forward_started = time.perf_counter()
        prediction = model.probe_model(batch, atom_representation)[0]
        synchronize(device)
        probe_forward_seconds += time.perf_counter() - forward_started
        reference = torch.as_tensor(
            targets[start:stop], dtype=torch.float32, device=device
        )
        absolute_error += float(torch.abs(prediction - reference).sum())
        absolute_reference += float(torch.abs(reference).sum())

    return {
        "formula": atoms.get_chemical_formula(),
        "num_atoms": len(atoms),
        "grid_shape": list(density.shape),
        "num_grid_points": len(targets),
        "num_probe_edges": num_probe_edges,
        "density_nmae": absolute_error / max(absolute_reference, 1e-12),
        "absolute_error_sum": absolute_error,
        "absolute_reference_sum": absolute_reference,
        "atom_graph_seconds": atom_graph_seconds,
        "atom_forward_seconds": atom_forward_seconds,
        "probe_graph_seconds": probe_graph_seconds,
        "probe_forward_seconds": probe_forward_seconds,
        "total_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--probe-chunk", type=int, default=2500)
    parser.add_argument("--sample-probes", type=int)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    (
        KdTreeGraphConstructor,
        E3DensityModel,
        calculate_grid_pos,
        decompress_tarmember,
        read_vasp,
    ) = load_official_modules(args.repository.resolve())
    device = torch.device(args.device)
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

    sample_ids = [
        int(Path(line.strip()).stem)
        for line in args.split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]
    members = archive_index(
        sorted(args.raw_dir.glob("cellrelax_*.tar")), set(sample_ids)
    )
    missing = sorted(set(sample_ids) - set(members))
    if missing:
        raise FileNotFoundError(f"Missing CHGCAR members: {missing[:10]}")

    records = []
    total_error = 0.0
    total_reference = 0.0
    for index, sample_id in enumerate(sample_ids, 1):
        archive_path, member = members[sample_id]
        load_started = time.perf_counter()
        with tarfile.open(archive_path, "r:") as archive:
            file_content = decompress_tarmember(archive, member)
        density, atoms, origin = read_vasp(file_content)
        atoms.set_pbc(True)
        grid_positions = calculate_grid_pos(
            density, origin, atoms.get_cell()
        )
        load_seconds = time.perf_counter() - load_started
        evaluation_density = density
        evaluation_grid_positions = grid_positions
        if (
            args.sample_probes is not None
            and args.sample_probes < density.size
        ):
            generator = np.random.default_rng(args.seed + sample_id)
            indices = np.sort(
                generator.choice(
                    density.size, args.sample_probes, replace=False
                )
            )
            evaluation_density = density.reshape(-1)[indices]
            evaluation_grid_positions = grid_positions.reshape(-1, 3)[
                indices
            ]
        record = evaluate_structure(
            model,
            constructor,
            atoms,
            evaluation_density,
            evaluation_grid_positions,
            device,
            args.probe_chunk,
        )
        record["sample_id"] = sample_id
        record["load_seconds"] = load_seconds
        records.append(record)
        total_error += float(record["absolute_error_sum"])
        total_reference += float(record["absolute_reference_sum"])
        print(
            f"{index}/{len(sample_ids)} id={sample_id} "
            f"NMAE={100 * float(record['density_nmae']):.4f}% "
            f"load={load_seconds:.3f}s "
            f"graph={float(record['probe_graph_seconds']):.3f}s "
            f"forward={float(record['probe_forward_seconds']):.3f}s",
            flush=True,
        )

    sample_nmae = np.asarray(
        [float(record["density_nmae"]) for record in records]
    )
    result = {
        "model": "official pretrained ChargE3Net NMC",
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_step": checkpoint.get("step"),
        "num_samples": len(records),
        "probe_chunk": args.probe_chunk,
        "sample_probes": args.sample_probes,
        "metric": "sum(abs(prediction-target))/sum(abs(target))",
        "density_nmae": total_error / max(total_reference, 1e-12),
        "sample_density_nmae_mean": float(sample_nmae.mean()),
        "sample_density_nmae_median": float(np.median(sample_nmae)),
        "sample_density_nmae_p90": float(np.quantile(sample_nmae, 0.9)),
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "mean_load_seconds": float(
            np.mean([record["load_seconds"] for record in records])
        ),
        "mean_atom_graph_seconds": float(
            np.mean([record["atom_graph_seconds"] for record in records])
        ),
        "mean_atom_forward_seconds": float(
            np.mean([record["atom_forward_seconds"] for record in records])
        ),
        "mean_probe_graph_seconds": float(
            np.mean([record["probe_graph_seconds"] for record in records])
        ),
        "mean_probe_forward_seconds": float(
            np.mean([record["probe_forward_seconds"] for record in records])
        ),
        "mean_total_seconds": float(
            np.mean([record["total_seconds"] for record in records])
        ),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    main()
