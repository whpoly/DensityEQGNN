"""Evaluate a compact coefficient model on complete NMC CHGCAR grids."""

from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import tempfile
import time
from pathlib import Path

import lz4.frame
import numpy as np
import torch
from ase.io.vasp import read_vasp_configuration
from pymatgen.io.vasp.outputs import Chgcar

from density_eqgnn.chgcar import vasp_grid
from density_eqgnn.compact_basis import (
    CompactBasisConfig,
    reconstruct_compact_density,
    check_basis_checkpoint,
)
from density_eqgnn.models import build_compact_coefficient_model, load_compact_model_state
from density_eqgnn.periodic import check_periodic_checkpoint
from density_eqgnn.training.config import load_config
from density_eqgnn.training.train import resolve_device


def load_archive_members(archive_paths: list[Path]) -> dict[int, tuple[Path, str]]:
    result = {}
    for archive_path in archive_paths:
        with tarfile.open(archive_path, "r:") as archive:
            for member in archive:
                if member.isfile() and member.name.endswith(".CHGCAR.lz4"):
                    sample_id = int(
                        Path(member.name).name.removesuffix(".CHGCAR.lz4")
                    )
                    result[sample_id] = (archive_path, member.name)
    return result


def extract_member(
    archive_path: Path, member_name: str, destination: Path
) -> None:
    with tarfile.open(archive_path, "r:") as archive:
        compressed = archive.extractfile(member_name)
        if compressed is None:
            raise IOError(f"Could not read {member_name} from {archive_path}")
        with lz4.frame.open(compressed, "rb") as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_sample(
    model,
    basis: CompactBasisConfig,
    chgcar_path: Path,
    device: torch.device,
    chunk_size: int,
) -> dict[str, float | int | list[int] | str]:
    with chgcar_path.open("r", encoding="utf-8") as handle:
        ase_atoms = read_vasp_configuration(handle)
    ase_atoms.set_pbc(True)
    chgcar = Chgcar.from_file(chgcar_path)
    raw_density = np.asarray(chgcar.data["total"], dtype=np.float64)
    target = raw_density.reshape(-1) / ase_atoms.get_volume()
    geometry_dtype = np.float64 if basis.radial_whitening else np.float32
    probes = vasp_grid(ase_atoms.cell.array, raw_density.shape).astype(geometry_dtype)

    atomic_numbers = torch.from_numpy(
        ase_atoms.get_atomic_numbers().astype(np.int64)
    )[None].to(device)
    positions = torch.from_numpy(
        ase_atoms.get_positions().astype(np.float32)
    )[None].to(device)
    cell = torch.from_numpy(ase_atoms.cell.array.astype(np.float32))[None].to(device)
    atom_mask = torch.ones(
        1, len(ase_atoms), dtype=torch.bool, device=device
    )
    decoder_positions = torch.as_tensor(ase_atoms.get_positions(),dtype=torch.float64,device=device)[None] if basis.radial_whitening else positions
    decoder_cell = torch.as_tensor(ase_atoms.cell.array,dtype=torch.float64,device=device)[None] if basis.radial_whitening else cell

    synchronize(device)
    coefficient_start = time.perf_counter()
    coefficients = model(
        atomic_numbers, positions, atom_mask, cell
    )["coefficients"]
    if basis.radial_whitening:
        coefficients = coefficients.double()
    synchronize(device)
    coefficient_seconds = time.perf_counter() - coefficient_start

    absolute_error = 0.0
    absolute_reference = 0.0
    reconstruction_start = time.perf_counter()
    for start in range(0, len(target), chunk_size):
        stop = min(start + chunk_size, len(target))
        probe_chunk = torch.from_numpy(probes[start:stop])[None].to(device)
        probe_mask = torch.ones(
            1, stop - start, dtype=torch.bool, device=device
        )
        prediction = reconstruct_compact_density(
            coefficients,
            decoder_positions,
            probe_chunk,
            atom_mask,
            probe_mask,
            decoder_cell,
            basis,
        )[0]
        reference = torch.from_numpy(
            target[start:stop].astype(geometry_dtype)
        ).to(device)
        absolute_error += float(torch.abs(prediction - reference).sum())
        absolute_reference += float(torch.abs(reference).sum())
    synchronize(device)
    reconstruction_seconds = time.perf_counter() - reconstruction_start
    return {
        "formula": ase_atoms.get_chemical_formula(),
        "num_atoms": len(ase_atoms),
        "grid_shape": list(raw_density.shape),
        "num_grid_points": len(target),
        "absolute_error_sum": absolute_error,
        "absolute_reference_sum": absolute_reference,
        "density_nmae": absolute_error / max(absolute_reference, 1e-12),
        "coefficient_seconds": coefficient_seconds,
        "reconstruction_seconds": reconstruction_seconds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["training"].get("device", "auto"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    check_periodic_checkpoint(
        checkpoint,
        pbc=bool(checkpoint["basis_config"].get("pbc", False))
        or bool(checkpoint["model_config"].get("pbc", False)),
    )
    model = build_compact_coefficient_model(checkpoint["model_config"]).to(device)
    load_compact_model_state(model,checkpoint)
    model.eval()
    basis = CompactBasisConfig(**checkpoint["basis_config"])
    check_basis_checkpoint(checkpoint,basis)

    names = [
        line.strip()
        for line in args.split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sample_ids = [int(Path(name).stem) for name in names]
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]
    archive_paths = sorted(args.raw_dir.glob("cellrelax_*.tar"))
    if len(archive_paths) != 2:
        raise FileNotFoundError(
            f"Expected two cellrelax tar archives under {args.raw_dir}"
        )
    member_index = load_archive_members(archive_paths)
    missing = [sample_id for sample_id in sample_ids if sample_id not in member_index]
    if missing:
        raise KeyError(f"Samples are absent from raw archives: {missing[:10]}")

    records = []
    total_error = 0.0
    total_reference = 0.0
    with tempfile.TemporaryDirectory(prefix="nmc_full_grid_") as temporary:
        chgcar_path = Path(temporary) / "CHGCAR"
        for index, sample_id in enumerate(sample_ids, 1):
            archive_path, member_name = member_index[sample_id]
            extract_member(archive_path, member_name, chgcar_path)
            record = evaluate_sample(
                model, basis, chgcar_path, device, args.chunk_size
            )
            record["sample_id"] = sample_id
            records.append(record)
            total_error += float(record["absolute_error_sum"])
            total_reference += float(record["absolute_reference_sum"])
            print(
                f"{index}/{len(sample_ids)} id={sample_id} "
                f"NMAE={100 * float(record['density_nmae']):.4f}% "
                f"coeff={float(record['coefficient_seconds']):.3f}s "
                f"grid={float(record['reconstruction_seconds']):.3f}s",
                flush=True,
            )

    sample_nmae = np.asarray([record["density_nmae"] for record in records])
    result = {
        "dataset": "NMC full-grid test",
        "num_samples": len(records),
        "metric": "sum(abs(prediction-target))/sum(abs(target))",
        "density_nmae": total_error / max(total_reference, 1e-12),
        "sample_density_nmae_mean": float(sample_nmae.mean()),
        "sample_density_nmae_median": float(np.median(sample_nmae)),
        "sample_density_nmae_p90": float(np.quantile(sample_nmae, 0.9)),
        "mean_coefficient_seconds": float(
            np.mean([record["coefficient_seconds"] for record in records])
        ),
        "mean_reconstruction_seconds": float(
            np.mean([record["reconstruction_seconds"] for record in records])
        ),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "basis_config": checkpoint["basis_config"],
        "model_config": checkpoint["model_config"],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "records"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
