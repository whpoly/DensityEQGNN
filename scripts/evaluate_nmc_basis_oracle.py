"""Estimate the representation oracle of the periodic compact density basis.

For every selected NMC structure, the atomic coefficients are optimized
directly against its DFT charge-density grid.  The structure-to-coefficient
network is therefore removed from the experiment: the remaining full-grid
NMAE measures the error of the analytic basis and numerical optimization.
"""

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
from density_eqgnn.training.config import load_config
from density_eqgnn.training.train import resolve_device
from evaluate_vasp_sad_nmc import (
    AtomicFormFactor,
    read_form_factors,
    sad_density,
)


def load_archive_members(
    archive_paths: list[Path],
) -> dict[int, tuple[Path, str]]:
    result: dict[int, tuple[Path, str]] = {}
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
        with lz4.frame.open(compressed, "rb") as source, destination.open(
            "wb"
        ) as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)


@torch.no_grad()
def density_nmae(
    coefficients: torch.Tensor,
    positions: torch.Tensor,
    probes: torch.Tensor,
    target: torch.Tensor,
    atom_mask: torch.Tensor,
    cell: torch.Tensor,
    basis: CompactBasisConfig,
    chunk_size: int,
    baseline: torch.Tensor | None = None,
) -> float:
    absolute_error = 0.0
    absolute_reference = 0.0
    for start in range(0, len(target), chunk_size):
        stop = min(start + chunk_size, len(target))
        probe_mask = torch.ones(
            1, stop - start, dtype=torch.bool, device=target.device
        )
        prediction = reconstruct_compact_density(
            coefficients,
            positions,
            probes[start:stop][None],
            atom_mask,
            probe_mask,
            cell,
            basis,
        )[0]
        if baseline is not None:
            prediction = prediction + baseline[start:stop]
        reference = target[start:stop]
        absolute_error += float(torch.abs(prediction - reference).sum())
        absolute_reference += float(torch.abs(reference).sum())
    return absolute_error / max(absolute_reference, 1e-12)


def optimize_one(
    chgcar_path: Path,
    sample_id: int,
    model,
    basis: CompactBasisConfig,
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    eval_every: int,
    eval_probes: int,
    chunk_size: int,
    seed: int,
    lbfgs_iterations: int,
    smooth_epsilon: float,
    form_factors: dict[str, AtomicFormFactor] | None,
) -> dict[str, float | int | str | list[int]]:
    with chgcar_path.open("r", encoding="utf-8") as handle:
        atoms = read_vasp_configuration(handle)
    atoms.set_pbc(True)
    chgcar = Chgcar.from_file(chgcar_path)
    raw_density = np.asarray(chgcar.data["total"], dtype=np.float64)
    target_numpy = raw_density.reshape(-1) / atoms.get_volume()
    probes_numpy = vasp_grid(atoms.cell.array, raw_density.shape).astype(
        np.float32
    )

    atomic_numbers = torch.from_numpy(
        atoms.get_atomic_numbers().astype(np.int64)
    )[None].to(device)
    positions = torch.from_numpy(
        atoms.get_positions().astype(np.float32)
    )[None].to(device)
    atom_mask = torch.ones(
        1, len(atoms), dtype=torch.bool, device=device
    )
    cell = torch.from_numpy(atoms.cell.array.astype(np.float32))[None].to(
        device
    )
    probes = torch.from_numpy(probes_numpy).to(device)
    target = torch.from_numpy(target_numpy.astype(np.float32)).to(device)
    baseline = None
    if form_factors is not None:
        sad_numpy = sad_density(
            np.asarray(atoms.get_chemical_symbols()),
            atoms.get_scaled_positions(wrap=True),
            atoms.cell.array,
            tuple(int(value) for value in raw_density.shape),
            form_factors,
        ).reshape(-1) / atoms.get_volume()
        baseline = torch.from_numpy(sad_numpy.astype(np.float32)).to(device)

    with torch.no_grad():
        model_initial = model(
            atomic_numbers, positions, atom_mask, cell
        )["coefficients"]
        initial = (
            torch.zeros_like(model_initial)
            if baseline is not None
            else model_initial
        )
    coefficients = torch.nn.Parameter(initial.detach().clone())
    optimizer = torch.optim.Adam([coefficients], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + sample_id)
    eval_count = min(eval_probes, len(target))
    eval_indices = torch.randperm(
        len(target), generator=generator, device=device
    )[:eval_count]
    initial_eval = density_nmae(
        coefficients,
        positions,
        probes[eval_indices],
        target[eval_indices],
        atom_mask,
        cell,
        basis,
        chunk_size,
        baseline[eval_indices] if baseline is not None else None,
    )
    best_eval = initial_eval
    best_coefficients = coefficients.detach().clone()
    started = time.perf_counter()
    for step in range(1, steps + 1):
        indices = torch.randint(
            len(target),
            (min(batch_size, len(target)),),
            generator=generator,
            device=device,
        )
        probe_mask = torch.ones(
            1, len(indices), dtype=torch.bool, device=device
        )
        prediction = reconstruct_compact_density(
            coefficients,
            positions,
            probes[indices][None],
            atom_mask,
            probe_mask,
            cell,
            basis,
        )[0]
        if baseline is not None:
            prediction = prediction + baseline[indices]
        reference = target[indices]
        loss = torch.abs(prediction - reference).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_every == 0 or step == steps:
            current_eval = density_nmae(
                coefficients,
                positions,
                probes[eval_indices],
                target[eval_indices],
                atom_mask,
                cell,
                basis,
                chunk_size,
                baseline[eval_indices] if baseline is not None else None,
            )
            scheduler.step(current_eval)
            if current_eval < best_eval:
                best_eval = current_eval
                best_coefficients = coefficients.detach().clone()
            print(
                f"id={sample_id} step={step}/{steps} "
                f"batch_l1={float(loss.detach()):.6f} "
                f"eval_nmae={100 * current_eval:.4f}% "
                f"best={100 * best_eval:.4f}%",
                flush=True,
            )

    lbfgs_closures = 0
    if lbfgs_iterations:
        coefficients = torch.nn.Parameter(best_coefficients.clone())
        full_optimizer = torch.optim.LBFGS(
            [coefficients],
            lr=1.0,
            max_iter=lbfgs_iterations,
            history_size=20,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def full_grid_closure() -> torch.Tensor:
            nonlocal lbfgs_closures
            full_optimizer.zero_grad(set_to_none=True)
            total = coefficients.new_zeros(())
            for start in range(0, len(target), chunk_size):
                stop = min(start + chunk_size, len(target))
                probe_mask = torch.ones(
                    1,
                    stop - start,
                    dtype=torch.bool,
                    device=device,
                )
                prediction = reconstruct_compact_density(
                    coefficients,
                    positions,
                    probes[start:stop][None],
                    atom_mask,
                    probe_mask,
                    cell,
                    basis,
                )[0]
                if baseline is not None:
                    prediction = prediction + baseline[start:stop]
                residual = prediction - target[start:stop]
                chunk_loss = torch.sqrt(
                    residual.square() + smooth_epsilon**2
                ).sum() / len(target)
                chunk_loss.backward()
                total += chunk_loss.detach()
            lbfgs_closures += 1
            if lbfgs_closures == 1 or lbfgs_closures % 5 == 0:
                print(
                    f"id={sample_id} lbfgs_closure={lbfgs_closures} "
                    f"smooth_l1={float(total):.7f}",
                    flush=True,
                )
            return total

        full_optimizer.step(full_grid_closure)
        refined_nmae = density_nmae(
            coefficients,
            positions,
            probes,
            target,
            atom_mask,
            cell,
            basis,
            chunk_size,
            baseline,
        )
        current_best_full = density_nmae(
            best_coefficients,
            positions,
            probes,
            target,
            atom_mask,
            cell,
            basis,
            chunk_size,
            baseline,
        )
        if refined_nmae < current_best_full:
            best_coefficients = coefficients.detach().clone()
        print(
            f"id={sample_id} lbfgs_full_grid_nmae="
            f"{100 * refined_nmae:.4f}%",
            flush=True,
        )

    full_grid_nmae = density_nmae(
        best_coefficients,
        positions,
        probes,
        target,
        atom_mask,
        cell,
        basis,
        chunk_size,
        baseline,
    )
    initial_full_grid_nmae = density_nmae(
        initial,
        positions,
        probes,
        target,
        atom_mask,
        cell,
        basis,
        chunk_size,
        baseline,
    )
    return {
        "sample_id": sample_id,
        "formula": atoms.get_chemical_formula(),
        "num_atoms": len(atoms),
        "grid_shape": list(raw_density.shape),
        "num_grid_points": len(target),
        "num_coefficients": int(coefficients.numel()),
        "baseline": "VASP POTCAR SAD" if baseline is not None else "none",
        "initial_full_grid_nmae": initial_full_grid_nmae,
        "optimized_full_grid_nmae": full_grid_nmae,
        "best_monitor_nmae": best_eval,
        "lbfgs_closures": lbfgs_closures,
        "optimization_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--eval-probes", type=int, default=32768)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--lbfgs-iterations", type=int, default=0)
    parser.add_argument("--smooth-epsilon", type=float, default=1e-5)
    parser.add_argument("--potcar-zip", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config["training"].get("device", "auto"))
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    model = build_compact_coefficient_model(checkpoint["model_config"]).to(
        device
    )
    load_compact_model_state(model,checkpoint)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    basis = CompactBasisConfig(**checkpoint["basis_config"])
    check_basis_checkpoint(checkpoint,basis)
    form_factors = (
        read_form_factors(args.potcar_zip)
        if args.potcar_zip is not None
        else None
    )

    sample_ids = [
        int(Path(line.strip()).stem)
        for line in args.split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][: args.limit]
    archive_paths = sorted(args.raw_dir.glob("cellrelax_*.tar"))
    member_index = load_archive_members(archive_paths)

    records = []
    with tempfile.TemporaryDirectory(prefix="nmc_basis_oracle_") as temporary:
        chgcar_path = Path(temporary) / "CHGCAR"
        for index, sample_id in enumerate(sample_ids, 1):
            archive_path, member_name = member_index[sample_id]
            extract_member(archive_path, member_name, chgcar_path)
            print(
                f"oracle sample {index}/{len(sample_ids)}: id={sample_id}",
                flush=True,
            )
            records.append(
                optimize_one(
                    chgcar_path,
                    sample_id,
                    model,
                    basis,
                    device,
                    args.steps,
                    args.batch_size,
                    args.learning_rate,
                    args.eval_every,
                    args.eval_probes,
                    args.chunk_size,
                    args.seed,
                    args.lbfgs_iterations,
                    args.smooth_epsilon,
                    form_factors,
                )
            )

    optimized = np.asarray(
        [record["optimized_full_grid_nmae"] for record in records]
    )
    initial = np.asarray(
        [record["initial_full_grid_nmae"] for record in records]
    )
    result = {
        "definition": (
            "Per-structure coefficients optimized directly against the DFT "
            "density; reported error is evaluated on the complete CHGCAR grid."
        ),
        "status": (
            "Numerical upper bound on the exact L1 projection error; lower is "
            "possible if coefficient optimization is improved."
        ),
        "num_samples": len(records),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lbfgs_iterations": args.lbfgs_iterations,
        "smooth_epsilon": args.smooth_epsilon,
        "baseline": (
            "VASP POTCAR SAD" if form_factors is not None else "none"
        ),
        "basis_config": checkpoint["basis_config"],
        "mean_initial_full_grid_nmae": float(initial.mean()),
        "mean_optimized_full_grid_nmae": float(optimized.mean()),
        "median_optimized_full_grid_nmae": float(np.median(optimized)),
        "min_optimized_full_grid_nmae": float(optimized.min()),
        "max_optimized_full_grid_nmae": float(optimized.max()),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
