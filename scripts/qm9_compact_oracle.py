"""Fit a compact real-spherical-harmonic density basis on QM9 test molecules.

This is an offline representation test, not an ML model.  Coefficients are
fitted on CHGCAR grid points and evaluated on the independent probe positions
stored by the benchmark preprocessor.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import lz4.frame
import numpy as np
import torch
from e3nn import o3
from pymatgen.io.vasp.outputs import Chgcar
from scipy.special import sph_harm_y

from density_eqgnn.chgcar import vasp_grid


def real_spherical_harmonics(
    directions: np.ndarray, lmax: int
) -> np.ndarray:
    """Return orthonormal real spherical harmonics through ``lmax``."""

    z = np.clip(directions[..., 2], -1.0, 1.0)
    theta = np.arccos(z)
    phi = np.arctan2(directions[..., 1], directions[..., 0])
    channels: list[np.ndarray] = []
    for degree in range(lmax + 1):
        for order in range(-degree, degree + 1):
            if order < 0:
                value = (
                    np.sqrt(2.0)
                    * (-1.0) ** order
                    * sph_harm_y(degree, -order, theta, phi).imag
                )
            elif order == 0:
                value = sph_harm_y(degree, 0, theta, phi).real
            else:
                value = (
                    np.sqrt(2.0)
                    * (-1.0) ** order
                    * sph_harm_y(degree, order, theta, phi).real
                )
            channels.append(value)
    return np.stack(channels, axis=-1)


def compact_design(
    atom_positions: np.ndarray,
    probes: np.ndarray,
    num_radial: int,
    lmax: int,
    cutoff: float,
    harmonics: str = "e3nn",
) -> np.ndarray:
    displacement = probes[:, None, :] - atom_positions[None, :, :]
    distances = np.linalg.norm(displacement, axis=-1)
    directions = displacement / np.maximum(distances[..., None], 1e-12)
    minimum = max(cutoff / 32.0, 0.08)
    scales = np.geomspace(minimum, cutoff, num_radial)
    radial = np.exp(-0.5 * (distances[..., None] / scales) ** 2)
    x = np.clip(distances / cutoff, 0.0, 1.0)
    envelope = 0.5 * (np.cos(np.pi * x) + 1.0)
    envelope = np.where(distances < cutoff, envelope, 0.0)
    radial *= envelope[..., None]
    if harmonics == "scipy":
        angular = real_spherical_harmonics(directions, lmax)
    elif harmonics == "e3nn":
        angular = (
            o3.spherical_harmonics(
                list(range(lmax + 1)),
                torch.from_numpy(directions),
                normalize=True,
                normalization="integral",
            )
            .numpy()
        )
    else:
        raise ValueError(f"Unknown harmonics backend: {harmonics}")
    features = radial[..., :, None] * angular[..., None, :]
    return features.reshape(len(probes), -1).astype(np.float32)


def fit_ridge_gpu(
    design: np.ndarray,
    density: np.ndarray,
    ridge: float,
    device: torch.device,
) -> np.ndarray:
    # Normal equations square the condition number.  Float64 is required for
    # the highly overlapping multiscale radial functions used at r=12.
    phi = torch.from_numpy(design).to(device=device, dtype=torch.float64)
    target = torch.from_numpy(density).to(device=device, dtype=torch.float64)
    gram = phi.T @ phi
    rhs = phi.T @ target
    scale = torch.trace(gram) / max(gram.shape[0], 1)
    gram.diagonal().add_(ridge * torch.clamp(scale, min=1.0))
    coefficients = torch.linalg.solve(gram, rhs)
    return coefficients.cpu().numpy()


def grid_indices_from_positions(
    positions: np.ndarray, cell: np.ndarray, shape: tuple[int, int, int]
) -> np.ndarray:
    fractional = positions @ np.linalg.inv(cell)
    indices = np.rint(fractional * np.asarray(shape)).astype(int)
    indices %= np.asarray(shape)
    return np.ravel_multi_index(indices.T, shape)


def fit_one(
    sample_path: Path,
    compressed_path: Path,
    num_radial: int,
    lmax: int,
    cutoff: float,
    fit_probes: int,
    ridge: float,
    device: torch.device,
    temp_root: Path,
    harmonics: str,
    output_sample_path: Path | None = None,
) -> dict[str, float | int | str]:
    sample_id = sample_path.stem
    with np.load(sample_path, allow_pickle=False) as sample:
        eval_positions = sample["probe_positions"].astype(np.float64)
        eval_density = sample["density"].astype(np.float64)

    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"oracle_{sample_id}_", dir=temp_root) as tmp:
        chgcar_path = Path(tmp) / "CHGCAR"
        with lz4.frame.open(compressed_path, "rb") as source, chgcar_path.open(
            "wb"
        ) as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        chgcar = Chgcar.from_file(chgcar_path)

    atom_positions = chgcar.structure.cart_coords
    cell = chgcar.structure.lattice.matrix
    raw = np.asarray(chgcar.data["total"], dtype=np.float64)
    density = raw.reshape(-1) / chgcar.structure.volume
    probes = vasp_grid(cell, raw.shape)

    excluded = np.unique(grid_indices_from_positions(eval_positions, cell, raw.shape))
    available_mask = np.ones(len(density), dtype=bool)
    available_mask[excluded] = False
    available = np.flatnonzero(available_mask)
    count = min(fit_probes, len(available))
    high_count = count // 4
    available_density = np.abs(density[available])
    high_local = np.argpartition(available_density, -high_count)[-high_count:]
    high = available[high_local]
    remaining = np.setdiff1d(available, high, assume_unique=False)
    rng = np.random.default_rng(int(sample_id) + 100_000)
    random = rng.choice(remaining, count - high_count, replace=False)
    fit_indices = np.sort(np.concatenate([high, random]))

    fit_design = compact_design(
        atom_positions,
        probes[fit_indices],
        num_radial,
        lmax,
        cutoff,
        harmonics,
    )
    coefficients = fit_ridge_gpu(
        fit_design, density[fit_indices], ridge, device
    )
    fit_prediction = fit_design @ coefficients
    del fit_design

    eval_design = compact_design(
        atom_positions,
        eval_positions,
        num_radial,
        lmax,
        cutoff,
        harmonics,
    )
    eval_prediction = eval_design @ coefficients
    if output_sample_path is not None:
        output_sample_path.parent.mkdir(parents=True, exist_ok=True)
        with np.load(sample_path, allow_pickle=False) as sample:
            payload = {name: sample[name] for name in sample.files}
        payload["coefficients"] = coefficients.reshape(
            len(atom_positions), num_radial * (lmax + 1) ** 2
        ).astype(np.float32)
        payload["compact_lmax"] = np.int64(lmax)
        payload["compact_num_radial"] = np.int64(num_radial)
        payload["compact_harmonics"] = np.asarray(harmonics)
        payload["compact_fit_nmae"] = np.float32(
            np.abs(fit_prediction - density[fit_indices]).sum()
            / np.abs(density[fit_indices]).sum()
        )
        payload["compact_eval_nmae"] = np.float32(
            np.abs(eval_prediction - eval_density).sum()
            / np.abs(eval_density).sum()
        )
        np.savez_compressed(output_sample_path, **payload)
    return {
        "sample_id": sample_id,
        "num_atoms": len(atom_positions),
        "num_coefficients": len(coefficients),
        "fit_nmae": float(
            np.abs(fit_prediction - density[fit_indices]).sum()
            / np.abs(density[fit_indices]).sum()
        ),
        "eval_nmae": float(
            np.abs(eval_prediction - eval_density).sum()
            / np.abs(eval_density).sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--members", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-radial", type=int, default=12)
    parser.add_argument("--lmax", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--fit-probes", type=int, default=8192)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--harmonics", choices=["e3nn", "scipy"], default="e3nn")
    parser.add_argument(
        "--split", choices=["test", "train", "val", "all"], default="test"
    )
    parser.add_argument("--output-dataset", type=Path)
    parser.add_argument(
        "--temp-root", type=Path, default=Path(r"D:\density-eqgnn-data\tmp")
    )
    args = parser.parse_args()

    if args.split == "all":
        names = [path.name for path in sorted(args.dataset.glob("*.npz"))]
    else:
        names = [
            line.strip()
            for line in (args.dataset / f"{args.split}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
    if args.limit:
        names = names[: args.limit]
    device = torch.device(args.device)
    records = []
    for index, name in enumerate(names, 1):
        sample_path = args.dataset / name
        sample_id = sample_path.stem
        record = fit_one(
            sample_path,
            args.members / f"{sample_id}.CHGCAR.lz4",
            args.num_radial,
            args.lmax,
            args.cutoff,
            args.fit_probes,
            args.ridge,
            device,
            args.temp_root,
            args.harmonics,
            (
                args.output_dataset / name
                if args.output_dataset is not None
                else None
            ),
        )
        records.append(record)
        print(
            f"{index}/{len(names)} {sample_id}: "
            f"fit={record['fit_nmae']:.5f} eval={record['eval_nmae']:.5f}",
            flush=True,
        )
    values = np.asarray([record["eval_nmae"] for record in records])
    result = {
        "num_samples": len(records),
        "num_radial": args.num_radial,
        "lmax": args.lmax,
        "angular_channels": (args.lmax + 1) ** 2,
        "coefficients_per_atom": args.num_radial * (args.lmax + 1) ** 2,
        "fit_probes": args.fit_probes,
        "eval_probes": 4096,
        "harmonics": args.harmonics,
        "mean_eval_nmae": float(values.mean()),
        "median_eval_nmae": float(np.median(values)),
        "p90_eval_nmae": float(np.quantile(values, 0.9)),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.output_dataset is not None:
        args.output_dataset.mkdir(parents=True, exist_ok=True)
        for split_name in ("train.txt", "val.txt", "test.txt", "development.txt"):
            source = args.dataset / split_name
            if source.exists():
                shutil.copy2(source, args.output_dataset / split_name)
        (args.output_dataset / "compact_basis.json").write_text(
            json.dumps(
                {
                    "num_radial": args.num_radial,
                    "lmax": args.lmax,
                    "cutoff": args.cutoff,
                    "harmonics": args.harmonics,
                    "fit_probes": args.fit_probes,
                    "ridge": args.ridge,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
