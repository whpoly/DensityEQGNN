"""Prepare 3,500 QM9/VASP densities as compact per-atom coefficient samples."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lz4.frame
import numpy as np

from density_eqgnn.basis import DensityBasisConfig
from density_eqgnn.chgcar import extract_chgcar_sample


def extract_members(
    archives: Path,
    members: Path,
    samples: int,
    start_index: int = 0,
    selected_ids: set[str] | None = None,
) -> list[Path]:
    """Extract a numeric slice of compressed CHGCAR members."""

    members.mkdir(parents=True, exist_ok=True)
    selected: list[Path] = []
    seen = 0
    for archive in sorted(archives.glob("*xxx.tar")):
        with tarfile.open(archive, mode="r:") as handle:
            for member in handle:
                if not member.isfile() or not member.name.endswith(".CHGCAR.lz4"):
                    continue
                sample_id = Path(member.name).name.removesuffix(".CHGCAR.lz4")
                if selected_ids is not None and sample_id not in selected_ids:
                    continue
                if seen < start_index:
                    seen += 1
                    continue
                seen += 1
                destination = members / Path(member.name).name
                if not destination.exists() or destination.stat().st_size != member.size:
                    source = handle.extractfile(member)
                    if source is None:
                        raise IOError(f"Cannot extract {member.name} from {archive}")
                    temporary = destination.with_suffix(destination.suffix + ".part")
                    with temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                    temporary.replace(destination)
                selected.append(destination)
                if len(selected) >= samples:
                    return selected
    raise RuntimeError(f"Only found {len(selected)} members; requested {samples}")


def _fit_one(
    compressed: str,
    output: str,
    basis_dict: dict,
    fit_probes: int,
    eval_probes: int,
    ridge: float,
    temp_root: str,
    fit_mode: str,
    max_atoms: int | None,
) -> dict:
    source = Path(compressed)
    destination = Path(output)
    sample_id = source.name.removesuffix(".CHGCAR.lz4")
    if destination.exists():
        with np.load(destination, allow_pickle=False) as data:
            nmae = float(data["fit_nmae"]) if "fit_nmae" in data else float("nan")
        return {"sample_id": sample_id, "path": destination.name, "oracle_nmae": nmae}

    basis = DensityBasisConfig(**basis_dict)
    Path(temp_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"qm9_{sample_id}_", dir=temp_root) as directory:
        chgcar = Path(directory) / "CHGCAR"
        with lz4.frame.open(source, "rb") as compressed_handle, chgcar.open("wb") as target:
            shutil.copyfileobj(compressed_handle, target, length=8 * 1024 * 1024)
        metrics = extract_chgcar_sample(
            chgcar,
            destination,
            basis,
            ridge=ridge,
            fit_probes=fit_probes,
            eval_probes=eval_probes,
            seed=int(sample_id),
            fit_mode=fit_mode,
        )
    with np.load(destination, allow_pickle=False) as data:
        atom_count = len(data["atomic_numbers"])
    if max_atoms is not None and atom_count > max_atoms:
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"{sample_id} contains {atom_count} atoms, above max_atoms={max_atoms}"
        )
    return {
        "sample_id": sample_id,
        "path": destination.name,
        "oracle_nmae": metrics["nmae"],
        "atom_count": atom_count,
    }


def write_splits(output: Path, names: list[str], seed: int) -> dict[str, list[str]]:
    """Create an 80/20 outer split and a validation subset inside development data."""

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(names)[rng.permutation(len(names))]
    test_count = int(round(0.20 * len(names)))
    development = shuffled[:-test_count]
    test = shuffled[-test_count:]
    val_count = int(round(0.10 * len(development)))
    splits = {
        "train": development[:-val_count].tolist(),
        "val": development[-val_count:].tolist(),
        "test": test.tolist(),
    }
    for split, values in splits.items():
        (output / f"{split}.txt").write_text("\n".join(values) + "\n", encoding="utf-8")
    (output / "development.txt").write_text(
        "\n".join(development.tolist()) + "\n", encoding="utf-8"
    )
    return splits


def main() -> None:
    data_root = Path(os.environ.get("DENSITY_EQGNN_DATA_ROOT", r"D:\density-eqgnn-data"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", type=Path, default=data_root / "qm9" / "archives")
    parser.add_argument("--members", type=Path, default=data_root / "qm9" / "members")
    parser.add_argument("--output", type=Path, default=data_root / "qm9_3500")
    parser.add_argument("--temp-root", type=Path, default=data_root / "tmp")
    parser.add_argument("--samples", type=int, default=3500)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--selection-file",
        type=Path,
        help="Optional text file containing six-digit QM9 member IDs to include",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fit-probes", type=int, default=20000)
    parser.add_argument("--eval-probes", type=int, default=4096)
    parser.add_argument("--num-radial", type=int, default=8)
    parser.add_argument("--lmax", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--cutoff", type=float, default=4.0)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument(
        "--fit-mode",
        choices=["global", "blockwise", "stockholder"],
        default="global",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-atoms", type=int)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    selected_ids = None
    if args.selection_file:
        selected_ids = {
            line.strip()
            for line in args.selection_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if len(selected_ids) < args.samples:
            raise ValueError(
                f"Selection file contains {len(selected_ids)} IDs, "
                f"but {args.samples} samples were requested"
            )
    selected = extract_members(
        args.archives,
        args.members,
        args.samples,
        start_index=args.start_index,
        selected_ids=selected_ids,
    )
    basis = DensityBasisConfig(
        num_radial=args.num_radial,
        cutoff=args.cutoff,
        lmax=args.lmax,
        pbc=False,
    )
    futures = []
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for compressed in selected:
            sample_id = compressed.name.removesuffix(".CHGCAR.lz4")
            futures.append(
                pool.submit(
                    _fit_one,
                    str(compressed),
                    str(args.output / f"{sample_id}.npz"),
                    basis.to_dict(),
                    args.fit_probes,
                    args.eval_probes,
                    args.ridge,
                    str(args.temp_root),
                    args.fit_mode,
                    args.max_atoms,
                )
            )
        for index, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"fitted {index}/{len(futures)}", flush=True)

    records.sort(key=lambda item: item["sample_id"])
    splits = write_splits(args.output, [item["path"] for item in records], args.seed)
    metadata = {
        "source": "QM9 Charge Densities and Energies Calculated with VASP",
        "figshare_article": 16794500,
        "license": "CC BY 4.0",
        "selection": (
            f"IDs from {args.selection_file}"
            if args.selection_file
            else f"numeric slice starting at {args.start_index}"
        ),
        "outer_split": "80% development / 20% test, seed 42",
        "optimization_train": len(splits["train"]),
        "validation": len(splits["val"]),
        "development_total": len(splits["train"]) + len(splits["val"]),
        "test": len(splits["test"]),
        "basis": basis.to_dict(),
        "fit_probes": args.fit_probes,
        "eval_probes": args.eval_probes,
        "ridge": args.ridge,
        "fit_mode": args.fit_mode,
        "max_atoms": args.max_atoms,
        "records": records,
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    oracle = np.asarray([item["oracle_nmae"] for item in records], dtype=float)
    print(
        f"prepared {len(records)} samples: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}; "
        f"mean oracle NMAE={np.nanmean(oracle):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
