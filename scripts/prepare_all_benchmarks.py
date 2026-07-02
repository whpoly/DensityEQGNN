"""Download and prepare all density benchmark datasets.

This script is intentionally self-contained:

- QM9 and NMC are downloaded from DTU/Figshare.
- Materials Project is downloaded through the Materials Project API.
- VASP CHGCAR files are converted into the repository NPZ format.
- train/val/test split files are generated automatically.

The full datasets are large. Use --limit-files and --limit-samples for a smoke
test before launching a long cluster job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from density_eqgnn.data.chgcar import chgcar_to_npz


FIGSHARE_ARTICLES = {
    "qm9": 16794500,
    "nmc": 16837721,
}

DATASET_ALIASES = {
    "mp": "materials_project",
    "materials-project": "materials_project",
    "materials_project": "materials_project",
    "qm9": "qm9",
    "nmc": "nmc",
}


def canonical_datasets(values: list[str]) -> list[str]:
    if values == ["all"] or "all" in values:
        return ["qm9", "nmc", "materials_project"]
    datasets = []
    for value in values:
        key = DATASET_ALIASES.get(value.lower())
        if key is None:
            raise ValueError(f"Unknown dataset: {value}")
        if key not in datasets:
            datasets.append(key)
    return datasets


def figshare_files(article_id: int) -> list[dict]:
    with urllib.request.urlopen(
        f"https://api.figshare.com/v2/articles/{article_id}/files", timeout=60
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def download_file(url: str, output_path: Path, expected_size: int | None = None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and expected_size and output_path.stat().st_size == expected_size:
        print(f"skip existing {output_path}")
        return

    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, tmp_path.open("wb") as handle:
        total = expected_size or int(response.headers.get("Content-Length") or 0)
        with tqdm(total=total, unit="B", unit_scale=True, desc=output_path.name) as progress:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                progress.update(len(chunk))
    tmp_path.replace(output_path)


def verify_md5(path: Path, expected_md5: str | None) -> None:
    if not expected_md5:
        return
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected_md5:
        raise ValueError(f"MD5 mismatch for {path}: expected {expected_md5}, got {digest.hexdigest()}")


def selected_figshare_files(dataset: str, limit_files: int | None) -> list[dict]:
    files = figshare_files(FIGSHARE_ARTICLES[dataset])
    if dataset == "qm9":
        selected = [item for item in files if item["name"].endswith(".tar")]
    elif dataset == "nmc":
        selected = [
            item
            for item in files
            if item["name"].endswith(".tar") or item["name"] == "split.json"
        ]
    else:
        selected = []
    tar_count = 0
    limited = []
    for item in selected:
        if item["name"].endswith(".tar"):
            tar_count += 1
            if limit_files is not None and tar_count > limit_files:
                continue
        limited.append(item)
    return limited


def download_figshare_dataset(
    dataset: str,
    raw_dir: Path,
    limit_files: int | None,
    skip_download: bool,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if skip_download:
        print(f"skip download for {dataset}; using existing files under {raw_dir}")
        return

    for item in selected_figshare_files(dataset, limit_files):
        path = raw_dir / item["name"]
        download_file(item["download_url"], path, expected_size=item.get("size"))
        verify_md5(path, item.get("computed_md5") or item.get("supplied_md5"))


def safe_sample_name(tar_path: Path, member_name: str) -> str:
    parent = Path(member_name).parent.as_posix().strip(".").strip("/")
    stem = parent.replace("/", "__") if parent else Path(member_name).stem
    if not stem or stem == ".":
        stem = Path(member_name).stem
    return f"{tar_path.stem}__{stem}".replace(" ", "_").replace(":", "_")


def iter_chgcar_members(tar_path: Path) -> Iterable[tarfile.TarInfo]:
    with tarfile.open(tar_path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name == "CHGCAR" or name.startswith("CHGCAR."):
                yield member


def convert_tar_chgcars(
    tar_path: Path,
    output_dir: Path,
    probes: int | None,
    density_normalization: str,
    seed: int,
    force: bool,
    remaining: int | None,
) -> int:
    converted = 0
    with tarfile.open(tar_path, mode="r:*") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
            and (Path(member.name).name == "CHGCAR" or Path(member.name).name.startswith("CHGCAR."))
        ]
        for member in members:
            if remaining is not None and converted >= remaining:
                break
            sample_name = safe_sample_name(tar_path, member.name)
            out_path = output_dir / f"{sample_name}.npz"
            if out_path.exists() and not force:
                converted += 1
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                chgcar_path = Path(tmp) / "CHGCAR"
                with chgcar_path.open("wb") as handle:
                    shutil.copyfileobj(extracted, handle)
                chgcar_to_npz(
                    chgcar_path,
                    out_path,
                    num_probes=probes,
                    seed=seed + converted,
                    density_normalization=density_normalization,
                )
            converted += 1
    return converted


def write_random_splits(
    output_dir: Path,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> None:
    files = sorted(path.name for path in output_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No prepared .npz files under {output_dir}")
    rng = random.Random(seed)
    rng.shuffle(files)
    n_total = len(files)
    n_train = max(1, int(n_total * train_frac))
    n_val = max(1, int(n_total * val_frac)) if n_total >= 3 else 0
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1 if n_total >= 2 else 0
    splits = {
        "train.txt": files[:n_train],
        "val.txt": files[n_train : n_train + n_val],
        "test.txt": files[n_train + n_val :],
    }
    if not splits["test.txt"]:
        splits["test.txt"] = splits["val.txt"] or splits["train.txt"][-1:]
    if not splits["val.txt"]:
        splits["val.txt"] = splits["test.txt"]
    for name, lines in splits.items():
        (output_dir / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"splits for {output_dir.name}: "
        f"train={len(splits['train.txt'])}, val={len(splits['val.txt'])}, test={len(splits['test.txt'])}"
    )


def prepare_figshare_benchmark(
    dataset: str,
    raw_root: Path,
    processed_root: Path,
    probes: int | None,
    density_normalization: str,
    limit_files: int | None,
    limit_samples: int | None,
    skip_download: bool,
    force: bool,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> None:
    raw_dir = raw_root / dataset
    output_dir = processed_root / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    download_figshare_dataset(dataset, raw_dir, limit_files, skip_download)

    tar_files = sorted(raw_dir.glob("*.tar"))
    if not tar_files:
        raise FileNotFoundError(f"No .tar files found under {raw_dir}")

    total_converted = 0
    for tar_path in tar_files:
        remaining = None if limit_samples is None else max(0, limit_samples - total_converted)
        if remaining == 0:
            break
        print(f"converting {tar_path}")
        total_converted += convert_tar_chgcars(
            tar_path,
            output_dir,
            probes=probes,
            density_normalization=density_normalization,
            seed=seed + total_converted,
            force=force,
            remaining=remaining,
        )
    print(f"converted/skipped {total_converted} samples for {dataset}")
    write_random_splits(output_dir, train_frac=train_frac, val_frac=val_frac, seed=seed)


def prepare_materials_project(
    raw_root: Path,
    processed_root: Path,
    probes: int | None,
    density_normalization: str,
    api_key: str | None,
    mp_ids: list[str] | None,
    mp_id_file: str | None,
    limit_samples: int | None,
    force: bool,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> None:
    api_key = api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        raise ValueError("Materials Project preparation needs --mp-api-key or MP_API_KEY.")

    try:
        from emmet.core.summary import HasProps
        from mp_api.client import MPRester
    except Exception as exc:
        raise RuntimeError("Install mp-api to prepare Materials Project data.") from exc

    raw_dir = raw_root / "materials_project"
    output_dir = processed_root / "materials_project"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with MPRester(api_key) as mpr:
        if mp_id_file:
            mpids = [
                line.strip()
                for line in Path(mp_id_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif mp_ids:
            mpids = mp_ids
        else:
            docs = mpr.materials.summary.search(
                has_props=[HasProps.charge_density],
                fields=["material_id"],
            )
            mpids = [str(doc.material_id) for doc in docs]

        converted = 0
        for mpid in mpids:
            if limit_samples is not None and converted >= limit_samples:
                break
            raw_chgcar = raw_dir / mpid / "CHGCAR"
            out_path = output_dir / f"{mpid}.npz"
            if out_path.exists() and not force:
                converted += 1
                continue
            raw_chgcar.parent.mkdir(parents=True, exist_ok=True)
            if not raw_chgcar.exists() or force:
                print(f"downloading Materials Project charge density for {mpid}")
                try:
                    result = mpr.get_charge_density_from_material_id(
                        mpid, inc_task_doc=True
                    )
                except Exception as exc:
                    print(f"skip {mpid}: MP charge-density download failed: {exc}")
                    continue
                if result is None:
                    print(f"skip {mpid}: MP API returned no charge density")
                    continue
                if isinstance(result, tuple):
                    chgcar, taskdoc = result
                else:
                    chgcar, taskdoc = result, None
                if chgcar is None:
                    print(f"skip {mpid}: empty CHGCAR")
                    continue
                chgcar.write_file(str(raw_chgcar))
                if taskdoc is not None:
                    (raw_chgcar.parent / "task_id.txt").write_text(
                        f"{taskdoc.task_id}\n", encoding="utf-8"
                    )
            chgcar_to_npz(
                raw_chgcar,
                out_path,
                num_probes=probes,
                seed=seed + converted,
                density_normalization=density_normalization,
            )
            converted += 1
    if converted == 0:
        raise RuntimeError("No Materials Project charge-density samples were prepared.")
    print(f"converted/skipped {converted} samples for materials_project")
    write_random_splits(output_dir, train_frac=train_frac, val_frac=val_frac, seed=seed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to prepare: all, qm9, nmc, mp/materials_project.",
    )
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--processed-root", default="data/processed")
    parser.add_argument("--probes", type=int, default=5000)
    parser.add_argument("--full-grid", action="store_true", help="Keep all grid points.")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument(
        "--density-normalization",
        default="vasp_divide_volume",
        choices=["raw", "vasp_divide_volume"],
    )
    parser.add_argument("--mp-api-key", default=None)
    parser.add_argument("--mp-ids", nargs="*", default=None, help="Optional MP material ids to prepare.")
    parser.add_argument("--mp-id-file", default=None, help="Optional file with one MP material id per line.")
    args = parser.parse_args()

    datasets = canonical_datasets(args.datasets)
    raw_root = Path(args.raw_root)
    processed_root = Path(args.processed_root)
    probes = None if args.full_grid else args.probes

    for dataset in datasets:
        if dataset in {"qm9", "nmc"}:
            prepare_figshare_benchmark(
                dataset,
                raw_root=raw_root,
                processed_root=processed_root,
                probes=probes,
                density_normalization=args.density_normalization,
                limit_files=args.limit_files,
                limit_samples=args.limit_samples,
                skip_download=args.skip_download,
                force=args.force,
                seed=args.seed,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
            )
        elif dataset == "materials_project":
            prepare_materials_project(
                raw_root=raw_root,
                processed_root=processed_root,
                probes=probes,
                density_normalization=args.density_normalization,
                api_key=args.mp_api_key,
                mp_ids=args.mp_ids,
                mp_id_file=args.mp_id_file,
                limit_samples=args.limit_samples,
                force=args.force,
                seed=args.seed,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
            )


if __name__ == "__main__":
    main()
