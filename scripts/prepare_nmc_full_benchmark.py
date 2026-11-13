"""Prepare the complete public NMC benchmark for density-coefficient training.

The two official, uncompressed tar archives contain 2,000 LZ4-compressed
CHGCAR members.  This script downloads the archives with resume support,
uses the official 1450/50/500 split, and stores uniformly sampled density
probes for training and validation.  The raw archives remain the source of
truth for full-grid evaluation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

import lz4.frame
import numpy as np
from ase.io.vasp import read_vasp_configuration
from pymatgen.io.vasp.outputs import Chgcar

from density_eqgnn.chgcar import vasp_grid


DATA_ROOT = Path(os.environ.get("DENSITY_EQGNN_DATA_ROOT", "data"))
ARCHIVES = (
    {
        "name": "cellrelax_000.tar",
        "url": "https://ndownloader.figshare.com/files/31138187",
        "size": 5_432_657_920,
        "md5": "0b7eeaa162d92b35c2fce3df31f0bf94",
    },
    {
        "name": "cellrelax_001.tar",
        "url": "https://ndownloader.figshare.com/files/31138196",
        "size": 5_432_483_840,
        "md5": "ecfd5a1f40caae853f3a33e536aeb1db",
    },
)
SPLIT_URL = "https://ndownloader.figshare.com/files/31138219"


def file_md5(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_resume(url: str, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = destination.stat().st_size if destination.exists() else 0
    if existing > expected_size:
        raise IOError(f"{destination} is larger than the expected remote file")
    if existing == expected_size:
        print(f"reuse {destination} ({existing / 1024**3:.2f} GiB)", flush=True)
        return

    # Small explicit ranges avoid large-response buffering by common desktop
    # HTTP proxies and make progress genuinely resumable.
    request_size = 8 * 1024 * 1024
    ranges = []
    start = existing
    while start < expected_size:
        stop = min(start + request_size, expected_size) - 1
        ranges.append((start, stop))
        start = stop + 1
    parts_dir = destination.parent / f"{destination.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    def fetch(byte_range: tuple[int, int]) -> tuple[int, int, Path]:
        start, stop = byte_range
        part = parts_dir / f"{start:012d}-{stop:012d}.part"
        expected_chunk = stop - start + 1
        if part.exists() and part.stat().st_size == expected_chunk:
            return start, stop, part
        chunk = None
        for attempt in range(5):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "density-eqgnn-nmc-benchmark/1.0",
                        "Range": f"bytes={start}-{stop}",
                    },
                )
                with urllib.request.urlopen(request, timeout=180) as response:
                    if response.status != 206:
                        raise IOError(
                            f"Server did not honor byte range for {destination}"
                        )
                    chunk = response.read()
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        assert chunk is not None
        if len(chunk) != expected_chunk:
            raise IOError(
                f"Received {len(chunk)} bytes for a {expected_chunk}-byte "
                f"range of {destination}"
            )
        part.write_bytes(chunk)
        return start, stop, part

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch, byte_range) for byte_range in ranges]
        for future in concurrent.futures.as_completed(futures):
            future.result()
            completed += 1
            if completed == 1 or completed % 8 == 0 or completed == len(ranges):
                ready = min(
                    existing + completed * request_size, expected_size
                )
                print(
                    f"{destination.name}: downloaded ranges through about "
                    f"{ready / 1024**3:.2f}/{expected_size / 1024**3:.2f} GiB",
                    flush=True,
                )

    with destination.open("ab" if existing else "wb") as output:
        for start, stop in ranges:
            part = parts_dir / f"{start:012d}-{stop:012d}.part"
            expected_chunk = stop - start + 1
            if part.stat().st_size != expected_chunk:
                raise IOError(f"Incomplete range file: {part}")
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            part.unlink()
        output.flush()
    parts_dir.rmdir()
    downloaded = destination.stat().st_size
    print(
        f"{destination.name}: assembled {downloaded / 1024**3:.2f} GiB",
        flush=True,
    )
    if downloaded != expected_size:
        raise IOError(
            f"{destination} has {downloaded} bytes; expected {expected_size}"
        )


def download_inputs(raw_dir: Path, verify_md5: bool) -> tuple[list[Path], Path]:
    archives = []
    for record in ARCHIVES:
        path = raw_dir / str(record["name"])
        download_with_resume(str(record["url"]), path, int(record["size"]))
        if verify_md5:
            actual = file_md5(path)
            if actual != record["md5"]:
                raise IOError(f"MD5 mismatch for {path}: {actual}")
        archives.append(path)

    split_path = raw_dir / "split.json"
    if not split_path.exists():
        urllib.request.urlretrieve(SPLIT_URL, split_path)
    return archives, split_path


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


def convert_member(
    archive: tarfile.TarFile,
    member: tarfile.TarInfo,
    destination: Path,
    probe_count: int,
    seed: int,
    temp_dir: Path,
) -> dict[str, object]:
    compressed = archive.extractfile(member)
    if compressed is None:
        raise IOError(f"Could not read {member.name}")
    chgcar_path = temp_dir / "CHGCAR"
    with lz4.frame.open(compressed, "rb") as source, chgcar_path.open("wb") as target:
        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)

    with chgcar_path.open("r", encoding="utf-8") as handle:
        ase_atoms = read_vasp_configuration(handle)
    ase_atoms.set_pbc(True)
    chgcar = Chgcar.from_file(chgcar_path)
    validate_structure(ase_atoms, chgcar, member.name)

    raw_density = np.asarray(chgcar.data["total"], dtype=np.float64)
    density = raw_density.reshape(-1) / ase_atoms.get_volume()
    all_probes = vasp_grid(ase_atoms.cell.array, raw_density.shape)
    count = min(probe_count, len(density))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(density), count, replace=False))
    np.savez_compressed(
        destination,
        atomic_numbers=ase_atoms.get_atomic_numbers().astype(np.int64),
        positions=ase_atoms.get_positions().astype(np.float32),
        cell=ase_atoms.cell.array.astype(np.float32),
        probe_positions=all_probes[indices].astype(np.float32),
        density=density[indices].astype(np.float32),
        probe_volume=np.float32(ase_atoms.get_volume() / len(density)),
        grid_shape=np.asarray(raw_density.shape, dtype=np.int64),
        source_member=np.asarray(member.name),
    )
    return {
        "sample_id": destination.stem,
        "formula": ase_atoms.get_chemical_formula(),
        "num_atoms": len(ase_atoms),
        "grid_shape": list(raw_density.shape),
        "stored_probes": count,
        "source_member": member.name,
    }


def prepare(
    archives: list[Path],
    split_path: Path,
    output_dir: Path,
    train_probes: int,
    eval_probes: int,
    force: bool,
    limit: int | None,
) -> dict[str, object]:
    official_split = json.loads(split_path.read_text(encoding="utf-8"))
    split_by_id = {
        int(sample_id): split
        for split, ids in official_split.items()
        for sample_id in ids
    }
    expected = {"train": 1450, "validation": 50, "test": 500}
    actual = {
        split: sum(value == split for value in split_by_id.values())
        for split in expected
    }
    if actual != expected:
        raise ValueError(f"Unexpected official split sizes: {actual}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    seen = 0
    temp_root = output_dir.parent / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nmc_full_", dir=temp_root) as temporary:
        temp_dir = Path(temporary)
        for archive_path in archives:
            with tarfile.open(archive_path, "r:") as archive:
                for member in archive:
                    if not member.isfile() or not member.name.endswith(".CHGCAR.lz4"):
                        continue
                    sample_id = int(Path(member.name).name.removesuffix(".CHGCAR.lz4"))
                    split = split_by_id[sample_id]
                    destination = output_dir / f"{sample_id}.npz"
                    if destination.exists() and not force:
                        with np.load(destination, allow_pickle=False) as saved:
                            record = {
                                "sample_id": str(sample_id),
                                "num_atoms": int(len(saved["atomic_numbers"])),
                                "grid_shape": saved["grid_shape"].tolist(),
                                "stored_probes": int(len(saved["density"])),
                                "source_member": str(saved["source_member"]),
                            }
                    else:
                        count = train_probes if split == "train" else eval_probes
                        record = convert_member(
                            archive,
                            member,
                            destination,
                            count,
                            sample_id + 20_260_724,
                            temp_dir,
                        )
                    record["split"] = split
                    record["archive"] = archive_path.name
                    records.append(record)
                    seen += 1
                    if seen == 1 or seen % 25 == 0:
                        print(
                            f"prepared {seen}/2000: {sample_id} "
                            f"({record['num_atoms']} atoms, {split})",
                            flush=True,
                        )
                    if limit is not None and seen >= limit:
                        break
            if limit is not None and seen >= limit:
                break

    available = {int(record["sample_id"]) for record in records}
    split_names = {"train": "train.txt", "validation": "val.txt", "test": "test.txt"}
    for official_name, filename in split_names.items():
        names = [
            f"{sample_id}.npz"
            for sample_id in official_split[official_name]
            if sample_id in available
        ]
        (output_dir / filename).write_text(
            "\n".join(names) + ("\n" if names else ""),
            encoding="utf-8",
        )
    metadata = {
        "source": "NMC Li-ion Battery Cathode Energies and Charge Densities",
        "figshare_article": 16837721,
        "license": "CC BY 4.0",
        "split_url": SPLIT_URL,
        "official_split_sizes": expected,
        "stored_train_probes": train_probes,
        "stored_eval_probes": eval_probes,
        "records": records,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=DATA_ROOT / "nmc_raw")
    parser.add_argument("--output", type=Path, default=DATA_ROOT / "nmc_full_probes")
    parser.add_argument("--train-probes", type=int, default=16384)
    parser.add_argument("--eval-probes", type=int, default=8192)
    parser.add_argument("--verify-md5", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if args.download_only and args.prepare_only:
        parser.error("--download-only and --prepare-only are mutually exclusive")
    if args.prepare_only:
        archives = [args.raw_dir / record["name"] for record in ARCHIVES]
        split_path = args.raw_dir / "split.json"
        missing = [path for path in [*archives, split_path] if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing downloaded inputs: {missing}")
    else:
        archives, split_path = download_inputs(args.raw_dir, args.verify_md5)
    if args.download_only:
        return

    metadata = prepare(
        archives,
        split_path,
        args.output,
        args.train_probes,
        args.eval_probes,
        args.force,
        args.limit,
    )
    counts = {
        split: sum(record["split"] == split for record in metadata["records"])
        for split in ("train", "validation", "test")
    }
    print(json.dumps({"prepared": len(metadata["records"]), "splits": counts}, indent=2))


if __name__ == "__main__":
    main()
