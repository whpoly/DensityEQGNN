"""Download only a small prefix of the public NMC tar and fit coefficients.

The Figshare archives are uncompressed tar files whose individual CHGCAR
members are LZ4-compressed.  HTTP Range requests let us fetch selected members
without downloading either complete ~5.4 GB archive.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import lz4.frame

from density_eqgnn.basis import DensityBasisConfig
from density_eqgnn.chgcar import extract_chgcar_sample


NMC_TAR_URL = "https://ndownloader.figshare.com/files/31138187"
DATA_ROOT = Path(os.environ.get("DENSITY_EQGNN_DATA_ROOT", "data"))


def read_range(url: str, start: int, stop: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{stop}"})
    with urllib.request.urlopen(request) as response:
        content = response.read()
    expected = stop - start + 1
    if len(content) != expected:
        raise IOError(f"Expected {expected} bytes, received {len(content)}")
    return content


def parse_tar_header(block: bytes) -> tuple[str, int]:
    if len(block) != 512:
        raise ValueError("A tar header must contain exactly 512 bytes")
    if not block.strip(b"\0"):
        return "", 0
    name = block[:100].split(b"\0", 1)[0].decode("utf-8")
    size_text = block[124:136].split(b"\0", 1)[0].strip() or b"0"
    return name, int(size_text, 8)


def iter_remote_tar_members(url: str):
    offset = 0
    while True:
        header = read_range(url, offset, offset + 511)
        name, size = parse_tar_header(header)
        if not name:
            return
        data_start = offset + 512
        yield name, size, data_start
        offset = data_start + ((size + 511) // 512) * 512


def prepare_subset(
    output: str | Path,
    samples: int,
    url: str = NMC_TAR_URL,
    fit_probes: int = 20000,
    eval_probes: int = 4096,
    ridge: float = 1e-4,
    force: bool = False,
    cache_dir: str | Path = DATA_ROOT / "nmc_cache",
) -> list[dict]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    basis = DensityBasisConfig(num_radial=4, cutoff=3.5, lmax=1, pbc=True)
    records: list[dict] = []

    temp_root = cache_dir.parent / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="density_eqgnn_nmc_", dir=temp_root
    ) as temp_dir:
        temp = Path(temp_dir)
        for name, size, data_start in iter_remote_tar_members(url):
            if not name.endswith(".CHGCAR.lz4"):
                continue
            sample_id = Path(name).name.removesuffix(".CHGCAR.lz4")
            destination = output / f"{sample_id}.npz"
            if destination.exists() and not force:
                print(f"reuse {destination}")
                records.append({"sample_id": sample_id, "path": destination.name})
            else:
                compressed = cache_dir / Path(name).name
                if not compressed.exists():
                    print(f"download {name} ({size / 1024**2:.1f} MiB compressed)")
                    compressed.write_bytes(
                        read_range(url, data_start, data_start + size - 1)
                    )
                else:
                    print(f"reuse cached {compressed}")
                chgcar = temp / f"{sample_id}.CHGCAR"
                with lz4.frame.open(compressed, "rb") as source, chgcar.open("wb") as target:
                    shutil.copyfileobj(source, target)
                metrics = extract_chgcar_sample(
                    chgcar,
                    destination,
                    basis,
                    ridge=ridge,
                    fit_probes=fit_probes,
                    eval_probes=eval_probes,
                    seed=int(sample_id),
                )
                records.append(
                    {
                        "sample_id": sample_id,
                        "path": destination.name,
                        "fit_nmae": metrics["nmae"],
                        "charge_error": metrics["charge_error"],
                    }
                )
                chgcar.unlink()
            if len(records) >= samples:
                break

    if len(records) < samples:
        raise RuntimeError(f"Only found {len(records)} NMC members, requested {samples}")
    names = [record["path"] for record in records]
    n_train = max(1, int(0.8 * len(names)))
    n_val = max(1, int(0.1 * len(names)))
    splits = {
        "train": names[:n_train],
        "val": names[n_train : n_train + n_val],
        "test": names[n_train + n_val :],
    }
    if not splits["test"]:
        splits["test"] = splits["val"]
    for split, split_names in splits.items():
        (output / f"{split}.txt").write_text(
            "\n".join(split_names) + "\n", encoding="utf-8"
        )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "source": "NMC Li-ion Battery Cathode Energies and Charge Densities",
                "figshare_article": 16837721,
                "archive_url": url,
                "license": "CC BY 4.0",
                "basis": basis.to_dict(),
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DATA_ROOT / "nmc_subset"))
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--fit-probes", type=int, default=20000)
    parser.add_argument("--eval-probes", type=int, default=4096)
    parser.add_argument("--ridge", type=float, default=1e-4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-dir", default=str(DATA_ROOT / "nmc_cache"))
    parser.add_argument("--url", default=NMC_TAR_URL)
    args = parser.parse_args()
    records = prepare_subset(
        args.output,
        args.samples,
        args.url,
        args.fit_probes,
        args.eval_probes,
        args.ridge,
        args.force,
        args.cache_dir,
    )
    fitted = [record.get("fit_nmae") for record in records if "fit_nmae" in record]
    if fitted:
        print(f"prepared {len(records)} samples; mean oracle fit NMAE={sum(fitted)/len(fitted):.6f}")
    else:
        print(f"prepared {len(records)} existing samples")


if __name__ == "__main__":
    main()
