"""Drop angular channels above a requested lmax from compact NPZ labels."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-radial", type=int, required=True)
    parser.add_argument("--source-lmax", type=int, required=True)
    parser.add_argument("--target-lmax", type=int, required=True)
    args = parser.parse_args()
    if args.target_lmax >= args.source_lmax:
        raise ValueError("target-lmax must be smaller than source-lmax")
    source_angular = (args.source_lmax + 1) ** 2
    target_angular = (args.target_lmax + 1) ** 2
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.source.glob("*.npz"))
    for index, path in enumerate(paths, 1):
        with np.load(path, allow_pickle=False) as sample:
            payload = {name: sample[name] for name in sample.files}
        coefficients = payload["coefficients"].reshape(
            len(payload["atomic_numbers"]),
            args.num_radial,
            source_angular,
        )
        payload["coefficients"] = coefficients[
            ..., :target_angular
        ].reshape(len(coefficients), -1).astype(np.float32)
        payload["compact_lmax"] = np.int64(args.target_lmax)
        np.savez_compressed(args.output / path.name, **payload)
        if index % 100 == 0 or index == len(paths):
            print(f"{index}/{len(paths)}", flush=True)
    for split_name in ("train.txt", "val.txt", "test.txt", "development.txt"):
        source = args.source / split_name
        if source.exists():
            shutil.copy2(source, args.output / split_name)


if __name__ == "__main__":
    main()
