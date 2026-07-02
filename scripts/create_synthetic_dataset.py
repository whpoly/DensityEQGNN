"""Create a tiny NPZ dataset for smoke testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_sample(seed: int, natoms: int, nprobes: int) -> dict:
    rng = np.random.default_rng(seed)
    atomic_numbers = rng.choice([1, 6, 7, 8], size=natoms).astype(np.int64)
    positions = rng.normal(scale=1.0, size=(natoms, 3)).astype(np.float32)
    probe_positions = rng.normal(scale=1.8, size=(nprobes, 3)).astype(np.float32)
    density = np.zeros(nprobes, dtype=np.float32)
    for z, pos in zip(atomic_numbers, positions):
        dist2 = np.sum((probe_positions - pos) ** 2, axis=1)
        density += 0.05 * z * np.exp(-dist2 / 0.8).astype(np.float32)
    return {
        "atomic_numbers": atomic_numbers,
        "positions": positions,
        "cell": np.eye(3, dtype=np.float32) * 20.0,
        "probe_positions": probe_positions,
        "density": density,
        "property": np.asarray([density.mean()], dtype=np.float32),
        "probe_volume": np.asarray(1.0 / nprobes, dtype=np.float32),
        "electron_count": np.asarray(float(density.mean()), dtype=np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/processed/example")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--natoms", type=int, default=5)
    parser.add_argument("--probes", type=int, default=128)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for idx in range(args.samples):
        name = f"synthetic_{idx:04d}.npz"
        np.savez_compressed(out / name, **make_sample(idx, args.natoms, args.probes))
        names.append(name)

    n_train = max(1, int(0.75 * len(names)))
    train = names[:n_train]
    val = names[n_train:] or names[-1:]
    (out / "train.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out / "val.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    (out / "test.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    print(f"wrote {len(names)} samples to {out}")


if __name__ == "__main__":
    main()
