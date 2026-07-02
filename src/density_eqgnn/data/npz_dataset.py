"""NPZ dataset used by the training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception:  # pragma: no cover - useful for metadata-only environments
    torch = None
    Dataset = object


REQUIRED_FIELDS = ("atomic_numbers", "positions", "probe_positions", "density")


class DensityNPZDataset(Dataset):
    """Read one density sample per .npz file.

    A split file is optional. When present it should be a plain text file with
    one sample path per line, relative to root or absolute.
    """

    def __init__(
        self,
        root: str | Path,
        split_file: str | Path | None = None,
        max_probes: int | None = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.max_probes = max_probes
        self.seed = seed
        if split_file is None:
            self.files = sorted(self.root.glob("*.npz"))
        else:
            split_path = Path(split_file)
            if not split_path.is_absolute():
                split_path = self.root / split_path
            lines = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
            self.files = [Path(line) if Path(line).is_absolute() else self.root / line for line in lines]
        if not self.files:
            raise FileNotFoundError(f"No .npz samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            missing = [field for field in REQUIRED_FIELDS if field not in data]
            if missing:
                raise KeyError(f"{path} missing required fields: {missing}")

            sample: dict[str, Any] = {
                "sample_id": path.stem,
                "atomic_numbers": data["atomic_numbers"].astype(np.int64),
                "positions": data["positions"].astype(np.float32),
                "cell": data["cell"].astype(np.float32)
                if "cell" in data
                else np.eye(3, dtype=np.float32),
                "probe_positions": data["probe_positions"].astype(np.float32),
                "density": data["density"].astype(np.float32),
            }
            if "property" in data:
                sample["property"] = np.atleast_1d(data["property"]).astype(np.float32)
            if "probe_volume" in data:
                sample["probe_volume"] = np.asarray(data["probe_volume"], dtype=np.float32)
            if "electron_count" in data:
                sample["electron_count"] = np.asarray(data["electron_count"], dtype=np.float32)

        if self.max_probes is not None and sample["density"].shape[0] > self.max_probes:
            rng = np.random.default_rng(self.seed + index)
            choice = np.sort(
                rng.choice(sample["density"].shape[0], size=self.max_probes, replace=False)
            )
            sample["probe_positions"] = sample["probe_positions"][choice]
            sample["density"] = sample["density"][choice]
            if "probe_volume" in sample and sample["probe_volume"].ndim > 0:
                sample["probe_volume"] = sample["probe_volume"][choice]
        return sample


def _pad_array(arrays: list[np.ndarray], fill_value: float = 0.0) -> tuple[Any, Any]:
    if torch is None:
        raise RuntimeError("collate_density_samples requires torch")
    max_shape = tuple(max(arr.shape[d] for arr in arrays) for d in range(arrays[0].ndim))
    output = []
    mask = []
    for arr in arrays:
        padded = np.full(max_shape, fill_value, dtype=arr.dtype)
        slices = tuple(slice(0, size) for size in arr.shape)
        padded[slices] = arr
        output.append(padded)
        mask_1d = np.zeros(max_shape[0], dtype=bool)
        mask_1d[: arr.shape[0]] = True
        mask.append(mask_1d)
    return torch.as_tensor(np.stack(output)), torch.as_tensor(np.stack(mask))


def collate_density_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable atom/probe counts into a batch dictionary."""
    if torch is None:
        raise RuntimeError("collate_density_samples requires torch")

    atomic_numbers, atom_mask = _pad_array([s["atomic_numbers"] for s in samples], fill_value=0)
    positions, _ = _pad_array([s["positions"] for s in samples], fill_value=0.0)
    probe_positions, probe_mask = _pad_array([s["probe_positions"] for s in samples], fill_value=0.0)
    density, _ = _pad_array([s["density"] for s in samples], fill_value=0.0)

    batch: dict[str, Any] = {
        "sample_id": [s["sample_id"] for s in samples],
        "atomic_numbers": atomic_numbers.long(),
        "positions": positions.float(),
        "atom_mask": atom_mask.bool(),
        "cell": torch.as_tensor(np.stack([s["cell"] for s in samples])).float(),
        "probe_positions": probe_positions.float(),
        "probe_mask": probe_mask.bool(),
        "density": density.float(),
    }

    if all("property" in s for s in samples):
        props = [s["property"] for s in samples]
        prop, _ = _pad_array(props, fill_value=0.0)
        batch["property"] = prop.float()

    if all("probe_volume" in s for s in samples):
        volumes = []
        for s in samples:
            v = np.asarray(s["probe_volume"], dtype=np.float32)
            if v.ndim == 0:
                v = np.full_like(s["density"], float(v), dtype=np.float32)
            volumes.append(v)
        probe_volume, _ = _pad_array(volumes, fill_value=0.0)
        batch["probe_volume"] = probe_volume.float()

    if all("electron_count" in s for s in samples):
        batch["electron_count"] = torch.as_tensor(
            [float(np.asarray(s["electron_count"])) for s in samples], dtype=torch.float32
        )

    return batch

