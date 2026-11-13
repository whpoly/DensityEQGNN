"""NPZ dataset containing one fitted coefficient vector per atom."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_FIELDS = ("atomic_numbers", "positions", "cell")


class CoefficientDataset(Dataset):
    """Read structures and their oracle atom-centred density coefficients."""

    def __init__(
        self,
        root: str | Path,
        split_file: str | Path | None = None,
        augment_rotations: bool = False,
        num_radial: int | None = None,
        seed: int = 0,
    ) -> None:
        self.root = Path(root)
        self.augment_rotations = augment_rotations
        self.num_radial = num_radial
        self.rng = np.random.default_rng(seed)
        if split_file is None:
            self.files = sorted(self.root.glob("*.npz"))
        else:
            path = Path(split_file)
            if not path.is_absolute():
                path = self.root / path
            names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
            self.files = [
                Path(name) if Path(name).is_absolute() else self.root / name
                for name in names
                if name
            ]
        if not self.files:
            raise FileNotFoundError(f"No coefficient samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            missing = [name for name in REQUIRED_FIELDS if name not in data]
            if missing:
                raise KeyError(f"{path} is missing fields: {missing}")
            sample: dict[str, Any] = {
                "sample_id": path.stem,
                "atomic_numbers": data["atomic_numbers"].astype(np.int64),
                "positions": data["positions"].astype(np.float32),
                "cell": data["cell"].astype(np.float32),
            }
            if "coefficients" in data:
                sample["coefficients"] = data["coefficients"].astype(np.float32)
            for name in ("probe_positions", "density", "probe_volume"):
                if name in data:
                    sample[name] = data[name].astype(np.float32)
        if self.augment_rotations:
            if self.num_radial is None:
                raise ValueError("num_radial is required for rotation augmentation")
            matrix = self.rng.normal(size=(3, 3))
            rotation, _ = np.linalg.qr(matrix)
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1
            rotation = rotation.astype(np.float32)
            sample["positions"] = sample["positions"] @ rotation.T
            sample["cell"] = sample["cell"] @ rotation.T
            if "probe_positions" in sample:
                sample["probe_positions"] = sample["probe_positions"] @ rotation.T
            if (
                "coefficients" in sample
                and sample["coefficients"].shape[1] >= 4 * self.num_radial
            ):
                vector_stop = 4 * self.num_radial
                vector = sample["coefficients"][
                    :, self.num_radial : vector_stop
                ].reshape(
                    len(sample["coefficients"]), self.num_radial, 3
                )
                vector = vector @ rotation.T
                parts = [
                    sample["coefficients"][:, : self.num_radial],
                    vector.reshape(len(vector), -1),
                ]
                if sample["coefficients"].shape[1] > vector_stop:
                    tensor = sample["coefficients"][:, vector_stop:].reshape(
                        len(sample["coefficients"]), self.num_radial, 3, 3
                    )
                    tensor = np.einsum(
                        "ab,inbc,dc->inad", rotation, tensor, rotation
                    )
                    parts.append(tensor.reshape(len(tensor), -1))
                sample["coefficients"] = np.concatenate(parts, axis=-1)
        return sample


def _pad_first_axis(
    arrays: list[np.ndarray], fill: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    max_items = max(array.shape[0] for array in arrays)
    trailing = arrays[0].shape[1:]
    output = np.full((len(arrays), max_items, *trailing), fill, dtype=arrays[0].dtype)
    mask = np.zeros((len(arrays), max_items), dtype=bool)
    for row, array in enumerate(arrays):
        output[row, : len(array)] = array
        mask[row, : len(array)] = True
    return torch.from_numpy(output), torch.from_numpy(mask)


def collate_coefficients(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad atoms and optional density probes into a training batch."""

    atomic_numbers, atom_mask = _pad_first_axis(
        [sample["atomic_numbers"] for sample in samples], fill=0
    )
    positions, _ = _pad_first_axis([sample["positions"] for sample in samples])
    batch: dict[str, Any] = {
        "sample_id": [sample["sample_id"] for sample in samples],
        "atomic_numbers": atomic_numbers.long(),
        "positions": positions.float(),
        "cell": torch.from_numpy(np.stack([sample["cell"] for sample in samples])).float(),
        "atom_mask": atom_mask,
    }
    if all("coefficients" in sample for sample in samples):
        coefficients, _ = _pad_first_axis(
            [sample["coefficients"] for sample in samples]
        )
        batch["coefficients"] = coefficients.float()

    if all("density" in sample and "probe_positions" in sample for sample in samples):
        probe_positions, probe_mask = _pad_first_axis(
            [sample["probe_positions"] for sample in samples]
        )
        density, _ = _pad_first_axis(
            [sample["density"][:, None] for sample in samples]
        )
        batch["probe_positions"] = probe_positions.float()
        batch["probe_mask"] = probe_mask
        batch["density"] = density.squeeze(-1).float()
        if all("probe_volume" in sample for sample in samples):
            batch["probe_volume"] = torch.as_tensor(
                [float(np.asarray(sample["probe_volume"])) for sample in samples],
                dtype=torch.float32,
            )
    return batch
