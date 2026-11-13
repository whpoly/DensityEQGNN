"""Spatial holdout utilities and unmodified official DeepDFT adapter.

All periodic caches contain geometry only. Density labels never enter a graph.
"""
from __future__ import annotations

import hashlib
import importlib
import itertools
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch

OFFICIAL_COMMIT = "a6bab4deb5cf05d9b46ae397b72253d04ea3c694"
DEFAULT_ROOT = Path("runs/nmc_probe_holdout_2026-10-26")
SPLIT_SEED = 20261026


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spatial_split(size, sample_id, seed=SPLIT_SEED):
    """Label-independent 80/10/10 partition, fixed across models and stages."""
    order = np.random.default_rng(np.random.SeedSequence([seed, sample_id])).permutation(size)
    test_size = size // 10
    validation_size = size // 10
    return {
        "train": np.sort(order[test_size + validation_size:]),
        "validation": np.sort(order[test_size:test_size + validation_size]),
        "test": np.sort(order[:test_size]),
        "selection": order[test_size:test_size + min(validation_size, 8192)],
    }


def save_split(folder, size, sample_id, source_hash):
    folder.mkdir(parents=True, exist_ok=True)
    split = spatial_split(size, sample_id)
    path = folder / f"sample_{sample_id:04d}.npz"
    metadata = {"sample_id": sample_id, "source_sha256": source_hash,
                "grid_points": size, "seed": SPLIT_SEED,
                "counts": {k: len(v) for k, v in split.items()},
                "indices_sha256": {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in split.items()}}
    if path.exists():
        with np.load(path) as old:
            for key, value in split.items():
                if not np.array_equal(value, old[key]):
                    raise ValueError("Existing spatial split differs")
        if json.loads(path.with_suffix(".json").read_text()) != metadata:
            raise ValueError("Split source or settings changed")
    else:
        np.savez_compressed(path, **split)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return split, metadata


def official_model(repository, device):
    repository = Path(repository).resolve()
    revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    if revision != OFFICIAL_COMMIT:
        raise ValueError(f"Official checkout differs: {revision}")
    if subprocess.check_output(["git", "-C", str(repository), "diff", "--name-only"], text=True).strip():
        raise ValueError("Official checkout has modified tracked sources")
    sys.path.insert(0, str(repository))
    module = importlib.import_module("densitymodel")
    layer_module = importlib.import_module("layer")
    if Path(module.__file__).parent != repository or Path(layer_module.__file__).parent != repository:
        raise ValueError("Official module name collision")
    # Matches the released nmc_painn arguments and runner, including default RBF=30.
    model = module.PainnDensityModel(num_interactions=3, hidden_state_size=128,
                                    cutoff=4.0, distance_embedding_size=30).to(device)
    provenance = {"repository": "https://github.com/peterbjorgensen/DeepDFT",
                  "commit": revision, "class": "densitymodel.PainnDensityModel",
                  "num_interactions": 3, "hidden_state_size": 128, "cutoff": 4.0,
                  "distance_embedding_size": 30,
                  "sources": {name: sha256(repository / name) for name in ("densitymodel.py", "layer.py")}}
    return model, provenance


class PeriodicQueryGraph:
    """CSR atom->query edges from all lattice images, arbitrary unwrapped points.

    cKDTree only replaces neighbour discovery; the official network computes
    displacements, radial features, messages and outputs without modification.
    """
    def __init__(self, positions, probes, cell, cutoff, *, exclude_self=False):
        self.device = positions.device
        self.positions, self.probes, self.cell = positions, probes, cell
        p = positions.detach().cpu().double().numpy()
        q = probes.detach().cpu().double().numpy()
        c = cell.detach().cpu().double().numpy()
        inv = np.linalg.inv(c)
        pa, pq = np.floor(p @ inv).astype(np.int64), np.floor(q @ inv).astype(np.int64)
        wp, wq = p - pa @ c, q - pq @ c
        bounds = np.ceil(cutoff * np.linalg.norm(inv, axis=0)).astype(int) + 1
        shifts = np.array(list(itertools.product(*(range(-b, b + 1) for b in bounds))), dtype=np.int64)
        image_positions = (wp[None] + (shifts @ c)[:, None]).reshape(-1, 3)
        image_atoms = np.tile(np.arange(len(p)), len(shifts))
        image_shifts = np.repeat(shifts, len(p), axis=0) - pa[image_atoms]
        tree = cKDTree(image_positions)
        atoms, translations, counts = [], [], []
        for start in range(0, len(q), 16384):
            stop = min(start + 16384, len(q))
            lists = tree.query_ball_point(wq[start:stop], cutoff, workers=1, return_sorted=True)
            n = np.array([len(x) for x in lists], dtype=np.int64)
            receiver = np.repeat(np.arange(start, stop), n)
            image = np.concatenate(lists).astype(np.int64) if n.sum() else np.empty(0, dtype=np.int64)
            a = image_atoms[image]
            shift = image_shifts[image] + pq[receiver]
            displacement = p[a] + shift @ c - q[receiver]
            valid = (displacement ** 2).sum(1) < cutoff ** 2
            if exclude_self:
                valid &= ~((a == receiver) & (shift == 0).all(1))
            atoms.append(a[valid])
            translations.append(shift[valid])
            counts.append(np.bincount(receiver[valid] - start, minlength=stop-start))
        atom_array = np.concatenate(atoms)
        shift_array = np.concatenate(translations)
        self.atoms = torch.tensor(atom_array, dtype=torch.int32, device=self.device)
        self.shifts = torch.tensor(shift_array, dtype=torch.int16, device=self.device)
        self.offsets = torch.tensor(np.r_[0, np.cumsum(np.concatenate(counts))], device=self.device)

    def select(self, indices):
        counts = self.offsets[indices + 1] - self.offsets[indices]
        count = int(counts.sum())
        receiver = torch.repeat_interleave(torch.arange(len(indices), device=self.device), counts, output_size=count)
        starts = counts.cumsum(0) - counts
        edge = self.offsets[indices][receiver] + torch.arange(count, device=self.device) - starts[receiver]
        return self.atoms[edge].long(), receiver, self.shifts[edge]


def atom_batch(sample):
    graph = PeriodicQueryGraph(sample["positions"], sample["positions"], sample["cell"], 4., exclude_self=True)
    device = sample["positions"].device
    source, target, shift = graph.select(torch.arange(len(sample["positions"]), device=device))
    return {"nodes": sample["numbers"][None], "atom_xyz": sample["positions"].float()[None],
            "cell": sample["cell"].float()[None],
            "num_nodes": torch.tensor([len(sample["numbers"])], device=device),
            "atom_edges": torch.stack((source, target), 1)[None],
            "atom_edges_displacement": shift.float()[None],
            "num_atom_edges": torch.tensor([len(source)], device=device)}


def probe_batch(sample, indices):
    source, target, shifts = sample["graph"].select(indices)
    batch = dict(sample["batch"])
    batch.update(probe_xyz=sample["probes"][indices].float()[None],
                 num_probes=torch.tensor([len(indices)], device=indices.device),
                 probe_edges=torch.stack((source, target), 1)[None],
                 probe_edges_displacement=shifts.float()[None],
                 num_probe_edges=torch.tensor([len(source)], device=indices.device))
    return batch


def compact_prediction(sample, coefficients, indices, basis):
    from density_eqgnn.compact_basis import reconstruct_compact_from_edges
    source, target, shift = sample["graph"].select(indices)
    delta = sample["probes"][indices][target] - sample["positions"][source] - shift.double() @ sample["cell"]
    return reconstruct_compact_from_edges(coefficients, delta, torch.zeros_like(source), source,
                                          target, len(indices), basis)
