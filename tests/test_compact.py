import numpy as np
import torch

from density_eqgnn.compact_basis import (
    CompactBasisConfig,
    reconstruct_compact_density,
)
from density_eqgnn.data import CoefficientDataset, collate_coefficients
from density_eqgnn.models import CompactCoefficientGNN, CompactModelConfig


def test_l3_compact_model_density_is_rotation_invariant() -> None:
    torch.manual_seed(7)
    config = CompactModelConfig(
        num_radial=3,
        lmax=3,
        hidden_dim=16,
        num_interactions=2,
        edge_basis=6,
        cutoff=4.0,
    )
    basis = CompactBasisConfig(num_radial=3, lmax=3, cutoff=5.0)
    model = CompactCoefficientGNN(config)
    for head in [model.scalar_head, *model.equivariant_heads.values()]:
        torch.nn.init.normal_(head[-1].weight, std=0.02)

    atomic_numbers = torch.tensor([[6, 1, 8]])
    positions = torch.randn(1, 3, 3)
    probes = torch.randn(1, 19, 3)
    atom_mask = torch.ones(1, 3, dtype=torch.bool)
    probe_mask = torch.ones(1, 19, dtype=torch.bool)
    cell = torch.eye(3)[None] * 12.0

    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotation[:, 0] *= torch.linalg.det(rotation)
    coefficients = model(
        atomic_numbers, positions, atom_mask, cell
    )["coefficients"]
    density = reconstruct_compact_density(
        coefficients,
        positions,
        probes,
        atom_mask,
        probe_mask,
        cell,
        basis,
    )
    rotated_positions = positions @ rotation.T
    rotated_probes = probes @ rotation.T
    rotated_coefficients = model(
        atomic_numbers,
        rotated_positions,
        atom_mask,
        cell @ rotation.T,
    )["coefficients"]
    rotated_density = reconstruct_compact_density(
        rotated_coefficients,
        rotated_positions,
        rotated_probes,
        atom_mask,
        probe_mask,
        cell @ rotation.T,
        basis,
    )
    torch.testing.assert_close(density, rotated_density, atol=2e-6, rtol=2e-6)


def test_density_only_dataset_does_not_require_fitted_coefficients(tmp_path) -> None:
    np.savez(
        tmp_path / "0.npz",
        atomic_numbers=np.asarray([3, 8], dtype=np.int64),
        positions=np.zeros((2, 3), dtype=np.float32),
        cell=np.eye(3, dtype=np.float32),
        probe_positions=np.zeros((5, 3), dtype=np.float32),
        density=np.ones(5, dtype=np.float32),
    )
    sample = CoefficientDataset(tmp_path)[0]
    batch = collate_coefficients([sample])

    assert "coefficients" not in sample
    assert "coefficients" not in batch
    assert batch["density"].shape == (1, 5)
