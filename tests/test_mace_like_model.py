import torch
import pytest

from density_eqgnn.compact_basis import (
    CompactBasisConfig,
    reconstruct_compact_density,
)
from density_eqgnn.models import MaceLikeCoefficientGNN, MaceLikeModelConfig


def test_mace_like_l2_density_is_rotation_invariant() -> None:
    torch.manual_seed(19)
    config = MaceLikeModelConfig(
        num_radial=3,
        lmax=2,
        scalar_channels=6,
        vector_channels=3,
        tensor_channels=2,
        num_interactions=2,
        edge_basis=5,
        radial_hidden=8,
        cutoff=4.0,
    )
    basis = CompactBasisConfig(num_radial=3, lmax=2, cutoff=4.0)
    model = MaceLikeCoefficientGNN(config)
    atomic_numbers = torch.tensor([[6, 1, 8, 1]])
    positions = torch.randn(1, 4, 3)
    probes = torch.randn(1, 13, 3)
    atom_mask = torch.ones(1, 4, dtype=torch.bool)
    probe_mask = torch.ones(1, 13, dtype=torch.bool)
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
    torch.testing.assert_close(
        density, rotated_density, atol=3e-5, rtol=3e-5
    )


def test_l2_trunk_reaches_l4_density_rotation_invariant() -> None:
    torch.manual_seed(23)
    config = MaceLikeModelConfig(
        num_radial=2,
        lmax=2,
        hidden_lmax=2,
        density_lmax=4,
        readout_edge_lmax=2,
        scalar_channels=5,
        vector_channels=3,
        tensor_channels=2,
        num_interactions=2,
        edge_basis=4,
        radial_hidden=8,
        cutoff=4.0,
    )
    basis = CompactBasisConfig(num_radial=2, lmax=4, cutoff=4.0)
    model = MaceLikeCoefficientGNN(config)
    atomic_numbers = torch.tensor([[6, 1, 8, 7]])
    positions = torch.randn(1, 4, 3)
    probes = torch.randn(1, 11, 3)
    atom_mask = torch.ones(1, 4, dtype=torch.bool)
    probe_mask = torch.ones(1, 11, dtype=torch.bool)
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
    rotated_coefficients = model(
        atomic_numbers,
        positions @ rotation.T,
        atom_mask,
        cell @ rotation.T,
    )["coefficients"]
    rotated_density = reconstruct_compact_density(
        rotated_coefficients,
        positions @ rotation.T,
        probes @ rotation.T,
        atom_mask,
        probe_mask,
        cell @ rotation.T,
        basis,
    )

    assert coefficients.shape[-1] == 2 * (4 + 1) ** 2
    l4 = coefficients.reshape(1, 4, 2, 25)[..., 16:25]
    assert torch.count_nonzero(l4).item() > 0
    torch.testing.assert_close(
        density, rotated_density, atol=5e-5, rtol=5e-5
    )


def test_readout_rejects_orders_above_clebsch_gordan_limit() -> None:
    with pytest.raises(ValueError, match="tensor-product limit"):
        MaceLikeCoefficientGNN(
            MaceLikeModelConfig(
                hidden_lmax=2,
                density_lmax=5,
                readout_edge_lmax=2,
            )
        )
