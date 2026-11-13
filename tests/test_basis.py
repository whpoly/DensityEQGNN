import numpy as np
import torch

from density_eqgnn.basis import (
    DensityBasisConfig,
    fit_density_coefficients,
    reconstruct_density_numpy,
    reconstruct_density_torch,
)
from density_eqgnn.data import CoefficientDataset


def test_global_fit_recovers_density_and_charge():
    config = DensityBasisConfig(num_radial=3, cutoff=3.0, lmax=1, pbc=False)
    positions = np.array([[2.0, 2.0, 2.0], [3.2, 2.0, 2.0]])
    axes = [(np.arange(9) + 0.5) / 9 * 6.0] * 3
    probes = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    coefficients = np.array(
        [
            [2.0, 0.8, 0.2, 0.10, 0.0, 0.0, 0.04, 0.0, 0.0, 0.01, 0.0, 0.0],
            [1.0, 0.4, 0.1, -0.05, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    density = reconstruct_density_numpy(coefficients, positions, probes, None, config)
    fitted, metrics = fit_density_coefficients(
        density,
        positions,
        probes,
        None,
        config,
        ridge=1e-12,
        conserve_charge=True,
        probe_volume=6.0**3 / len(probes),
    )
    reconstructed = reconstruct_density_numpy(fitted, positions, probes, None, config)

    assert np.mean(np.abs(reconstructed - density)) / np.mean(np.abs(density)) < 1e-6
    assert abs(metrics["charge_error"]) < 1e-9


def test_torch_reconstruction_matches_numpy_with_pbc():
    config = DensityBasisConfig(num_radial=2, cutoff=2.5, lmax=1, pbc=True)
    cell = np.diag([5.0, 5.0, 5.0])
    positions = np.array([[0.2, 0.2, 0.2], [4.7, 0.2, 0.2]], dtype=np.float32)
    probes = np.array([[4.9, 0.2, 0.2], [2.5, 2.5, 2.5]], dtype=np.float32)
    coefficients = np.arange(16, dtype=np.float32).reshape(2, 8) / 20.0
    expected = reconstruct_density_numpy(coefficients, positions, probes, cell, config)
    actual = reconstruct_density_torch(
        torch.from_numpy(coefficients)[None],
        torch.from_numpy(positions)[None],
        torch.from_numpy(probes)[None],
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2, dtype=torch.bool),
        torch.from_numpy(cell.astype(np.float32))[None],
        config,
    )
    assert np.allclose(actual.numpy()[0], expected, atol=1e-6)


def test_rotation_augmentation_preserves_reconstructed_density(tmp_path):
    config = DensityBasisConfig(num_radial=2, cutoff=2.5, lmax=2, pbc=False)
    positions = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]], dtype=np.float32)
    probes = np.array([[1.2, 1.1, 1.0], [1.8, 0.9, 1.1]], dtype=np.float32)
    coefficients = np.arange(52, dtype=np.float32).reshape(2, 26) / 50.0
    density = reconstruct_density_numpy(coefficients, positions, probes, None, config)
    np.savez(
        tmp_path / "sample.npz",
        atomic_numbers=np.array([1, 8]),
        positions=positions,
        cell=np.eye(3, dtype=np.float32) * 5.0,
        coefficients=coefficients,
        probe_positions=probes,
        density=density,
    )
    sample = CoefficientDataset(
        tmp_path, augment_rotations=True, num_radial=2, seed=9
    )[0]
    augmented = reconstruct_density_numpy(
        sample["coefficients"],
        sample["positions"],
        sample["probe_positions"],
        sample["cell"],
        config,
    )
    assert np.allclose(augmented, density, atol=1e-6)
