"""Atom-centred density basis, fitting, and reconstruction.

The minimal representation implemented here is

    rho(r) = sum_i [sum_n c0[i,n] R_n(r_i)
                    + sum_n c1[i,n] . rhat_i R_n(r_i)
                    + sum_n rhat_i.T C2[i,n] rhat_i R_n(r_i)]

with ``r_i = r - R_i``.  The scalar coefficients are invariant and the
three-component coefficients transform as vectors.  This is the l <= 1 real
spherical-harmonic basis used by the first validation model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .periodic import image_shifts_numpy


@dataclass(frozen=True)
class DensityBasisConfig:
    """Configuration of the compact atom-centred density basis."""

    num_radial: int = 4
    cutoff: float = 4.0
    lmax: int = 2
    radial_width: float | None = None
    pbc: bool = False

    def __post_init__(self) -> None:
        if self.num_radial < 1:
            raise ValueError("num_radial must be positive")
        if self.cutoff <= 0:
            raise ValueError("cutoff must be positive")
        if self.lmax not in {0, 1, 2}:
            raise ValueError("The compact prototype supports lmax=0, 1, or 2")

    @property
    def coefficient_size(self) -> int:
        return self.num_radial * (
            1 + (3 if self.lmax >= 1 else 0) + (9 if self.lmax >= 2 else 0)
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _radial_scales(config: DensityBasisConfig) -> np.ndarray:
    """Return multiscale atom-centred Gaussian widths.

    Charge densities are steepest near nuclei, so logarithmic widths are much
    more efficient here than equally spaced Gaussian shell centres.
    """

    if config.radial_width is not None:
        return np.full(config.num_radial, config.radial_width, dtype=np.float64)
    minimum = max(config.cutoff / 32.0, 0.08)
    return np.geomspace(minimum, config.cutoff, config.num_radial, dtype=np.float64)


def _minimum_image_numpy(displacements: np.ndarray, cell: np.ndarray) -> np.ndarray:
    inv_cell = np.linalg.inv(np.asarray(cell, dtype=np.float64))
    fractional = displacements @ inv_cell
    fractional -= np.round(fractional)
    wrapped = fractional @ cell
    if not wrapped.size:
        return wrapped
    best = wrapped.copy()
    best_squared = np.sum(best**2, axis=-1)
    # Used only for distances/partition weights. The density expansion below
    # instead sums ALL images within support.
    cutoff = float(np.sqrt(best_squared.max()))
    for shift in image_shifts_numpy(cell, cutoff):
        candidate = wrapped + shift @ cell
        squared = np.sum(candidate**2, axis=-1)
        improved = squared < best_squared
        best[improved] = candidate[improved]
        best_squared = np.minimum(best_squared, squared)
    return best


def atom_probe_displacements_numpy(
    atom_positions: np.ndarray,
    probe_positions: np.ndarray,
    cell: np.ndarray | None,
    pbc: bool,
) -> np.ndarray:
    """Return ``probe - atom`` displacements, using true nearest images for PBC.

    This is a distance/partition helper, not the periodic density summation.
    """

    atom_positions = np.asarray(atom_positions, dtype=np.float64)
    probe_positions = np.asarray(probe_positions, dtype=np.float64)
    displacements = probe_positions[:, None, :] - atom_positions[None, :, :]
    if pbc:
        if cell is None:
            raise ValueError("A cell is required when pbc=True")
        displacements = _minimum_image_numpy(displacements, np.asarray(cell, dtype=np.float64))
    return displacements


def radial_basis_numpy(distances: np.ndarray, config: DensityBasisConfig) -> np.ndarray:
    """Evaluate smooth Gaussian radial functions."""

    scales = _radial_scales(config)
    gaussian = np.exp(-0.5 * (distances[..., None] / scales) ** 2)
    x = np.clip(distances / config.cutoff, 0.0, 1.0)
    envelope = 0.5 * (np.cos(np.pi * x) + 1.0)
    envelope = np.where(distances < config.cutoff, envelope, 0.0)
    return gaussian * envelope[..., None]


def design_matrix_numpy(
    atom_positions: np.ndarray,
    probe_positions: np.ndarray,
    cell: np.ndarray | None,
    config: DensityBasisConfig,
) -> np.ndarray:
    """Build the dense design matrix for a chunk of density probes."""

    if config.pbc:
        if cell is None:
            raise ValueError("A cell is required when pbc=True")
        cell = np.asarray(cell, dtype=np.float64)
        delta = np.asarray(probe_positions)[:, None, :] - np.asarray(atom_positions)[None]
        frac = delta @ np.linalg.inv(cell)
        wrapped = (frac - np.round(frac)) @ cell
        features = np.zeros((*delta.shape[:2], config.coefficient_size), dtype=np.float64)
        for shift in image_shifts_numpy(cell, config.cutoff):
            disp = wrapped + shift @ cell
            row, atom = np.nonzero(np.sum(disp**2, axis=-1) < config.cutoff**2)
            features[row, atom] += _basis_features_numpy(disp[row, atom], config)
        return features.reshape(len(probe_positions), -1)

    disp = atom_probe_displacements_numpy(atom_positions, probe_positions, cell, False)
    return _basis_features_numpy(disp, config).reshape(len(probe_positions), -1)


def _basis_features_numpy(disp: np.ndarray, config: DensityBasisConfig) -> np.ndarray:
    distances = np.linalg.norm(disp, axis=-1)
    directions = disp / np.maximum(distances[..., None], 1e-12)
    radial = radial_basis_numpy(distances, config)

    per_atom = [radial]
    if config.lmax >= 1:
        vector_basis = radial[..., :, None] * directions[..., None, :]
        per_atom.append(vector_basis.reshape(*radial.shape[:-1], 3 * config.num_radial))
    if config.lmax >= 2:
        identity = np.eye(3, dtype=np.float64) / 3.0
        quadrupole = (
            directions[..., :, None] * directions[..., None, :] - identity
        )
        tensor_basis = radial[..., :, None, None] * quadrupole[..., None, :, :]
        per_atom.append(tensor_basis.reshape(*radial.shape[:-1], 9 * config.num_radial))
    return np.concatenate(per_atom, axis=-1)


def reconstruct_density_numpy(
    coefficients: np.ndarray,
    atom_positions: np.ndarray,
    probe_positions: np.ndarray,
    cell: np.ndarray | None,
    config: DensityBasisConfig,
    chunk_size: int = 8192,
) -> np.ndarray:
    """Reconstruct density values from per-atom coefficients."""

    coefficients = np.asarray(coefficients, dtype=np.float64)
    expected = (len(atom_positions), config.coefficient_size)
    if coefficients.shape != expected:
        raise ValueError(f"coefficients must have shape {expected}, got {coefficients.shape}")
    flat = coefficients.reshape(-1)
    output = np.empty(len(probe_positions), dtype=np.float64)
    for start in range(0, len(probe_positions), chunk_size):
        stop = min(start + chunk_size, len(probe_positions))
        phi = design_matrix_numpy(atom_positions, probe_positions[start:stop], cell, config)
        output[start:stop] = phi @ flat
    return output


def fit_density_coefficients(
    density: np.ndarray,
    atom_positions: np.ndarray,
    probe_positions: np.ndarray,
    cell: np.ndarray | None,
    config: DensityBasisConfig,
    ridge: float = 1e-8,
    conserve_charge: bool = True,
    probe_volume: float | np.ndarray = 1.0,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit all atoms' coefficients together using chunked normal equations.

    The optional linear constraint preserves the integral of the input density
    on the supplied grid.  Ridge regularisation selects a stable canonical
    solution when overlapping atom-centred functions are nearly dependent.
    """

    density = np.asarray(density, dtype=np.float64).reshape(-1)
    probe_positions = np.asarray(probe_positions, dtype=np.float64)
    if len(density) != len(probe_positions):
        raise ValueError("density and probe_positions must have the same length")

    n_coeff = len(atom_positions) * config.coefficient_size
    gram = np.zeros((n_coeff, n_coeff), dtype=np.float64)
    rhs = np.zeros(n_coeff, dtype=np.float64)
    integral_basis = np.zeros(n_coeff, dtype=np.float64)
    target_charge = 0.0
    volumes = np.asarray(probe_volume, dtype=np.float64)

    for start in range(0, len(density), chunk_size):
        stop = min(start + chunk_size, len(density))
        phi = design_matrix_numpy(atom_positions, probe_positions[start:stop], cell, config)
        rho = density[start:stop]
        weight = (
            np.full(stop - start, float(volumes), dtype=np.float64)
            if volumes.ndim == 0
            else volumes[start:stop]
        )
        weighted_phi = phi * weight[:, None]
        gram += phi.T @ weighted_phi
        rhs += phi.T @ (weight * rho)
        integral_basis += weighted_phi.sum(axis=0)
        target_charge += float(np.dot(weight, rho))

    scale = float(np.trace(gram) / max(n_coeff, 1))
    regularized = gram + (ridge * max(scale, 1.0)) * np.eye(n_coeff)

    if conserve_charge:
        kkt = np.zeros((n_coeff + 1, n_coeff + 1), dtype=np.float64)
        kkt[:n_coeff, :n_coeff] = regularized
        kkt[:n_coeff, n_coeff] = integral_basis
        kkt[n_coeff, :n_coeff] = integral_basis
        target = np.concatenate([rhs, [target_charge]])
        try:
            solution = np.linalg.solve(kkt, target)[:n_coeff]
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(kkt, target, rcond=1e-10)[0][:n_coeff]
    else:
        try:
            solution = np.linalg.solve(regularized, rhs)
        except np.linalg.LinAlgError:
            solution = np.linalg.lstsq(regularized, rhs, rcond=1e-10)[0]

    coefficients = solution.reshape(len(atom_positions), config.coefficient_size)
    reconstructed = reconstruct_density_numpy(
        coefficients, atom_positions, probe_positions, cell, config, chunk_size
    )
    residual = reconstructed - density
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    scale_density = float(np.mean(np.abs(density)))
    fitted_charge = float(
        np.sum(reconstructed * (float(volumes) if volumes.ndim == 0 else volumes))
    )
    diagnostics = {
        "mae": mae,
        "rmse": rmse,
        "nmae": mae / max(scale_density, 1e-12),
        "target_charge": target_charge,
        "fitted_charge": fitted_charge,
        "charge_error": fitted_charge - target_charge,
    }
    return coefficients.astype(np.float32), diagnostics


def fit_density_coefficients_stockholder(
    density: np.ndarray,
    atom_positions: np.ndarray,
    probe_positions: np.ndarray,
    cell: np.ndarray | None,
    config: DensityBasisConfig,
    ridge: float = 1e-6,
    partition_width: float = 1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit one compact vector per atom after a smooth stockholder partition.

    The positive weights

    ``w_i(r) = exp(-|r-R_i|²/(2 sigma²)) / sum_j exp(...)``

    sum to one at every probe.  Thus ``rho_i = w_i rho`` is a deterministic,
    smooth atomic decomposition without Bader-style hard boundaries.  Each
    atomic target is then fitted independently in the same equivariant basis.
    This changes the expensive global solve into linearly many small solves.
    """

    density = np.asarray(density, dtype=np.float64).reshape(-1)
    atom_positions = np.asarray(atom_positions, dtype=np.float64)
    probe_positions = np.asarray(probe_positions, dtype=np.float64)
    displacements = atom_probe_displacements_numpy(
        atom_positions, probe_positions, cell, config.pbc
    )
    distances = np.linalg.norm(displacements, axis=-1)
    logits = -0.5 * (distances / partition_width) ** 2
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True).clip(min=1e-300)

    coefficients = np.zeros(
        (len(atom_positions), config.coefficient_size), dtype=np.float64
    )
    for atom_index in range(len(atom_positions)):
        local = distances[:, atom_index] < config.cutoff
        if not np.any(local):
            continue
        phi = design_matrix_numpy(
            atom_positions[atom_index : atom_index + 1],
            probe_positions[local],
            cell,
            config,
        )
        target = density[local] * weights[local, atom_index]
        gram = phi.T @ phi
        scale = float(np.trace(gram) / max(config.coefficient_size, 1))
        regularized = gram + ridge * max(scale, 1.0) * np.eye(
            config.coefficient_size
        )
        rhs = phi.T @ target
        try:
            coefficients[atom_index] = np.linalg.solve(regularized, rhs)
        except np.linalg.LinAlgError:
            coefficients[atom_index] = np.linalg.lstsq(
                regularized, rhs, rcond=1e-10
            )[0]

    reconstructed = reconstruct_density_numpy(
        coefficients, atom_positions, probe_positions, cell, config
    )
    residual = reconstructed - density
    reference = max(float(np.mean(np.abs(density))), 1e-12)
    diagnostics = {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "nmae": float(np.mean(np.abs(residual))) / reference,
        "target_charge": float(density.sum()),
        "fitted_charge": float(reconstructed.sum()),
        "charge_error": float((reconstructed - density).sum()),
    }
    return coefficients.astype(np.float32), diagnostics


def fit_density_coefficients_blockwise(
    density: np.ndarray,
    atom_positions: np.ndarray,
    probe_positions: np.ndarray,
    cell: np.ndarray | None,
    config: DensityBasisConfig,
    ridge: float = 1e-6,
    sweeps: int = 40,
) -> tuple[np.ndarray, dict[str, float]]:
    """Solve the global density fit with atom-block coordinate descent.

    This minimizes the same total-density least-squares objective as
    :func:`fit_density_coefficients`, but each update solves only one atom's
    coefficient block.  It avoids constructing or factoring the full
    multi-thousand-dimensional Gram matrix for larger QM9 molecules.
    """

    density = np.asarray(density, dtype=np.float64).reshape(-1)
    atom_positions = np.asarray(atom_positions, dtype=np.float64)
    probe_positions = np.asarray(probe_positions, dtype=np.float64)
    distances = np.linalg.norm(
        atom_probe_displacements_numpy(
            atom_positions, probe_positions, cell, config.pbc
        ),
        axis=-1,
    )
    coefficients = np.zeros(
        (len(atom_positions), config.coefficient_size), dtype=np.float64
    )
    prediction = np.zeros_like(density)
    identity = np.eye(config.coefficient_size, dtype=np.float64)

    for sweep in range(sweeps):
        order = (
            range(len(atom_positions))
            if sweep % 2 == 0
            else range(len(atom_positions) - 1, -1, -1)
        )
        for atom_index in order:
            local = distances[:, atom_index] < config.cutoff
            if not np.any(local):
                continue
            phi = design_matrix_numpy(
                atom_positions[atom_index : atom_index + 1],
                probe_positions[local],
                cell,
                config,
            )
            old = phi @ coefficients[atom_index]
            target = density[local] - prediction[local] + old
            gram = phi.T @ phi
            scale = float(np.trace(gram) / max(config.coefficient_size, 1))
            regularized = gram + ridge * max(scale, 1.0) * identity
            rhs = phi.T @ target
            try:
                updated = np.linalg.solve(regularized, rhs)
            except np.linalg.LinAlgError:
                updated = np.linalg.lstsq(regularized, rhs, rcond=1e-10)[0]
            prediction[local] += phi @ updated - old
            coefficients[atom_index] = updated

    residual = prediction - density
    reference = max(float(np.mean(np.abs(density))), 1e-12)
    diagnostics = {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "nmae": float(np.mean(np.abs(residual))) / reference,
        "target_charge": float(density.sum()),
        "fitted_charge": float(prediction.sum()),
        "charge_error": float(residual.sum()),
    }
    return coefficients.astype(np.float32), diagnostics


def reconstruct_density_torch(
    coefficients: "torch.Tensor",
    atom_positions: "torch.Tensor",
    probe_positions: "torch.Tensor",
    atom_mask: "torch.Tensor",
    probe_mask: "torch.Tensor",
    cell: "torch.Tensor | None",
    config: DensityBasisConfig,
) -> "torch.Tensor":
    """Differentiable batched density reconstruction."""

    import torch

    if config.pbc:
        from .periodic import cutoff_edges

        edges = cutoff_edges(
            probe_positions, atom_positions, probe_mask, atom_mask,
            cell, config.cutoff, True,
        )
        density = coefficients.new_zeros(probe_mask.numel())
        density = density + coefficients.sum() * 0 + edges.displacement.sum() * 0
        scales = torch.as_tensor(
            _radial_scales(config), dtype=coefficients.dtype, device=coefficients.device
        )
        for start in range(0, len(edges.displacement), 16384):
            stop = start + 16384
            disp = -edges.displacement[start:stop]
            distances = torch.linalg.vector_norm(disp, dim=-1)
            directions = disp / distances[:, None].clamp_min(1e-12)
            radial = torch.exp(-0.5 * (distances[:, None] / scales).square())
            radial = radial * (0.5 * (torch.cos(torch.pi * distances / config.cutoff) + 1))[:, None]
            b, a = edges.batch[start:stop], edges.neighbor[start:stop]
            coeff = coefficients[b, a]
            angular = coeff[:, :config.num_radial]
            if config.lmax >= 1:
                vector = coeff[:, config.num_radial:4 * config.num_radial].reshape(-1, config.num_radial, 3)
                angular = angular + torch.einsum("end,ed->en", vector, directions)
            if config.lmax >= 2:
                tensor = coeff[:, 4 * config.num_radial:].reshape(-1, config.num_radial, 3, 3)
                angular = angular + torch.einsum("enij,ei,ej->en", tensor, directions, directions)
                angular = angular - tensor.diagonal(dim1=-2, dim2=-1).sum(-1) / 3
            values = (radial * angular).sum(-1)
            target = b * probe_positions.shape[1] + edges.center[start:stop]
            density = density.index_add(0, target, values)
        return density.reshape_as(probe_mask)

    disp = probe_positions[:, None, :, :] - atom_positions[:, :, None, :]

    distances = torch.linalg.vector_norm(disp, dim=-1)
    directions = disp / distances.unsqueeze(-1).clamp_min(1e-12)
    scales = torch.as_tensor(
        _radial_scales(config), dtype=distances.dtype, device=distances.device
    )
    radial = torch.exp(-0.5 * (distances[..., None] / scales).square())
    x = (distances / config.cutoff).clamp(0.0, 1.0)
    envelope = 0.5 * (torch.cos(torch.pi * x) + 1.0)
    envelope = envelope * (distances < config.cutoff).to(envelope.dtype)
    radial = radial * envelope[..., None]

    scalar = coefficients[..., : config.num_radial]
    valid_atoms = atom_mask[:, :, None].to(coefficients.dtype)
    scalar_contrib = torch.einsum("ban,bapn->bap", scalar, radial)
    density = (scalar_contrib * valid_atoms).sum(dim=1)
    if config.lmax >= 1:
        vector = coefficients[..., config.num_radial :].reshape(
            *coefficients.shape[:-1],
            -1,
        )
        vector = vector[..., : 3 * config.num_radial].reshape(
            *coefficients.shape[:-1], config.num_radial, 3
        )
        angular = torch.einsum("band,bapd->bapn", vector, directions)
        vector_contrib = (angular * radial).sum(dim=-1)
        density = density + (vector_contrib * valid_atoms).sum(dim=1)
    if config.lmax >= 2:
        tensor_start = config.num_radial * 4
        tensor = coefficients[..., tensor_start:].reshape(
            *coefficients.shape[:-1], config.num_radial, 3, 3
        )
        angular = torch.einsum(
            "banij,bapi,bapj->bapn", tensor, directions, directions
        )
        angular = angular - tensor.diagonal(dim1=-2, dim2=-1).sum(-1)[:, :, None] / 3
        tensor_contrib = (angular * radial).sum(dim=-1)
        density = density + (tensor_contrib * valid_atoms).sum(dim=1)
    return density * probe_mask.to(density.dtype)
