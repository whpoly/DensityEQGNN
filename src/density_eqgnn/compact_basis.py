"""Compact real-spherical-harmonic density reconstruction."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from e3nn import o3

from .periodic import cutoff_edges
from .radial_basis import RadialSpec, radial_transform, radial_values


@dataclass(frozen=True)
class CompactBasisConfig:
    num_radial: int = 12
    lmax: int = 3
    cutoff: float = 5.0
    pbc: bool = False
    regular_radial: bool = False
    radial_whitening: bool = False
    radial_minimum: float | None = None

    @property
    def angular_channels(self) -> int:
        return (self.lmax + 1) ** 2

    @property
    def coefficient_size(self) -> int:
        return self.num_radial * self.angular_channels

    @property
    def radial_spec(self) -> RadialSpec:
        return RadialSpec(
            num_radial=self.num_radial, lmax=self.lmax, cutoff=self.cutoff,
            minimum=self.radial_minimum if self.radial_minimum is not None
            else max(self.cutoff / 32.0, 0.08), regular=self.regular_radial,
        )


def check_basis_checkpoint(checkpoint: dict, basis: CompactBasisConfig) -> None:
    """Same output shape does not imply the same coefficient coordinates."""
    saved = CompactBasisConfig(**checkpoint["basis_config"])
    if saved != basis:
        raise ValueError("Checkpoint basis/whitening differs; use a new training.output_dir")
    if basis.radial_whitening:
        expected = basis_checkpoint_metadata(basis)["radial_whitening_matrices"]
        actual = checkpoint.get("radial_whitening_matrices")
        if actual is None or actual.shape != expected.shape or not torch.allclose(
            actual.detach().cpu().double(), expected, rtol=1e-10, atol=1e-10,
        ):
            raise ValueError("Checkpoint radial whitening matrices differ from this runtime")


def basis_checkpoint_metadata(basis: CompactBasisConfig) -> dict:
    result = {"basis_config":asdict(basis)}
    if basis.radial_whitening:
        result["radial_whitening_matrices"] = torch.from_numpy(np.stack([
            radial_transform(basis.radial_spec,l)[0] for l in range(basis.lmax+1)
        ]))
    return result


def radial_scales(
    config: CompactBasisConfig, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    minimum = config.radial_spec.minimum
    return torch.exp(
        torch.linspace(
            math.log(minimum),
            math.log(config.cutoff),
            config.num_radial,
            dtype=dtype,
            device=device,
        )
    )


def radial_to_degree_major(coefficients: torch.Tensor, config: CompactBasisConfig) -> torch.Tensor:
    """Reorder model outputs into the oracle/design-matrix column convention."""
    shaped = coefficients.reshape(*coefficients.shape[:-1],config.num_radial,config.angular_channels)
    return torch.cat([
        shaped[...,l*l:(l+1)*(l+1)].flatten(-2) for l in range(config.lmax+1)
    ],dim=-1)


def reconstruct_compact_density(
    coefficients: torch.Tensor,
    atom_positions: torch.Tensor,
    probe_positions: torch.Tensor,
    atom_mask: torch.Tensor,
    probe_mask: torch.Tensor,
    cell: torch.Tensor | None,
    config: CompactBasisConfig,
) -> torch.Tensor:
    """Reconstruct scalar density from ``n radial × (lmax+1)^2`` coefficients."""

    if config.regular_radial or config.radial_whitening:
        return _reconstruct_transformed_density(
            coefficients, atom_positions, probe_positions, atom_mask,
            probe_mask, cell, config,
        )

    if config.pbc:
        edges = cutoff_edges(
            probe_positions, atom_positions, probe_mask, atom_mask,
            cell, config.cutoff, True,
        )
        # The graph displacement points toward the atom image; density harmonics
        # use the opposite direction, from that image to the probe (odd l matters).
        displacement = -edges.displacement
        shaped = coefficients.reshape(
            *coefficients.shape[:2], config.num_radial, config.angular_channels
        )
        num_probes = probe_positions.shape[1]
        density = coefficients.new_zeros(probe_mask.numel())
        density = density + coefficients.sum() * 0 + displacement.sum() * 0
        scales = radial_scales(config, coefficients.dtype, coefficients.device)
        for start in range(0, len(displacement), 16384):
            stop = start + 16384
            disp = displacement[start:stop]
            distances = torch.linalg.vector_norm(disp, dim=-1)
            angular = o3.spherical_harmonics(
                list(range(config.lmax + 1)), disp,
                normalize=True, normalization="integral",
            )
            radial = torch.exp(-0.5 * (distances[:, None] / scales).square())
            radial = radial * (0.5 * (torch.cos(torch.pi * distances / config.cutoff) + 1))[:, None]
            b, a = edges.batch[start:stop], edges.neighbor[start:stop]
            values = torch.einsum("enm,en,em->e", shaped[b, a], radial, angular)
            targets = b * num_probes + edges.center[start:stop]
            density = density.index_add(0, targets, values)
        return density.reshape_as(probe_mask)

    displacement = probe_positions[:, None, :, :] - atom_positions[:, :, None, :]
    distances = torch.linalg.vector_norm(displacement, dim=-1)
    directions = displacement / distances[..., None].clamp_min(1e-12)
    angular = o3.spherical_harmonics(
        list(range(config.lmax + 1)),
        directions,
        normalize=True,
        normalization="integral",
    )
    scales = radial_scales(config, distances.dtype, distances.device)
    radial = torch.exp(-0.5 * (distances[..., None] / scales).square())
    x = (distances / config.cutoff).clamp(0.0, 1.0)
    envelope = 0.5 * (torch.cos(torch.pi * x) + 1.0)
    envelope *= (distances < config.cutoff).to(envelope.dtype)
    radial *= envelope[..., None]

    shaped = coefficients.reshape(
        *coefficients.shape[:2], config.num_radial, config.angular_channels
    )
    density = torch.einsum("banm,bapn,bapm->bp", shaped, radial, angular)
    density *= probe_mask.to(density.dtype)
    # Padded atoms have zero coefficients in the model, but this also protects
    # reconstruction when externally supplied arrays are not zero padded.
    if not torch.all(atom_mask):
        masked = shaped * atom_mask[..., None, None].to(shaped.dtype)
        density = torch.einsum("banm,bapn,bapm->bp", masked, radial, angular)
        density *= probe_mask.to(density.dtype)
    return density


def _reconstruct_transformed_density(
    coefficients, atom_positions, probe_positions, atom_mask, probe_mask, cell, config,
):
    # Promote coordinates before subtraction and wrapping, not after computing
    # potentially cancellation-sensitive distances in single precision.
    dtype = torch.float64 if config.radial_whitening else coefficients.dtype
    atoms, probes = atom_positions.to(dtype), probe_positions.to(dtype)
    cells = cell.to(dtype) if cell is not None else None
    edges = cutoff_edges(probes, atoms, probe_mask, atom_mask, cells,
                         config.cutoff, config.pbc, chunk_size=4096)
    displacement = -edges.displacement
    targets = edges.batch * probe_positions.shape[1] + edges.center
    return reconstruct_compact_from_edges(
        coefficients, displacement, edges.batch, edges.neighbor, targets,
        probe_mask.numel(), config,
    ).reshape_as(probe_mask)


def reconstruct_compact_from_edges(coefficients, displacement, batch, atoms, targets,
                                   num_targets, config):
    """Decode already-selected edges; used by the fixed-grid training cache too.

    Displacements point from an atom image to its probe. The caller must supply
    every image inside cutoff, with masks already applied. All transforms and
    reductions use float64 when radial whitening is enabled.
    """
    dtype = torch.float64 if config.radial_whitening else coefficients.dtype
    shaped = coefficients.reshape(*coefficients.shape[:2], config.num_radial,
                                  config.angular_channels)
    density = torch.zeros(num_targets, dtype=dtype, device=coefficients.device)
    density = density + coefficients.sum() * 0 + displacement.sum() * 0
    spec = config.radial_spec
    for start in range(0, len(displacement), 8192):
        stop = start + 8192
        disp = displacement[start:stop].to(dtype)
        radius = torch.linalg.vector_norm(disp, dim=-1)
        harmonics = o3.spherical_harmonics(list(range(config.lmax + 1)), disp,
                                          normalize=not config.regular_radial, normalization="integral")
        b, a = batch[start:stop], atoms[start:stop]
        selected = shaped[b, a].to(dtype)
        values = radius * 0
        for degree in range(config.lmax + 1):
            radial = radial_values(radius, spec, degree, whiten=config.radial_whitening,
                                   solid_harmonics=config.regular_radial)
            angular = harmonics[:, degree**2:(degree + 1)**2]
            block = selected[:, :, degree**2:(degree + 1)**2]
            values = values + torch.einsum("enm,en,em->e", block, radial, angular)
        density = density.index_add(0, targets[start:stop], values)
    return density.to(coefficients.dtype)
