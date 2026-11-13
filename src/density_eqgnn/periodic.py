"""Cutoff neighbours including every lattice image, for row-vector cells."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import warnings

import numpy as np
import torch


PERIODIC_GEOMETRY_VERSION = "all_images_v1"


def check_periodic_checkpoint(checkpoint: dict, *, pbc: bool, resume: bool = False) -> None:
    """Keep pre-fix training histories from silently spanning different operators."""
    if not pbc or checkpoint.get("periodic_geometry_version") == PERIODIC_GEOMETRY_VERSION:
        return
    message = (
        "This checkpoint predates or differs from the all-image PBC implementation. "
        "Its metrics are not metrics of the corrected model. Retrain with a new "
        "training.output_dir (e.g. configs/nmc_full_l2_to_l4_mace_pbc.yaml); "
        "do not resume the old optimizer/history."
    )
    if resume:
        raise ValueError(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def image_shifts_numpy(cell: np.ndarray, cutoff: float) -> np.ndarray:
    """Bound images of a fractionally wrapped displacement, even in skew cells.

    For |d| < cutoff, |(d @ inv(cell))[k]| <= cutoff * |inv(cell)[:, k]|.
    The wrapped fractional displacement is in [-1/2, 1/2], so these bounds
    include every possible integer translation. Wrapping is only an enumeration
    aid; it is deliberately NOT treated as a nearest-image calculation.
    """
    inverse = np.linalg.inv(np.asarray(cell, dtype=np.float64))
    bounds = np.ceil(cutoff * np.linalg.norm(inverse, axis=0) + 0.5).astype(int)
    return np.asarray(list(itertools.product(*(range(-n, n + 1) for n in bounds))))


@dataclass
class CutoffEdges:
    batch: torch.Tensor
    center: torch.Tensor
    neighbor: torch.Tensor
    displacement: torch.Tensor  # neighbor image - center


def cutoff_edges(
    centers: torch.Tensor,
    neighbors: torch.Tensor,
    center_mask: torch.Tensor,
    neighbor_mask: torch.Tensor,
    cell: torch.Tensor | None,
    cutoff: float,
    pbc: bool,
    *,
    exclude_self: bool = False,
    chunk_size: int = 512,
) -> CutoffEdges:
    """Build batched edges without differentiating discrete neighbour selection.

    Returned displacements retain gradients w.r.t. positions and cell. With
    exclude_self, the two point sets must have identical indexing: only the
    zero-translation i=j edge is removed, not the atom's periodic images.
    Masks are applied before distance calculations, including padded positions.
    """
    if cutoff <= 0 or chunk_size < 1:
        raise ValueError("cutoff and chunk_size must be positive")
    if pbc and cell is None:
        raise ValueError("cell is required when pbc=True")
    device = centers.device
    batch_parts, center_parts, neighbor_parts, shift_parts = [], [], [], []
    with torch.no_grad():
        for b in range(centers.shape[0]):
            ci = torch.nonzero(center_mask[b], as_tuple=True)[0]
            ni = torch.nonzero(neighbor_mask[b], as_tuple=True)[0]
            if not ci.numel() or not ni.numel():
                continue
            if pbc:
                inverse = torch.linalg.inv(cell[b].detach())
                shifts = image_shifts_numpy(cell[b].detach().cpu().numpy(), cutoff)
            else:
                shifts = np.zeros((1, 3), dtype=np.int64)
            for start in range(0, len(ci), chunk_size):
                selected = ci[start : start + chunk_size]
                delta = neighbors[b, ni][None] - centers[b, selected][:, None]
                wrap = torch.round(delta @ inverse).to(torch.long) if pbc else None
                wrapped = delta - wrap.to(delta.dtype) @ cell[b] if pbc else delta
                for shift_np in shifts:
                    shift = torch.as_tensor(shift_np, device=device, dtype=torch.long)
                    disp = wrapped + shift.to(delta.dtype) @ cell[b] if pbc else wrapped
                    valid = disp.square().sum(dim=-1) < cutoff**2
                    if exclude_self:
                        same = selected[:, None] == ni[None]
                        zero_image = (wrap == shift).all(dim=-1) if pbc else True
                        valid &= ~(same & zero_image)
                    row, col = torch.nonzero(valid, as_tuple=True)
                    if not row.numel():
                        continue
                    batch_parts.append(torch.full_like(row, b))
                    center_parts.append(selected[row])
                    neighbor_parts.append(ni[col])
                    shift_parts.append(
                        shift[None] - wrap[row, col] if pbc
                        else torch.zeros((len(row), 3), dtype=torch.long, device=device)
                    )
    empty = torch.empty(0, dtype=torch.long, device=device)
    if not batch_parts:
        # Empty indexing also keeps the zero result connected to autograd.
        disp = neighbors[empty, empty] - centers[empty, empty]
        if pbc:
            disp = disp + cell[empty].sum(dim=1)
        return CutoffEdges(empty, empty, empty, disp)
    batch = torch.cat(batch_parts)
    center = torch.cat(center_parts)
    neighbor = torch.cat(neighbor_parts)
    displacement = neighbors[batch, neighbor] - centers[batch, center]
    if pbc:
        shifts = torch.cat(shift_parts).to(displacement.dtype)
        displacement = displacement + torch.einsum("ei,eij->ej", shifts, cell[batch])
    return CutoffEdges(batch, center, neighbor, displacement)


def atom_graph_geometry(
    positions: torch.Tensor,
    atom_mask: torch.Tensor,
    cell: torch.Tensor | None,
    cutoff: float,
    pbc: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Padded per-receiver neighbours, including duplicate source image indices."""
    edges = cutoff_edges(
        positions, positions, atom_mask, atom_mask, cell, cutoff, pbc,
        exclude_self=True,
    )
    batch, atoms = atom_mask.shape
    receivers = edges.batch * atoms + edges.center
    order = torch.argsort(receivers, stable=True)
    counts = torch.bincount(receivers, minlength=batch * atoms)
    width = max(int(counts.max()), 1) if counts.numel() else 1
    starts = counts.cumsum(0) - counts
    slots = torch.arange(len(order), device=positions.device) - starts[receivers[order]]
    b, a = edges.batch[order], edges.center[order]
    displacement = positions.new_zeros(batch, atoms, width, 3)
    displacement[b, a, slots] = edges.displacement[order]
    source = torch.zeros((batch, atoms, width), dtype=torch.long, device=positions.device)
    source[b, a, slots] = edges.neighbor[order]
    mask = torch.zeros_like(source, dtype=torch.bool)
    mask[b, a, slots] = True
    distances = torch.linalg.vector_norm(displacement, dim=-1)
    directions = displacement / distances[..., None].clamp_min(1e-12)
    return distances, directions, mask, source


def gather_neighbors(features: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(features.shape[0], device=features.device)[:, None, None]
    return features[batch, source]
