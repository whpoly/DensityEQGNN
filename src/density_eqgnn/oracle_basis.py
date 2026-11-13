"""Fixed basis candidates for numerical density projection, not trained models.

Radial SVD whitening is an invertible change of coordinates: all radial
directions are retained. Features are constructed in float64 before an optional
float32 storage conversion. Periodic images use the production neighbour rule.
"""

import numpy as np
import torch
from e3nn import o3

from .periodic import cutoff_edges
from .radial_basis import RadialSpec, radial_transform



def edge_features(displacements: torch.Tensor, spec: RadialSpec):
    """Degree-major whitened basis evaluated on probe-minus-center vectors."""
    r = torch.linalg.vector_norm(displacements, dim=-1)
    scales = torch.as_tensor(
        np.geomspace(spec.minimum, spec.cutoff, spec.num_radial),
        device=r.device, dtype=r.dtype,
    )
    radial = torch.exp(-0.5 * (r[:, None] / scales).square())
    radial *= (0.5 * (1 + torch.cos(torch.pi * r / spec.cutoff)))[:, None]
    radial *= (r < spec.cutoff)[:, None]
    harmonics = o3.spherical_harmonics(
        list(range(spec.lmax + 1)), displacements,
        normalize=True, normalization="integral",
    )
    blocks = []
    for degree in range(spec.lmax + 1):
        raw = radial * (r[:, None] / scales) ** degree if spec.regular else radial
        transform = torch.as_tensor(radial_transform(spec, degree)[0], device=r.device, dtype=r.dtype)
        whitened = raw @ transform
        angular = harmonics[:, degree**2:(degree+1)**2]
        blocks.append((whitened[:, :, None] * angular[:, None, :]).flatten(1))
    return torch.cat(blocks, dim=1)


def raw_to_whitened(coefficients: torch.Tensor, spec: RadialSpec):
    """Map production radial-major coefficients to degree-major coordinates."""
    raw = coefficients.reshape(-1, spec.num_radial, (spec.lmax + 1)**2)
    result = []
    for degree in range(spec.lmax + 1):
        inverse = torch.as_tensor(radial_transform(spec, degree)[1], device=raw.device, dtype=raw.dtype)
        result.append(torch.einsum(
            "ij,ajm->aim", inverse, raw[:, :, degree**2:(degree+1)**2]
        ).flatten(1))
    return torch.cat(result, dim=1).flatten()


def periodic_bond_centers(positions, numbers, cell, cutoff=2.6):
    """Unique cation--O coordination midpoints modulo the cell.

    This geometric definition includes Li--O and transition-metal--O links;
    it is not a quantum-chemical bond-order assignment. Each cation-to-O edge
    is included once; reverse edges are excluded by the species predicate.
    """
    mask = torch.ones(1, len(positions), dtype=torch.bool, device=positions.device)
    edges = cutoff_edges(
        positions[None], positions[None], mask, mask, cell[None], cutoff, True,
        exclude_self=True,
    )
    selected = (numbers[edges.center] != 8) & (numbers[edges.neighbor] == 8)
    midpoint = positions[edges.center[selected]] + edges.displacement[selected] / 2
    frac = torch.remainder(midpoint @ torch.linalg.inv(cell), 1.0)
    # Quantization is only a deduplication key, never a change to coordinates.
    keys = torch.remainder(torch.round(frac * 1e8).long(), 100000000).cpu().numpy()
    _, indices = np.unique(keys, axis=0, return_index=True)
    return frac[torch.as_tensor(np.sort(indices), device=positions.device)] @ cell


class OracleBasis:
    def __init__(self, blocks, cell):
        self.blocks = blocks  # [(positions, RadialSpec), ...]
        self.cell = cell
        self.size = sum(len(centers) * spec.size for centers, spec in blocks)

    @torch.no_grad()
    def design(self, probes, dtype=torch.float32):
        """Build a small dense query chunk, summing duplicate periodic images."""
        result = torch.zeros(len(probes), self.size, device=probes.device, dtype=dtype)
        offset = 0
        for centers, spec in self.blocks:
            width = len(centers) * spec.size
            view = result[:, offset:offset+width].view(len(probes), len(centers), spec.size)
            edges = cutoff_edges(
                probes[None], centers[None],
                torch.ones(1, len(probes), dtype=torch.bool, device=probes.device),
                torch.ones(1, len(centers), dtype=torch.bool, device=probes.device),
                self.cell[None], spec.cutoff, True, chunk_size=max(len(probes), 1),
            )
            for start in range(0, len(edges.center), 16384):
                stop = start + 16384
                values = edge_features(-edges.displacement[start:stop], spec).to(dtype)
                view.index_put_(
                    (edges.center[start:stop], edges.neighbor[start:stop]), values,
                    accumulate=True,
                )
            offset += width
        return result


class _TransposeDesign:
    def __init__(self, design):
        self.design = design

    def __matmul__(self, vector):
        return torch.cat([features.T @ vector[rows] for rows, features in self.design.parts])


class _PreciseMatvec(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vector, design):
        ctx.design = design
        result = torch.zeros(design.shape[0],device=design.device,dtype=torch.float64)
        offset = 0
        for rows,features in design.parts:
            width = features.shape[1]
            result.index_add_(0,rows,features.double()@vector[offset:offset+width])
            offset += width
        return result

    @staticmethod
    def backward(ctx, gradient):
        # Recreate one double block at a time; retaining every converted block
        # for autograd would double GPU storage for the entire grid.
        return ctx.design.precise_transpose(gradient), None


class FullGridDesign:
    """All grid rows, storing only each center's compact support on the GPU.

    Unlike a global sparse matrix, columns within one center are dense. This
    avoids storing a column index for every spherical/radial basis value.
    Multiple images of a center are summed before storing its row block.
    """
    def __init__(self, basis, probes):
        self.shape = (len(probes), basis.size)
        self.device = probes.device
        self.dtype = torch.float32
        self.parts = []
        # First block is the atom-centered block in these oracle experiments.
        self.nearest_atom = torch.full((len(probes),), float("inf"), device=self.device, dtype=torch.float64)
        with torch.no_grad():
            for block_index, (centers, spec) in enumerate(basis.blocks):
                for center in centers:
                    edges = cutoff_edges(
                        probes[None], center[None, None],
                        torch.ones(1, len(probes), dtype=torch.bool, device=probes.device),
                        torch.ones(1, 1, dtype=torch.bool, device=probes.device),
                        basis.cell[None], spec.cutoff, True, chunk_size=len(probes),
                    )
                    rows, inverse = torch.unique(edges.center, sorted=True, return_inverse=True)
                    if block_index == 0:
                        self.nearest_atom.scatter_reduce_(
                            0, edges.center, edges.displacement.norm(dim=-1), reduce="amin",
                        )
                    features = torch.zeros(len(rows), spec.size, device=self.device, dtype=self.dtype)
                    for start in range(0, len(edges.center), 16384):
                        stop = start + 16384
                        values = edge_features(-edges.displacement[start:stop], spec).float()
                        features.index_add_(0, inverse[start:stop], values)
                    self.parts.append((rows, features))

    @property
    def T(self):
        return _TransposeDesign(self)

    def __matmul__(self, vector):
        result = torch.zeros(self.shape[0], device=self.device, dtype=self.dtype)
        offset = 0
        for rows, features in self.parts:
            width = features.shape[1]
            result.index_add_(0, rows, features @ vector[offset:offset+width])
            offset += width
        return result

    def normalize_columns(self):
        scales = []
        for _, features in self.parts:
            scale = torch.linalg.vector_norm(features, dim=0).clamp_min(1e-12)
            features /= scale[None]
            scales.append(scale)
        return torch.cat(scales)

    def precise_matvec(self,vector):
        return _PreciseMatvec.apply(vector,self)

    def precise_transpose(self,vector):
        return torch.cat([features.double().T@vector[rows] for rows,features in self.parts])

    @torch.no_grad()
    def gram_cholesky(self, ridge=1e-9):
        """Global preconditioner from the exact stored full-grid operator.

        Double precision accumulation and factorization; the ridge stabilizes
        the invertible coordinate transform, not the final L1 objective.
        """
        gram = torch.zeros(self.shape[1],self.shape[1],device=self.device,dtype=torch.float64)
        offsets = np.cumsum([0]+[f.shape[1] for _,f in self.parts])
        for i,(rows_i,features_i) in enumerate(self.parts):
            for j in range(i,len(self.parts)):
                rows_j,features_j = self.parts[j]
                where = torch.searchsorted(rows_i,rows_j)
                valid = (where<len(rows_i)) & (rows_i[where.clamp_max(len(rows_i)-1)]==rows_j)
                ii = where[valid]
                jj = torch.nonzero(valid,as_tuple=True)[0]
                block = torch.zeros(features_i.shape[1],features_j.shape[1],device=self.device,dtype=torch.float64)
                for start in range(0,len(ii),16384):
                    stop = start+16384
                    block += features_i[ii[start:stop]].double().T @ features_j[jj[start:stop]].double()
                gram[offsets[i]:offsets[i+1],offsets[j]:offsets[j+1]] = block
                if i != j:
                    gram[offsets[j]:offsets[j+1],offsets[i]:offsets[i+1]] = block.T
            if i % 20 == 0:
                print(f"Gram blocks {i+1}/{len(self.parts)}",flush=True)
        gram.diagonal().add_(ridge)
        info = torch.empty((),device=self.device,dtype=torch.int32)
        torch.linalg.cholesky_ex(gram,out=(gram,info))
        if int(info):
            raise RuntimeError(f"Full-grid Gram Cholesky failed at minor {int(info)}")
        return gram
