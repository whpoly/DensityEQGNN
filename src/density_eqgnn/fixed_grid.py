"""Geometry-only cache for density learning on fixed structures and full grids.

No density features, labels, or fitted coefficients enter this cache. Coordinates
are detached deliberately; use reconstruct_compact_density when learning atomic
positions or requiring coordinate derivatives.
"""
import torch

from .compact_basis import reconstruct_compact_from_edges
from .periodic import cutoff_edges


class FixedGridDensity:
    @torch.no_grad()
    def __init__(self, positions, probes, cell, basis):
        self.basis = basis
        self.num_probes = len(probes)
        self.device = probes.device
        self.num_atoms = len(positions)
        edges = cutoff_edges(
            probes.detach().double()[None],positions.detach().double()[None],
            torch.ones(1,len(probes),device=probes.device,dtype=torch.bool),
            torch.ones(1,len(positions),device=probes.device,dtype=torch.bool),
            cell.detach().double()[None] if cell is not None else None,
            basis.cutoff,basis.pbc,chunk_size=16384,
        )
        order = torch.argsort(edges.center,stable=True)
        counts = torch.bincount(edges.center,minlength=len(probes))
        self.nearest_atom = torch.full((len(probes),),float("inf"),device=probes.device,dtype=torch.float64)
        self.nearest_atom.scatter_reduce_(0,edges.center,edges.displacement.norm(dim=-1),reduce="amin")
        self.offsets = torch.cat([counts.new_zeros(1),counts.cumsum(0)])
        self.atoms = edges.neighbor[order].to(torch.int32)
        self.displacement = -edges.displacement[order]

    def __call__(self, coefficients, indices):
        counts = self.offsets[indices+1]-self.offsets[indices]
        size = int(counts.sum())
        target = torch.repeat_interleave(torch.arange(len(indices),device=self.device),
                                        counts,output_size=size)
        starts = counts.cumsum(0)-counts
        edge = self.offsets[indices][target]+torch.arange(size,device=self.device)-starts[target]
        return reconstruct_compact_from_edges(
            coefficients,self.displacement[edge],torch.zeros(size,device=self.device,dtype=torch.long),
            self.atoms[edge].long(),target,len(indices),self.basis,
        )
