"""Controlled readout ablation on the existing compact-model atom backbone.

The query head is a local equivariant network, not official DeepDFT. Both
variants initialize the original atom model first, so their backbone weights
are identical for a fixed seed. Query points never exchange messages.
"""
from __future__ import annotations

import torch
from torch import nn
from e3nn import o3
from e3nn.nn import NormActivation

from .mace_like_model import MaceLikeCoefficientGNN
from .model import GaussianEdgeBasis, _mlp


def atom_representations(model, numbers, positions, mask, cell):
    """Original forward up to each interaction; no readout or density labels."""
    distances,directions,pair_mask,source_indices=model._geometry(positions,mask,cell)
    radial=model.edge_basis(distances)
    harmonics=o3.spherical_harmonics(list(range(model.config.resolved_hidden_lmax+1)),
        directions,normalize=True,normalization="integral")
    features=model.embedding(numbers)
    states=[]
    for interaction in model.interactions:
        features=interaction(features,harmonics,radial,pair_mask,source_indices)
        features=features*mask[...,None].to(features.dtype)
        states.append(features)
    return states


class QueryInteraction(nn.Module):
    def __init__(self, irreps, edge_irreps, radial_size=30, radial_hidden=64):
        super().__init__()
        self.product=o3.FullyConnectedTensorProduct(irreps,edge_irreps,irreps,
            internal_weights=False,shared_weights=False,irrep_normalization="component",
            path_normalization="element")
        self.radial=_mlp(radial_size,radial_hidden,self.product.weight_numel)
        self.skip=o3.Linear(irreps,irreps)
        self.activation=NormActivation(irreps,torch.nn.functional.silu,normalize=True,bias=False)

    def forward(self, atom_features, probes, source, receiver, harmonics, radial, counts, envelope):
        message=self.product(atom_features[source],harmonics,self.radial(radial))
        # MLP biases otherwise remain nonzero when the radial input vanishes.
        message=message*envelope[:,None]
        aggregate=torch.zeros_like(probes).index_add(0,receiver,message)
        aggregate=aggregate/counts.clamp_min(1).sqrt()[:,None]
        return self.activation(self.skip(probes)+aggregate)


class EquivariantQueryReadout(nn.Module):
    def __init__(self, irreps, layers=3, cutoff=3., radial_size=30, radial_hidden=64):
        super().__init__()
        self.irreps=o3.Irreps(irreps)
        self.cutoff=cutoff
        self.edge_irreps=o3.Irreps.spherical_harmonics(2)
        self.radial=GaussianEdgeBasis(radial_size,cutoff)
        self.interactions=nn.ModuleList([QueryInteraction(self.irreps,self.edge_irreps,
            radial_size,radial_hidden) for _ in range(layers)])
        invariant_channels=sum(mul for mul,_ in self.irreps)
        self.readout=nn.Sequential(nn.Linear(invariant_channels,128),nn.SiLU(),
                                   nn.Linear(128,128),nn.SiLU(),nn.Linear(128,1))

    def forward(self, atom_states, source, receiver, displacement, num_queries):
        dtype=atom_states[0].dtype
        displacement=displacement.to(dtype)
        distances=displacement.norm(dim=-1)
        harmonics=o3.spherical_harmonics(self.edge_irreps,displacement,
            normalize=True,normalization="integral")
        radial=self.radial(distances)
        envelope=.5*(1+torch.cos(torch.pi*(distances/self.cutoff).clamp(0,1)))
        envelope=envelope*(distances<self.cutoff).to(dtype)
        # Smooth support mass, rather than a discontinuous integer neighbour count.
        counts=distances.new_zeros(num_queries).index_add(0,receiver,envelope)
        probes=atom_states[0].new_zeros(num_queries,self.irreps.dim)
        # Both readouts see the same final backbone features. No skip access to
        # earlier layers is added only for the query arm.
        for interaction in self.interactions:
            probes=interaction(atom_states[-1][0],probes,source,receiver,harmonics,radial,counts,envelope)
        invariants=[]
        for (mul,ir),sl in zip(self.irreps,self.irreps.slices()):
            block=probes[:,sl].reshape(num_queries,mul,ir.dim)
            invariants.append(block[...,0] if ir.l==0 else
                              (block.square().sum(-1)+1e-12).sqrt())
        return self.readout(torch.cat(invariants,-1)).squeeze(-1)


class PairedReadoutModel(nn.Module):
    def __init__(self, config, readout="coefficient", query_radial_hidden=25):
        super().__init__()
        config=dict(config)
        config.pop("architecture",None)
        self.atom_model=MaceLikeCoefficientGNN(config)
        self.kind=readout
        if readout=="query":
            # Construct the exact original model before removing its unused head.
            # This preserves RNG consumption and the atom backbone initialization.
            self.query_head=EquivariantQueryReadout(self.atom_model.hidden_irreps,
                layers=self.atom_model.config.num_interactions,cutoff=3.,radial_hidden=query_radial_hidden)
            del self.atom_model.readout
            del self.atom_model.learned_scalar_baseline
            del self.atom_model.atomic_scalar_baseline
        elif readout!="coefficient":
            raise ValueError(readout)

    def representation(self,sample):
        args=(sample["numbers"][None],sample["model_positions"],sample["mask"],sample["model_cell"])
        if self.kind=="coefficient":
            return self.atom_model(*args)["coefficients"]
        return atom_representations(self.atom_model,*args)

    def query(self,sample,indices,states):
        source,receiver,shifts=sample["graph"].select(indices)
        displacement=sample["probes"][indices][receiver]-sample["positions"][source]-shifts.double()@sample["cell"]
        return self.query_head(states,source,receiver,displacement,len(indices))
