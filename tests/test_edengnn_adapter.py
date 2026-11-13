"""Geometry and official-forward equivalence for the NMC benchmark adapter."""
from pathlib import Path
import types

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from scripts.nmc_edengnn import EdenDensity, EdenStencil, REPO, official_classes, stencil_numpy


def synthetic_sample():
    cell=torch.tensor([[4.,0.,0.],[1.2,4.5,0.],[.4,.3,5.]],dtype=torch.float64)
    positions=torch.tensor([[.31,.21,.12],[3.6,3.9,4.4]],dtype=torch.float64)
    return {"cell":cell,"positions":positions,"numbers":torch.tensor([3,8]),
            "metadata":{"grid_shape":[7,8,9]},"sad":torch.zeros(7*8*9)}


def test_native_stencil_csr_preserves_periodic_multiplicity():
    sample=synthetic_sample()
    stencil=EdenStencil(sample)
    ids=torch.arange(stencil.num_points)
    atoms,receiver,offset=stencil.select(ids)
    assert len(atoms)==stencil.num_atoms*stencil.npb
    target=(stencil.pos_n[atoms]+stencil.map_probe[offset])%stencil.grid_shape
    expected=(target[:,0]*8+target[:,1])*9+target[:,2]
    torch.testing.assert_close(receiver,expected)
    # The sphere exceeds half a cell: multiple images must not be deduplicated.
    assert len(torch.unique(torch.stack((atoms,receiver),1),dim=0))<len(atoms)
    assert len(torch.unique(atoms*stencil.npb+offset))==len(atoms)
    moved=dict(sample,positions=sample["positions"]+sample["cell"][0])
    translated=EdenStencil(moved)
    torch.testing.assert_close(stencil.pairs,translated.pairs)
    torch.testing.assert_close(stencil.pos_res,translated.pos_res,atol=1e-6,rtol=1e-6)


@pytest.mark.skipif(not (REPO/"src/edengnn/model/model.py").exists(),reason="optional pinned upstream checkout absent")
def test_sampled_official_density_values_gradients_and_duplicate_queries():
    torch.manual_seed(91)
    Encoder,DensityLayer,_=official_classes()
    model=EdenDensity.__new__(EdenDensity)
    torch.nn.Module.__init__(model)
    irreps="4x0e+3x1o+2x2e"
    model.encoder=Encoder(l_max=2,num_types=100,irreps_node_features=irreps,num_conv_layers=2)
    model.density=DensityLayer(cutoff=4.,l_max=2,
        irreps_node_attribute=model.encoder.one_hot.irreps_out["node_attrs"],
        irreps_node_features=irreps,n_channels=2,num_radial_filter_neurons=8)
    sample=synthetic_sample()
    from scripts.nmc_edengnn import atom_data
    sample["atom_data"]=atom_data(sample)
    sample["stencil"]=EdenStencil(sample)
    sample["native_data"]=sample["atom_data"].clone()
    for key,value in sample["stencil"].native_fields().items():
        sample["native_data"][key]=value
    ids=torch.tensor([0,1,19,81,81,90,200,503])
    sparse=model.sampled(sample,ids)
    native=model.native(sample)[ids]
    torch.testing.assert_close(sparse,native,rtol=2e-5,atol=2e-6)
    sparse.square().sum().backward()
    gradients={name:p.grad.clone() for name,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.native(sample)[ids].square().sum().backward()
    for name,p in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad,gradients[name],rtol=2e-4,atol=2e-6)
    with torch.no_grad():
        rep=model.encode(sample["atom_data"])
        batched=torch.cat([model.sampled(sample,chunk,rep) for chunk in ids.split(3)])
    torch.testing.assert_close(batched,sparse,rtol=2e-5,atol=2e-6)
