from dataclasses import replace

import numpy as np
import pytest
import torch

from density_eqgnn.compact_basis import (
    CompactBasisConfig, check_basis_checkpoint, reconstruct_compact_density, basis_checkpoint_metadata,
    radial_to_degree_major,
)
from density_eqgnn.oracle_basis import OracleBasis
from density_eqgnn.radial_basis import radial_transform, radial_values
from density_eqgnn.models import MaceLikeCoefficientGNN, MaceLikeModelConfig
from density_eqgnn.fixed_grid import FixedGridDensity


def degree_to_radial(vector, basis):
    blocks, offset = [], 0
    for l in range(basis.lmax + 1):
        size = basis.num_radial * (2*l+1)
        blocks.append(vector[:, offset:offset+size].reshape(-1, basis.num_radial, 2*l+1))
        offset += size
    return torch.cat(blocks, dim=-1).flatten(1)


@pytest.mark.parametrize("nradial,regular", [(12, False), (16, True)])
def test_transformed_decoder_matches_independent_oracle_and_adjoint(nradial, regular):
    torch.manual_seed(73)
    basis = CompactBasisConfig(num_radial=nradial, lmax=4, cutoff=3., pbc=True,
                               regular_radial=regular, radial_whitening=True)
    cell = torch.tensor([[3.,0.,0.],[1.2,2.7,0.],[.3,.2,3.1]],dtype=torch.float64)
    atoms = torch.rand(2,3,dtype=torch.float64) @ cell
    probes = torch.rand(13,3,dtype=torch.float64) @ cell
    c = torch.randn(2, basis.coefficient_size,dtype=torch.float64,requires_grad=True)
    torch.testing.assert_close(radial_to_degree_major(degree_to_radial(c,basis),basis),c)
    matrix = OracleBasis([(atoms,basis.radial_spec)],cell).design(probes,dtype=torch.float64)
    pred = reconstruct_compact_density(degree_to_radial(c,basis)[None],atoms[None],probes[None],
                                      torch.ones(1,2,dtype=torch.bool),torch.ones(1,13,dtype=torch.bool),
                                      cell[None],basis)[0]
    torch.testing.assert_close(pred,matrix@c.flatten(),rtol=2e-7,atol=2e-7)
    v = torch.randn_like(pred)
    (pred*v).sum().backward()
    torch.testing.assert_close(c.grad.flatten(),matrix.T@v,rtol=2e-7,atol=2e-7)


def test_r16_whitening_uses_float64_even_inside_autocast_and_is_orthonormal():
    basis = CompactBasisConfig(num_radial=16,lmax=4,cutoff=3.,regular_radial=True,radial_whitening=True)
    nodes, weights = np.polynomial.legendre.leggauss(512)
    radius = torch.from_numpy((nodes+1)*1.5)
    weights = torch.from_numpy(weights*1.5)*radius.square()
    for l in range(5):
        with torch.autocast("cpu",dtype=torch.bfloat16):
            white = radial_values(radius,basis.radial_spec,l,whiten=True)
        assert white.dtype == torch.float64
        torch.testing.assert_close(white.T@(weights[:,None]*white),torch.eye(16,dtype=torch.float64),
                                   rtol=1e-6,atol=2e-7)
        # The unsafe single-precision transform really loses significant digits.
        raw32 = radial_values(radius.float(),basis.radial_spec,l,whiten=False)
        naive = raw32 @ torch.tensor(radial_transform(basis.radial_spec,l)[0],dtype=torch.float32)
        assert (naive.double()-white).abs().max() > .01


def test_regular_r16_has_correct_nonzero_l1_derivative_at_exact_origin():
    basis = CompactBasisConfig(num_radial=16,lmax=4,cutoff=3.,
                               regular_radial=True,radial_whitening=True)
    c = torch.zeros(1,1,basis.coefficient_size,dtype=torch.float64)
    c[0,0,1] = 1.  # first radial channel, l=1 x component
    atom = torch.zeros(1,1,3,dtype=torch.float64)
    mask = torch.ones(1,1,dtype=torch.bool)
    probe = torch.zeros_like(atom,requires_grad=True)
    def decode(p):
        return reconstruct_compact_density(c,atom,p,mask,mask,None,basis)
    derivative = torch.autograd.grad(decode(probe).sum(),probe)[0]
    h = torch.zeros_like(probe); h[...,0] = 1e-5
    finite = (decode(h)-decode(-h))/(2e-5)
    assert abs(float(derivative[...,0])) > 1e-3
    torch.testing.assert_close(derivative[...,0],finite,rtol=1e-6,atol=1e-8)


def test_transformed_decoder_coordinate_gradients_padding_and_empty_support():
    basis = CompactBasisConfig(num_radial=4,lmax=2,cutoff=3.,pbc=True,
                               regular_radial=True,radial_whitening=True)
    c = torch.randn(1,2,basis.coefficient_size,dtype=torch.float64,requires_grad=True)
    a = torch.tensor([[[.2,.4,.3],[1.1,.2,.3]]],dtype=torch.float64,requires_grad=True)
    p = torch.tensor([[[.9,.8,1.2],[1.8,1.1,1.7]]],dtype=torch.float64,requires_grad=True)
    cell = (torch.eye(3,dtype=torch.float64)[None]*4).requires_grad_()
    am, pm = torch.tensor([[True,False]]), torch.tensor([[True,False]])
    def decode(c,a,p,cell):
        return reconstruct_compact_density(c,a,p,am,pm,cell,basis)
    assert torch.autograd.gradcheck(decode,(c,a,p,cell),fast_mode=True,atol=1e-5)
    assert decode(c,a,p,cell)[0,1] == 0
    empty = reconstruct_compact_density(c,a,p,torch.zeros_like(am),pm,cell,basis)
    assert torch.all(empty == 0)
    empty.sum().backward()
    assert torch.all(c.grad == 0)


def test_r16_gnn_density_rotation_and_primitive_supercell_consistency():
    torch.manual_seed(9)
    basis = CompactBasisConfig(num_radial=16,lmax=4,cutoff=3.,pbc=True,
                               regular_radial=True,radial_whitening=True)
    model = MaceLikeCoefficientGNN(MaceLikeModelConfig(
        num_radial=16,hidden_lmax=2,density_lmax=4,readout_edge_lmax=2,
        scalar_channels=3,vector_channels=2,tensor_channels=1,num_interactions=1,
        edge_basis=3,radial_hidden=4,cutoff=4.,pbc=True,
    )).double()
    cell = torch.tensor([[3.,0,0],[.8,3.2,0],[.2,.4,3.3]],dtype=torch.float64)
    atoms = torch.tensor([[.2,.3,.4]],dtype=torch.float64)
    z = torch.tensor([[8]])
    probes = torch.tensor([[.6,.8,.9],[2.8,.4,.7]],dtype=torch.float64)
    def evaluate(a,z,c,p):
        mask = torch.ones(z.shape,dtype=torch.bool)
        coeff = model(z,a[None],mask,c[None])["coefficients"]
        return reconstruct_compact_density(coeff,a[None],p[None],mask,
                                           torch.ones(1,len(p),dtype=torch.bool),c[None],basis)
    density = evaluate(atoms,z,cell,probes)
    expanded = cell.clone(); expanded[0] *= 2
    replicas = torch.cat([atoms,atoms+cell[0]])
    torch.testing.assert_close(density,evaluate(replicas,z.repeat(1,2),expanded,probes),rtol=2e-6,atol=2e-6)
    rotation,_ = torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
    torch.testing.assert_close(density,evaluate(atoms@rotation.T,z,cell@rotation.T,probes@rotation.T),
                               rtol=2e-6,atol=2e-6)


def test_checkpoint_rejects_silent_change_of_coefficient_coordinates():
    basis = CompactBasisConfig(num_radial=12,lmax=4,cutoff=3.,pbc=True)
    old = {"basis_config":{"num_radial":12,"lmax":4,"cutoff":3.,"pbc":True}}
    check_basis_checkpoint(old,basis)
    for changed in (replace(basis,regular_radial=True),replace(basis,radial_whitening=True)):
        with pytest.raises(ValueError,match="whitening differs"):
            check_basis_checkpoint(old,changed)
        check_basis_checkpoint(basis_checkpoint_metadata(changed),changed)
    white = replace(basis,radial_whitening=True)
    corrupted = basis_checkpoint_metadata(white)
    corrupted["radial_whitening_matrices"] *= -1
    with pytest.raises(ValueError,match="matrices differ"):
        check_basis_checkpoint(corrupted,white)


def test_fixed_grid_cache_handles_repeated_random_probes_and_coefficients_gradients():
    torch.manual_seed(91)
    basis = CompactBasisConfig(num_radial=16,lmax=4,cutoff=3.,pbc=True,
                               regular_radial=True,radial_whitening=True)
    cell = torch.tensor([[3.,0,0],[.7,2.8,0],[.1,.2,3.2]],dtype=torch.float64)
    atoms = torch.rand(2,3,dtype=torch.float64)@cell
    probes = torch.rand(9,3,dtype=torch.float64)@cell
    c = torch.randn(1,2,basis.coefficient_size,dtype=torch.float64,requires_grad=True)
    cache = FixedGridDensity(atoms,probes,cell,basis)
    ids = torch.tensor([8,0,0,7,2])
    expected = reconstruct_compact_density(c,atoms[None],probes[ids][None],
                                          torch.ones(1,2,dtype=torch.bool),torch.ones(1,5,dtype=torch.bool),
                                          cell[None],basis)[0]
    actual = cache(c,ids)
    torch.testing.assert_close(actual,expected,rtol=1e-8,atol=1e-8)
    grad1 = torch.autograd.grad(actual.square().sum(),c,retain_graph=True)[0]
    grad2 = torch.autograd.grad(expected.square().sum(),c)[0]
    torch.testing.assert_close(grad1,grad2,rtol=1e-8,atol=1e-8)
