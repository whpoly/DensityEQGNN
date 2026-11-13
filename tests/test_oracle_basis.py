from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from density_eqgnn.compact_basis import CompactBasisConfig, reconstruct_compact_density
from density_eqgnn.oracle_basis import (
    FullGridDesign, OracleBasis, RadialSpec, edge_features, periodic_bond_centers, raw_to_whitened,
)


def test_whitening_preserves_production_periodic_density():
    torch.manual_seed(10)
    cell = torch.tensor([[3.,0,0],[1.,2.7,0],[.2,.3,3.5]], dtype=torch.float64)
    atoms = torch.rand(2,3,dtype=torch.float64) @ cell
    probes = torch.rand(11,3,dtype=torch.float64) @ cell
    spec = RadialSpec()
    coeff = torch.randn(2,spec.size,dtype=torch.float64)
    basis = OracleBasis([(atoms,spec)],cell)
    pred = basis.design(probes,dtype=torch.float64) @ raw_to_whitened(coeff,spec)
    expected = reconstruct_compact_density(
        coeff[None],atoms[None],probes[None],torch.ones(1,2,dtype=torch.bool),
        torch.ones(1,11,dtype=torch.bool),cell[None],
        CompactBasisConfig(num_radial=12,lmax=4,cutoff=3.,pbc=True),
    )[0]
    torch.testing.assert_close(pred,expected,atol=1e-9,rtol=1e-9)


def test_regular_angular_channels_vanish_at_origin():
    spec = RadialSpec(regular=True)
    vectors = torch.eye(3,dtype=torch.float64)*1e-8
    features = edge_features(torch.cat([vectors,-vectors,torch.zeros_like(vectors[:1])]),spec)
    assert torch.all(features[-1,spec.num_radial:] == 0)
    assert features[:-1,spec.num_radial:].abs().max() < 1e-4
    smaller = edge_features(vectors/10,spec)
    torch.testing.assert_close(smaller[:,12:48],features[:3,12:48]/10,rtol=1e-7,atol=1e-12)


def test_full_grid_operator_matches_dense_and_has_correct_adjoint_and_gradient():
    torch.manual_seed(23)
    cell = torch.eye(3,dtype=torch.float64)*2.5
    atoms = torch.rand(2,3,dtype=torch.float64) @ cell
    probes = torch.rand(17,3,dtype=torch.float64) @ cell
    spec = RadialSpec(num_radial=3,lmax=2,regular=True)
    basis = OracleBasis([(atoms,spec),(atoms+.1,replace(spec,lmax=0))],cell)
    dense = basis.design(probes)
    operator = FullGridDesign(basis,probes)
    x = torch.randn(basis.size,requires_grad=True)
    y = torch.randn(len(probes))
    torch.testing.assert_close(operator@x,dense@x)
    torch.testing.assert_close(operator.T@y,dense.T@y)
    ((operator@x)*y).sum().backward()
    torch.testing.assert_close(x.grad,dense.T@y)
    scales = operator.normalize_columns()
    torch.testing.assert_close(operator@(x.detach()*scales),dense@x.detach())
    factor = operator.gram_cholesky(ridge=1e-7)
    normalized = (dense/scales[None]).double()
    expected = normalized.T@normalized+torch.eye(basis.size,dtype=torch.float64)*1e-7
    torch.testing.assert_close(factor@factor.T,expected,rtol=1e-6,atol=1e-6)
    x64 = x.detach().double().requires_grad_()
    assert torch.autograd.gradcheck(operator.precise_matvec,(x64,),fast_mode=True)


def test_bond_centers_and_density_are_supercell_consistent():
    cell = torch.eye(3,dtype=torch.float64)*3
    atoms = torch.tensor([[.2,.2,.2],[1.7,.2,.2]],dtype=torch.float64)
    z = torch.tensor([3,8])
    centers = periodic_bond_centers(atoms,z,cell,2.)
    assert len(centers) == 2  # Includes the bond through the unit-cell face.
    replicas = torch.cat([atoms,atoms+cell[0]])
    expanded = cell.clone(); expanded[0] *= 2
    repeated = periodic_bond_centers(replicas,z.repeat(2),expanded,2.)
    assert len(repeated) == 2*len(centers)
    mapped = torch.remainder(repeated @ torch.linalg.inv(cell),1.) @ cell
    for point in mapped:
        assert torch.linalg.vector_norm(centers-point,dim=1).min() < 1e-10
    spec = RadialSpec(num_radial=3,lmax=2,cutoff=2.,minimum=.15,regular=True)
    probes = torch.tensor([[.4,.6,.3],[2.8,.1,.1]],dtype=torch.float64)
    coeff = torch.ones(len(centers)*spec.size,dtype=torch.float64)
    rho = OracleBasis([(centers,spec)],cell).design(probes,dtype=torch.float64) @ coeff
    other = OracleBasis([(repeated,spec)],expanded).design(probes,dtype=torch.float64) @ coeff.repeat(2)
    torch.testing.assert_close(rho,other,rtol=1e-9,atol=1e-9)


def test_oracle_optimizer_finds_l1_median_and_dual_diagnostic():
    from scripts.compare_local_nmc_oracles import fit

    # A constant field with one outlier has a known L1 optimum (the median),
    # different from its least-squares optimum (the mean).
    operator = FullGridDesign.__new__(FullGridDesign)
    operator.device = torch.device("cpu")
    operator.dtype = torch.float32
    operator.shape = (5,1)
    operator.parts = [(torch.arange(5),torch.ones(5,1))]
    target = torch.tensor([1.,1.,1.,1.,9.],dtype=torch.float64)
    args = SimpleNamespace(gram_precondition=True,gram_ridge=1e-9,
                           precise_steps=-1,round_steps=50,rounds=1,dual_check=True)
    coefficient,history = fit(operator,target,args)
    torch.testing.assert_close(coefficient,torch.ones_like(coefficient),rtol=1e-3,atol=1e-3)
    dual = history["dual_diagnostic"]
    assert abs(dual["primal_nmae"]-8/13) < 1e-4
    assert abs(dual["numerical_primal_dual_gap"]) < 1e-4
    assert dual["max_abs_adjoint_constraint"] < 1e-8
    assert dual["max_abs_dual"] <= 1
