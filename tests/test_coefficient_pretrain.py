import torch
from scripts.train_nmc_coefficient_pretrain import coefficient_loss,fast_prediction,initialize
from scripts.fit_nmc_holdout_oracle import BASIS
from density_eqgnn.oracle_basis import OracleBasis
from scripts.nmc_probe_holdout import PeriodicQueryGraph


def radial_to_degree(coeff):
    blocks=coeff.reshape(-1,16,25)
    return torch.cat([blocks[...,l*l:(l+1)**2].flatten(1) for l in range(5)],1).flatten()


def test_density_weighted_coefficient_objective_matches_independent_design_and_gradient():
    torch.manual_seed(91)
    cell=torch.tensor([[3.2,0.,0.],[.3,3.3,0.],[.2,.4,3.4]],dtype=torch.float64)
    atoms=torch.rand(2,3,dtype=torch.float64)@cell
    probes=torch.rand(9,3,dtype=torch.float64)@cell
    sample={'positions':atoms,'probes':probes,'cell':cell}
    sample['graph']=PeriodicQueryGraph(atoms,probes,cell,3.)
    coeff=torch.randn(1,2,400,dtype=torch.float64,requires_grad=True)
    teacher=torch.randn_like(coeff)
    # Includes repeated query ids and a nontrivial permutation.
    ids=torch.tensor([7,1,7,4,0])
    matrix=OracleBasis([(atoms,BASIS.radial_spec)],cell).design(probes[ids],dtype=torch.float64)
    expected=matrix@radial_to_degree(coeff-teacher)
    actual=fast_prediction(sample,coeff-teacher,ids)
    torch.testing.assert_close(actual,expected,rtol=2e-7,atol=2e-7)
    a=torch.autograd.grad(actual.square().mean(),coeff,retain_graph=True)[0]
    b=torch.autograd.grad(expected.square().mean(),coeff)[0]
    assert torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b)<2e-7


def test_coefficient_mse_has_zero_at_teacher_and_expected_normalized_gradient():
    torch.manual_seed(12)
    teacher=torch.randn(1,3,400,dtype=torch.float64)
    assert coefficient_loss(teacher,teacher)==0
    value=torch.randn_like(teacher,requires_grad=True)
    grad=torch.autograd.grad(coefficient_loss(value,teacher),value)[0]
    torch.testing.assert_close(grad,2*(value-teacher)/teacher.square().sum())


def test_pretrain_initializes_exact_original_r16_model():
    model,settings=initialize(torch.device('cpu'))
    assert sum(p.numel() for p in model.parameters())==521472
    assert settings['steps']==16000
