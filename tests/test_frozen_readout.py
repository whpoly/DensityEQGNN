import torch

from density_eqgnn.models import build_compact_coefficient_model, load_compact_model_state
from density_eqgnn.compact_basis import CompactBasisConfig, radial_to_degree_major
from density_eqgnn.frozen_readout import (promote_linear_readout, frozen_features,
    linear_parameters, set_linear_parameters, frozen_digest, blocks_to_degree_major)


def test_frozen_linear_map_matches_network_and_preserves_nonlinear_parameters():
    torch.manual_seed(63)
    model = build_compact_coefficient_model(dict(architecture='mace_like',num_radial=3,
        hidden_lmax=2,density_lmax=4,readout_edge_lmax=2,scalar_channels=4,
        vector_channels=3,tensor_channels=2,num_interactions=2,radial_hidden=5,
        edge_basis=4,cutoff=3.2,pbc=True,learnable_atomic_baseline=True))
    species = torch.tensor([3,8,28])
    positions = torch.tensor([[[.1,.3,.2],[1.2,.4,.8],[.5,1.6,.3]]])
    cell = torch.tensor([[[3.,0.,0.],[.7,3.1,0.],[.2,.6,3.3]]])
    numbers = species[None]
    mask = torch.ones_like(numbers,dtype=torch.bool)
    digest = frozen_digest(model)
    promote_linear_readout(model)
    matrices = frozen_features(model,numbers,positions,mask,cell,species)
    theta = linear_parameters(model,species)
    basis = CompactBasisConfig(num_radial=3,lmax=4)
    def forward():
        return radial_to_degree_major(model(numbers,positions,mask,cell)['coefficients'],basis)[0]
    def expected(p):
        return blocks_to_degree_major([f@v for f,v in zip(matrices,p)],3)
    torch.testing.assert_close(forward(),expected(theta),rtol=1e-10,atol=1e-10)
    alternate = [t+torch.randn_like(t)*.2 for t in theta]
    set_linear_parameters(model,species,alternate)
    torch.testing.assert_close(forward(),expected(alternate),rtol=1e-10,atol=1e-10)
    assert frozen_digest(model) == digest
    midpoint = [(a+b)/2 for a,b in zip(theta,alternate)]
    set_linear_parameters(model,species,midpoint)
    torch.testing.assert_close(forward(),(expected(theta)+expected(alternate))/2,rtol=1e-10,atol=1e-10)
    state={'model':model.state_dict(),'linear_readout_dtype':'float64'}
    from dataclasses import asdict
    restored=build_compact_coefficient_model(dict(architecture='mace_like',**asdict(model.config)))
    load_compact_model_state(restored,state)
    torch.testing.assert_close(restored(numbers,positions,mask,cell)['coefficients'],
                               model(numbers,positions,mask,cell)['coefficients'],rtol=0,atol=0)


def test_frozen_readout_feature_adjoint_and_parameter_roundtrip():
    torch.manual_seed(64)
    model = build_compact_coefficient_model(dict(architecture='mace_like',num_radial=2,
        hidden_lmax=1,density_lmax=2,readout_edge_lmax=1,scalar_channels=3,
        vector_channels=2,num_interactions=1,radial_hidden=4,edge_basis=3,
        cutoff=2.,pbc=False,learnable_atomic_baseline=True))
    promote_linear_readout(model)
    species = torch.tensor([1,8])
    params = linear_parameters(model,species)
    random = [torch.randn_like(p) for p in params]
    set_linear_parameters(model,species,random)
    for a,b in zip(random,linear_parameters(model,species)):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    numbers = torch.tensor([[1,8,1]])
    pos = torch.randn(1,3,3)
    features = frozen_features(model,numbers,pos,torch.ones_like(numbers,dtype=torch.bool),None,species)
    for f,theta in zip(features,random):
        v = torch.randn(f.shape[0],theta.shape[1],dtype=torch.float64)
        torch.testing.assert_close(((f@theta)*v).sum(),(theta*(f.T@v)).sum(),rtol=1e-12,atol=1e-12)


def test_reduced_readout_preserves_parent_and_has_correct_density_adjoint():
    from scripts.calibrate_local_nmc_readout import ReducedReadout, DensityOperator, cgls
    torch.manual_seed(65)
    counts=[2,3]
    radial=2
    features=[torch.randn(sum(counts)*(2*l+1),4+l,dtype=torch.float64) for l in range(3)]
    theta=[torch.randn(f.shape[1],radial,dtype=torch.float64) for f in features]
    reduced=ReducedReadout(features,theta,counts,1e-12)
    zero=torch.zeros(reduced.size,dtype=torch.float64)
    expected=blocks_to_degree_major([f@t for f,t in zip(features,theta)],radial)
    torch.testing.assert_close(torch.cat(reduced.coefficients(zero)),expected.flatten())
    matrices=[torch.randn(100,count*radial*9) for count in counts]
    operator=DensityOperator(reduced,matrices,[1.,2.],torch.linspace(.1,2.,200,dtype=torch.float64))
    x=torch.randn_like(zero)
    y=torch.randn(200,dtype=torch.float64)
    torch.testing.assert_close((operator.forward(x)*y).sum(),(x*operator.adjoint(y)).sum(),rtol=2e-6,atol=2e-6)
    target=operator.forward(x,baseline=True)
    fitted,history=cgls(operator,target,zero,100)
    torch.testing.assert_close(fitted,x,rtol=2e-5,atol=2e-5)


def test_stratified_probe_weights_recover_full_grid_mean_in_expectation():
    from scripts.calibrate_local_nmc_readout import stratified_probes
    nearest=torch.tensor([.1,.2,.5,1.,2.])
    values=torch.tensor([20.,8.,3.,2.,1.],dtype=torch.float64)
    estimates={}
    for seed in range(30):
        ids,weights,core=stratified_probes(nearest,.35,1,torch.Generator().manual_seed(seed))
        assert core==2
        torch.testing.assert_close(weights.sum(),torch.tensor(1.,dtype=torch.float64))
        estimates[int(ids[-1])]=(values[ids]*weights).sum()
    assert len(estimates)==3
    torch.testing.assert_close(torch.stack(list(estimates.values())).mean(),values.mean())
    for radius in (0.,3.):
        ids,weights,_=stratified_probes(nearest,radius,10,torch.Generator().manual_seed(1))
        torch.testing.assert_close((values[ids]*weights).sum(),values.mean())
