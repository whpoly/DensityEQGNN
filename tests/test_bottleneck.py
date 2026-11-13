import torch

from density_eqgnn.compact_basis import reconstruct_compact_density
from density_eqgnn.oracle_basis import OracleBasis
from scripts.fit_nmc_holdout_oracle import BASIS, degree_to_radial
from density_eqgnn.models import build_compact_coefficient_model
from density_eqgnn.models.paired_readout import PairedReadoutModel, atom_representations
from density_eqgnn.training.config import load_config
from scripts.nmc_probe_holdout import PeriodicQueryGraph
from scripts.train_local_nmc_overfit import state_digest


def test_holdout_oracle_export_matches_production_decoder():
    torch.manual_seed(92)
    cell=torch.tensor([[2.8,0.,0.],[.4,3.1,0.],[.3,.2,3.2]],dtype=torch.float64)
    positions=torch.rand(2,3,dtype=torch.float64)@cell
    probes=torch.rand(7,3,dtype=torch.float64)@cell
    coeff=torch.randn(2*BASIS.coefficient_size,dtype=torch.float64)
    expected=OracleBasis([(positions,BASIS.radial_spec)],cell).design(probes,dtype=torch.float64)@coeff
    actual=reconstruct_compact_density(degree_to_radial(coeff,2),positions[None],probes[None],
        torch.ones(1,2,dtype=torch.bool),torch.ones(1,7,dtype=torch.bool),cell[None],BASIS)[0]
    # The independent oracle multiplies ill-conditioned raw radial functions;
    # production uses regular solid harmonics. Float64 cancellation is ~1e-8.
    torch.testing.assert_close(actual,expected,rtol=2e-8,atol=2e-8)


def model_config():
    return dict(load_config("configs/nmc_full_l2_to_l4_mace_pbc.yaml")["model"],num_radial=16)


def synthetic_sample():
    cell=torch.tensor([[3.1,0.,0.],[.4,3.2,0.],[.2,.3,3.3]],dtype=torch.float64)
    positions=torch.tensor([[.1,.2,.3],[.7,.4,.5]],dtype=torch.float64)@cell
    probes=torch.tensor([[.2,.3,.4],[.4,.6,.7],[1.1,-.1,.3]],dtype=torch.float64)@cell
    return {"positions":positions,"probes":probes,"cell":cell,"numbers":torch.tensor([8,28]),
        "model_positions":positions.float()[None],"model_cell":cell.float()[None],
        "mask":torch.ones(1,2,dtype=torch.bool)}


def test_readout_variants_have_identical_backbones_and_production_encoding():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    original=build_compact_coefficient_model(model_config())
    torch.manual_seed(42)
    coefficient=PairedReadoutModel(model_config(),"coefficient")
    torch.manual_seed(42)
    query=PairedReadoutModel(model_config(),"query")
    assert state_digest(original.state_dict())==state_digest(coefficient.atom_model.state_dict())
    assert state_digest(original.state_dict(),True)==state_digest(query.atom_model.state_dict(),True)
    sample=synthetic_sample()
    saved=[]
    hook=original.readout.register_forward_pre_hook(lambda module,inputs:saved.append(inputs[0]))
    args=(sample["numbers"][None],sample["model_positions"],sample["mask"],sample["model_cell"])
    expected=original(*args)["coefficients"]
    hook.remove()
    actual=coefficient.representation(sample)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    features=atom_representations(query.atom_model,*args)[-1]
    torch.testing.assert_close(features,saved[0],rtol=0,atol=0)
    expected_grad=torch.autograd.grad(saved[0].square().sum(),original.embedding.weight)[0]
    actual_grad=torch.autograd.grad(features.square().sum(),query.atom_model.embedding.weight)[0]
    torch.testing.assert_close(actual_grad,expected_grad,rtol=0,atol=0)


def test_query_rotation_batch_independence_and_backbone_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(73)
    model=PairedReadoutModel(model_config(),"query")
    sample=synthetic_sample()
    sample["graph"]=PeriodicQueryGraph(sample["positions"],sample["probes"],sample["cell"],3.)
    states=model.representation(sample)
    output=model.query(sample,torch.arange(3),states)
    duplicate=model.query(sample,torch.tensor([2,0,2]),states)
    torch.testing.assert_close(duplicate,output[torch.tensor([2,0,2])],rtol=2e-5,atol=2e-6)
    output.square().sum().backward()
    assert torch.isfinite(model.atom_model.embedding.weight.grad).all()
    assert model.atom_model.embedding.weight.grad.abs().sum()>0
    q,_=torch.linalg.qr(torch.randn(3,3,dtype=torch.float64))
    rotated=synthetic_sample()
    for key in ("positions","probes","cell"): rotated[key]=rotated[key]@q
    rotated["model_positions"]=rotated["positions"].float()[None]
    rotated["model_cell"]=rotated["cell"].float()[None]
    rotated["graph"]=PeriodicQueryGraph(rotated["positions"],rotated["probes"],rotated["cell"],3.)
    with torch.no_grad():
        actual=model.query(rotated,torch.arange(3),model.representation(rotated))
    torch.testing.assert_close(actual,output,rtol=3e-5,atol=2e-6)


def test_query_cutoff_is_continuous_when_an_edge_disappears():
    torch.set_num_threads(2)
    torch.manual_seed(12)
    head=PairedReadoutModel(model_config(),"query").query_head
    states=[torch.randn(1,2,head.irreps.dim)]
    source=torch.tensor([0,1])
    receiver=torch.tensor([0,0])
    inside=torch.tensor([[.5,0.,0.],[3.-1e-5,0.,0.]])
    at_boundary=torch.tensor([[.5,0.,0.],[3.,0.,0.]])
    with torch.no_grad():
        a=head(states,source,receiver,inside,1)
        b=head(states,source,receiver,at_boundary,1)
        c=head(states,source[:1],receiver[:1],inside[:1],1)
    torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(b,c,rtol=1e-5,atol=1e-6)
