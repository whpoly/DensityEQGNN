import torch
from e3nn import o3
from scripts.nmc_eden_r16 import EdenR16,prediction
from scripts.nmc_edengnn import atom_data
from scripts.nmc_probe_holdout import PeriodicQueryGraph


def sample():
    cell=torch.tensor([[4.,0.,0.],[.2,4.1,0.],[.1,.3,4.2]],dtype=torch.float64)
    positions=torch.tensor([[.15,.25,.3],[.65,.55,.45]],dtype=torch.float64)@cell
    probes=torch.tensor([[.3,.2,.4],[.6,.7,.5],[.8,.1,.2]],dtype=torch.float64)@cell
    return {'positions':positions,'cell':cell,'probes':probes,'numbers':torch.tensor([8,28]),
            'sad':torch.tensor([.2,.3,.1])}


def test_official_encoder_r16_bridge_is_rotation_equivariant():
    torch.set_num_threads(2)
    torch.manual_seed(23)
    model=EdenR16().eval()
    s=sample()
    with torch.no_grad():
        coeff=model.coefficients(atom_data(s))
        rotation=o3.rand_matrix(dtype=torch.float64)
        turned=dict(s)
        for key in ['positions','cell','probes']:
            turned[key]=s[key]@rotation.T
        rotated=model.coefficients(atom_data(turned))
        d=o3.Irreps.spherical_harmonics(4).D_from_matrix(rotation).float()
        expected=(coeff.reshape(1,2,16,25)@d.T).reshape_as(coeff)
        torch.testing.assert_close(rotated,expected,rtol=3e-4,atol=3e-5)
    # The readout and official trunk both receive gradients.
    model.train()
    model.coefficients(atom_data(s)).square().mean().backward()
    assert model.readout.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())


def test_zero_residual_is_exact_sad_and_fast_decoder_matches():
    s=sample()
    s['graph']=PeriodicQueryGraph(s['positions'],s['probes'],s['cell'],3.)
    ids=torch.tensor([2,0,2])
    coeff=torch.zeros(1,2,400)
    torch.testing.assert_close(prediction(s,coeff,ids),s['sad'][ids].double(),rtol=0,atol=0)
    torch.manual_seed(6)
    coeff=torch.randn_like(coeff)*.03
    torch.testing.assert_close(prediction(s,coeff,ids,True),prediction(s,coeff,ids),rtol=3e-5,atol=3e-6)
