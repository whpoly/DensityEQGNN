import torch
from e3nn import o3
from scripts.nmc_adaptive_r16 import AdaptiveEdenR16
from scripts.nmc_eden_r16 import EdenR16
from scripts.nmc_edengnn import atom_data
from test_eden_r16 import sample


def test_adaptive_warm_start_and_nonzero_head_equivariance():
    torch.set_num_threads(2)
    torch.manual_seed(45)
    base = EdenR16().eval()
    model = AdaptiveEdenR16().eval()
    model.load_state_dict(base.state_dict(),strict=False)
    s = sample()
    with torch.no_grad():
        torch.testing.assert_close(model.coefficients(atom_data(s)),base.coefficients(atom_data(s)),rtol=0,atol=0)
        model.mixing[-1].weight.normal_(std=.01)
        model.mixing[-1].bias.normal_(std=.01)
        coeff = model.coefficients(atom_data(s))
        # Include an improper rotation to check parity, not SO(3) alone.
        rotation = -o3.rand_matrix(dtype=torch.float64)
        turned = dict(s)
        for key in ('positions','cell','probes'):
            turned[key] = s[key]@rotation.T
        rotated = model.coefficients(atom_data(turned))
        d = o3.Irreps.spherical_harmonics(4).D_from_matrix(rotation).float()
        expected = (coeff.reshape(1,2,16,25)@d.T).reshape_as(coeff)
        torch.testing.assert_close(rotated,expected,rtol=5e-4,atol=5e-5)
    model.train()
    model.coefficients(atom_data(s)).square().mean().backward()
    for part in (model.mixing[0],model.mixing[-1],model.element_context):
        assert part.weight.grad is not None and part.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.encoder.parameters())
