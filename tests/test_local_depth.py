import pytest
import torch

from scripts.deepdft_local_depth import DeepLocalProbe
from scripts.deepdft_mixed_embedding import MixedProbe
from scripts.deepdft_inference_fast import ProbeEdges
from test_mixed_embedding import setup


def test_depth_one_matches_existing_head_and_gradients():
    sample,rep = setup()
    old = MixedProbe('local',32)
    new = DeepLocalProbe(32,1)
    new.load_state_dict(old.state_dict())
    graph = ProbeEdges.build(sample)
    geometry = graph.chunk(sample,0,29)
    x = old(*geometry,*old.atoms(rep),29)
    y = new(*geometry,*new.atoms(rep),29)
    torch.testing.assert_close(x,y,atol=0,rtol=0)
    x.square().sum().backward();y.square().sum().backward()
    for a,b in zip(old.parameters(),new.parameters()):
        if a.requires_grad:
            torch.testing.assert_close(a.grad,b.grad,atol=1e-7,rtol=1e-5)


@pytest.mark.parametrize('depth',[2,3])
def test_deep_local_rotation_chunking_and_all_layers_receive_gradients(depth):
    sample,rep = setup()
    head = DeepLocalProbe(128,depth)
    graph = ProbeEdges.build(sample)
    states = head.atoms(rep)
    a,b,d = graph.chunk(sample,0,24)
    output = head(a,b,d,*states,24)
    parts = [head(*graph.chunk(sample,i,i+8),*states,8) for i in range(0,24,8)]
    torch.testing.assert_close(output,torch.cat(parts),atol=1e-5,rtol=1e-4)
    q,_ = torch.linalg.qr(torch.randn(3,3))
    rotated = (rep[0],[torch.einsum('ij,ajc->aic',q,v) for v in rep[1]])
    other = head(a,b,d@q.T,*head.atoms(rotated),24)
    torch.testing.assert_close(output,other,atol=2e-5,rtol=1e-4)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in head.parameters() if p.requires_grad)
    for layer in head.probe.message_layers:
        assert layer.filter_layer.weight.grad.abs().sum()>0
