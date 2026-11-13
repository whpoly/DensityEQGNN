import pytest
import torch

from scripts.deepdft_mixed_embedding import MixedProbe
from scripts.deepdft_inference_fast import ProbeEdges
from test_mixed_embedding import setup


@pytest.mark.parametrize('width',[128,256])
def test_wide_local_backward_and_chunking(width):
    sample,rep = setup()
    head = MixedProbe('local',width)
    graph = ProbeEdges.build(sample)
    states = head.atoms(rep)
    output = head(*graph.chunk(sample,0,24),*states,24)
    separate = torch.cat([head(*graph.chunk(sample,i,i+8),*states,8) for i in range(0,24,8)])
    torch.testing.assert_close(output,separate,atol=1e-5,rtol=1e-4)
    output.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in head.parameters() if p.requires_grad)
    assert all(p.grad is None for p in head.global_encoder.parameters())
