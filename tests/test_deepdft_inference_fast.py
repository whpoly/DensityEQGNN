import numpy as np
import pytest
import torch

from scripts.nmc_probe_holdout import DEFAULT_ROOT, official_model, atom_batch, probe_batch, PeriodicQueryGraph
from scripts.deepdft_inference_fast import FlatProbe, ProbeEdges, ProbeReplay, NeighborReuse, flat_density


def sample(device='cpu'):
    cell=torch.tensor([[2.6,0.,0.],[1.1,2.9,0.],[.5,.7,3.1]],dtype=torch.float64,device=device)
    shape=(4,5,6)
    frac=torch.stack(torch.meshgrid(*[torch.arange(n,device=device,dtype=torch.float64)/n for n in shape],indexing='ij'),-1).reshape(-1,3)
    return {'positions':torch.tensor([[1.1,-.2,.45],[.32,.64,.11]],dtype=torch.float64,device=device)@cell,
            'probes':frac@cell,'cell':cell,'numbers':torch.tensor([8,28],device=device),
            'metadata':{'grid_shape':shape}}


def reference(model,s):
    prepared=dict(s,batch=atom_batch(s),graph=PeriodicQueryGraph(s['positions'],s['probes'],s['cell'],4.))
    return model(probe_batch(prepared,torch.arange(len(s['probes']),device=s['positions'].device)))[0]


@torch.no_grad()
def test_flat_probe_geometry_and_neighbor_reuse_refresh():
    torch.set_num_threads(2)
    model,_=official_model(DEFAULT_ROOT/'DeepDFT','cpu')
    model.eval(); core=FlatProbe(model).eval(); s=sample()
    direct=reference(model,s)
    graph=ProbeEdges.build(s)
    result=flat_density(model,core,s,graph,chunk=37)
    torch.testing.assert_close(result,direct,rtol=3e-5,atol=3e-6)
    cache=NeighborReuse(s)
    shifted=dict(s,positions=s['positions']+torch.tensor([[.02,-.01,.03],[-.02,.01,-.01]]))
    # Changing elements must also refresh learned atom features.
    shifted['numbers']=torch.tensor([28,8])
    cached=flat_density(model,core,shifted,cache.get(shifted),chunk=29)
    torch.testing.assert_close(cached,reference(model,shifted),rtol=3e-5,atol=3e-6)
    with pytest.raises(ValueError,match='Motion'):
        cache.get(dict(s,positions=s['positions']+.2))
    with pytest.raises(ValueError,match='cell'):
        cache.get(dict(s,cell=s['cell']*1.01))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@torch.no_grad()
def test_probe_graph_refresh_and_padding():
    torch.set_num_threads(2)
    model,_=official_model(DEFAULT_ROOT/'DeepDFT','cuda')
    model.eval();core=FlatProbe(model).eval();s=sample('cuda')
    graph=ProbeEdges.build(s)
    chunk=40
    capacity=max(int(graph.offsets[min(i+chunk,graph.size)]-graph.offsets[i]) for i in range(0,graph.size,chunk))+100
    replay=ProbeReplay(core,chunk,capacity,atom_capacity=4)
    a=flat_density(model,core,s,graph,chunk,replay)
    torch.testing.assert_close(a,reference(model,s),rtol=3e-5,atol=3e-6)
    s2=dict(s,numbers=torch.tensor([28,8],device='cuda'),positions=s['positions']+.01)
    g2=ProbeEdges.build(s2)
    b=flat_density(model,core,s2,g2,chunk,replay)
    torch.testing.assert_close(b,reference(model,s2),rtol=3e-5,atol=3e-6)
    assert not torch.allclose(a,b)
