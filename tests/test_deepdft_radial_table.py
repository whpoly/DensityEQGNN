import pytest
import torch

from scripts.deepdft_factored_messages import FactoredProbe
from scripts.deepdft_radial_table import RadialTableProbe
from scripts.nmc_probe_holdout import DEFAULT_ROOT,official_model


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@torch.no_grad()
def test_table_fused_aggregation_boundaries_empty_queries_and_refresh():
    torch.manual_seed(17)
    model,_ = official_model(DEFAULT_ROOT/'DeepDFT','cuda')
    model.eval().requires_grad_(False)
    exact = FactoredProbe(model)
    table = RadialTableProbe(model)
    fused = RadialTableProbe(model,fused=True)
    source = torch.tensor([0,1,2,0,2,1,0,2],device='cuda')
    receiver = torch.tensor([0,0,0,1,1,3,3,3],device='cuda')
    radii = torch.tensor([0.,1e-7,.2,3.9999,4.,4.1,1.72,2.91],device='cuda')
    direction = torch.randn(8,3,device='cuda')
    direction = direction/direction.norm(dim=1,keepdim=True)
    diff = direction*radii[:,None]
    for change in (0,1):
        scalars = [torch.randn(3,128,device='cuda') for _ in range(3)]
        vectors = [torch.randn(3,3,128,device='cuda') for _ in range(3)]
        ss,vv,pp = exact.atoms((scalars,vectors))
        if change:
            diff = diff+.013
        geometries = [c.geometry(diff) for c in (exact,table,fused)]
        for k,layer in enumerate(exact.probe.message_layers):
            results = [c.aggregate(layer,ss[k],vv[k],pp[k],source,receiver,g,5)
                for c,g in zip((exact,table,fused),geometries)]
            for reference,interpolated,combined in zip(*results):
                torch.testing.assert_close(interpolated,reference,rtol=5e-5,atol=3e-5)
                torch.testing.assert_close(combined,interpolated,rtol=2e-5,atol=2e-6)
                assert torch.count_nonzero(combined[[2,4]])==0
        # Same distance tensor can safely reuse the within-call CSR offsets.
        again = fused.aggregate(exact.probe.message_layers[0],ss[0],vv[0],pp[0],source,receiver,geometries[2],5)
        assert all(torch.isfinite(x).all() for x in again)
    empty = torch.empty(0,dtype=torch.long,device='cuda')
    geo = fused.geometry(torch.empty(0,3,device='cuda'))
    result = fused.aggregate(exact.probe.message_layers[0],ss[0],vv[0],pp[0],empty,empty,geo,7)
    assert all(torch.count_nonzero(x)==0 for x in result)
