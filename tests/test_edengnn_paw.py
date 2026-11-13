import torch
from e3nn import o3
from scripts.deepdft_paw import IRREPS,WIDTH,BLOCKS
from scripts.edengnn_paw import EdenPAW


def test_official_augmentation_layer_nontrivial_covariance_and_mask():
    torch.manual_seed(71)
    model = EdenPAW(torch.zeros(29,WIDTH),torch.ones(29,WIDTH),pretrained=False)
    for p in model.aug.parameters():
        torch.nn.init.normal_(p,std=.1)
    incoming = o3.Irreps(model.encoder_config['conv']['irreps_node_features'])
    for (mul,ir),section in zip(incoming,incoming.slices()):
        model.feature_rms[section] = (torch.rand(mul)+.3).repeat_interleave(ir.dim)
    x = torch.randn(5,incoming.dim)
    z = torch.tensor([3,8,25,27,28])
    out = model.readout(x,z)
    # The adapter must equal the exact official linear map with support mask.
    reference = model.aug.conv_to_output_node.linear(x/model.feature_rms)*model.supported
    torch.testing.assert_close(out,reference)
    for rotation in (o3.rand_matrix(),-o3.rand_matrix()):
        actual = model.readout(x@incoming.D_from_matrix(rotation).T,z)
        expected = out@IRREPS.D_from_matrix(rotation).T
        torch.testing.assert_close(actual,expected,rtol=3e-4,atol=1e-5)
    for b in BLOCKS:
        if b['l']>2:
            assert torch.count_nonzero(out[:,b['start']:b['stop']])==0
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.aug.parameters())
