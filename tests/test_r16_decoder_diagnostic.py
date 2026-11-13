import torch
from density_eqgnn.compact_basis import CompactBasisConfig,reconstruct_compact_from_edges
from scripts.r16_decoder_diagnostic import decode


def test_reordered_whitened_decoder_values_and_gradients():
    torch.manual_seed(42)
    config=CompactBasisConfig(num_radial=16,lmax=4,cutoff=3.,pbc=True,
                              regular_radial=True,radial_whitening=True,radial_minimum=3./32.)
    coeff=torch.randn(1,3,400,dtype=torch.float64,requires_grad=True)
    disp=(torch.rand(257,3,dtype=torch.float64)-.5)*3.8
    disp.requires_grad_()
    batch=torch.zeros(257,dtype=torch.long)
    atoms=torch.arange(257)%3
    targets=torch.arange(257)%19
    expected=reconstruct_compact_from_edges(coeff,disp,batch,atoms,targets,19,config)
    actual=decode(coeff,disp,batch,atoms,targets,19,config,chunk=71)
    torch.testing.assert_close(actual,expected,rtol=3e-7,atol=3e-7)
    weight=torch.randn(19,dtype=torch.float64)
    grad_expected=torch.autograd.grad((expected*weight).sum(),(coeff,disp),retain_graph=True)
    grad_actual=torch.autograd.grad((actual*weight).sum(),(coeff,disp))
    for a,b in zip(grad_actual,grad_expected):
        relative=torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b)
        assert relative<2e-7
