import torch
from scripts.r16_grid_diagnostic import uniform_grid_edges
from scripts.r16_inference_fast import decode_grouped
from scripts.r16_inference_graph import decode_uniform,decode_angular_first
from scripts.fit_nmc_holdout_oracle import BASIS


def test_direct_grid_decoder_skew_cell_images_values_and_gradients():
    torch.set_num_threads(2)
    torch.manual_seed(152)
    cell=torch.tensor([[2.6,0.,0.],[1.1,2.9,0.],[.5,.7,3.1]],dtype=torch.float64)
    shape=(7,8,9)
    positions=(torch.tensor([[1.1,-.2,.45],[.32,.64,.11]],dtype=torch.float64)@cell).requires_grad_()
    coeff=torch.randn(1,2,400,dtype=torch.float64,requires_grad=True)
    edges=uniform_grid_edges(positions,cell,shape,BASIS.cutoff)
    expected=decode_grouped(coeff,-edges.displacement,edges.batch,edges.neighbor,edges.center,504,BASIS)
    actual=decode_uniform(coeff,positions,cell,shape,BASIS,atoms_per_chunk=1)
    torch.testing.assert_close(actual,expected,rtol=3e-7,atol=3e-7)
    weights=torch.randn(504,dtype=torch.float64)
    a=torch.autograd.grad((expected*weights).sum(),(coeff,positions),retain_graph=True)
    b=torch.autograd.grad((actual*weights).sum(),(coeff,positions))
    for x,y in zip(a,b):
        assert torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x)<2e-7


def test_angular_first_contraction_values_and_gradients():
    torch.manual_seed(178)
    coeff=torch.randn(2,4,400,dtype=torch.float64,requires_grad=True)
    disp=((torch.rand(137,3,dtype=torch.float64)-.5)*4).requires_grad_()
    batch=torch.arange(137)%2
    atoms=torch.arange(137)%3
    targets=torch.arange(137)%17
    expected=decode_grouped(coeff,disp,batch,atoms,targets,17,BASIS,atoms_per_chunk=3,point_chunk=13)
    actual=decode_angular_first(coeff,disp,batch,atoms,targets,17,BASIS,atoms_per_chunk=3,point_chunk=13)
    torch.testing.assert_close(actual,expected,rtol=3e-7,atol=3e-7)
    weights=torch.randn(17,dtype=torch.float64)
    a=torch.autograd.grad((expected*weights).sum(),(coeff,disp),retain_graph=True)
    b=torch.autograd.grad((actual*weights).sum(),(coeff,disp))
    for x,y in zip(a,b):
        assert torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x)<2e-7
