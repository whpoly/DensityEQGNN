import numpy as np
import torch
from scipy.interpolate import CubicSpline
from density_eqgnn.compact_basis import CompactBasisConfig
from scripts.r16_decoder_diagnostic import decode
from scripts.r16_inference_fast import decode_atomwise,decode_grouped,TorchSAD
from scripts.evaluate_vasp_sad_nmc import AtomicFormFactor,sad_density


def test_grouped_decoder_arbitrary_order_duplicate_targets_and_gradients():
    torch.manual_seed(13)
    config=CompactBasisConfig(num_radial=16,lmax=4,cutoff=3.,pbc=True,
                              regular_radial=True,radial_whitening=True,radial_minimum=3/32)
    coeff=torch.randn(2,3,400,dtype=torch.float64,requires_grad=True)
    disp=((torch.rand(131,3,dtype=torch.float64)-.5)*4).requires_grad_()
    batch=torch.arange(131)%2
    atoms=torch.arange(131)%3
    targets=torch.arange(131)%17
    expected=decode(coeff,disp,batch,atoms,targets,17,config,chunk=73)
    actual=decode_atomwise(coeff,disp,batch,atoms,targets,17,config,chunk=19)
    torch.testing.assert_close(actual,expected,rtol=3e-7,atol=3e-7)
    weights=torch.randn(17,dtype=torch.float64)
    a=torch.autograd.grad((expected*weights).sum(),(coeff,disp),retain_graph=True)
    b=torch.autograd.grad((actual*weights).sum(),(coeff,disp))
    for x,y in zip(a,b):
        assert torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x)<2e-7


def test_batched_grouped_decoder_padding_empty_atoms_and_gradients():
    torch.manual_seed(37)
    config=CompactBasisConfig(num_radial=16,lmax=4,cutoff=3.,pbc=True,
                              regular_radial=True,radial_whitening=True,radial_minimum=3/32)
    coeff=torch.randn(2,4,400,dtype=torch.float64,requires_grad=True)
    disp=((torch.rand(137,3,dtype=torch.float64)-.5)*4).requires_grad_()
    batch=torch.arange(137)%2
    atoms=torch.arange(137)%3
    targets=torch.arange(137)%17
    expected=decode(coeff,disp,batch,atoms,targets,17,config)
    actual=decode_grouped(coeff,disp,batch,atoms,targets,17,config,atoms_per_chunk=3,point_chunk=13)
    torch.testing.assert_close(actual,expected,rtol=3e-7,atol=3e-7)
    weights=torch.randn(17,dtype=torch.float64)
    a=torch.autograd.grad((expected*weights).sum(),(coeff,disp),retain_graph=True)
    b=torch.autograd.grad((actual*weights).sum(),(coeff,disp))
    for x,y in zip(a,b):
        assert torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x)<2e-7


def test_torch_sad_matches_numpy_skew_cell_odd_even_grids():
    factors={}
    for symbol,zval,cutoff in [('Li',1.,12.),('O',6.,10.)]:
        x=np.arange(1000,dtype=float)
        values=zval*np.exp(-.003*x*x)
        factors[symbol]=AtomicFormFactor(symbol,'synthetic',zval,cutoff,values,
                                         CubicSpline(x,values,bc_type='natural',extrapolate=False))
    cell=np.array([[2.1,0.,0.],[.6,2.2,0.],[.3,.4,2.5]])
    frac=np.array([[.23,.13,.49],[1.2,-.3,.7],[.63,.73,.29]])
    for shape in [(12,11,10),(11,10,13)]:
        sample={'cell':torch.tensor(cell),'positions':torch.tensor(frac@cell),
                'numbers':torch.tensor([3,8,8]),'metadata':{'grid_shape':shape,'volume':np.linalg.det(cell)}}
        actual=TorchSAD(factors,device='cpu')(sample).double()
        expected=sad_density(np.array(['Li','O','O']),frac,cell,shape,factors).flatten()/np.linalg.det(cell)
        torch.testing.assert_close(actual,torch.tensor(expected).float().double(),rtol=2e-6,atol=2e-7)
