import torch
from scripts.benchmark_deepdft_sparse import periodic_double


def test_periodic_interpolation_samples_wrap_and_constant():
    x=torch.arange(24,dtype=torch.float64).reshape(2,3,4)
    y=periodic_double(x)
    torch.testing.assert_close(y[::2,::2,::2],x)
    torch.testing.assert_close(y[-1,0,0],(x[-1,0,0]+x[0,0,0])/2)
    torch.testing.assert_close(y[-1,-1,-1],x[[0,-1]][:,[0,-1]][:,:,[0,-1]].mean())
    torch.testing.assert_close(periodic_double(torch.ones_like(x)),torch.ones(4,6,8,dtype=torch.float64))
