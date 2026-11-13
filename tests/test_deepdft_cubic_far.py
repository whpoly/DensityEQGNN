import itertools

import pytest
import torch

from scripts.deepdft_cubic_far import interpolation_weights, periodic_expand, expanded_far_density
from scripts.deepdft_near_far import near_far_density
from scripts.deepdft_inference_fast import flat_density
from test_deepdft_near_far import fixture


@pytest.mark.parametrize('stride',[2,4])
def test_cubic_tensor_product_and_periodic_boundary(stride):
    torch.manual_seed(12)
    shape = (4,5,6)
    values = torch.randn(*shape,2,dtype=torch.float64)
    expanded = periodic_expand(values,stride,'cubic')
    torch.testing.assert_close(expanded[::stride,::stride,::stride],values,rtol=0,atol=0)
    for coord in [(1,2,3),tuple(n*stride-1 for n in shape),(0,0,0)]:
        choices = [list(zip(*interpolation_weights(c%stride,stride,'cubic'))) for c in coord]
        direct = torch.zeros(2,dtype=torch.float64)
        for x,y,z in itertools.product(*choices):
            offsets = (x,y,z)
            index = tuple((coord[a]//stride+offsets[a][0])%shape[a] for a in range(3))
            direct += values[index]*x[1]*y[1]*z[1]
        torch.testing.assert_close(expanded[coord],direct,rtol=1e-12,atol=1e-12)


def test_cubic_reproduces_polynomials_and_constants():
    for stride in (2,4):
        for phase in range(stride):
            nodes,weights = interpolation_weights(phase,stride,'cubic')
            for degree in range(4):
                actual = sum(w*x**degree for x,w in zip(nodes,weights))
                assert abs(actual-(phase/stride)**degree)<1e-12
    ones = torch.ones(4,5,6,2)
    torch.testing.assert_close(periodic_expand(ones,2,'cubic'),torch.ones(8,10,12,2))


@torch.no_grad()
def test_expanded_density_matches_linear_and_preserves_cubic_anchors():
    model,core,sample = fixture()
    reference = flat_density(model,core,sample,chunk=31)
    linear = expanded_far_density(model,core,sample,order='linear',chunk=29)
    torch.testing.assert_close(linear,near_far_density(model,core,sample,chunk=37),rtol=3e-5,atol=3e-6)
    cubic = expanded_far_density(model,core,sample,order='cubic',chunk=29)
    shape = sample['metadata']['grid_shape']
    torch.testing.assert_close(cubic.reshape(shape)[::2,::2,::2],reference.reshape(shape)[::2,::2,::2],rtol=3e-5,atol=3e-6)
    exact = expanded_far_density(model,core,sample,order='cubic',stride=1,chunk=29)
    torch.testing.assert_close(exact,reference,rtol=3e-5,atol=3e-6)
