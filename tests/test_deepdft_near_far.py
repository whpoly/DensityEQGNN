import torch

from scripts.deepdft_inference_fast import FlatProbe, ProbeEdges, flat_density
from scripts.deepdft_near_far import (near_weight, weighted_geometry, near_far_density,
    interpolation_stencil, interpolate)
from scripts.nmc_probe_holdout import DEFAULT_ROOT, official_model, atom_batch


def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(70)
    cell = torch.tensor([[2.6, 0., 0.], [1.1, 2.9, 0.], [.5, .7, 3.1]], dtype=torch.float64)
    shape = (6, 4, 8)
    frac = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float64)/n for n in shape], indexing='ij'), -1)
    sample = {'positions':torch.tensor([[1.1, -.2, .45], [.32, .64, .11]], dtype=torch.float64)@cell,
        'cell':cell, 'probes':frac.reshape(-1,3)@cell, 'numbers':torch.tensor([8,28]),
        'metadata':{'grid_shape':shape}}
    model, _ = official_model(DEFAULT_ROOT/'DeepDFT', 'cpu')
    model.eval()
    return model, FlatProbe(model).eval(), sample


@torch.no_grad()
def test_split_aggregates_reconstruct_each_layer():
    model, core, sample = fixture()
    states = core.atoms(model.atom_model(atom_batch(sample)))
    inputs = ProbeEdges.build(sample).chunk(sample, 0, len(sample['probes']))
    source, receiver, diff = inputs
    for k, layer in enumerate(core.probe.message_layers):
        args = (layer, *(state[k] for state in states), source, receiver)
        full = core.aggregate(*args, core.geometry(diff), len(sample['probes']))
        near = core.aggregate(*args, weighted_geometry(core, diff, 2., 1.), len(sample['probes']))
        far = core.aggregate(*args, weighted_geometry(core, diff, 2., 1., True), len(sample['probes']))
        for a,b,c in zip(full, near, far):
            torch.testing.assert_close(a, b+c, rtol=2e-5, atol=2e-6)


@torch.no_grad()
def test_density_stride_one_exact_and_stride_two_anchor_exact():
    model, core, sample = fixture()
    reference = flat_density(model, core, sample, chunk=31)
    exact = near_far_density(model, core, sample, stride=1, chunk=29)
    torch.testing.assert_close(exact, reference, rtol=3e-5, atol=3e-6)
    approx = near_far_density(model, core, sample, stride=2, chunk=37)
    shape = sample['metadata']['grid_shape']
    torch.testing.assert_close(approx.reshape(shape)[::2,::2,::2],
                              reference.reshape(shape)[::2,::2,::2], rtol=3e-5, atol=3e-6)
    changed = dict(sample, positions=sample['positions']+.03, numbers=torch.tensor([28,8]))
    refreshed = near_far_density(model, core, changed, stride=1, chunk=29)
    torch.testing.assert_close(refreshed, flat_density(model,core,changed,chunk=31), rtol=3e-5, atol=3e-6)
    assert not torch.allclose(refreshed, exact)


def test_interpolation_wrap_and_complementary_weights():
    shape = (4, 4, 4)
    field = torch.arange(8.).reshape(8,1)
    out = interpolate(field, interpolation_stencil(torch.arange(64), shape, 2)).reshape(shape)
    torch.testing.assert_close(out[::2,::2,::2], field.reshape(2,2,2))
    assert out[3,3,3] == field.mean()
    assert out[1,0,0] == 2.
    d = torch.tensor([0.,1.,1.5,2.,3.,4.])
    w = near_weight(d, 2., 1.)
    torch.testing.assert_close(w, torch.tensor([1.,1.,.5,0.,0.,0.]))
