"""Fixed-weight, per-layer DeepDFT message sharing on a periodic coarse grid.

Only far aggregated messages are interpolated. Near messages, query gates,
PaiNN updates and readout retain the original model and channel count.
"""
import itertools

import torch

from scripts.deepdft_inference_fast import ProbeEdges
from scripts.nmc_probe_holdout import atom_batch


def near_weight(distance, radius, transition):
    if not 0 < transition <= radius <= 4.:
        raise ValueError('Require 0 < transition <= radius <= 4 Angstrom')
    t = ((distance - (radius-transition))/transition).clamp(0, 1)
    return .5*(1+torch.cos(torch.pi*t))


def weighted_geometry(core, diff, radius, transition, far=False):
    direction, radial, envelope = core.geometry(diff)
    weight = near_weight(diff.norm(dim=1, keepdim=True), radius, transition)
    return direction, radial, envelope*((1-weight) if far else weight)


def coarse_sample(sample, stride):
    shape = tuple(sample['metadata']['grid_shape'])
    if stride < 1 or any(n % stride for n in shape):
        raise ValueError('Positive stride must divide all grid dimensions')
    return dict(sample,
        probes=sample['probes'].reshape(*shape, 3)[::stride, ::stride, ::stride].reshape(-1, 3),
        metadata=dict(sample['metadata'], grid_shape=tuple(n//stride for n in shape)))


def interpolation_stencil(ids, shape, stride):
    """Periodic trilinear weights in fractional grid coordinates, including wrap."""
    coords = torch.stack((ids//(shape[1]*shape[2]), ids//shape[2] % shape[1], ids % shape[2]), 1)
    base = coords//stride
    frac = (coords % stride).float()/stride
    dimensions = torch.tensor([n//stride for n in shape], device=ids.device)
    corners, weights = [], []
    for bits in itertools.product((0, 1), repeat=3):
        offset = torch.tensor(bits, device=ids.device)
        xyz = (base+offset) % dimensions
        corners.append((xyz[:, 0]*dimensions[1]+xyz[:, 1])*dimensions[2]+xyz[:, 2])
        weights.append(torch.where(offset.bool()[None], frac, 1-frac).prod(1))
    return corners, weights


def interpolate(field, stencil):
    corners, weights = stencil
    out = torch.zeros((len(corners[0]), field.shape[-1]), device=field.device, dtype=field.dtype)
    for ids, weight in zip(corners, weights):
        out.add_(field[ids]*weight[:, None])
    return out


@torch.no_grad()
def far_fields(core, sample, states, radius, transition, stride, chunk):
    coarse = coarse_sample(sample, stride)
    graph = ProbeEdges.build(coarse, cutoff=core.probe.cutoff)
    scalars, vectors, projections = states
    width = scalars.shape[-1]
    fields = scalars.new_empty(len(scalars), graph.size, 4*width)
    for start in range(0, graph.size, chunk):
        stop = min(start+chunk, graph.size)
        source, receiver, diff = graph.chunk(coarse, start, stop)
        keep = diff.norm(dim=1) > radius-transition
        source, receiver, diff = source[keep], receiver[keep], diff[keep]
        geometry = weighted_geometry(core, diff, radius, transition, far=True)
        for k, layer in enumerate(core.probe.message_layers):
            a, b = core.aggregate(layer, scalars[k], vectors[k], projections[k],
                                 source, receiver, geometry, stop-start)
            fields[k, start:stop, :width] = a
            fields[k, start:stop, width:] = b.flatten(1)
    return fields


@torch.no_grad()
def near_far_density(model, core, sample, radius=2., transition=1., stride=2,
                     chunk=4096, near_only=False):
    # Every call refreshes atom features, both graphs and the coarse feature field.
    states = core.atoms(model.atom_model(atom_batch(sample)))
    scalars, vectors, projections = states
    fields = None if near_only else far_fields(core, sample, states, radius, transition, stride, chunk)
    graph = ProbeEdges.build(sample, cutoff=radius)
    shape = tuple(sample['metadata']['grid_shape'])
    width = scalars.shape[-1]
    out = scalars.new_empty(graph.size)
    for start in range(0, graph.size, chunk):
        stop = min(start+chunk, graph.size)
        size = stop-start
        source, receiver, diff = graph.chunk(sample, start, stop)
        geometry = weighted_geometry(core, diff, radius, transition)
        stencil = None if near_only else interpolation_stencil(
            torch.arange(start, stop, device=scalars.device), shape, stride)
        s, v = scalars.new_zeros(size, width), vectors.new_zeros(size, 3, width)
        for k, (layer, update) in enumerate(zip(core.probe.message_layers, core.probe.scalar_vector_update)):
            a, b = core.aggregate(layer, scalars[k], vectors[k], projections[k],
                                 source, receiver, geometry, size)
            if fields is not None:
                shared = interpolate(fields[k], stencil)
                a = a+shared[:, :width]
                b = b+shared[:, width:].reshape(size, 3, width)
            gs, gv = layer.update_gate_mlp(a).split(width, dim=1)
            s, v = update(gs*s+(1-gs)*a, gv[:, None]*v+(1-gv[:, None])*b)
        out[start:stop] = core.probe.readout_function(s).squeeze(1)
    return out
