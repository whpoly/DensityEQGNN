"""Separable periodic interpolation of fixed-weight DeepDFT far messages.

Expand each spatial axis once instead of gathering 64 corners at every query.
This trades dense feature storage for fewer interpolation kernels. The linear
control uses the same expansion path to separate implementation and order gains.
"""
import torch

from scripts.deepdft_inference_fast import ProbeEdges
from scripts.deepdft_near_far import far_fields, weighted_geometry
from scripts.nmc_probe_holdout import atom_batch


def interpolation_weights(phase, stride, order):
    t = phase/stride
    if order == 'linear':
        return (0, 1), (1-t, t)
    if order != 'cubic':
        raise ValueError('Expected linear or cubic interpolation')
    # Lagrange polynomial through nodes -1,0,1,2; exact for degree <=3.
    return (-1, 0, 1, 2), (-t*(t-1)*(t-2)/6,
        (t+1)*(t-1)*(t-2)/2, -(t+1)*t*(t-2)/2, (t+1)*t*(t-1)/6)


def periodic_expand(values, stride=2, order='cubic'):
    """Channel-last tensor [nx,ny,nz,C]; wrap independently in fractional axes."""
    if stride < 1 or int(stride) != stride:
        raise ValueError('Positive integer stride required')
    interpolation_weights(0, stride, order)
    if stride == 1:
        return values
    for axis in range(3):
        shape = list(values.shape)
        shape[axis] *= stride
        out = values.new_empty(shape)
        index = torch.arange(values.shape[axis],device=values.device)
        for phase in range(stride):
            destination = [slice(None)]*4
            destination[axis] = slice(phase,None,stride)
            if phase == 0:
                out[tuple(destination)].copy_(values)
                continue
            offsets, weights = interpolation_weights(phase,stride,order)
            accum = torch.zeros_like(values)
            for offset, weight in zip(offsets,weights):
                accum.add_(values.index_select(axis,(index+offset)%values.shape[axis]),alpha=weight)
            out[tuple(destination)].copy_(accum)
            del accum
        values = out
    return values


@torch.no_grad()
def expanded_far_density(model, core, sample, radius=2., transition=1., stride=2,
                         order='cubic', chunk=4096):
    states = core.atoms(model.atom_model(atom_batch(sample)))
    scalars, vectors, projections = states
    width = scalars.shape[-1]
    shape = tuple(sample['metadata']['grid_shape'])
    coarse_shape = tuple(n//stride for n in shape)
    coarse = far_fields(core,sample,states,radius,transition,stride,chunk)
    fields = [periodic_expand(coarse[k].reshape(*coarse_shape,4*width),stride,order).reshape(-1,4*width)
              for k in range(len(scalars))]
    del coarse
    graph = ProbeEdges.build(sample,cutoff=radius)
    out = scalars.new_empty(graph.size)
    for start in range(0,graph.size,chunk):
        stop = min(start+chunk,graph.size)
        size = stop-start
        source,receiver,diff = graph.chunk(sample,start,stop)
        geometry = weighted_geometry(core,diff,radius,transition)
        s,v = scalars.new_zeros(size,width),vectors.new_zeros(size,3,width)
        for k,(layer,update) in enumerate(zip(core.probe.message_layers,core.probe.scalar_vector_update)):
            a,b = core.aggregate(layer,scalars[k],vectors[k],projections[k],source,receiver,geometry,size)
            shared = fields[k][start:stop]
            a,b = a+shared[:,:width],b+shared[:,width:].reshape(size,3,width)
            gs,gv = layer.update_gate_mlp(a).split(width,dim=1)
            s,v = update(gs*s+(1-gs)*a,gv[:,None]*v+(1-gv[:,None])*b)
        out[start:stop] = core.probe.readout_function(s).squeeze(1)
    return out
