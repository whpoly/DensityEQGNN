"""Fixed-weight DeepDFT inference: flat probe tensors and reusable execution.

No upstream files are changed. Query geometry and atom features are refreshed;
only model constants and optional CUDA execution storage persist.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from scripts.nmc_probe_holdout import atom_batch, PeriodicQueryGraph
from scripts.r16_grid_diagnostic import uniform_grid_edges


@dataclass
class ProbeEdges:
    source: torch.Tensor
    receiver: torch.Tensor
    shifts: torch.Tensor
    offsets: np.ndarray
    size: int

    @classmethod
    def build(cls, sample, local=True, cutoff=4.):
        n = len(sample['probes'])
        if local:
            e = uniform_grid_edges(sample['positions'], sample['cell'],
                                   sample['metadata']['grid_shape'], cutoff)
            source, receiver = e.neighbor, e.center
            shifts = torch.round((sample['probes'][receiver] - sample['positions'][source]
                                  + e.displacement) @ torch.linalg.inv(sample['cell'])).long()
            order = torch.argsort(receiver, stable=True)
            source, receiver, shifts = source[order], receiver[order], shifts[order]
            count = torch.bincount(receiver, minlength=n)
            offsets = torch.cat((count.new_zeros(1), count.cumsum(0))).cpu().numpy()
        else:
            g = PeriodicQueryGraph(sample['positions'], sample['probes'], sample['cell'], cutoff)
            source, shifts = g.atoms.long(), g.shifts.long()
            offsets = g.offsets.cpu().numpy()
            receiver = torch.repeat_interleave(torch.arange(n, device=source.device),
                                               g.offsets[1:] - g.offsets[:-1], output_size=len(source))
        return cls(source, receiver, shifts, offsets, n)

    def chunk(self, sample, start, stop):
        a, b = int(self.offsets[start]), int(self.offsets[stop])
        source, receiver = self.source[a:b], self.receiver[a:b]
        # Match upstream float32 geometry and batched matrix multiplication.
        shift = torch.bmm(self.shifts[a:b].float()[:, None],
                          sample['cell'].float().expand(b-a, -1, -1)).squeeze(1)
        diff = sample['probes'][receiver].float() - (sample['positions'][source].float() + shift)
        return source, receiver-start, diff


class FlatProbe(torch.nn.Module):
    """Upstream probe algebra, hoisting atom-only MLPs outside grid chunks."""
    def __init__(self, model):
        super().__init__()
        self.probe = model.probe_model

    def atoms(self, representation, capacity=None):
        scalars, vectors = representation
        projections = [m.scalar_message_mlp(s) for m,s in zip(self.probe.message_layers, scalars)]
        result = (torch.stack(scalars), torch.stack(vectors), torch.stack(projections))
        if capacity is not None:
            if scalars[0].shape[0] > capacity:
                raise ValueError('Atom capacity exceeded')
            padded = []
            for x in result:
                out = x.new_zeros((x.shape[0], capacity, *x.shape[2:]))
                out[:, :x.shape[1]].copy_(x)
                padded.append(out)
            result = tuple(padded)
        return result

    def geometry(self, diff):
        distance = diff.square().sum(1, keepdim=True).sqrt()
        direction = diff / torch.linalg.vector_norm(diff, dim=1, keepdim=True).clamp_min(1e-12)
        n = torch.arange(self.probe.distance_embedding_size, device=diff.device, dtype=diff.dtype)+1
        radial = torch.sinc(n/self.probe.cutoff*distance)*np.pi*n/self.probe.cutoff
        envelope = torch.where(distance < self.probe.cutoff,
            .5*(torch.cos(np.pi*distance/self.probe.cutoff)+1), torch.zeros_like(distance))
        return direction, radial, envelope

    def aggregate(self, layer, scalar, vector, projection, source, receiver, geometry, size):
        direction, radial, envelope = geometry
        filters = layer.filter_layer(radial)*envelope
        gates = filters*projection[source]
        gv, ge, gs = gates.split(scalar.shape[-1], dim=1)
        ms = scalar[source]*gs
        mv = vector[source]*gv[:,None] + ge[:,None]*direction[:,:,None]
        agg_s = scalar.new_zeros(size, scalar.shape[-1]).index_add_(0, receiver, ms)
        agg_v = vector.new_zeros(size, 3, scalar.shape[-1]).index_add_(0, receiver, mv)
        return agg_s, agg_v

    def forward(self, source, receiver, diff, scalars, vectors, projections, size):
        geometry = self.geometry(diff)
        s = scalars.new_zeros(size, scalars.shape[-1])
        v = vectors.new_zeros(size, 3, scalars.shape[-1])
        for k,(layer,update) in enumerate(zip(self.probe.message_layers, self.probe.scalar_vector_update)):
            a,b = self.aggregate(layer,scalars[k],vectors[k],projections[k],
                                 source,receiver,geometry,size)
            gs,gv = layer.update_gate_mlp(a).split(s.shape[-1],dim=1)
            s = gs*s + (1.-gs)*a
            v = gv[:,None]*v + (1.-gv[:,None])*b
            s,v = update(s,v)
        return self.probe.readout_function(s).squeeze(1)


@torch.no_grad()
def flat_density(model, core, sample, graph=None, chunk=8192, replay=None):
    if graph is None:
        graph = ProbeEdges.build(sample)
    rep = model.atom_model(atom_batch(sample))
    states = core.atoms(rep, replay.atom_capacity if replay is not None else None)
    if replay is not None:
        replay.set_atoms(states)
    out = sample['positions'].new_empty(graph.size,dtype=torch.float32)
    for start in range(0,graph.size,chunk):
        stop = min(start+chunk,graph.size)
        inputs = graph.chunk(sample,start,stop)
        value = (core(*inputs,*states,stop-start) if replay is None
                 else replay.query(*inputs,stop-start))
        out[start:stop].copy_(value)
    return out


class ProbeReplay:
    """One padded probe CUDA Graph; copies new features and geometry each call."""
    def __init__(self, core, query_capacity, edge_capacity, atom_capacity=64):
        self.query_capacity = query_capacity
        self.edge_capacity = edge_capacity
        self.atom_capacity = atom_capacity
        self.padding_distance = core.probe.cutoff+1.
        device = next(core.parameters()).device
        width = core.probe.hidden_state_size
        layers = core.probe.num_interactions
        self.source = torch.zeros(edge_capacity,device=device,dtype=torch.long)
        self.receiver = torch.zeros_like(self.source)
        self.diff = torch.zeros(edge_capacity,3,device=device)
        self.diff[:,0] = core.probe.cutoff+1.
        self.states = (torch.zeros(layers,atom_capacity,width,device=device),
                       torch.zeros(layers,atom_capacity,3,width,device=device),
                       torch.zeros(layers,atom_capacity,3*width,device=device))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                self.output = core(self.source,self.receiver,self.diff,*self.states,query_capacity)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.output = core(self.source,self.receiver,self.diff,*self.states,query_capacity)

    def set_atoms(self, states):
        for destination,value in zip(self.states,states):
            destination.copy_(value)

    def query(self, source, receiver, diff, size):
        n = len(source)
        if n>self.edge_capacity or size>self.query_capacity:
            raise ValueError('Replay capacity exceeded; rebuild with larger capacities')
        self.source.zero_(); self.receiver.zero_(); self.diff.zero_()
        self.diff[:,0].fill_(self.padding_distance)
        self.source[:n].copy_(source); self.receiver[:n].copy_(receiver); self.diff[:n].copy_(diff)
        self.graph.replay()
        # Caller must consume before next replay, as with a workspace buffer.
        return self.output[:size]


class NeighborReuse:
    """Fixed-cell/grid skin cache. Does not cache any learned representations."""
    def __init__(self, sample, skin=.2):
        if skin <= 0:
            raise ValueError('Positive skin required')
        self.skin = skin
        self.positions = sample['positions'].clone()
        self.cell = sample['cell'].clone()
        self.probes = sample['probes'].clone()
        self.shape = tuple(sample['metadata']['grid_shape'])
        self.graph = ProbeEdges.build(sample,cutoff=4.+skin)

    def get(self, sample):
        if (sample['positions'].shape != self.positions.shape or
            tuple(sample['metadata']['grid_shape']) != self.shape or
            not torch.equal(sample['cell'],self.cell) or
            not torch.equal(sample['probes'],self.probes)):
            raise ValueError('Changed cell, grid, or atom count: rebuild neighbor cache')
        if float((sample['positions']-self.positions).norm(dim=1).max()) > self.skin/2:
            raise ValueError('Motion exceeds skin bound: rebuild neighbor cache')
        return self.graph
