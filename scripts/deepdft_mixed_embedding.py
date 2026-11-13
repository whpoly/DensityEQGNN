"""Small local query encoder with an optional once-per-structure global embedding.

The atom encoder is supplied by the caller. No query labels enter embeddings.
"""
import importlib
from types import SimpleNamespace

import torch
from torch import nn

from scripts.deepdft_inference_fast import FlatProbe, ProbeEdges
from scripts.nmc_probe_holdout import atom_batch


def descriptor(rep):
    s, v = rep[0][-1], rep[1][-1]
    return torch.cat((s.mean(0), s.std(0, unbiased=False), v.norm(dim=1).mean(0)))


class MixedProbe(FlatProbe):
    def __init__(self, mode='global', width=64, global_width=32):
        if mode not in ('local', 'constant', 'global'):
            raise ValueError(mode)
        module = importlib.import_module('densitymodel')
        probe = module.PainnProbeMessageModel(1, width, 4., 30)
        super().__init__(SimpleNamespace(probe_model=probe))
        self.mode = mode
        self.scalar_projection = nn.Linear(3 * 128, width)
        self.vector_projection = nn.Linear(3 * 128, width, bias=False)
        self.global_encoder = nn.Sequential(nn.Linear(384, global_width), nn.SiLU(),
                                            nn.Linear(global_width, 2 * width))
        nn.init.zeros_(self.global_encoder[-1].weight)
        nn.init.zeros_(self.global_encoder[-1].bias)
        self.register_buffer('global_mean', torch.zeros(384))
        self.register_buffer('global_scale', torch.ones(384))
        if mode == 'local':
            self.global_encoder.requires_grad_(False)

    def atoms(self, representation, context_override=None):
        s = self.scalar_projection(torch.cat(representation[0], -1))
        v = self.vector_projection(torch.cat(representation[1], -1))
        p = self.probe.message_layers[0].scalar_message_mlp(s)
        if self.mode == 'local':
            context = s.new_zeros(2 * s.shape[-1])
        else:
            desc = descriptor(representation) if context_override is None else context_override
            z = (desc - self.global_mean) / self.global_scale
            if self.mode == 'constant':
                z = torch.zeros_like(z)
            context = self.global_encoder(z)
        return s, v, p, context

    def forward(self, source, receiver, diff, s, v, projection, context, size):
        layer = self.probe.message_layers[0]
        a, b = self.aggregate(layer, s, v, projection, source, receiver,
                              self.geometry(diff), size)
        gs, gv = layer.update_gate_mlp(a).split(s.shape[-1], -1)
        a, b = self.probe.scalar_vector_update[0]((1-gs)*a, (1-gv[:, None])*b)
        scale, shift = context.chunk(2)
        a = a * (1 + .1 * torch.tanh(scale)) + shift
        return self.probe.readout_function(a).squeeze(-1)


class TrainingEdges:
    """Compact geometry-only CSR; supports repeated/random query indices."""
    def __init__(self, graph):
        self.source = graph.source.int()
        self.shifts = graph.shifts.short()
        self.offsets = torch.as_tensor(graph.offsets, device=self.source.device)

    def select(self, sample, ids):
        counts = self.offsets[ids+1] - self.offsets[ids]
        n = int(counts.sum())
        receiver = torch.repeat_interleave(torch.arange(len(ids), device=ids.device),
                                           counts, output_size=n)
        starts = counts.cumsum(0) - counts
        edges = self.offsets[ids][receiver] + torch.arange(n, device=ids.device) - starts[receiver]
        source = self.source[edges].long()
        shift = self.shifts[edges].float() @ sample['cell'].float()
        diff = sample['probes'][ids[receiver]].float() - (sample['positions'][source].float()+shift)
        return source, receiver, diff


@torch.no_grad()
def mixed_density(backbone, head, sample, chunk=8192):
    # New-input inference includes atom graph, atom encoder, global pooling,
    # projections, query graph and all point predictions. Nothing learned cached.
    rep = backbone.atom_model(atom_batch(sample))
    states = head.atoms(rep)
    graph = ProbeEdges.build(sample)
    output = sample['positions'].new_empty(graph.size, dtype=torch.float32)
    for start in range(0, graph.size, chunk):
        stop = min(start+chunk, graph.size)
        output[start:stop] = head(*graph.chunk(sample, start, stop), *states, stop-start)
    return output
