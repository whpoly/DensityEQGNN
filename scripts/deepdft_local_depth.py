"""Depth extension of the existing local query encoder; atom projections stay shared."""
import importlib

import torch

from scripts.deepdft_mixed_embedding import MixedProbe


class DeepLocalProbe(MixedProbe):
    def __init__(self, width=128, depth=2):
        if depth < 1:
            raise ValueError('Positive depth required')
        # Preserve the existing one-layer initialization and state-dict keys.
        super().__init__('local', width)
        self.depth = depth
        if depth > 1:
            module = importlib.import_module('densitymodel')
            extra = module.PainnProbeMessageModel(depth-1, width, 4., 30)
            self.probe.message_layers.extend(extra.message_layers)
            self.probe.scalar_vector_update.extend(extra.scalar_vector_update)
        self.probe.num_interactions = depth

    def atoms(self, representation, context_override=None):
        s = self.scalar_projection(torch.cat(representation[0], -1))
        v = self.vector_projection(torch.cat(representation[1], -1))
        projections = torch.stack([m.scalar_message_mlp(s) for m in self.probe.message_layers])
        return s, v, projections, s.new_zeros(2*s.shape[-1])

    def forward(self, source, receiver, diff, s, v, projections, context, size):
        geometry = self.geometry(diff)
        qs = s.new_zeros(size, s.shape[-1])
        qv = v.new_zeros(size, 3, s.shape[-1])
        for k, (message, update) in enumerate(zip(self.probe.message_layers,
                                                self.probe.scalar_vector_update)):
            a,b = self.aggregate(message,s,v,projections[k],source,receiver,geometry,size)
            gs,gv = message.update_gate_mlp(a).split(s.shape[-1],-1)
            qs,qv = update(gs*qs+(1-gs)*a, gv[:,None]*qv+(1-gv[:,None])*b)
        return self.probe.readout_function(qs).squeeze(-1)
