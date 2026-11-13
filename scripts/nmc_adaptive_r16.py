"""Atom-local invariant-conditioned equivariant residual readout for Eden/R16."""
from pathlib import Path
import math

import numpy as np
import torch
from e3nn import o3

from scripts.nmc_eden_r16 import EdenR16
from scripts.nmc_probe_holdout import sha256

ROOT = Path('runs/nmc_adaptive_r16_2026-10-29')
REPORT = Path('reports/nmc_adaptive_r16_2026-10-29')
BASE = Path('runs/nmc_eden_r16_2026-10-29/eden_r16_sad/best.pt')
BASE_SHA = '261a6a4b1568cf821e3a9dd227817bda2b884da40ff6bda1adab143fe2644358'
ARMS = ('continued', 'adaptive')
STEPS = 8000


class AdaptiveEdenR16(EdenR16):
    """Keep the 400-coefficient decoder; predict extra mixing weights per atom.

    Only even scalars and element identity condition the weights, preserving O(3)
    equivariance. Each degree mixes matching-parity channels, identically for
    every m. The final layer starts at zero for an exact pretrained warm start.
    """
    def __init__(self):
        super().__init__()
        irreps = o3.Irreps(self.encoder_config['conv']['irreps_node_features'])
        self.channels = []
        self.scalar_slices = []
        for (mul, ir), section in zip(irreps, irreps.slices()):
            if ir.l == 0 and ir.p == 1:
                self.scalar_slices.append(section)
            if ir.l <= 4 and ir.p == (-1)**ir.l:
                self.channels.append((ir.l, mul, section))
        scalar_dim = sum(s.stop-s.start for s in self.scalar_slices)
        self.context_norm = torch.nn.LayerNorm(scalar_dim)
        self.element_context = torch.nn.Embedding(100, 16, padding_idx=0)
        self.mixing = torch.nn.Sequential(
            torch.nn.Linear(scalar_dim+16, 64), torch.nn.SiLU(),
            torch.nn.Linear(64, sum(16*mul for _, mul, _ in self.channels)))
        torch.nn.init.zeros_(self.mixing[-1].weight)
        torch.nn.init.zeros_(self.mixing[-1].bias)

    def adaptive_degree(self, features, numbers):
        scalars = torch.cat([features[:, s] for s in self.scalar_slices], dim=-1)
        context = torch.cat((self.context_norm(scalars), self.element_context(numbers)), dim=-1)
        weights = self.mixing(context)
        pieces = [features.new_zeros((len(features),16,2*l+1)) for l in range(5)]
        offset = 0
        for l, mul, section in self.channels:
            w = weights[:, offset:offset+16*mul].reshape(-1,16,mul)
            x = features[:, section].reshape(-1,mul,2*l+1)
            pieces[l] = pieces[l] + torch.einsum('ani,aim->anm',w,x)/math.sqrt(mul)
            offset += 16*mul
        return torch.cat([p.flatten(1) for p in pieces],dim=-1)

    def coefficients(self, data):
        inputs = (data.pos,data.z,data.edge_index,data.nbr_shift)
        key = tuple(x.data_ptr() for x in inputs)
        if self.training and torch.is_grad_enabled() and self.use_graphs:
            features = self._encoder_graphs[key](*inputs)
        else:
            encoded = data.clone()
            self.encoder(encoded)
            features = encoded['node_features']
        degree = self.readout(features) + self.adaptive_degree(features,data.z)
        pieces = [degree[:,16*l*l:16*(l+1)**2].reshape(len(data.pos),16,2*l+1) for l in range(5)]
        blocks = torch.cat(pieces,dim=-1)@self.frame_back
        blocks = blocks + torch.nn.functional.pad(self.scalar_baseline(data.z)[...,None],(0,24))
        return blocks.reshape(1,len(data.pos),400)


def build(arm, device='cuda'):
    if arm not in ARMS:
        raise ValueError(arm)
    assert sha256(BASE) == BASE_SHA
    torch.manual_seed(42)
    np.random.seed(42)
    model = (AdaptiveEdenR16() if arm == 'adaptive' else EdenR16()).to(device)
    saved = torch.load(BASE,map_location=device,weights_only=False)
    incompatible = model.load_state_dict(saved['model'],strict=(arm == 'continued'))
    if arm == 'adaptive':
        expected = {n for n in model.state_dict() if n.startswith(('context_norm.','element_context.','mixing.'))}
        assert set(incompatible.missing_keys) == expected and not incompatible.unexpected_keys
    return model
