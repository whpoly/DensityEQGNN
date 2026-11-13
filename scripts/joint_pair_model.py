"""One atomic encoding and two outputs for both NMC/EC pipelines."""
from pathlib import Path
import numpy as np
import torch
from scripts.deepdft_paw import PAWReadout, geometry_features
from scripts.deepdft_radial_table import RadialTableProbe
from scripts.edengnn_paw import augmentation_class
from scripts.joint_density_predictor import decode_states
from scripts.nmc_edengnn import EdenDensity, atom_data, EdenStencil, make_sad
from scripts.nmc_probe_holdout import DEFAULT_ROOT, official_model, atom_batch, probe_batch, PeriodicQueryGraph


class JointModel(torch.nn.Module):
    def __init__(self, arm, mean, scale):
        super().__init__()
        self.arm = arm
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        if arm == 'deep':
            self.net, _ = official_model(DEFAULT_ROOT/'DeepDFT', 'cuda')
            self.head = PAWReadout(mean, scale, lmax=2)
        else:
            self.net = EdenDensity()
            self.head = augmentation_class()(self.net.config['atom']['conv']['irreps_node_features'])
            for p in self.head.parameters():
                torch.nn.init.zeros_(p)

    def eden_paw(self, encoded):
        self.head(encoded)
        z = encoded.z
        return self.mean[z] + self.scale[z] * encoded['node_aug']

    def sampled(self, s, ids, sad_values):
        ijk, shape, xyz = coordinates(s, ids)
        order = torch.arange(len(ids), device='cuda')
        if self.arm == 'deep':
            batch = atom_batch(s)
            rep = self.net.atom_model(batch)
            graph = PeriodicQueryGraph(s['positions'], xyz, s['cell'], 4.)
            query = probe_batch(dict(graph=graph, probes=xyz, batch=batch), order)
            density = self.net.probe_model(query, *rep)[0]
            paw = self.head(torch.cat(rep[0], -1), torch.cat(rep[1], -1), s['numbers'], geometry_features(batch))
        else:
            h,z,data = self.net.encode(atom_data(s))
            paw = self.eden_paw(data)
            # Match the native regular stencil, which is centred at each atom's
            # nearest grid node, then apply the sub-grid Cartesian residual.
            grid = s['cell'] / shape[:, None]
            centres = (s['positions'] @ torch.linalg.inv(s['cell']) * shape).round().long()
            snapped = centres.double() @ grid
            graph = PeriodicQueryGraph(snapped, xyz, s['cell'], 4.00000001)
            atoms, recv, shifts = graph.select(order)
            offsets = ijk[recv] - centres[atoms] - shifts.long()*shape
            delta = offsets.double() @ grid
            keep = delta.norm(dim=1) <= 4.
            atoms, recv, delta = atoms[keep], recv[keep], delta[keep]
            residual = s['positions'].float() - centres.float() @ grid.float()
            displacement = (delta.float()-residual[atoms])[:,[1,2,0]][:,None,:]
            dist = torch.linalg.vector_norm(displacement, dim=-1)
            values = self.net.density.compute_chunk(displacement, dist, z[atoms,None], h[atoms]).flatten()
            density = values.new_zeros(len(ids)).index_add(0,recv,values) + sad_values
        return density, paw

    @torch.no_grad()
    def full(self, s, factors, core=None):
        if self.arm == 'deep':
            batch = atom_batch(s)
            rep = self.net.atom_model(batch)
            paw = self.head(torch.cat(rep[0], -1), torch.cat(rep[1], -1), s['numbers'], geometry_features(batch))
            density = decode_states(core, s, core.atoms(rep), dict(chunk=8192))
        else:
            data = atom_data(s)
            for k,v in EdenStencil(s,csr=False).native_fields().items():
                data[k] = v
            self.net.encoder(data)
            paw = self.eden_paw(data)
            self.net.density(data)
            density = data['charge'].flatten() + make_sad(s, factors)
        return density, paw


def coordinates(s, ids):
    shape = torch.tensor(s['metadata']['grid_shape'],device='cuda')
    ix = torch.as_tensor(ids, device='cuda')
    ijk = torch.stack((ix//(shape[1]*shape[2]), ix//shape[2]%shape[1], ix%shape[2]), dim=1)
    return ijk, shape, (ijk.double()/shape) @ s['cell']
