"""PAW augmentation readout for a frozen, unmodified DeepDFT encoder.

Targets are VASP's real spherical-tensor augmentation coefficients, not band
occupations. POTCAR projector order is read explicitly; no element-default
pseudopotential mapping is assumed. This module supports nonmagnetic NMC.
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from pathlib import Path

import lz4.frame
import numpy as np
import torch
from e3nn import o3
from pymatgen.io.vasp.inputs import Poscar

ELEMENTS = {'Li':3, 'O':8, 'Mn':25, 'Co':27, 'Ni':28}
UNIVERSAL_BASIS = [0,0,1,1,2,2]


def blocks(basis):
    occurrences = {}
    ids = []
    for l in basis:
        ids.append((l,occurrences.get(l,0)))
        occurrences[l] = occurrences.get(l,0)+1
    result, offset = [], 0
    for i,li in enumerate(basis):
        for j in range(i,len(basis)):
            lj = basis[j]
            pair = tuple(sorted((ids[i],ids[j])))
            for l in range(abs(li-lj),li+lj+1,2):
                result.append(dict(key=(*pair,l),l=l,start=offset,stop=offset+2*l+1))
                offset += 2*l+1
    return result


BLOCKS = blocks(UNIVERSAL_BASIS)
WIDTH = BLOCKS[-1]['stop']
IRREPS = o3.Irreps([(1,(b['l'],(-1)**b['l'])) for b in BLOCKS])


def potcar_specs(path=Path('potpaw54.zip')):
    result = {}
    with zipfile.ZipFile(path) as archive:
        for symbol,z in ELEMENTS.items():
            name = f'potpaw54/potpaw_PBE/PAW_GGA_PBE/{symbol}/POTCAR'
            raw = archive.read(name)
            text = raw.decode()
            basis = []
            for l,n in re.findall(r'Non local Part\s*\n\s*(\d+)\s+(\d+)',text):
                basis.extend([int(l)]*int(n))
            if not basis or max(basis)>2:
                raise ValueError((symbol,basis))
            result[z] = dict(symbol=symbol,basis=basis,
                title=re.search(r'TITEL\s*=\s*([^\n]+)',text).group(1).strip(),
                sha256=hashlib.sha256(raw).hexdigest(),member=name)
    return result


def read_structure_and_aug(path,specs,lmax=4):
    path = Path(path)
    raw = path.read_bytes()
    text = (lz4.frame.decompress(raw) if path.suffix=='.lz4' else raw).decode()
    # Poscar reader does not consume density values or augmentation labels.
    lines = text.splitlines()
    count_line = 5 if all(t.isdigit() for t in lines[5].split()) else 6
    nat = sum(int(t) for t in lines[count_line].split())
    coordinate_line = count_line+1
    if lines[coordinate_line].strip().lower().startswith('s'):
        coordinate_line += 1
    structure = Poscar.from_str('\n'.join(lines[:coordinate_line+1+nat]),read_velocities=False).structure
    numbers = np.array(structure.atomic_numbers)
    matches = list(re.finditer(r'augmentation\s+occupancies\s+(\d+)\s+(\d+)',text))
    if len(matches)!=len(numbers):
        raise ValueError('Expected exactly one nonmagnetic augmentation block per atom')
    target = np.zeros((len(numbers),WIDTH),dtype=np.float64)
    mask = np.zeros_like(target,dtype=bool)
    canonical = {b['key']:b for b in BLOCKS}
    raw_values, mappings = [], []
    for a,(z,match) in enumerate(zip(numbers,matches)):
        if int(match.group(1))!=a+1:
            raise ValueError('Augmentation atom ordering mismatch')
        n = int(match.group(2))
        tail = text[match.end():matches[a+1].start() if a+1<len(matches) else len(text)]
        values = np.fromstring(tail,sep=' ')
        source_blocks = blocks(specs[int(z)]['basis'])
        if len(values)!=n or n!=source_blocks[-1]['stop']:
            raise ValueError(f'POTCAR / augmentation shape mismatch: {z}, {len(values)}, {n}')
        mapping = np.empty(n,dtype=np.int64)
        for b in source_blocks:
            destination = canonical[b['key']]
            s = slice(b['start'],b['stop'])
            d = slice(destination['start'],destination['stop'])
            target[a,d] = values[s]
            mask[a,d] = b['l']<=lmax
            mapping[s] = np.arange(destination['start'],destination['stop'])
        raw_values.append(values)
        mappings.append(mapping)
    geometry = np.concatenate((np.asarray(structure.lattice.matrix).ravel(),
        np.asarray(structure.cart_coords).ravel(),numbers)).astype('<f8').tobytes()
    return dict(positions=torch.tensor(structure.cart_coords,dtype=torch.float64),
        cell=torch.tensor(structure.lattice.matrix,dtype=torch.float64),
        numbers=torch.tensor(numbers), target_aug=torch.tensor(target,dtype=torch.float32),
        aug_mask=torch.tensor(mask), raw_aug=raw_values, raw_mapping=mappings,
        metadata=dict(source=str(path.resolve()),source_sha256=hashlib.sha256(raw).hexdigest(),
            geometry_sha256=hashlib.sha256(geometry).hexdigest(),atoms=len(numbers),formula=structure.formula),
        _source_text=text)


def write_predicted_aug(source, prediction, destination):
    """Replace only augmentation numbers; preserve structure and density text."""
    text = source['_source_text']
    matches = list(re.finditer(r'augmentation\s+occupancies\s+(\d+)\s+(\d+)',text))
    parts = [text[:matches[0].start()]]
    prediction = np.asarray(prediction)
    for a,match in enumerate(matches):
        values = prediction[a,source['raw_mapping'][a]]
        parts.append(match.group(0)+'\n')
        for start in range(0,len(values),5):
            parts.append(' '.join(f'{v: .10E}' for v in values[start:start+5])+'\n')
    Path(destination).write_text(''.join(parts),encoding='utf-8')


def geometry_features(batch):
    edges = batch['atom_edges'][0]
    source,receiver = edges.T
    shift = batch['atom_edges_displacement'][0]@batch['cell'][0]
    delta = batch['atom_xyz'][0][source]+shift-batch['atom_xyz'][0][receiver]
    distance = delta.norm(dim=-1)
    direction = delta/distance[:,None].clamp_min(1e-8)
    n = torch.arange(1,17,device=delta.device,dtype=delta.dtype)
    envelope = .5*(1+torch.cos(torch.pi*distance.clamp(max=4)/4))
    radial = torch.sinc(distance[:,None]*n[None]/4)*envelope[:,None]
    # VASP real harmonics use (y,z,x) for l=1; match Eden's convention.
    harmonics = o3.spherical_harmonics(list(range(5)),direction[:,[1,2,0]],
        normalize=True,normalization='component')
    return dict(source=source,receiver=receiver,direction=direction,radial=radial,
        harmonics=harmonics,envelope=envelope)


class PAWReadout(torch.nn.Module):
    """Invariant edge weights times spherical harmonics supply l=0..4.

    Scalar/vector features are from all three original PaiNN layers. No density
    target, augmentation target, or query-grid feature enters this readout.
    """
    def __init__(self,mean,scale,lmax=4):
        super().__init__()
        self.lmax = lmax
        self.register_buffer('mean',mean.clone())
        self.register_buffer('scale',scale.clone())
        self.scalar = torch.nn.Sequential(torch.nn.LayerNorm(384),torch.nn.Linear(384,64),
            torch.nn.SiLU(),torch.nn.Linear(64,64),torch.nn.SiLU())
        self.vector = torch.nn.Linear(384,8,bias=False)
        self.edge = torch.nn.Sequential(torch.nn.Linear(160,64),torch.nn.SiLU(),
            torch.nn.Linear(64,64),torch.nn.SiLU(),torch.nn.Linear(64,len(BLOCKS)))
        self.onsite = torch.nn.Linear(64,sum(b['l']==0 for b in BLOCKS))
        torch.nn.init.zeros_(self.edge[-1].weight)
        torch.nn.init.zeros_(self.edge[-1].bias)
        torch.nn.init.zeros_(self.onsite.weight)
        torch.nn.init.zeros_(self.onsite.bias)

    def forward(self,scalars,vectors,numbers,geometry):
        h = self.scalar(scalars)
        v = self.vector(vectors)
        src,rec = geometry['source'],geometry['receiver']
        d = geometry['direction']
        x = torch.cat((h[rec],h[src],geometry['radial'],
            (v[rec]*d[:,:,None]).sum(1),(v[src]*d[:,:,None]).sum(1)),dim=-1)
        weights = self.edge(x)*geometry['envelope'][:,None]
        degree = torch.bincount(rec,minlength=len(h)).clamp_min(1).sqrt()
        weights = weights/degree[rec,None]
        onsite = self.onsite(h)
        parts = []
        scalar_index = 0
        for k,b in enumerate(BLOCKS):
            l = b['l']
            y = geometry['harmonics'][:,l*l:(l+1)**2]
            value = h.new_zeros(len(h),2*l+1).index_add(0,rec,weights[:,k,None]*y)
            if l==0:
                value = value+onsite[:,scalar_index,None]
                scalar_index += 1
            if l>self.lmax:
                value = value*0
            parts.append(value)
        return self.mean[numbers]+self.scale[numbers]*torch.cat(parts,dim=-1)


def training_statistics(samples):
    mean = torch.zeros(29,WIDTH)
    scale = torch.ones_like(mean)
    for z in ELEMENTS.values():
        values = torch.cat([s['target_aug'][s['numbers']==z] for s in samples])
        mask = torch.cat([s['aug_mask'][s['numbers']==z] for s in samples])
        if not len(values):
            raise ValueError(f'No training atoms for element {z}')
        for b in BLOCKS:
            section = slice(b['start'],b['stop'])
            if not mask[:,section].any():
                continue
            if b['l']==0:
                mean[z,section] = values[:,section].mean()
            residual = values[:,section]-mean[z,section]
            # One scale for all m in a block preserves rotation equivariance.
            scale[z,section] = residual.square().mean().sqrt().clamp_min(1e-3)
    return mean,scale


def features(model,sample):
    from scripts.nmc_probe_holdout import atom_batch
    device = next(model.parameters()).device
    sample_device = {k:sample[k].to(device) for k in ('positions','cell','numbers')}
    batch = atom_batch(sample_device)
    ss,vv = model.atom_model(batch)
    return dict(scalars=torch.cat(ss,dim=-1),vectors=torch.cat(vv,dim=-1),
        numbers=sample_device['numbers'],geometry=geometry_features(batch))


@torch.no_grad()
def density_and_paw(model,head,sample,chunk=4096):
    """Both branches share one freshly computed atom representation.

    Return canonical PAW coefficients; map native slots with the matching POTCAR
    specification. Unsupported higher ranks are zero and not validated.
    """
    from scripts.nmc_probe_holdout import atom_batch
    from scripts.deepdft_inference_fast import FlatProbe,ProbeEdges
    batch = atom_batch(sample)
    ss,vv = model.atom_model(batch)
    paw = head(torch.cat(ss,dim=-1),torch.cat(vv,dim=-1),sample['numbers'],geometry_features(batch))
    core = FlatProbe(model)
    states = core.atoms((ss,vv))
    graph = ProbeEdges.build(sample)
    density = sample['positions'].new_empty(graph.size,dtype=torch.float32)
    for start in range(0,graph.size,chunk):
        stop = min(start+chunk,graph.size)
        density[start:stop] = core(*graph.chunk(sample,start,stop),*states,stop-start)
    return dict(density=density,augmentation=paw)
