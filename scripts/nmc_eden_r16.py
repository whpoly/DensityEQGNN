"""Eden-inspired SAD residual ablation and official-encoder/R16 hybrid."""
from __future__ import annotations

import json
from pathlib import Path
import types
import numpy as np
import torch
from e3nn import o3

from scripts.nmc_edengnn import EdenDensity,atom_data,make_sad,write_json
from scripts.edengnn_graph_training import TensorEncoder
from scripts.nmc_probe_holdout import DEFAULT_ROOT,PeriodicQueryGraph,save_split,compact_prediction,sha256
from scripts.compare_local_nmc_oracles import load_sample
from scripts.evaluate_vasp_sad_nmc import read_form_factors
from scripts.fit_nmc_holdout_oracle import BASIS
from scripts.train_nmc_coefficient_pretrain import initialize,fast_prediction

ROOT=Path('runs/nmc_eden_r16_2026-10-29')
ARMS=['r16_sad','eden_r16_sad']


class EdenR16(torch.nn.Module):
    """Official l<=6 encoder -> equivariant linear R16 residual coefficients.

    Eden uses cyclic y,z,x coordinates. Rotate output irreps back into the
    physical x,y,z convention used by the R16 spherical harmonics.
    """
    def __init__(self):
        super().__init__()
        source=EdenDensity()
        self.encoder=source.encoder
        self.provenance=source.provenance
        self.encoder_config=source.config['atom']
        incoming=self.encoder_config['conv']['irreps_node_features']
        outgoing='16x0e+16x1o+16x2e+16x3o+16x4e'
        self.readout=o3.Linear(incoming,outgoing)
        self.scalar_baseline=torch.nn.Embedding(100,16,padding_idx=0)
        torch.nn.init.zeros_(self.scalar_baseline.weight)
        rotation=torch.eye(3,dtype=torch.float64)[[1,2,0]]
        self.register_buffer('frame_back',o3.Irreps.spherical_harmonics(4).D_from_matrix(rotation).float())
        self.use_graphs=False
        self._encoder_graphs={}

    def coefficients(self,data):
        inputs=(data.pos,data.z,data.edge_index,data.nbr_shift)
        key=tuple(x.data_ptr() for x in inputs)
        if self.training and torch.is_grad_enabled() and self.use_graphs:
            features=self._encoder_graphs[key](*inputs)
        else:
            data=data.clone()
            self.encoder(data)
            features=data['node_features']
        degree=self.readout(features)
        pieces=[degree[:,16*l*l:16*(l+1)**2].reshape(len(data.pos),16,2*l+1) for l in range(5)]
        blocks=torch.cat(pieces,dim=-1)@self.frame_back
        scalar=self.scalar_baseline(data.z)
        blocks=blocks+torch.nn.functional.pad(scalar[...,None],(0,24))
        return blocks.reshape(1,len(data.pos),400)

    def capture(self,samples):
        self.train()
        self.encoder.spharm_edges.coord_change=self.encoder.spharm_edges.coord_change.to(next(self.parameters()).device)
        for sample in samples:
            data=sample['atom_data']
            inputs=(data.pos,data.z,data.edge_index,data.nbr_shift)
            key=tuple(x.data_ptr() for x in inputs)
            self._encoder_graphs[key]=torch.cuda.make_graphed_callables(TensorEncoder(self.encoder),inputs,num_warmup_iters=3)
        self.use_graphs=True


def build(arm,device):
    if arm=='r16_sad':
        return initialize(device)[0]
    torch.manual_seed(42)
    np.random.seed(42)
    return EdenR16().to(device)


def prepare(model,arm,device='cuda'):
    factors=read_form_factors(Path('potpaw54.zip'))
    samples=[]
    for sid in range(8):
        sample=load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),device)
        split,metadata=save_split(DEFAULT_ROOT/'spatial_splits',len(sample['target']),sid,sample['metadata']['sha256'])
        sample.update(id=sid,split={k:torch.tensor(v,device=device) for k,v in split.items()},split_metadata=metadata)
        sample['normalizer']=sample['target'][sample['split']['train']].abs().mean()
        sample['sad']=make_sad(sample,factors)
        sample['graph']=PeriodicQueryGraph(sample['positions'],sample['probes'],sample['cell'],3.)
        if arm=='r16_sad':
            sample['model_positions']=sample['positions'].float()[None]
            sample['model_cell']=sample['cell'].float()[None]
            sample['mask']=torch.ones(1,len(sample['numbers']),dtype=torch.bool,device=device)
            with torch.no_grad():
                sample['geometry']=model._geometry(sample['model_positions'],sample['mask'],sample['model_cell'])
        else:
            sample['atom_data']=atom_data(sample)
        samples.append(sample)
        print(f'PREPARED arm={arm} sample={sid}',flush=True)
    if arm=='r16_sad':
        original=model._geometry
        cache={s['model_positions'].data_ptr():s['geometry'] for s in samples}
        def geometry(self,positions,mask,cell):
            return cache[positions.data_ptr()] if positions.data_ptr() in cache else original(positions,mask,cell)
        model._geometry=types.MethodType(geometry,model)
    return samples


def coefficients(model,sample,arm):
    if arm=='r16_sad':
        return model(sample['numbers'][None],sample['model_positions'],sample['mask'],sample['model_cell'])['coefficients']
    return model.coefficients(sample['atom_data'])


def prediction(sample,coeff,ids,fast=False):
    residual=fast_prediction(sample,coeff,ids) if fast else compact_prediction(sample,coeff,ids,BASIS)
    return residual.double()+sample['sad'][ids].double()


@torch.no_grad()
def evaluate(model,samples,arm,final=False,sad_only=False):
    model.eval()
    rows=[]
    for sample in samples:
        coeff=None if sad_only else coefficients(model,sample,arm)
        groups=['train','validation','test'] if final else ['selection']
        row={'sample_id':sample['id']}
        for group in groups:
            accum=torch.zeros(4,dtype=torch.float64,device=sample['target'].device)
            for ids in sample['split'][group].split(2048):
                pred=sample['sad'][ids].double() if sad_only else prediction(sample,coeff,ids)
                target=sample['target'][ids]
                accum+=torch.stack(((pred-target).abs().sum(),target.abs().sum(),(pred-target).sum(),(-pred).clamp_min(0).sum()))
            row[group+'_nmae']=float(accum[0]/accum[1])
            row[group+'_charge_error']=float(accum[2]/accum[1])
            row[group+'_negative_fraction']=float(accum[3]/accum[1])
        rows.append(row)
    return {'records':rows,**{g+'_macro_nmae':float(np.mean([r[g+'_nmae'] for r in rows])) for g in groups}}
