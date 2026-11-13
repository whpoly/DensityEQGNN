"""Pinned official Eden encoder/augmentation layer on the locked NMC PAW split."""
import ast
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

from scripts.deepdft_paw import BLOCKS,IRREPS,WIDTH
from scripts.nmc_edengnn import EdenDensity,REPO,ROOT as EDEN_ROOT,atom_data

CHECKPOINT = EDEN_ROOT/'eden_eight/best.pt'


def augmentation_class():
    # official_classes has already checked the pinned checkout and loaded the
    # exact module imports; add only the unchanged official augmentation class.
    module = sys.modules['edengnn.model.density_benchmark']
    module.AUG_IRREPS = IRREPS
    path = REPO/'src/edengnn/model/model.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AugmentationLayer')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),module.__dict__)
    return module.AugmentationLayer


class EdenPAW(torch.nn.Module):
    def __init__(self,mean,scale,pretrained=True,feature_rms=None):
        super().__init__()
        density = EdenDensity()
        if pretrained:
            density.load_state_dict(torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['model'])
        self.encoder = density.encoder
        self.encoder_config = density.config['atom']
        self.provenance = density.provenance
        self.aug = augmentation_class()(self.encoder_config['conv']['irreps_node_features'])
        from e3nn import o3
        dim = o3.Irreps(self.encoder_config['conv']['irreps_node_features']).dim
        self.register_buffer('feature_rms',torch.ones(dim) if feature_rms is None else feature_rms.clone())
        # Same training-only affine residual scaling as the DeepDFT comparison.
        self.register_buffer('mean',mean.clone())
        self.register_buffer('scale',scale.clone())
        valid = torch.zeros(WIDTH)
        for b in BLOCKS:
            if b['l']<=2:
                valid[b['start']:b['stop']] = 1
        self.register_buffer('supported',valid)
        for p in self.aug.parameters():
            torch.nn.init.zeros_(p)

    def encode(self,positions,numbers,edges,shifts):
        data = Data(pos=positions,z=numbers,edge_index=edges,nbr_shift=shifts)
        self.encoder(data)
        return data['node_features']

    def readout(self,node_features,numbers):
        data = {'node_features':node_features/self.feature_rms}
        self.aug(data)
        return self.mean[numbers]+self.scale[numbers]*data['node_aug']*self.supported

    def forward(self,positions,numbers,edges,shifts):
        return self.readout(self.encode(positions,numbers,edges,shifts),numbers)


def inputs(sample,device='cuda'):
    data = atom_data({k:sample[k].to(device) for k in ('positions','numbers','cell')})
    return data.pos,data.z,data.edge_index,data.nbr_shift


class TensorPAW(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        self.model = model

    def forward(self,*args):
        return self.model(*args)


def capture(model,samples):
    model.train()
    model.encoder.spharm_edges.coord_change = model.encoder.spharm_edges.coord_change.cuda()
    graphs = {}
    for s in samples:
        graphs[s['id']] = torch.cuda.make_graphed_callables(TensorPAW(model),s['inputs'],num_warmup_iters=3)
        print('CAPTURE',s['id'],round(torch.cuda.memory_allocated()/2**20,1),'MiB',flush=True)
    return graphs
