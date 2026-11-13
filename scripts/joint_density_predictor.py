"""One geometry-only inference interface for the unified NMC benchmark."""
from pathlib import Path

import torch

from scripts.benchmark_deepdft_reuse import CHECKPOINT
from scripts.nmc_probe_holdout import DEFAULT_ROOT,official_model,atom_batch
from scripts.deepdft_inference_fast import FlatProbe,ProbeEdges
from scripts.deepdft_factored_messages import FactoredProbe
from scripts.deepdft_radial_table import RadialTableProbe
from scripts.deepdft_near_far import far_fields,weighted_geometry
from scripts.deepdft_cubic_far import periodic_expand
from scripts.deepdft_paw import PAWReadout,geometry_features

DEEP_PAW = Path('runs/deepdft_paw_2026-10-30')
EDEN_PAW = Path('runs/edengnn_paw_scaled_2026-10-30')
VARIANTS = {
    'deep_reference':dict(core='original',chunk=4096),
    'deep_factored':dict(core='factored',chunk=4096),
    'deep_table':dict(core='table',chunk=4096),
    'deep_fused':dict(core='fused',chunk=8192),
    **{f'fused_cubic_r{r:g}':dict(core='fused',chunk=8192,radius=r) for r in (3.,3.25,3.5)},
    'eden':dict(core='eden'),
}


def load_deep_head(seed,device):
    saved = torch.load(DEEP_PAW/f'seed_{seed}/best.pt',map_location=device,weights_only=False)
    state = saved['head']
    head = PAWReadout(state['mean'],state['scale'],lmax=2).to(device)
    head.load_state_dict(state)
    return head.eval().requires_grad_(False)


def load_eden_paw(seed,device):
    from scripts.edengnn_paw import EdenPAW
    state = torch.load(EDEN_PAW/f'finetuned_{seed}/best.pt',map_location=device,weights_only=False)['model']
    # Construct on CPU because EdenDensity's official modules initialize there.
    head = EdenPAW(state['mean'].cpu(),state['scale'].cpu(),pretrained=False,
        feature_rms=state['feature_rms'].cpu()).to(device)
    head.load_state_dict(state)
    return head.eval().requires_grad_(False)


@torch.no_grad()
def decode_states(core,sample,states,config):
    chunk = config['chunk']
    radius = config.get('radius')
    if radius is None:
        graph = ProbeEdges.build(sample)
        output = sample['positions'].new_empty(graph.size,dtype=torch.float32)
        for start in range(0,graph.size,chunk):
            stop = min(start+chunk,graph.size)
            output[start:stop] = core(*graph.chunk(sample,start,stop),*states,stop-start)
        return output
    scalars,vectors,projections = states
    width = scalars.shape[-1]
    shape = tuple(sample['metadata']['grid_shape'])
    coarse_shape = tuple(n//2 for n in shape)
    coarse = far_fields(core,sample,states,radius,1.,2,chunk)
    fields = [periodic_expand(coarse[k].reshape(*coarse_shape,4*width),2,'cubic').reshape(-1,4*width)
        for k in range(len(scalars))]
    del coarse
    graph = ProbeEdges.build(sample,cutoff=radius)
    output = scalars.new_empty(graph.size)
    for start in range(0,graph.size,chunk):
        stop = min(start+chunk,graph.size)
        size = stop-start
        source,receiver,diff = graph.chunk(sample,start,stop)
        geometry = weighted_geometry(core,diff,radius,1.)
        s,v = scalars.new_zeros(size,width),vectors.new_zeros(size,3,width)
        for k,(layer,update) in enumerate(zip(core.probe.message_layers,core.probe.scalar_vector_update)):
            a,b = core.aggregate(layer,scalars[k],vectors[k],projections[k],source,receiver,geometry,size)
            shared = fields[k][start:stop]
            a,b = a+shared[:,:width],b+shared[:,width:].reshape(size,3,width)
            gs,gv = layer.update_gate_mlp(a).split(width,1)
            s,v = update(gs*s+(1-gs)*a,gv[:,None]*v+(1-gv[:,None])*b)
        output[start:stop] = core.probe.readout_function(s).squeeze(1)
    return output


class JointPredictor:
    def __init__(self,variant,device='cuda'):
        self.variant,self.config,self.device = variant,VARIANTS[variant],device
        if variant=='eden':
            from scripts.nmc_edengnn import EdenDensity,ROOT
            from scripts.evaluate_vasp_sad_nmc import read_form_factors
            self.model = EdenDensity().to(device)
            self.model.load_state_dict(torch.load(ROOT/'eden_eight/best.pt',map_location=device,weights_only=False)['model'])
            self.model.eval().requires_grad_(False)
            self.head = load_eden_paw(42,device)
            self.factors = read_form_factors(Path('potpaw54.zip'))
            self.core = None
        else:
            self.model,self.provenance = official_model(DEFAULT_ROOT/'DeepDFT',device)
            self.model.load_state_dict(torch.load(CHECKPOINT,map_location=device,weights_only=False)['model'])
            self.model.eval().requires_grad_(False)
            core = self.config['core']
            self.core = (FlatProbe(self.model) if core=='original' else FactoredProbe(self.model) if core=='factored'
                else RadialTableProbe(self.model,fused=core=='fused')).eval()
            self.head = load_deep_head(42,device)

    @torch.no_grad()
    def predict(self,sample,joint=True):
        if 'target' in sample or 'target_aug' in sample:
            raise ValueError('Pass geometry-only model inputs')
        if self.variant=='eden':
            from scripts.benchmark_nmc_edengnn import prepare_native_eden
            prepared = prepare_native_eden(sample,self.factors)
            density = self.model.native(prepared)
            augmentation = None
            if joint:
                data = prepared['atom_data']
                augmentation = self.head(data.pos,data.z,data.edge_index,data.nbr_shift)
        else:
            batch = atom_batch(sample)
            scalars,vectors = self.model.atom_model(batch)
            augmentation = self.head(torch.cat(scalars,-1),torch.cat(vectors,-1),sample['numbers'],
                geometry_features(batch)) if joint else None
            density = decode_states(self.core,sample,self.core.atoms((scalars,vectors)),self.config)
        return dict(density=density,augmentation=augmentation)

    @torch.no_grad()
    def alternate_paw(self,sample):
        # Used only after timing, then released before another timed structure.
        if self.variant=='eden':
            from scripts.edengnn_paw import inputs
            head = load_eden_paw(43,self.device)
            output = head(*inputs(sample))
        else:
            head = load_deep_head(43,self.device)
            batch = atom_batch(sample)
            scalars,vectors = self.model.atom_model(batch)
            output = head(torch.cat(scalars,-1),torch.cat(vectors,-1),sample['numbers'],geometry_features(batch))
        return output
