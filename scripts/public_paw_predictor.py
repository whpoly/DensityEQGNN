"""Geometry-only official DeepDFT density + trained NMC PAW inference."""
from pathlib import Path
import json

import torch

from scripts.benchmark_public_density import model_for
from scripts.deepdft_paw import PAWReadout, geometry_features, potcar_specs, blocks, BLOCKS, WIDTH
from scripts.deepdft_radial_table import RadialTableProbe
from scripts.joint_density_predictor import decode_states
from scripts.nmc_probe_holdout import atom_batch, sha256


class PublicPAWPredictor:
    """Only L<=2 nonmagnetic NMC PAW coefficients are supported.

    Outputs use the existing canonical real-spherical augmentation layout.
    Higher ranks are placeholders, not complete deployable PAW occupancies.
    """
    def __init__(self,root=Path('runs/public_paw_2026-11-02'),seed=42):
        self.root=Path(root)
        cfg=json.loads((self.root/'settings.json').read_text())
        assert sha256(cfg['deep_checkpoint'])==cfg['deep_checkpoint_sha256']
        folder=self.root/f'deep_frozen_{seed}'
        selection=json.loads((folder/'selection.json').read_text())
        assert sha256(folder/'best.pt')==selection['checkpoint_sha256']
        saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
        assert saved['settings_sha256']==sha256(self.root/'settings.json')
        state=saved['model']
        self.model,_=model_for('nmc')
        self.head=PAWReadout(state['mean'],state['scale'],lmax=2).cuda()
        self.head.load_state_dict(state)
        self.head.eval().requires_grad_(False)
        self.core=RadialTableProbe(self.model,fused=True).eval()
        self.valid=torch.zeros(29,WIDTH,dtype=torch.bool,device='cuda')
        self.native_mapping={}
        canonical={b['key']:b for b in BLOCKS}
        for z,spec in potcar_specs().items():
            mapping=[]
            for b in blocks(spec['basis']):
                c=canonical[b['key']]
                mapping.extend(range(c['start'],c['stop']))
                if b['l']<=2: self.valid[z,c['start']:c['stop']]=True
            self.native_mapping[z]=torch.tensor(mapping,dtype=torch.int64,device='cuda')

    @torch.no_grad()
    def predict(self,geometry,density=True,occupation=True):
        if set(geometry)-{'positions','numbers','cell','probes','metadata'}:
            raise ValueError('Pass geometry only; targets and cached atom features are forbidden')
        if not set(geometry['numbers'].tolist()) <= {3,8,25,27,28}:
            raise ValueError('Only Li/O/Mn/Co/Ni supported')
        batch=atom_batch(geometry)
        scalars,vectors=self.model.atom_model(batch)
        aug=self.head(torch.cat(scalars,-1),torch.cat(vectors,-1),geometry['numbers'],
                      geometry_features(batch)) if occupation else None
        rho=decode_states(self.core,geometry,self.core.atoms((scalars,vectors)),dict(chunk=8192)) if density else None
        return dict(density=rho,augmentation=aug,
                    augmentation_mask=self.valid[geometry['numbers']] if occupation else None,
                    supported_lmax=2)

    def native_augmentation(self,augmentation,numbers):
        """Native ordering for the recorded local POTCARs; high ranks remain unvalidated."""
        return [row[self.native_mapping[int(z)]] for row,z in zip(augmentation,numbers)]
