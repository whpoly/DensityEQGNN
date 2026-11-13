"""Reusable density inference from structure + grid shape, without DFT labels.

Usage in Python:
    predictor = FastAdaptiveR16()
    density = predictor(numbers, positions_angstrom, cell_angstrom, (56,56,140))

Returns a GPU tensor in electrons/angstrom^3. Constructor loads the checkpoint
and POTCAR constants once; every call rebuilds geometry, SAD and learned output.
"""
from pathlib import Path

import torch

from scripts.nmc_adaptive_r16 import AdaptiveEdenR16
from scripts.nmc_edengnn import atom_data
from scripts.evaluate_vasp_sad_nmc import read_form_factors
from scripts.fit_nmc_holdout_oracle import BASIS
from scripts.r16_grid_diagnostic import uniform_grid_edges
from scripts.r16_inference_fast import TorchSAD,decode_grouped


class FastAdaptiveR16:
    def __init__(self,checkpoint=Path('runs/nmc_adaptive_r16_2026-10-29/adaptive/best.pt'),
                 potcar_zip=Path('potpaw54.zip'),device='cuda'):
        self.device=torch.device(device)
        self.model=AdaptiveEdenR16().to(self.device).eval()
        saved=torch.load(checkpoint,map_location=self.device,weights_only=False)
        self.model.load_state_dict(saved['model'],strict=True)
        self.sad=TorchSAD(read_form_factors(Path(potcar_zip)),device=self.device)

    @torch.no_grad()
    def __call__(self,numbers,positions,cell,grid_shape):
        numbers=torch.as_tensor(numbers,dtype=torch.long,device=self.device)
        positions=torch.as_tensor(positions,dtype=torch.float64,device=self.device)
        cell=torch.as_tensor(cell,dtype=torch.float64,device=self.device)
        shape=tuple(int(n) for n in grid_shape)
        if len(shape)!=3 or any(n<=0 for n in shape):
            raise ValueError('Three positive grid dimensions required')
        if cell.shape!=(3,3) or numbers.ndim!=1 or positions.shape!=(len(numbers),3):
            raise ValueError('Expected [atoms], [atoms,3] positions and [3,3] row-vector cell')
        sample={'numbers':numbers,'positions':positions,'cell':cell,
                'metadata':{'grid_shape':shape,'volume':cell.det().abs()}}
        coefficients=self.model.coefficients(atom_data(sample))
        edges=uniform_grid_edges(positions,cell,shape,BASIS.cutoff)
        residual=decode_grouped(coefficients,-edges.displacement,edges.batch,edges.neighbor,
                                edges.center,shape[0]*shape[1]*shape[2],BASIS)
        return (residual.double()+self.sad(sample).double()).reshape(shape)
