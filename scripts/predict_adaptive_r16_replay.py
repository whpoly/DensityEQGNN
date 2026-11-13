"""Optional CUDA Graph inference for repeated compatible input shapes.

First use of a new atom/edge shape records an execution graph and is slower.
The bounded cache stores executions, not geometry, coefficients or densities.
For isolated new structures use FastAdaptiveR16 instead.
"""
from pathlib import Path
import torch
from scripts.predict_adaptive_r16_fast import FastAdaptiveR16
from scripts.r16_inference_graph import GraphCoefficients
from scripts.nmc_edengnn import atom_data
from scripts.r16_grid_diagnostic import uniform_grid_edges
from scripts.r16_inference_fast import decode_grouped
from scripts.fit_nmc_holdout_oracle import BASIS


class ReplayAdaptiveR16(FastAdaptiveR16):
    def __init__(self,checkpoint=Path('runs/nmc_adaptive_r16_2026-10-29/adaptive/best.pt'),
                 potcar_zip=Path('potpaw54.zip'),device='cuda',max_graphs=2):
        super().__init__(checkpoint,potcar_zip,device)
        if self.device.type!='cuda':
            raise ValueError('ReplayAdaptiveR16 requires CUDA')
        self.graphs=GraphCoefficients(self.model,max_graphs=max_graphs)

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
        coeff=self.graphs(atom_data(sample))
        e=uniform_grid_edges(positions,cell,shape,BASIS.cutoff)
        residual=decode_grouped(coeff,-e.displacement,e.batch,e.neighbor,e.center,
                               shape[0]*shape[1]*shape[2],BASIS)
        return (residual.double()+self.sad(sample).double()).reshape(shape)

    def clear_graphs(self):
        torch.cuda.synchronize(self.device)
        self.graphs.cache.clear()
