"""Predict density and supported PAW coefficients from a POSCAR only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from pymatgen.io.vasp.inputs import Poscar

from density_eqgnn.chgcar import vasp_grid
from scripts.public_paw_predictor import PublicPAWPredictor


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--structure',type=Path,required=True)
    p.add_argument('--grid',type=int,nargs=3,required=True,metavar=('NX','NY','NZ'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[42,43],default=42)
    args=p.parse_args()
    if args.output.suffix.lower()!='.npz': raise ValueError('Output must end in .npz')
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError('Preserve existing prediction')
    if min(args.grid)<1: raise ValueError('Positive grid dimensions required')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    structure=Poscar.from_file(args.structure).structure
    sample=dict(positions=torch.tensor(structure.cart_coords,dtype=torch.float64,device='cuda'),
        numbers=torch.tensor(structure.atomic_numbers,dtype=torch.int64,device='cuda'),
        cell=torch.tensor(structure.lattice.matrix,dtype=torch.float64,device='cuda'),
        probes=torch.tensor(vasp_grid(structure.lattice.matrix,args.grid),dtype=torch.float64,device='cuda'),
        metadata=dict(grid_shape=args.grid,volume=structure.volume))
    predictor=PublicPAWPredictor(seed=args.seed)
    torch.cuda.synchronize();start=time.perf_counter()
    output=predictor.predict(sample);torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    native=predictor.native_augmentation(output['augmentation'],sample['numbers'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,density=output['density'].cpu().numpy().reshape(args.grid),
        augmentation=output['augmentation'].cpu().numpy(),augmentation_mask=output['augmentation_mask'].cpu().numpy(),
        native_augmentation=np.concatenate([v.cpu().numpy() for v in native]),
        native_offsets=np.cumsum([0]+[len(v) for v in native]),
        positions=structure.cart_coords,numbers=structure.atomic_numbers,cell=structure.lattice.matrix,
        supported_lmax=np.array(2),grid_shape=np.array(args.grid))
    args.output.with_suffix('.json').write_text(json.dumps(dict(source=str(args.structure.resolve()),seed=args.seed,
        first_forward_seconds=elapsed,grid_shape=args.grid,supported_lmax=2,
        density_units='electrons/angstrom^3',augmentation='canonical real spherical coefficients and local POTCAR native order',
        limitation='Nonmagnetic Li/O/Mn/Co/Ni; L3/L4 unavailable. This NPZ is not a complete VASP restart CHGCAR.'),indent=2),encoding='utf-8')
    print(str(args.output.resolve()))


if __name__=='__main__': main()
