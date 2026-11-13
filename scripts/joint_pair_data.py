"""Prepare the native grids and PAW labels, retaining official archive order."""
import argparse
import json
import re
import tempfile
from pathlib import Path
import lz4.frame
import numpy as np
import torch
from ase.data import chemical_symbols
from scripts.benchmark_public_density import save
from scripts.deepdft_paw import read_structure_and_aug, ELEMENTS
from scripts.evaluate_vasp_sad_nmc import sad_density
from scripts.joint_paw_tables import potcar_specs, read_form_factors
from scripts.nmc_probe_holdout import sha256

ROOT = Path('runs/joint_nmc_ec')
SPECIES = {'nmc': ELEMENTS, 'ec': {'H': 1, 'C': 6, 'O': 8}}


def manifest(ds):
    cfg = json.loads(Path(f'runs/public_benchmark_2026-10-30/{ds}/settings.json').read_text())
    path = Path(f'data/public_density/{ds}/split.json')
    assert cfg['split'] == json.loads(path.read_text()) and sha256(path) == cfg['split_sha256']
    return cfg


def convert(entry, temporary, specs, factors, dest, training):
    with Path(entry['archive']).open('rb') as f:
        f.seek(entry['offset'])
        raw = lz4.frame.decompress(f.read(entry['size']))
    temporary.write_bytes(raw)
    sample = read_structure_and_aug(temporary, specs, lmax=2)
    # Use the first total-density grid. VASP writes x fastest.
    prefix = raw[:raw.index(b'augmentation occupancies')]
    start = prefix.index(b'Direct') if b'Direct' in prefix else prefix.index(b'Cartesian')
    grid = re.search(rb'(?m)^\s*(\d+)[ \t]+(\d+)[ \t]+(\d+)[ \t\r]*\n', prefix[start:])
    shape = tuple(map(int, grid.groups()))
    values = prefix[start+grid.end():]
    cell = sample['cell'].numpy()
    volume = abs(np.linalg.det(cell))
    rho = np.fromstring(values, sep=' ').reshape(shape, order='F') / volume
    assert len(shape) == 3 and np.isfinite(rho).all() and np.isfinite(sample['target_aug'].numpy()).all()
    row = {k: sample[k] for k in ('positions','numbers','cell','target_aug','aug_mask')}
    row.update(metadata=dict(grid_shape=shape, volume=volume, geometry_sha256=sample['metadata']['geometry_sha256']),
               id=entry['index'], norm=float(np.abs(rho).mean()))
    assert row['norm'] > 0 and row['aug_mask'].any()
    folder = dest / str(entry['index'])
    folder.mkdir(exist_ok=True)
    np.save(folder/'rho.npy', (rho.astype('float32') if training else rho).flatten())
    if training:
        symbols = np.array([chemical_symbols[z] for z in sample['numbers'].tolist()])
        sad = sad_density(symbols, sample['positions'].numpy() @ np.linalg.inv(cell), cell, shape, factors)
        assert abs(sad.mean() - sum(factors[z].zval for z in symbols)) < 2e-7
        np.save(folder/'sad.npy', (sad.flatten()/volume).astype('float32'))
    path = folder/'sample.tmp'
    torch.save(row, path)
    path.replace(folder/'sample.pt')


def prepare(ds, stage):
    cfg = manifest(ds)
    dest = ROOT/ds/'data'
    dest.mkdir(parents=True, exist_ok=True)
    ids = cfg['split']['test'] if stage=='test' else cfg['split']['train']+cfg['split']['validation']
    specs = potcar_specs(elements=SPECIES[ds])
    factors = read_form_factors(Path('potpaw54.zip'), SPECIES[ds])
    with tempfile.TemporaryDirectory() as tmp:
        for j, i in enumerate(ids):
            if not (dest/str(i)/'sample.pt').exists():
                convert(cfg['entries'][i], Path(tmp)/'CHGCAR', specs, factors, dest, stage!='test')
            if j % 100 == 0:
                print(ds,stage,j+1,len(ids),flush=True)
    save(ROOT/ds/f'{stage}_prepared.json',dict(count=len(ids),split_sha256=cfg['split_sha256'],specs=specs))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset', choices=SPECIES)
    parser.add_argument('stage', choices=('fit','test'))
    args=parser.parse_args()
    prepare(args.dataset,args.stage)
