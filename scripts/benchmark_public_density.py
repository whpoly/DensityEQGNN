"""Official pretrained NMC/EC benchmarks with immutable manifests and resume."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import time

import lz4.frame
import numpy as np
import torch
from pymatgen.io.vasp.outputs import Chgcar
from density_eqgnn.chgcar import vasp_grid

from scripts.download_public_density import digest,DISCOVERY
from scripts.deepdft_inference_fast import FlatProbe,flat_density
from scripts.deepdft_radial_table import RadialTableProbe
from scripts.nmc_probe_holdout import DEFAULT_ROOT,official_model,sha256

ROOT=Path('runs/public_benchmark_2026-10-30')
VARIANTS={'reference':4096,'fused_same_chunk':4096,'fused':8192}
SOURCES=['scripts/benchmark_public_density.py','scripts/deepdft_radial_table.py',
    'scripts/radial_message_kernel.cu','scripts/deepdft_inference_fast.py',
    'scripts/deepdft_factored_messages.py','scripts/r16_grid_diagnostic.py',
    'scripts/nmc_probe_holdout.py','src/density_eqgnn/chgcar.py']


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def checkpoint(dataset):
    name='nmc' if dataset=='nmc' else 'ethylenecarbonate'
    return DEFAULT_ROOT/'DeepDFT'/f'pretrained_models/{name}_painn/best_model.pth'


def prepare(dataset):
    root=ROOT/dataset;path=root/'settings.json'
    if path.exists():check(dataset);return
    preflight=json.loads((ROOT/f'{dataset}_preflight.json').read_text())
    assert preflight['passed']
    source=Path('data/public_density')/dataset
    metadata=json.loads((DISCOVERY/f'{dataset}.json').read_text())
    names=([f'cellrelax_{n:03d}.tar' for n in (0,1)] if dataset=='nmc' else
           (DISCOVERY/'ec_filelist.txt').read_text().splitlines())
    entries=[];receipts={}
    for name in names:
        archive=source/name
        published=next(r for r in metadata['files'] if r['name']==name)
        receipt=json.loads((source/(name+'.verified.json')).read_text())
        assert receipt['verified'] and archive.stat().st_size==published['size']
        assert digest(archive,'md5')==published['computed_md5']==receipt['md5']
        receipts[name]=receipt
        with tarfile.open(archive,'r:') as tf:
            for member in tf:
                assert member.isfile() and member.name.endswith('.CHGCAR.lz4')
                index=len(entries)
                expected=index+(1000 if dataset=='ec' else 0)
                assert int(Path(member.name).name.split('.')[0])==expected,'Official order mismatch'
                entries.append(dict(index=index,archive=str(archive),name=member.name,
                                    offset=member.offset_data,size=member.size))
    split=json.loads((source/'split.json').read_text())
    counts={'nmc':dict(train=1450,validation=50,test=500),
            'ec':dict(train=7330,validation=50,test=4000)}[dataset]
    assert {k:len(v) for k,v in split.items()}==counts
    flattened=[i for values in split.values() for i in values]
    assert len(set(flattened))==len(flattened)==len(entries)
    assert sorted(flattened)==list(range(len(entries)))
    rng=np.random.default_rng(20261030)
    timing=sorted(int(i) for i in rng.choice(split['test'],50,replace=False))
    cfg=dict(dataset=dataset,counts=counts,split=split,entries=entries,timing_indices=timing,
        checkpoint=str(checkpoint(dataset)),checkpoint_sha256=sha256(checkpoint(dataset)),
        source_hashes={s:sha256(s) for s in SOURCES},archive_receipts=receipts,
        split_sha256=sha256(source/'split.json'),preflight_sha256=sha256(ROOT/f'{dataset}_preflight.json'),
        protocol_sha256=sha256('reports/public_benchmark_2026-10-30/protocol.md'),
        variants=VARIANTS,table_points=16385,precision='FP32; TF32 disabled',
        accuracy_mode='all official test structures; every original grid point',
        timing_mode='50 fixed random test structures; 2 warmups + 5 synchronized repetitions; fresh geometry',
        device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda)
    save(path,cfg);print('PREPARED '+dataset,flush=True)


def check(dataset):
    cfg=json.loads((ROOT/dataset/'settings.json').read_text())
    assert all(sha256(s)==h for s,h in cfg['source_hashes'].items())
    assert sha256(cfg['checkpoint'])==cfg['checkpoint_sha256']
    assert sha256('reports/public_benchmark_2026-10-30/protocol.md')==cfg['protocol_sha256']
    assert sha256(Path('data/public_density')/dataset/'split.json')==cfg['split_sha256']
    return cfg


def model_for(dataset):
    model,provenance=official_model(DEFAULT_ROOT/'DeepDFT','cuda')
    model.load_state_dict(torch.load(checkpoint(dataset),map_location='cuda',weights_only=False)['model'])
    return model.eval().requires_grad_(False),provenance


def load(entry):
    with Path(entry['archive']).open('rb') as f:
        f.seek(entry['offset']);payload=f.read(entry['size'])
    return decode(payload,entry['name'])


def decode(payload,name):
    with tempfile.TemporaryDirectory(prefix='public_density_') as tmp:
        path=Path(tmp)/'CHGCAR'
        path.write_bytes(lz4.frame.decompress(payload))
        chgcar=Chgcar.from_file(path)
    structure=chgcar.structure
    raw=np.asarray(chgcar.data['total'],dtype=np.float64)
    # VASP stores density multiplied by cell volume; report electrons / Angstrom^3.
    density=raw/structure.volume
    sample=dict(positions=torch.tensor(structure.cart_coords,device='cuda',dtype=torch.float64),
        numbers=torch.tensor(structure.atomic_numbers,device='cuda'),
        cell=torch.tensor(structure.lattice.matrix,device='cuda',dtype=torch.float64),
        probes=torch.tensor(vasp_grid(structure.lattice.matrix,density.shape),device='cuda'),
        metadata=dict(grid_shape=list(density.shape),volume=structure.volume,atoms=len(structure),
            points=density.size,formula=structure.formula,name=name,
            member_sha256=hashlib.sha256(payload).hexdigest()))
    target=torch.tensor(density.reshape(-1),device='cuda',dtype=torch.float64)
    assert torch.isfinite(target).all() and target.abs().sum()>0
    return sample,target


def metric(pred,target,sample):
    p=pred.double();diff=p-target;den=target.abs().sum()
    return dict(nmae=float(diff.abs().sum()/den),mae=float(diff.abs().mean()),
        rmse=float(diff.square().mean().sqrt()),abs_error_sum=float(diff.abs().sum()),
        target_abs_sum=float(den),charge_bias_electrons=float(diff.sum())*sample['metadata']['volume']/len(target),
        negative_integral_fraction=float(-p.clamp(max=0).sum()/den))


def previous(path,cfg):
    if path.exists():
        result=json.loads(path.read_text());assert result['settings_sha256']==sha256(ROOT/cfg['dataset']/'settings.json')
        return result
    return dict(completed=False,settings_sha256=sha256(ROOT/cfg['dataset']/'settings.json'),records=[])


@torch.no_grad()
def accuracy(dataset):
    cfg=check(dataset);path=ROOT/dataset/'accuracy.json';result=previous(path,cfg)
    if result['completed']:return
    model,provenance=model_for(dataset)
    cores={'reference':FlatProbe(model),'fused':RadialTableProbe(model,fused=True)}
    finished={r['index'] for r in result['records']}
    for i in cfg['split']['test']:
        if i in finished:continue
        row=dict(index=i);sample=target=pred=delta=None
        try:
            start=time.perf_counter();sample,target=load(cfg['entries'][i])
            row.update(metadata=sample['metadata'],read_decode_transfer_seconds=time.perf_counter()-start)
            pred={}
            for variant in (('reference','fused') if i%2==0 else ('fused','reference')):
                pred[variant]=flat_density(model,cores[variant],sample,chunk=VARIANTS[variant])
            row['density']={n:metric(v,target,sample) for n,v in pred.items()}
            delta=(pred['fused'].double()-pred['reference'].double()).abs()
            relative=float(delta.sum()/pred['reference'].double().abs().sum())
            close=bool(torch.isclose(pred['fused'],pred['reference'],rtol=5e-5,atol=5e-5).all())
            row['agreement']=dict(relative_l1=relative,max_abs=float(delta.max()),
                allclose=close,passed=close and relative<1e-6)
        except torch.cuda.OutOfMemoryError:
            row['error']='CUDA OOM; no automatic chunk/precision change'
        result['records'].append(row);save(path,result)
        print('ACCURACY '+json.dumps(dict(index=i,done=len(result['records']),total=len(cfg['split']['test']),
            agreement=row.get('agreement'),error=row.get('error'))),flush=True)
        del sample,target,pred,delta;gc.collect();torch.cuda.empty_cache()
    check(dataset);result.update(completed=True,provenance=provenance);save(path,result)


@torch.no_grad()
def timing(dataset,variant):
    cfg=check(dataset);path=ROOT/dataset/f'timing_{variant}.json';result=previous(path,cfg)
    if result['completed']:return
    started=time.perf_counter();model,_=model_for(dataset)
    core=FlatProbe(model) if variant=='reference' else RadialTableProbe(model,fused=True)
    torch.cuda.synchronize();result['setup_seconds']=time.perf_counter()-started
    finished={r['index'] for r in result['records']}
    for i in cfg['timing_indices']:
        if i in finished:continue
        row=dict(index=i);sample=target=value=None
        try:
            sample,target=load(cfg['entries'][i]);del target;target=None
            row['metadata']=sample['metadata']
            for _ in range(2):
                value=flat_density(model,core,sample,chunk=VARIANTS[variant]);del value
            values=[];peaks=[];absolute=[]
            for _ in range(5):
                torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                started=time.perf_counter();value=flat_density(model,core,sample,chunk=VARIANTS[variant])
                torch.cuda.synchronize();values.append(time.perf_counter()-started)
                peaks.append((torch.cuda.max_memory_allocated()-baseline)/2**30)
                absolute.append(torch.cuda.max_memory_allocated()/2**30);del value
            row.update(seconds=values,median_seconds=float(np.median(values)),
                       peak_increment_gib=max(peaks),peak_absolute_gib=max(absolute))
        except torch.cuda.OutOfMemoryError:row['error']='CUDA OOM; no automatic chunk/precision change'
        result['records'].append(row);save(path,result)
        print('TIMING '+json.dumps(dict(variant=variant,index=i,done=len(result['records']),
            median_seconds=row.get('median_seconds'),error=row.get('error'))),flush=True)
        del sample,target;gc.collect();torch.cuda.empty_cache()
    check(dataset);result['completed']=True;save(path,result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('nmc','ec'),required=True)
    p.add_argument('--stage',choices=('prepare','accuracy','timing'),required=True)
    p.add_argument('--variant',choices=tuple(VARIANTS));args=p.parse_args()
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    if args.stage=='prepare':prepare(args.dataset)
    elif args.stage=='accuracy':accuracy(args.dataset)
    else:
        assert args.variant is not None;timing(args.dataset,args.variant)


if __name__=='__main__':main()
