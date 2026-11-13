"""Locked fresh-geometry density+PAW benchmark, selection and scaling."""
import argparse
import gc
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
import torch
from pymatgen.io.vasp.outputs import Chgcar

from density_eqgnn.chgcar import vasp_grid
from scripts.benchmark_deepdft_reuse import CHECKPOINT,save
from scripts.compare_local_nmc_oracles import load_sample
from scripts.deepdft_paw import BLOCKS
from scripts.joint_density_predictor import JointPredictor,VARIANTS,DEEP_PAW,EDEN_PAW
from scripts.nmc_probe_holdout import sha256,spatial_split
from scripts.nmc_edengnn import ROOT as EDEN_ROOT
from scripts.train_deepdft_paw import measures

ROOT = Path('runs/joint_density_benchmark_2026-10-30')
SOURCES = ['scripts/benchmark_joint_density.py','scripts/joint_density_predictor.py',
    'scripts/deepdft_radial_table.py','scripts/radial_message_kernel.cu','scripts/audit_joint_density.py',
    'scripts/deepdft_inference_fast.py','scripts/deepdft_factored_messages.py','scripts/deepdft_near_far.py',
    'scripts/deepdft_cubic_far.py','scripts/deepdft_paw.py','scripts/edengnn_paw.py',
    'scripts/nmc_probe_holdout.py','scripts/nmc_edengnn.py','scripts/r16_grid_diagnostic.py',
    'scripts/benchmark_nmc_edengnn.py','scripts/evaluate_vasp_sad_nmc.py']
CHECKPOINTS = [CHECKPOINT,EDEN_ROOT/'eden_eight/best.pt',
    *[DEEP_PAW/f'seed_{s}/best.pt' for s in (42,43)],
    *[EDEN_PAW/f'finetuned_{s}/best.pt' for s in (42,43)]]


def source_path(sid):
    if sid.startswith('cache_'):
        return Path(f'data/nmc_cache/{sid.split("_")[1]}.CHGCAR.lz4')
    return Path(f'runs/nmc_scf_benchmark_10/sample_{int(sid.split("_")[1]):04d}/oracle_full/CHGCAR')


def prepare(root):
    if (root/'settings.json').exists():
        raise ValueError('Preserve prepared benchmark')
    audit = json.loads((root/'numerical_audit.json').read_text())
    assert audit['passed'] and all(sha256(p)==h for p,h in audit['sources'].items())
    base_cfg = json.loads((DEEP_PAW/'settings.json').read_text())
    assert base_cfg['structure_matcher_no_duplicates']
    base = torch.load(DEEP_PAW/'dataset.pt',map_location='cpu',weights_only=False)
    records = []
    for split,rows in base['samples'].items():
        for row in rows:
            sid = row['id']
            path = source_path(sid)
            start = time.perf_counter()
            if path.suffix=='.lz4':
                sample = load_sample(path,'cpu')
            else:
                chg = Chgcar.from_file(path)
                st,raw = chg.structure,chg.data['total']
                sample = dict(positions=torch.tensor(st.cart_coords,dtype=torch.float64),
                    numbers=torch.tensor(st.atomic_numbers),cell=torch.tensor(st.lattice.matrix,dtype=torch.float64),
                    probes=torch.tensor(vasp_grid(st.lattice.matrix,raw.shape)),
                    target=torch.tensor(raw.reshape(-1)/st.volume,dtype=torch.float64),
                    metadata=dict(source=str(path),sha256=sha256(path),formula=st.formula,atoms=len(st),
                        volume=st.volume,grid_shape=list(raw.shape)))
            parse_seconds = time.perf_counter()-start
            for key in ('positions','cell','numbers'):
                torch.testing.assert_close(sample[key],row[key])
            sample.update(id=sid,split=split,target_aug=row['target_aug'],aug_mask=row['aug_mask'])
            cell,pos = sample['cell'].numpy(),sample['positions'].numpy()
            inv = np.linalg.inv(cell)
            pos = pos-np.floor(pos@inv)@cell
            bounds = np.ceil(4*np.linalg.norm(inv,axis=0)).astype(int)+1
            shifts = np.array(list(itertools.product(*[range(-b,b+1) for b in bounds])))
            tree = cKDTree((pos[None]+(shifts@cell)[:,None]).reshape(-1,3))
            distances,_ = tree.query(sample['probes'].numpy(),workers=1)
            sample['nearest_distance'] = torch.tensor(distances,dtype=torch.float32)
            sample['metadata']['parse_seconds'] = parse_seconds
            destination = root/'data'/f'{sid}.pt'
            destination.parent.mkdir(parents=True,exist_ok=True)
            torch.save(sample,destination)
            records.append(dict(id=sid,split=split,source=str(path),source_sha256=sha256(path),
                data_sha256=sha256(destination),metadata=sample['metadata']))
            print('PREPARED '+sid,flush=True)
    save(root/'settings.json',dict(records=records,variants=VARIANTS,
        source_hashes={p:sha256(p) for p in SOURCES},checkpoint_hashes={str(p):sha256(p) for p in CHECKPOINTS},
        protocol_sha256=sha256('reports/joint_density_benchmark_2026-10-30/protocol.md'),
        base_paw_dataset_sha256=sha256(DEEP_PAW/'dataset.pt'),device=torch.cuda.get_device_name(),
        torch=torch.__version__,cuda=torch.version.cuda,tf32=False,dtype='float32',
        compute_capability=torch.cuda.get_device_capability(),
        device_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        warmup=2,repeat=5,validation_warmup=1,validation_repeat=3,
        numerical_audit_sha256=sha256(root/'numerical_audit.json')))


def check(root):
    cfg = json.loads((root/'settings.json').read_text())
    assert all(sha256(p)==h for p,h in cfg['source_hashes'].items()),'Sources changed'
    assert all(sha256(p)==h for p,h in cfg['checkpoint_hashes'].items()),'Checkpoint changed'
    assert sha256('reports/joint_density_benchmark_2026-10-30/protocol.md')==cfg['protocol_sha256']
    assert sha256(root/'numerical_audit.json')==cfg['numerical_audit_sha256']
    return cfg


def load(root,record):
    path = root/'data'/f"{record['id']}.pt"
    assert sha256(path)==record['data_sha256']
    s = torch.load(path,map_location='cpu',weights_only=False)
    return {k:v.cuda() if torch.is_tensor(v) else v for k,v in s.items()}


def inputs(sample):
    return {k:sample[k] for k in ('positions','numbers','cell','probes','metadata')}


def density_metrics(output,sample):
    pred,target = output.double(),sample['target'].double()
    error = pred-target
    denom = float(target.abs().sum())
    volume_element = sample['metadata']['volume']/len(target)
    row = dict(count=len(target),nmae=float(error.abs().sum())/denom,
        mae=float(error.abs().mean()),rmse=float(error.square().mean().sqrt()),
        absolute_error_sum=float(error.abs().sum()),target_absolute_sum=denom,
        target_electrons=float(target.sum())*volume_element,
        charge_bias_electrons=float(error.sum())*volume_element,
        absolute_charge_bias_percent=100*float(error.sum().abs()/target.sum().abs()),
        negative_integral_electrons=float(-pred.clamp(max=0).sum())*volume_element,
        negative_integral_fraction=float(-pred.clamp(max=0).sum())/denom)
    if 'nearest_distance' in sample:
        row['distance_bands'] = {}
        for low,high in ((0.,.5),(.5,1.),(1.,2.),(2.,float('inf'))):
            mask = (sample['nearest_distance']>=low)&(sample['nearest_distance']<high)
            if mask.any():
                row['distance_bands'][f'{low:g}-{high:g}'] = dict(count=int(mask.sum()),
                    mae=float(error[mask].abs().mean()),
                    nmae=float(error[mask].abs().sum()/target[mask].abs().sum()))
    if sample['id'].startswith('cache_') and sample.get('replica') is None:
        sid = int(sample['id'].split('_')[1])
        row['seen_spatial_holdout'] = {name:float(error[indices].abs().sum()/target[indices].abs().sum())
            for name,indices in spatial_split(len(target),sid).items() if name!='selection'}
    return row


def paw_metrics(output,sample):
    row = measures(output,sample['target_aug'],sample['aug_mask'])
    row['by_l'] = {}
    for l in range(3):
        mask = torch.zeros_like(sample['aug_mask'])
        for block in BLOCKS:
            if block['l']==l:
                mask[:,block['start']:block['stop']] = sample['aug_mask'][:,block['start']:block['stop']]
        row['by_l'][str(l)] = measures(output,sample['target_aug'],mask)
    return row


@torch.no_grad()
def timed(fn,warmup,repeat):
    torch.cuda.empty_cache()
    for _ in range(warmup):
        output = fn()
        del output
    seconds,absolute,increment,reserved,baselines = [],[],[],[],[]
    for _ in range(repeat):
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        baselines.append(baseline/2**20)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = fn()
        torch.cuda.synchronize()
        seconds.append(time.perf_counter()-start)
        absolute.append(torch.cuda.max_memory_allocated()/2**20)
        increment.append((torch.cuda.max_memory_allocated()-baseline)/2**20)
        reserved.append(torch.cuda.max_memory_reserved()/2**20)
        del output
    return dict(seconds=seconds,median_seconds=float(np.median(seconds)),p90_seconds=float(np.percentile(seconds,90)),
        peak_absolute_mib=max(absolute),peak_increment_mib=max(increment),peak_reserved_mib=max(reserved),
        baseline_allocated_mib=baselines)


@torch.no_grad()
def evaluate(root,variant,stage):
    cfg = check(root)
    if stage=='final':
        json.loads((root/'selection.json').read_text())
    folder = root/stage
    path = folder/f'{variant}.json'
    if path.exists():
        raise ValueError('Preserve existing results')
    started = time.perf_counter()
    predictor = JointPredictor(variant)
    torch.cuda.synchronize()
    result = dict(variant=variant,setup_seconds=time.perf_counter()-started,
        persistent_allocated_mib=torch.cuda.memory_allocated()/2**20,records=[],completed=False,
        density_model_parameters=sum(p.numel() for p in predictor.model.parameters()),
        occupation_model_parameters=sum(p.numel() for p in predictor.head.parameters()),
        provenance=getattr(predictor,'provenance',getattr(predictor.model,'provenance',None)))
    if hasattr(predictor.core,'tables'):
        result['table_mib'] = predictor.core.tables.numel()*predictor.core.tables.element_size()/2**20
        result['table_points'] = predictor.core.points
        if predictor.core.kernel:
            result['compiler_options'] = predictor.core.kernel.options
    chosen = [r for r in cfg['records'] if stage=='final' or r['split']=='validation']
    for record in chosen:
        sample = load(root,record)
        x = inputs(sample)
        row = dict(id=record['id'],split=record['split'],metadata=record['metadata'])
        try:
            warmup = cfg['warmup'] if stage=='final' else cfg['validation_warmup']
            repeat = cfg['repeat'] if stage=='final' else cfg['validation_repeat']
            row['joint_timing'] = timed(lambda:predictor.predict(x),warmup,repeat)
            if stage=='final':
                row['density_timing'] = timed(lambda:predictor.predict(x,joint=False),warmup,repeat)
            output = predictor.predict(x)
            row['density'] = density_metrics(output['density'],sample)
            row['paw'] = {'42':paw_metrics(output['augmentation'],sample)}
            if stage=='final':
                alternate = predictor.alternate_paw(x)
                row['paw']['43'] = paw_metrics(alternate,sample)
                destination = root/'predictions'/variant
                destination.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(destination/f"{record['id']}.npz",density=output['density'].cpu().numpy(),
                    augmentation42=output['augmentation'].cpu().numpy(),augmentation43=alternate.cpu().numpy())
                del alternate
            del output
            print('RESULT '+json.dumps(dict(variant=variant,id=row['id'],nmae_percent=100*row['density']['nmae'],
                joint_seconds=row['joint_timing']['median_seconds'])),flush=True)
        except torch.cuda.OutOfMemoryError:
            row['error'] = 'CUDA OOM; no chunk size or precision adjustment'
            print('OOM '+variant+' '+record['id'],flush=True)
        result['records'].append(row)
        save(path,result)
        del sample,x
        gc.collect()
        torch.cuda.empty_cache()
    check(root)
    result.update(completed=True,source_hashes_match=True,checkpoint_hashes_match=True)
    save(path,result)


def select(root):
    cfg = check(root)
    if (root/'selection.json').exists():
        raise ValueError('Preserve selection')
    rows = {}
    for name in VARIANTS:
        result = json.loads((root/'validation'/f'{name}.json').read_text())
        assert result['completed'] and len(result['records'])==2
        if any('error' in r for r in result['records']):
            rows[name] = dict(error='Unavailable on validation')
        else:
            rows[name] = dict(nmae=float(np.mean([r['density']['nmae'] for r in result['records']])),
                seconds=float(np.mean([r['joint_timing']['median_seconds'] for r in result['records']])))
    base,eden = rows['deep_reference'],rows['eden']
    selected = {}
    for label,tolerance in (('strict',1.01),('relaxed',1.05)):
        candidates = [name for name,r in rows.items() if name!='eden' and 'error' not in r and r['nmae']<=tolerance*base['nmae']]
        name = min(candidates,key=lambda n:rows[n]['seconds'])
        selected[label] = dict(name=name,accuracy_ratio=rows[name]['nmae']/base['nmae'],
            speed_ratio_to_eden=rows[name]['seconds']/eden['seconds'],
            speed_target_met=rows[name]['seconds']<=1.10*eden['seconds'])
    save(root/'selection.json',dict(selected=selected,validation=rows,
        settings_sha256=sha256(root/'settings.json'),test_metrics_opened=False))
    print(json.dumps(selected,indent=2))


def replicate(sample,factors):
    count = int(np.prod(factors))
    shape = sample['metadata']['grid_shape']
    shifts = torch.tensor(list(itertools.product(*[range(n) for n in factors])),device='cuda',dtype=torch.float64)@sample['cell']
    cell = sample['cell']*torch.tensor(factors,device='cuda')[:,None]
    full_shape = tuple(int(n*m) for n,m in zip(shape,factors))
    result = dict(id=sample['id'],replica=list(factors),split='replica_diagnostic',
        positions=(sample['positions'][None]+shifts[:,None]).reshape(-1,3),cell=cell,
        numbers=sample['numbers'].repeat(count),probes=torch.tensor(vasp_grid(cell.cpu().numpy(),full_shape),device='cuda'),
        target=sample['target'].reshape(*shape).repeat(*factors).flatten(),
        target_aug=sample['target_aug'].repeat(count,1),aug_mask=sample['aug_mask'].repeat(count,1),
        metadata=dict(sample['metadata'],grid_shape=full_shape,volume=sample['metadata']['volume']*count,
            atoms=sample['metadata']['atoms']*count))
    return result


@torch.no_grad()
def scaling(root,variant):
    cfg = check(root)
    selection = json.loads((root/'selection.json').read_text())
    allowed = {'deep_reference','eden',selection['selected']['strict']['name']}
    assert variant in allowed
    path = root/'scaling'/f'{variant}.json'
    if path.exists():
        raise ValueError('Preserve scaling')
    predictor = JointPredictor(variant)
    base = load(root,next(r for r in cfg['records'] if r['id']=='cache_0'))
    base_output = predictor.predict(inputs(base))
    records = []
    for factors in ((1,1,1),(2,1,1),(2,2,1)):
        sample = replicate(base,factors)
        row = dict(replica=factors,atoms=sample['metadata']['atoms'],points=len(sample['probes']))
        try:
            row['joint_timing'] = timed(lambda:predictor.predict(inputs(sample)),cfg['warmup'],cfg['repeat'])
            output = predictor.predict(inputs(sample))
            reference = base_output['density'].reshape(*base['metadata']['grid_shape']).repeat(*factors).flatten()
            aug_reference = base_output['augmentation'].repeat(int(np.prod(factors)),1)
            row['density_periodic_relative_l1'] = float((output['density']-reference).double().abs().sum()/reference.double().abs().sum())
            row['paw_periodic_max_abs'] = float((output['augmentation']-aug_reference).abs().max())
            row['density'] = density_metrics(output['density'],sample)
            row['paw'] = paw_metrics(output['augmentation'],sample)
            del output,reference,aug_reference
        except torch.cuda.OutOfMemoryError:
            row['error'] = 'CUDA OOM; unchanged configuration'
        records.append(row)
        save(path,dict(variant=variant,records=records,completed=False))
        print('SCALING '+json.dumps(dict(variant=variant,**row)),flush=True)
        del sample
        gc.collect()
        torch.cuda.empty_cache()
    check(root)
    save(path,dict(variant=variant,records=records,completed=True,source_hashes_match=True,checkpoint_hashes_match=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('prepare','validation','select','final','scaling'),required=True)
    parser.add_argument('--variant',choices=tuple(VARIANTS))
    parser.add_argument('--root',type=Path,default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.stage=='prepare': prepare(args.root)
    elif args.stage=='select': select(args.root)
    elif args.stage=='scaling': scaling(args.root,args.variant)
    else: evaluate(args.root,args.variant,args.stage)


if __name__=='__main__':
    main()
