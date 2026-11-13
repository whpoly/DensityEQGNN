"""Locked small-data PAW readout experiment; test labels only after selection."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

from scripts.benchmark_deepdft_reuse import CHECKPOINT,save
from scripts.deepdft_paw import (ELEMENTS,BLOCKS,WIDTH,IRREPS,PAWReadout,potcar_specs,
    read_structure_and_aug,training_statistics,features,density_and_paw)
from scripts.nmc_probe_holdout import DEFAULT_ROOT,official_model,sha256
from scripts.train_local_nmc_overfit import state_digest

ROOT = Path('runs/deepdft_paw_2026-10-30')
SOURCES = ['scripts/deepdft_paw.py','scripts/train_deepdft_paw.py','scripts/nmc_probe_holdout.py']


def load_model():
    model,provenance = official_model(DEFAULT_ROOT/'DeepDFT','cuda')
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cuda',weights_only=False)['model'])
    return model.eval().requires_grad_(False),provenance


def device_sample(sample):
    return {k:(v.to('cuda') if torch.is_tensor(v) else
        {kk:vv.to('cuda') for kk,vv in v.items()} if k=='geometry' else v)
        for k,v in sample.items()}


def apply(head,sample):
    return head(sample['scalars'],sample['vectors'],sample['numbers'],sample['geometry'])


def measures(out,target,mask):
    error = (out.double()-target.double())[mask]
    truth = target.double()[mask]
    denominator = float(truth.abs().sum())
    return dict(count=len(error),mae=float(error.abs().mean()),rmse=float(error.square().mean().sqrt()),
        relative_l1=float(error.abs().sum())/denominator if denominator else None,
        absolute_error_sum=float(error.abs().sum()),target_absolute_sum=denominator,
        squared_error_sum=float(error.square().sum()))


def summarize(records):
    return dict(macro_mae=float(np.mean([r['mae'] for r in records])),
        macro_relative_l1=float(np.mean([r['relative_l1'] for r in records])),
        pooled_mae=sum(r['absolute_error_sum'] for r in records)/sum(r['count'] for r in records),
        pooled_relative_l1=sum(r['absolute_error_sum'] for r in records)/sum(r['target_absolute_sum'] for r in records))


@torch.no_grad()
def evaluate(head,samples,mean=None,details=False):
    if head is not None:
        head.eval()
    rows = []
    for s in samples:
        out = mean[s['numbers']] if head is None else apply(head,s)
        row = dict(id=s['id'],**measures(out,s['target_aug'],s['aug_mask']))
        if details:
            row['by_l'] = {}
            for l in range(3):
                mask = torch.zeros_like(s['aug_mask'])
                for b in BLOCKS:
                    if b['l']==l:
                        mask[:,b['start']:b['stop']] = s['aug_mask'][:,b['start']:b['stop']]
                row['by_l'][str(l)] = measures(out,s['target_aug'],mask)
            anisotropic = s['aug_mask'].clone()
            for b in BLOCKS:
                if b['l']==0:
                    anisotropic[:,b['start']:b['stop']] = False
            row['anisotropic'] = measures(out,s['target_aug'],anisotropic)
            row['by_element'] = {symbol:measures(out,s['target_aug'],s['aug_mask'] & (s['numbers']==z)[:,None])
                for symbol,z in ELEMENTS.items() if (s['numbers']==z).any()}
        rows.append(row)
    return dict(**summarize(rows),records=rows)


def prepare(root):
    if (root/'dataset.pt').exists():
        raise ValueError('Preserve prepared data')
    specs = potcar_specs()
    model,provenance = load_model()
    groups = {'train':[('cache_'+str(i),Path(f'data/nmc_cache/{i}.CHGCAR.lz4')) for i in range(8)],
        'validation':[(f'scf_{i}',Path(f'runs/nmc_scf_benchmark_10/sample_{i:04d}/oracle_full/CHGCAR')) for i in (170,314)],
        'test':[(f'scf_{i}',Path(f'runs/nmc_scf_benchmark_10/sample_{i:04d}/oracle_full/CHGCAR')) for i in (361,796,825,1056,1749)]}
    samples,metadata,structures = {},{},{}
    for split,paths in groups.items():
        samples[split],metadata[split] = [],[]
        for name,path in paths:
            s = read_structure_and_aug(path,specs,lmax=2)
            structures[name] = Structure(s['cell'].numpy(),s['numbers'].tolist(),s['positions'].numpy(),coords_are_cartesian=True)
            rank_max = {str(l):float(s['target_aug'][:,[i for b in BLOCKS if b['l']==l
                for i in range(b['start'],b['stop'])]].abs().max()) for l in range(5)}
            assert rank_max['3']==rank_max['4']==0
            with torch.no_grad():
                f = features(model,s)
            prepared = dict(id=name,**{k:s[k] for k in ('positions','cell','target_aug','aug_mask','metadata')},
                **{k:v.cpu() if torch.is_tensor(v) else {kk:vv.cpu() for kk,vv in v.items()} for k,v in f.items()})
            samples[split].append(prepared)
            metadata[split].append(dict(id=name,**s['metadata'],rank_max_abs=rank_max,
                valid_coefficients=int(s['aug_mask'].sum())))
            print('PREPARED',split,name,flush=True)
    matcher = StructureMatcher(ltol=1e-5,stol=1e-5,angle_tol=1e-3,primitive_cell=False,scale=False)
    names = list(structures)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if matcher.fit(structures[a],structures[b]):
                raise ValueError(f'Equivalent structures in split: {a}, {b}')
    # Also audit that copied SCF POTCARs match the exact local specifications.
    for split in ('validation','test'):
        for _,path in groups[split]:
            text = path.with_name('POTCAR').read_text()
            for symbol in Poscar_symbols(path.with_name('POSCAR')):
                assert specs[ELEMENTS[symbol]]['title'] in text
    mean,scale = training_statistics(samples['train'])
    settings = dict(provenance=provenance,checkpoint=str(CHECKPOINT),checkpoint_sha256=sha256(CHECKPOINT),
        frozen_model_state_sha256=state_digest(model.state_dict()),source_hashes={p:sha256(p) for p in SOURCES},
        potcar_specs=specs,metadata=metadata,excluded_duplicate_scf_ids=[1,3,6],
        structure_matcher_no_duplicates=True,lmax=2,width=WIDTH,irreps=str(IRREPS),seeds=[42,43],
        steps=6000,eval_every=200,batch=2,lr=1e-3,min_lr=1e-5,
        selection='Validation macro raw coefficient MAE; no test selection',
        limitation='Only L<=2 labels available; 8 train / 2 validation / 5 test structures; frozen eight-structure density encoder; original POTCAR versions not independently recoverable from CHGCAR',
        torch=torch.__version__,device=torch.cuda.get_device_name())
    save(root/'settings.json',settings)
    torch.save(dict(samples=samples,mean=mean,scale=scale),root/'dataset.pt')


def Poscar_symbols(path):
    from pymatgen.io.vasp.inputs import Poscar
    return Poscar.from_file(path).site_symbols


def train(root,seed):
    cfg = json.loads((root/'settings.json').read_text())
    assert all(sha256(p)==h for p,h in cfg['source_hashes'].items())
    if seed not in cfg['seeds']:
        raise ValueError('Seed not predeclared')
    folder = root/f'seed_{seed}'
    if folder.exists():
        raise ValueError('Preserve training results')
    folder.mkdir()
    torch.manual_seed(seed)
    data = torch.load(root/'dataset.pt',weights_only=False)
    samples = {split:[device_sample(s) for s in data['samples'][split]] for split in ('train','validation')}
    head = PAWReadout(data['mean'],data['scale'],lmax=cfg['lmax']).cuda()
    optimizer = torch.optim.Adam(head.parameters(),lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['steps'],eta_min=cfg['min_lr'])
    rng = np.random.default_rng(seed)
    best = float('inf')
    history = []
    start = time.perf_counter()
    for step in range(1,cfg['steps']+1):
        head.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in rng.choice(len(samples['train']),size=cfg['batch'],replace=False):
            s = samples['train'][index]
            error = (apply(head,s)-s['target_aug'])/head.scale[s['numbers']]
            losses.append(error[s['aug_mask']].square().mean())
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),5)
        optimizer.step()
        scheduler.step()
        if step%cfg['eval_every']==0:
            val = evaluate(head,samples['validation'])
            fit = evaluate(head,samples['train'])
            row = dict(step=step,loss=float(loss),validation_mae=val['macro_mae'],train_mae=fit['macro_mae'],
                elapsed_seconds=time.perf_counter()-start)
            if val['macro_mae']<best:
                best = val['macro_mae']
                torch.save(dict(head=head.state_dict(),step=step,validation=val,seed=seed,lmax=cfg['lmax'],
                    frozen_checkpoint_sha256=cfg['checkpoint_sha256']),folder/'best.pt')
            history.append(row)
            save(folder/'history.json',history)
            print('TRAIN '+json.dumps(row),flush=True)
    saved = torch.load(folder/'best.pt',weights_only=False,map_location='cpu')
    save(folder/'selection.json',dict(seed=seed,step=saved['step'],validation_mae=saved['validation']['macro_mae'],
        best_checkpoint_sha256=sha256(folder/'best.pt'),training_seconds=time.perf_counter()-start,
        parameters=sum(p.numel() for p in head.parameters()),source_hashes_match=all(sha256(p)==h for p,h in cfg['source_hashes'].items())))


@torch.no_grad()
def final(root):
    from e3nn import o3
    from scripts.compare_local_nmc_oracles import load_sample
    from scripts.deepdft_inference_fast import FlatProbe,flat_density
    cfg = json.loads((root/'settings.json').read_text())
    assert all(sha256(p)==h for p,h in cfg['source_hashes'].items())
    if (root/'results.json').exists():
        raise ValueError('Preserve final results')
    data = torch.load(root/'dataset.pt',weights_only=False)
    samples = {split:[device_sample(s) for s in ss] for split,ss in data['samples'].items()}
    # Require both validation selections before opening any test metric.
    selections = {seed:json.loads((root/f'seed_{seed}/selection.json').read_text()) for seed in cfg['seeds']}
    results = dict(baseline={split:evaluate(None,ss,mean=data['mean'].cuda(),details=True) for split,ss in samples.items()},seeds={})
    model,_ = load_model()
    initial_digest = state_digest(model.state_dict())
    for seed in cfg['seeds']:
        folder = root/f'seed_{seed}'
        assert sha256(folder/'best.pt')==selections[seed]['best_checkpoint_sha256']
        head = PAWReadout(data['mean'],data['scale'],lmax=cfg['lmax']).cuda().eval()
        head.load_state_dict(torch.load(folder/'best.pt',map_location='cuda',weights_only=False)['head'])
        row = dict(selection=selections[seed],metrics={split:evaluate(head,ss,details=True) for split,ss in samples.items()})
        timing,equivariance = [],[]
        for s in samples['test']:
            expected = apply(head,s)
            np.savez_compressed(folder/(s['id']+'_prediction.npz'),prediction=expected.cpu().numpy(),
                target=s['target_aug'].cpu().numpy(),mask=s['aug_mask'].cpu().numpy(),numbers=s['numbers'].cpu().numpy())
            for _ in range(3):
                apply(head,s)
                head(**features(model,s))
            times = dict(head=[],encoder_graph_and_head=[])
            for _ in range(10):
                torch.cuda.synchronize()
                t = time.perf_counter()
                actual = apply(head,s)
                torch.cuda.synchronize()
                times['head'].append(time.perf_counter()-t)
                t = time.perf_counter()
                actual = head(**features(model,s))
                torch.cuda.synchronize()
                times['encoder_graph_and_head'].append(time.perf_counter()-t)
                torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
            timing.append(dict(id=s['id'],seconds=times))
            # Rotate/invert physical structure, rerun actual DeepDFT features.
            torch.manual_seed(900+seed)
            q = torch.eye(3,device='cuda')[[1,2,0]]
            rotation = -o3.rand_matrix(device='cuda')
            transformed = dict(s,positions=s['positions']@rotation.double().T+.13,
                cell=s['cell']@rotation.double().T)
            out = head(**features(model,transformed))
            ref = expected@IRREPS.D_from_matrix(q@rotation@q.T).T
            torch.testing.assert_close(out,ref,rtol=3e-3,atol=3e-4)
            equivariance.append(dict(id=s['id'],max_abs=float((out-ref).abs().max()),
                relative_l1=float((out-ref).abs().sum()/ref.abs().sum())))
        row.update(timing=timing,equivariance=equivariance)
        results['seeds'][str(seed)] = row
        print('TEST '+json.dumps(dict(seed=seed,mae=row['metrics']['test']['macro_mae'],
            relative_l1=row['metrics']['test']['macro_relative_l1'])),flush=True)
    # One full-grid audit of the combined entry point; model state is identical
    # for both seeds and the PAW head never changes density intermediates.
    density_checks = []
    for sid in range(8):
        sample = load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        old = flat_density(model,FlatProbe(model),sample,chunk=4096)
        new = density_and_paw(model,head,sample,chunk=4096)['density']
        torch.testing.assert_close(new,old,rtol=3e-5,atol=5e-5)
        density_checks.append(dict(id=sid,max_abs=float((old-new).abs().max()),
            relative_l1=float((old-new).abs().sum()/old.abs().sum())))
        del old,new,sample
        print('DENSITY_UNCHANGED',sid,flush=True)
    results['audit'] = dict(density_checks=density_checks,
        density_model_state_unchanged=state_digest(model.state_dict())==initial_digest==cfg['frozen_model_state_sha256'],
        source_hashes_match=all(sha256(p)==h for p,h in cfg['source_hashes'].items()),
        checkpoint_hash_matches=sha256(CHECKPOINT)==cfg['checkpoint_sha256'])
    save(root/'results.json',results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('prepare','train','final'),required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--root',type=Path,default=ROOT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    args.root.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare': prepare(args.root)
    elif args.stage=='train': train(args.root,args.seed)
    else: final(args.root)


if __name__=='__main__':
    main()
