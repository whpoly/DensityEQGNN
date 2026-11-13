"""Locked test evaluation, fresh-inference timing and reports for full NMC PAW."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.public_paw_experiment import (ROOT,ARMS,SEEDS,check,gpu_tree,make_head,
    predict,collate,evaluate,SOURCE_MANIFEST)
from scripts.benchmark_public_density import save,load,metric,model_for
from scripts.deepdft_paw import BLOCKS,features
from scripts.nmc_probe_holdout import sha256
from scripts.train_deepdft_paw import measures,summarize


def selections():
    cfg=check()
    assert sha256(ROOT/'audit.json')==cfg['audit_sha256']
    for arm in ARMS:
        assert sha256(ROOT/f'{arm}_features.pt')==json.loads((ROOT/f'{arm}_features.json').read_text())['sha256']
        for seed in SEEDS:
            folder=ROOT/f'{arm}_{seed}'
            s=json.loads((folder/'selection.json').read_text())
            assert s['settings_sha256']==sha256(ROOT/'settings.json')
            assert sha256(folder/'best.pt')==s['checkpoint_sha256']
            assert sha256('scripts/public_paw_experiment.py')==s['experiment_source_sha256']
    return cfg


def add_details(out,s):
    row=dict(id=s['id'],**measures(out,s['target_aug'],s['aug_mask']))
    row['by_l']={}
    for l in range(3):
        mask=torch.zeros_like(s['aug_mask'])
        for b in BLOCKS:
            if b['l']==l: mask[:,b['start']:b['stop']]=s['aug_mask'][:,b['start']:b['stop']]
        row['by_l'][str(l)]=measures(out,s['target_aug'],mask)
    row['by_element']={str(int(z)):measures(out,s['target_aug'],s['aug_mask']&(s['numbers']==z)[:,None])
        for z in s['numbers'].unique()}
    return row


@torch.no_grad()
def final(arm):
    cfg=selections();path=ROOT/f'{arm}_final.json'
    if path.exists(): raise ValueError('Preserve final results')
    data=torch.load(ROOT/f'{arm}_features.pt',weights_only=False)
    structures=torch.load(ROOT/'structures.pt',weights_only=False)
    heads={}
    for seed in SEEDS:
        heads[seed]=make_head(data,arm).eval().requires_grad_(False)
        heads[seed].load_state_dict(torch.load(ROOT/f'{arm}_{seed}/best.pt',map_location='cuda',weights_only=False)['model'])
    samples={i:gpu_tree(s) for i,s in enumerate(data['samples'])}
    results={str(seed):{split:evaluate(head,samples,cfg['split'][split],arm,details=True)
        for split in ('train','validation')} for seed,head in heads.items()}
    if arm=='deep_frozen': encoder,_=model_for('nmc')
    else:
        from scripts.edengnn_paw import inputs
        encoder=heads[42]
    test=cfg['split']['test'] if arm=='deep_frozen' else cfg['paired_test']
    records={seed:[] for seed in SEEDS};predictions={seed:[] for seed in SEEDS}
    baseline=[];discrepancy=[]
    for num,i in enumerate(test):
        s=structures[i];cached=samples[i]
        if arm=='deep_frozen': fresh=features(encoder,s)
        else: fresh=dict(eden_features=encoder.encode(*inputs(s)),numbers=s['numbers'].cuda())
        for seed,head in heads.items():
            out=predict(head,fresh,arm)
            ref=predict(head,cached,arm)
            delta=float((out-ref).abs().max())
            torch.testing.assert_close(out,ref,rtol=5e-5,atol=5e-5)
            discrepancy.append(delta)
            records[seed].append(add_details(out,cached))
            predictions[seed].append(dict(id=i,prediction=out.cpu(),mask=cached['aug_mask'].cpu()))
        baseline.append(add_details(heads[42].mean[cached['numbers']],cached))
        if (num+1)%50==0: print('FINAL',arm,num+1,len(test),flush=True)
    for seed in SEEDS:
        results[str(seed)]['test']=dict(**summarize(records[seed]),records=records[seed])
        common=[r for r in records[seed] if r['id'] in cfg['paired_test']]
        results[str(seed)]['paired_test']=dict(**summarize(common),records=common)
        torch.save(predictions[seed],ROOT/f'{arm}_{seed}/test_predictions.pt')
    common=[r for r in baseline if r['id'] in cfg['paired_test']]
    save(path,dict(completed=True,arm=arm,seeds=results,baseline=dict(**summarize(baseline),records=baseline),
        paired_baseline=dict(**summarize(common),records=common),fresh_vs_cached_max_abs=max(discrepancy),
        settings_sha256=sha256(ROOT/'settings.json'),finalizer_sha256=sha256(__file__)))


@torch.no_grad()
def verify():
    from e3nn import o3
    from scripts.public_paw_predictor import PublicPAWPredictor
    from scripts.deepdft_inference_fast import FlatProbe,flat_density
    from scripts.deepdft_paw import IRREPS
    cfg=selections();manifest=json.loads(SOURCE_MANIFEST.read_text())
    path=ROOT/'joint_verification.json'
    if path.exists(): raise ValueError('Preserve verification')
    predictor=PublicPAWPredictor();reference=FlatProbe(predictor.model)
    rows=[]
    # Fixed subset defined before this experiment, never selected by its errors.
    for i in manifest['timing_indices']:
        sample,target=load(manifest['entries'][i])
        original=flat_density(predictor.model,reference,sample,chunk=4096)
        result=predictor.predict(sample)
        delta=(result['density'].double()-original.double()).abs()
        relative=float(delta.sum()/original.double().abs().sum())
        torch.testing.assert_close(result['density'],original,rtol=5e-5,atol=5e-5)
        assert relative<1e-6
        row=dict(id=i,density_reference=metric(original,target,sample),density_joint=metric(result['density'],target,sample),
                 relative_l1=relative,max_abs=float(delta.max()))
        if len(rows)<3:
            directory=ROOT/'geometry_only_examples';directory.mkdir(exist_ok=True)
            np.savez_compressed(directory/f'nmc_{i}.npz',
                density=result['density'].cpu().numpy().reshape(sample['metadata']['grid_shape']),
                augmentation=result['augmentation'].cpu().numpy(),
                augmentation_mask=result['augmentation_mask'].cpu().numpy(),
                positions=sample['positions'].cpu().numpy(),cell=sample['cell'].cpu().numpy(),
                numbers=sample['numbers'].cpu().numpy(),supported_lmax=np.array(2))
        # No grid rotation is needed when testing the atom-only occupation branch.
        if len(rows)<5:
            torch.manual_seed(1357+i)
            rotation=o3.rand_matrix(dtype=torch.float64)
            if len(rows)%2: rotation=-rotation
            rotated={k:v for k,v in sample.items() if k in ('positions','numbers','cell')}
            r=rotation.cuda()
            rotated.update(positions=sample['positions']@r.T+.17,cell=sample['cell']@r.T)
            changed=predictor.predict(rotated,density=False)['augmentation']
            q=torch.eye(3,dtype=torch.float64)[[1,2,0]]
            d=IRREPS.D_from_matrix(q@rotation@q.T).float().cuda()
            expected=result['augmentation']@d.T
            torch.testing.assert_close(changed,expected,rtol=5e-4,atol=5e-4)
            row['rotation_max_abs']=float((changed-expected).abs().max())
        rows.append(row)
        print('VERIFY',len(rows),len(manifest['timing_indices']),relative,flush=True)
        del sample,target,original,result,delta
        gc.collect();torch.cuda.empty_cache()
    save(path,dict(completed=True,passed=True,records=rows,
        deep_checkpoint_unchanged=sha256(cfg['deep_checkpoint'])==cfg['deep_checkpoint_sha256']))


@torch.no_grad()
def timing(mode):
    from scripts.public_paw_predictor import PublicPAWPredictor
    cfg=selections();manifest=json.loads(SOURCE_MANIFEST.read_text())
    path=ROOT/f'timing_{mode}.json'
    if path.exists(): raise ValueError('Preserve timing')
    start=time.perf_counter()
    if mode=='eden_occupation':
        from scripts.edengnn_paw import inputs
        data=torch.load(ROOT/'eden_frozen_features.pt',weights_only=False)
        predictor=make_head(data,'eden_frozen').eval().requires_grad_(False)
        predictor.load_state_dict(torch.load(ROOT/'eden_frozen_42/best.pt',map_location='cuda',weights_only=False)['model'])
        del data
        call=lambda s:predictor(*inputs(s))
    else:
        predictor=PublicPAWPredictor()
        call=lambda s:predictor.predict(s,density=mode!='deep_occupation',occupation=mode!='density')
    torch.cuda.synchronize();setup=time.perf_counter()-start
    rows=[]
    ids=manifest['timing_indices'] if mode in ('density','joint') else [i for i in manifest['timing_indices'] if i in cfg['paired_test']]
    for i in ids:
        sample,target=load(manifest['entries'][i]);del target
        if mode.endswith('occupation'): sample={k:sample[k] for k in ('positions','cell','numbers')}
        for _ in range(2): output=call(sample);del output
        times=[];peak=[];absolute=[]
        for _ in range(5):
            torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter();output=call(sample);torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
            peak.append((torch.cuda.max_memory_allocated()-base)/2**30)
            absolute.append(torch.cuda.max_memory_allocated()/2**30);del output
        rows.append(dict(id=i,seconds=times,median_seconds=float(np.median(times)),
                         peak_increment_gib=max(peak),peak_absolute_gib=max(absolute)))
        print('TIMING',mode,len(rows),len(ids),round(rows[-1]['median_seconds'],5),flush=True)
        del sample;gc.collect();torch.cuda.empty_cache()
    save(path,dict(completed=True,mode=mode,setup_seconds=setup,records=rows,
        mean_seconds=float(np.mean([r['median_seconds'] for r in rows])),
        median_seconds=float(np.median([r['median_seconds'] for r in rows]))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['final','verify','timing'],required=True)
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--mode',choices=['density','joint','deep_occupation','eden_occupation'])
    args=p.parse_args();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if args.stage=='final': final(args.arm)
    elif args.stage=='verify': verify()
    else: timing(args.mode)


if __name__=='__main__': main()
