"""Validation-only selection of higher-order fixed-weight far-field sharing."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.benchmark_deepdft_reuse import CHECKPOINT,save
from scripts.compare_local_nmc_oracles import load_sample
from scripts.deepdft_inference_fast import FlatProbe,flat_density
from scripts.deepdft_near_far import near_far_density
from scripts.deepdft_cubic_far import expanded_far_density
from scripts.nmc_probe_holdout import DEFAULT_ROOT,official_model,sha256,spatial_split
from scripts.verify_deepdft_near_far import metric,timed


def candidates():
    items = [('linear',2.,1.,2),('linear',3.5,1.,2)]+[
        ('cubic',r,1.,2) for r in (2.,2.5,3.,3.5)]+[
        ('cubic',2.5,.5,2),('cubic',3.,1.,4)]
    return {f'{order}_r{r:g}_w{w:g}_s{s}':dict(order=order,radius=r,transition=w,stride=s)
            for order,r,w,s in items}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('runs/deepdft_cubic_far_2026-10-30'))
    p.add_argument('--samples',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--chunk',type=int,default=4096)
    p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--smoke',action='store_true',help='Only validation for fixed R=2 linear/cubic; never reads test labels')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    model,provenance = official_model(DEFAULT_ROOT/'DeepDFT','cuda')
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cuda',weights_only=False)['model'])
    model.eval().requires_grad_(False)
    core = FlatProbe(model).eval()
    configs = candidates()
    if args.smoke:
        configs = {n:c for n,c in configs.items() if n in ('linear_r2_w1_s2','cubic_r2_w1_s2')}
    sources = ['scripts/verify_deepdft_cubic_far.py','scripts/deepdft_cubic_far.py',
        'scripts/deepdft_near_far.py','scripts/deepdft_inference_fast.py',
        'scripts/nmc_probe_holdout.py','scripts/r16_grid_diagnostic.py','scripts/verify_deepdft_near_far.py']
    settings = dict(samples=args.samples,chunk=args.chunk,repeat=args.repeat,smoke=args.smoke,
        candidates=configs,checkpoint_sha256=sha256(CHECKPOINT),provenance=provenance,
        source_hashes={s:sha256(s) for s in sources},torch=torch.__version__,device=torch.cuda.get_device_name(),
        selection='All candidates fixed before validation. Fastest cubic candidate within 1% and 5% of original macro validation NMAE. Always include cubic R2, expanded linear R2 and prior gather-linear R3.5. If none within 5%, include lowest-error cubic. Lock selection before evaluating test metrics. Historical data reuse is not a new independent test.',
        timing='Every call recomputes atom features, all graphs, far messages and interpolation. Warmup then three rotated synchronized runs. Includes dense feature materialization; excludes file/checkpoint loading and metrics.')
    settings_path = args.output/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        raise ValueError('Settings/source changed; choose another output directory')
    save(settings_path,settings)
    val_path = args.output/'validation.json'
    validation = json.loads(val_path.read_text()) if val_path.exists() else []
    for sid in args.samples:
        if any(r['id']==sid for r in validation):
            continue
        sample = load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        ids = torch.as_tensor(spatial_split(len(sample['probes']),sid)['validation'],device='cuda')
        reference = flat_density(model,core,sample,chunk=args.chunk)
        row = dict(id=sid,metadata=sample['metadata'],original_validation_nmae=metric(reference,sample['target'],ids),variants={})
        for name,config in configs.items():
            out,stat = timed(lambda:expanded_far_density(model,core,sample,chunk=args.chunk,**config))
            stat['validation_nmae'] = metric(out,sample['target'],ids)
            stat['validation_relative_l1_vs_original'] = metric(out,reference,ids)
            shape = tuple(sample['metadata']['grid_shape'])
            stride = config['stride']
            torch.testing.assert_close(out.reshape(shape)[::stride,::stride,::stride],
                reference.reshape(shape)[::stride,::stride,::stride],rtol=3e-5,atol=5e-5)
            if config['order']=='linear':
                old = near_far_density(model,core,sample,chunk=args.chunk,
                    **{k:v for k,v in config.items() if k!='order'})
                torch.testing.assert_close(out,old,rtol=3e-5,atol=5e-5)
                stat['max_abs_vs_gather_linear'] = float((out-old).abs().max())
                del old
            row['variants'][name] = stat
            print('VALIDATION '+json.dumps(dict(id=sid,name=name,**stat)),flush=True)
            del out
        validation.append(row)
        save(val_path,validation)
        del reference,sample
    if args.smoke:
        print('SMOKE COMPLETE (validation only)',flush=True)
        return
    baseline = float(np.mean([r['original_validation_nmae'] for r in validation]))
    stats = {n:{'validation_nmae':float(np.mean([r['variants'][n]['validation_nmae'] for r in validation])),
                'provisional_seconds':float(np.mean([r['variants'][n]['seconds'] for r in validation]))}
             for n in configs}
    cubic = [n for n,c in configs.items() if c['order']=='cubic']
    best = min(cubic,key=lambda n:stats[n]['validation_nmae'])
    budget = {}
    for percent in (1,5):
        eligible = [n for n in cubic if stats[n]['validation_nmae']<=(1+percent/100)*baseline]
        budget[str(percent)] = min(eligible,key=lambda n:stats[n]['provisional_seconds']) if eligible else None
    selected = list(dict.fromkeys(['linear_r2_w1_s2','cubic_r2_w1_s2']+
        [n for n in budget.values() if n]+([best] if budget['5'] is None else [])))
    selection = dict(original_validation_nmae=baseline,candidates=stats,best_accuracy=best,
        fastest_within_percent=budget,selected=selected)
    selection_path = args.output/'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text())!=selection:
        raise ValueError('Locked selection changed')
    save(selection_path,selection)
    print('SELECTION LOCKED '+json.dumps(selection),flush=True)
    result_path = args.output/'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else dict(selection=selection,records=[])
    for sid in args.samples:
        if any(r['id']==sid for r in result['records']):
            continue
        sample = load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        split = {k:torch.as_tensor(v,device='cuda') for k,v in spatial_split(len(sample['probes']),sid).items()}
        reference = flat_density(model,core,sample,chunk=args.chunk)
        functions = {'original':lambda:flat_density(model,core,sample,chunk=args.chunk),
            'previous_linear_r3.5':lambda:near_far_density(model,core,sample,radius=3.5,transition=1.,stride=2,chunk=args.chunk)}
        functions.update({name:(lambda config=configs[name]:expanded_far_density(model,core,sample,chunk=args.chunk,**config)) for name in selected})
        row = dict(id=sid,variants={})
        for name,fn in functions.items():
            out = fn()
            row['variants'][name] = dict(test_nmae=metric(out,sample['target'],split['test']),
                validation_nmae=metric(out,sample['target'],split['validation']),
                relative_l1_vs_original=float((out.double()-reference.double()).abs().sum()/reference.double().abs().sum()),
                max_abs_vs_original=float((out-reference).abs().max()),seconds=[],peak_increment_mib=0.)
            del out
        names = list(functions)
        for repeat in range(args.repeat):
            for name in names[repeat%len(names):]+names[:repeat%len(names)]:
                out,stat = timed(functions[name])
                entry = row['variants'][name]
                entry['seconds'].append(stat['seconds'])
                entry['peak_increment_mib'] = max(entry['peak_increment_mib'],stat['peak_increment_mib'])
                del out
        for value in row['variants'].values():
            value['median_seconds'] = float(np.median(value['seconds']))
        result['records'].append(row)
        save(result_path,result)
        print('FINAL '+json.dumps(row),flush=True)
        del reference,sample
    result['source_hashes_match'] = all(sha256(s)==digest for s,digest in settings['source_hashes'].items())
    result['checkpoint_hash_matches'] = sha256(CHECKPOINT)==settings['checkpoint_sha256']
    save(result_path,result)
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
