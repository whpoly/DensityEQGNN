"""Validation-selected, fixed-checkpoint near/far message reuse experiment."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.benchmark_deepdft_reuse import CHECKPOINT, save
from scripts.compare_local_nmc_oracles import load_sample
from scripts.deepdft_inference_fast import FlatProbe, flat_density
from scripts.deepdft_near_far import near_far_density
from scripts.nmc_probe_holdout import DEFAULT_ROOT, official_model, sha256, spatial_split


def candidates():
    items = [(r,1.,2) for r in (2.,2.5,3.,3.5,4.)]+[(2.,.5,2),(3.,.5,2),(4.,2.,2),(3.,1.,4)]
    return {f'near{r:g}_blend{w:g}_stride{s}':dict(radius=r,transition=w,stride=s) for r,w,s in items}


def metric(out, target, ids):
    return float((out[ids].double()-target[ids].double()).abs().sum()/target[ids].double().abs().sum())


def timed(fn):
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, {'seconds':time.perf_counter()-start,
        'peak_increment_mib':(torch.cuda.max_memory_allocated()-baseline)/2**20}


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('runs/deepdft_near_far_2026-10-30'))
    p.add_argument('--samples',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--chunk',type=int,default=4096)
    p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--smoke',action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    model, provenance = official_model(DEFAULT_ROOT/'DeepDFT','cuda')
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cuda',weights_only=False)['model'])
    model.eval().requires_grad_(False)
    core = FlatProbe(model).eval()
    configs = candidates()
    if args.smoke:
        configs = {k:v for k,v in configs.items() if k=='near2_blend1_stride2'}
    sources = ['scripts/verify_deepdft_near_far.py','scripts/deepdft_near_far.py',
               'scripts/deepdft_inference_fast.py','scripts/nmc_probe_holdout.py',
               'scripts/r16_grid_diagnostic.py']
    settings = dict(samples=args.samples,chunk=args.chunk,repeat=args.repeat,smoke=args.smoke,
        candidates=configs,checkpoint_sha256=sha256(CHECKPOINT),provenance=provenance,
        source_hashes={s:sha256(s) for s in sources},torch=torch.__version__,device=torch.cuda.get_device_name(),
        selection='Macro NMAE on full validation split; best accuracy, plus fastest provisional full-grid time among <=1.05x original validation NMAE. Always include predeclared radius2/transition1/stride2 and its near-only control. Test opened only after selection saved.',
        timing='Fresh atom graph, atom features, both query graphs, coarse fields and interpolation in every call. Warmup then rotated order, synchronized wall time. Excludes loading checkpoint/data and error metrics.')
    settings_path = args.output/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        raise ValueError('Settings or code changed; choose a new output directory')
    save(settings_path,settings)
    val_path = args.output/'validation.json'
    validation = json.loads(val_path.read_text()) if val_path.exists() else []
    for sid in args.samples:
        if any(row['id']==sid for row in validation):
            continue
        sample = load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        ids = torch.as_tensor(spatial_split(len(sample['probes']),sid)['validation'],device='cuda')
        reference = flat_density(model,core,sample,chunk=args.chunk)
        row = {'id':sid,'metadata':sample['metadata'],'original_validation_nmae':metric(reference,sample['target'],ids),'variants':{}}
        for name,config in configs.items():
            out,stat = timed(lambda:near_far_density(model,core,sample,chunk=args.chunk,**config))
            stat['validation_nmae'] = metric(out,sample['target'],ids)
            stat['validation_relative_l1_vs_original'] = metric(out,reference,ids)
            shape = tuple(sample['metadata']['grid_shape'])
            stride = config['stride']
            torch.testing.assert_close(out.reshape(shape)[::stride,::stride,::stride],
                reference.reshape(shape)[::stride,::stride,::stride],rtol=3e-5,atol=5e-5)
            row['variants'][name] = stat
            print('VALIDATION '+json.dumps({'id':sid,'name':name,**stat}),flush=True)
            del out
        validation.append(row)
        save(val_path,validation)
        del reference, sample
    baseline = float(np.mean([r['original_validation_nmae'] for r in validation]))
    stats = {name:{'validation_nmae':float(np.mean([r['variants'][name]['validation_nmae'] for r in validation])),
                   'provisional_seconds':float(np.mean([r['variants'][name]['seconds'] for r in validation]))}
             for name in configs}
    best = min(stats,key=lambda name:stats[name]['validation_nmae'])
    eligible = [n for n,s in stats.items() if s['validation_nmae']<=1.05*baseline]
    budget = min(eligible,key=lambda name:stats[name]['provisional_seconds']) if eligible else None
    selected = list(dict.fromkeys(['near2_blend1_stride2',best]+([budget] if budget else [])))
    selection = dict(original_validation_nmae=baseline,candidates=stats,best_accuracy=best,
        eligible_5percent=eligible,fastest_within_5percent=budget,selected=selected)
    selection_path = args.output/'selection.json'
    if selection_path.exists() and json.loads(selection_path.read_text())!=selection:
        raise ValueError('Selection changed after being locked')
    save(selection_path,selection)
    print('SELECTION LOCKED '+json.dumps(selection),flush=True)
    result_path = args.output/'result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {'selection':selection,'records':[]}
    for sid in args.samples:
        if any(row['id']==sid for row in result['records']):
            continue
        sample = load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        split = {k:torch.as_tensor(v,device='cuda') for k,v in spatial_split(len(sample['probes']),sid).items()}
        reference = flat_density(model,core,sample,chunk=args.chunk)
        functions = {'original':lambda:flat_density(model,core,sample,chunk=args.chunk),
            'near_only2':lambda:near_far_density(model,core,sample,radius=2.,transition=1.,stride=2,chunk=args.chunk,near_only=True)}
        functions.update({name:(lambda config=configs[name]:near_far_density(model,core,sample,chunk=args.chunk,**config)) for name in selected})
        row = {'id':sid,'variants':{}}
        for name,fn in functions.items():
            out = fn()  # warmup, also the sealed test evaluation
            row['variants'][name] = {
                'test_nmae':metric(out,sample['target'],split['test']),
                'validation_nmae':metric(out,sample['target'],split['validation']),
                'relative_l1_vs_original':float((out.double()-reference.double()).abs().sum()/reference.double().abs().sum()),
                'max_abs_vs_original':float((out-reference).abs().max()),
                'seconds':[],'peak_increment_mib':0.}
            del out
        names = list(functions)
        for repeat in range(args.repeat):
            order = names[repeat%len(names):]+names[:repeat%len(names)]
            for name in order:
                out,stat = timed(functions[name])
                item = row['variants'][name]
                item['seconds'].append(stat['seconds'])
                item['peak_increment_mib'] = max(item['peak_increment_mib'],stat['peak_increment_mib'])
                del out
        for value in row['variants'].values():
            value['median_seconds'] = float(np.median(value['seconds']))
        result['records'].append(row)
        save(result_path,result)
        print('FINAL '+json.dumps(row),flush=True)
        del reference, sample
    result['source_hashes_match'] = all(sha256(s)==digest for s,digest in settings['source_hashes'].items())
    result['checkpoint_hash_matches'] = sha256(CHECKPOINT)==settings['checkpoint_sha256']
    save(result_path,result)
    print('COMPLETE',flush=True)


if __name__=='__main__':
    main()
