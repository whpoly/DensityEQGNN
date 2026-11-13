"""Label-free chunk/algebra tuning followed by isolated eight-structure tests."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.benchmark_deepdft_reuse import CHECKPOINT,save
from scripts.compare_local_nmc_oracles import load_sample
from scripts.deepdft_inference_fast import FlatProbe,flat_density
from scripts.deepdft_cubic_far import expanded_far_density
from scripts.deepdft_factored_messages import FactoredProbe
from scripts.nmc_probe_holdout import DEFAULT_ROOT,official_model,sha256,spatial_split
from scripts.verify_deepdft_near_far import timed,metric


def candidates():
    result = {f'cubic_{kind}_{chunk}':dict(mode='cubic',factored=kind=='factored',chunk=chunk,radius=3.)
        for kind in ('plain','factored') for chunk in (4096,8192,16384)}
    result.update({f'full_factored_{chunk}':dict(mode='full',factored=True,chunk=chunk) for chunk in (4096,8192,16384)})
    return result


def run(model,core,sample,config):
    if config['mode']=='full':
        return flat_density(model,core,sample,chunk=config['chunk'])
    return expanded_far_density(model,core,sample,radius=config['radius'],chunk=config['chunk'])


def compare(out,ref):
    torch.testing.assert_close(out,ref,rtol=3e-5,atol=5e-5)
    relative = float((out.double()-ref.double()).abs().sum()/ref.double().abs().sum())
    if relative>=1e-6:
        raise AssertionError(f'Numerical equivalence failed: {relative}')
    return dict(max_abs=float((out-ref).abs().max()),relative_l1=relative)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage',choices=('select','final'),required=True)
    p.add_argument('--variant',choices=('original','previous_cubic3','fast_full','fast_cubic3','fast_cubic3.5'))
    p.add_argument('--output',type=Path,default=Path('runs/deepdft_cubic_fast_2026-10-30'))
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    model,provenance = official_model(DEFAULT_ROOT/'DeepDFT','cuda')
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cuda',weights_only=False)['model'])
    model.eval().requires_grad_(False)
    cores = {False:FlatProbe(model).eval(),True:FactoredProbe(model).eval()}
    sources = ['scripts/verify_cubic_fast.py','scripts/deepdft_factored_messages.py','scripts/deepdft_cubic_far.py',
        'scripts/deepdft_near_far.py','scripts/deepdft_inference_fast.py','scripts/nmc_probe_holdout.py','scripts/r16_grid_diagnostic.py']
    settings = dict(checkpoint_sha256=sha256(CHECKPOINT),provenance=provenance,
        source_hashes={s:sha256(s) for s in sources},candidates=candidates(),torch=torch.__version__,device=torch.cuda.get_device_name(),
        selection='Samples 0 and 3; no density labels used. Every candidate must match its existing full/cubic prediction with relative L1 <1e-6 and assert_close. Select fastest mean of per-structure medians separately for cubic and full.',
        final='One variant per fresh process. Eight structures, two warmups and three synchronized timed calls per structure. Refresh graphs, atom features, coarse/dense fields on every call. Exclude reading files, loading checkpoint, warmup and metric evaluation.')
    settings_path = args.output/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        raise ValueError('Settings/source changed: choose a new output directory')
    save(settings_path,settings)
    if args.stage=='select':
        records = []
        for sid in (0,3):
            sample = load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
            refs = {'full':flat_density(model,cores[False],sample,chunk=4096),
                    'cubic':expanded_far_density(model,cores[False],sample,radius=3.,chunk=4096)}
            row = dict(id=sid,variants={})
            for name,config in settings['candidates'].items():
                try:
                    out = run(model,cores[config['factored']],sample,config)
                    stat = dict(equivalence=compare(out,refs[config['mode']]),seconds=[],peak_increment_mib=0.)
                    del out
                    for _ in range(3):
                        out,t = timed(lambda:run(model,cores[config['factored']],sample,config))
                        stat['seconds'].append(t['seconds'])
                        stat['peak_increment_mib'] = max(stat['peak_increment_mib'],t['peak_increment_mib'])
                        del out
                    stat['median_seconds'] = float(np.median(stat['seconds']))
                    row['variants'][name] = stat
                    print('TUNE '+json.dumps(dict(id=sid,name=name,**stat)),flush=True)
                except torch.cuda.OutOfMemoryError:
                    row['variants'][name] = dict(error='CUDA OOM')
                    torch.cuda.empty_cache()
                    print('OOM '+name,flush=True)
            records.append(row)
            save(args.output/'tuning.json',records)
            del sample,refs
        stats = {name:float(np.mean([r['variants'][name]['median_seconds'] for r in records]))
            for name in settings['candidates'] if all('error' not in r['variants'][name] for r in records)}
        chosen = {mode:min([n for n in stats if settings['candidates'][n]['mode']==mode],key=stats.get)
                  for mode in ('full','cubic')}
        selection = dict(chosen=chosen,seconds=stats,configs={m:settings['candidates'][n] for m,n in chosen.items()})
        save(args.output/'selection.json',selection)
        print('SELECTION '+json.dumps(selection),flush=True)
        return
    if args.variant is None:
        p.error('--variant required for final')
    selection = json.loads((args.output/'selection.json').read_text())
    if args.variant=='original':
        config = dict(mode='full',factored=False,chunk=4096)
    elif args.variant=='previous_cubic3':
        config = dict(mode='cubic',factored=False,chunk=4096,radius=3.)
    elif args.variant=='fast_full':
        config = dict(selection['configs']['full'])
    else:
        config = dict(selection['configs']['cubic'])
        config['radius'] = 3.5 if args.variant=='fast_cubic3.5' else 3.
    path = args.output/'final'/f'{args.variant}.json'
    if path.exists():
        raise ValueError('Preserve existing final results')
    result = dict(variant=args.variant,config=config,records=[])
    for sid in range(8):
        torch.cuda.empty_cache()
        sample = load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        reference_config = dict(config,factored=False,chunk=4096)
        ref = run(model,cores[False],sample,reference_config)
        out = run(model,cores[config['factored']],sample,config)
        ids = torch.as_tensor(spatial_split(len(out),sid)['test'],device='cuda')
        row = dict(id=sid,equivalence=compare(out,ref),test_nmae=metric(out,sample['target'],ids),
            seconds=[],peak_increment_mib=0.)
        del ref,out,ids
        # Clear references' allocator history before this variant's warmup/timing.
        torch.cuda.empty_cache()
        fn = lambda:run(model,cores[config['factored']],sample,config)
        for _ in range(2):
            out = fn()
            del out
        for _ in range(3):
            out,stat = timed(fn)
            row['seconds'].append(stat['seconds'])
            row['peak_increment_mib'] = max(row['peak_increment_mib'],stat['peak_increment_mib'])
            del out
        row['median_seconds'] = float(np.median(row['seconds']))
        result['records'].append(row)
        save(path,result)
        print('FINAL '+json.dumps(dict(variant=args.variant,**row)),flush=True)
        del sample,fn
    result['source_hashes_match'] = all(sha256(s)==h for s,h in settings['source_hashes'].items())
    result['checkpoint_hash_matches'] = sha256(CHECKPOINT)==settings['checkpoint_sha256']
    save(path,result)
    print('COMPLETE '+args.variant,flush=True)


if __name__=='__main__':
    main()
