"""Widen the single-layer local query head, retaining the mixed-embedding protocol."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts import verify_mixed_embedding as base
from scripts.deepdft_mixed_embedding import MixedProbe, mixed_density
from scripts.deepdft_inference_fast import FlatProbe, flat_density
from scripts.nmc_probe_holdout import DEFAULT_ROOT, official_model, sha256

REFERENCE = Path('runs/mixed_embedding_2026-10-29')


def checkpoint_path(args, width, seed):
    root = REFERENCE if width == 64 else args.output/f'width_{width}'
    return root/f'seed_{seed}'/'local_best.pt'


@torch.no_grad()
def final(args, backbone, samples):
    widths = [64]+args.widths
    result = {'accuracy':[], 'timing':[], 'reference_sha256':sha256(base.CHECKPOINT)}
    heads = {}
    for width in widths:
        for seed in args.seeds:
            path = checkpoint_path(args, width, seed)
            state = torch.load(path, map_location='cuda', weights_only=False)
            head = MixedProbe('local', width).cuda().eval()
            head.load_state_dict(state['model'])
            row = {'width':width, 'seed':seed, 'step':state['step'],
                   'head_parameters':sum(p.numel() for p in head.parameters() if p.requires_grad),
                   'sha256':sha256(path), 'selection':state['selection'],
                   'test':base.evaluate(head, samples, 'test'),
                   'validation':base.evaluate(head, samples, 'validation')}
            result['accuracy'].append(row)
            if seed == args.seeds[0]:
                heads[width] = head
            base.save(args.output/'result.json', result)
            print('FINAL '+json.dumps({'width':width,'seed':seed,'test':row['test']['macro_nmae']}),flush=True)
    core = FlatProbe(backbone).eval()
    for s in samples:
        functions = {'original':lambda:flat_density(backbone, core, s, None, args.chunk)}
        functions.update({f'local_{w}':(lambda h=h:mixed_density(backbone,h,s,args.chunk))
                          for w,h in heads.items()})
        row = {'id':s['id'], 'variants':{}, 'test_nmae':{}}
        for name,fn in functions.items():
            out = fn()
            ids = s['split']['test']
            value = float((out[ids].double()-s['target'][ids]).abs().sum()/s['target'][ids].abs().sum())
            row['test_nmae'][name] = value
            if name != 'original':
                width = int(name.split('_')[1])
                expected = next(r for r in result['accuracy'] if r['width']==width and r['seed']==args.seeds[0])
                expected = next(r['nmae'] for r in expected['test']['records'] if r['id']==s['id'])
                assert abs(value-expected)<1e-6
            del out
        names = list(functions)
        for repeat in range(args.repeat):
            for name in names[repeat%len(names):]+names[:repeat%len(names)]:
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                out = functions[name]()
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                entry = row['variants'].setdefault(name, {'seconds':[], 'peak_increment_mib':0.})
                entry['seconds'].append(elapsed)
                entry['peak_increment_mib'] = max(entry['peak_increment_mib'],
                    (torch.cuda.max_memory_allocated()-allocated)/2**20)
                del out
        for stat in row['variants'].values():
            stat['median_seconds'] = float(np.median(stat['seconds']))
        result['timing'].append(row)
        base.save(args.output/'result.json', result)
        print('TIMING '+json.dumps(row),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('runs/local_width_2026-10-29'))
    p.add_argument('--widths',type=int,nargs='+',default=[128,256])
    p.add_argument('--samples',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--probes',type=int,default=1024)
    p.add_argument('--eval-every',type=int,default=500)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--chunk',type=int,default=4096)
    p.add_argument('--repeat',type=int,default=3)
    p.add_argument('--skip-final',action='store_true')
    args = p.parse_args()
    if 64 in args.widths or len(set(args.widths))!=len(args.widths):
        p.error('64 is the existing reference; use distinct new widths')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    backbone,provenance = official_model(DEFAULT_ROOT/'DeepDFT','cuda')
    backbone.load_state_dict(torch.load(base.CHECKPOINT,map_location='cuda',weights_only=False)['model'])
    backbone.eval().requires_grad_(False)
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='skip_final'}
    settings.update(backbone_sha256=sha256(base.CHECKPOINT),provenance=provenance,
        torch=torch.__version__,device=torch.cuda.get_device_name(),
        source_hashes={str(p):sha256(p) for p in [Path(__file__),Path('scripts/verify_mixed_embedding.py'),
            Path('scripts/deepdft_mixed_embedding.py'),Path('scripts/deepdft_inference_fast.py')]})
    settings_path = args.output/'settings.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        raise ValueError('Settings/source changed: choose a new output directory')
    base.save(settings_path,settings)
    samples = base.prepare(backbone,args.samples)
    base.save(args.output/'sources.json',[{'id':s['id'],**s['metadata']} for s in samples])
    base.MODES = ('local',)
    for width in args.widths:
        run = copy.copy(args)
        run.width = width
        run.output = args.output/f'width_{width}'
        print(f'TRAIN WIDTH {width}',flush=True)
        base.train(run,backbone,samples)
    if not args.skip_final:
        final(args,backbone,samples)


if __name__=='__main__':
    main()
