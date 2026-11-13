"""Timing robustness check: exactly one variant per fresh Python/CUDA process.

Triggered by large spread in the original interleaved experiment. Retains those
measurements, fixes the evaluated variants before this rerun, and records CUDA
reserved memory without assuming allocation history is the cause of variance.
"""
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
from scripts.nmc_probe_holdout import DEFAULT_ROOT,official_model,sha256
from scripts.verify_deepdft_near_far import timed
from scripts.verify_deepdft_cubic_far import candidates


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant',required=True)
    p.add_argument('--run',type=Path,default=Path('runs/deepdft_cubic_far_2026-10-30'))
    args = p.parse_args()
    locked = json.loads((args.run/'selection.json').read_text())
    settings = json.loads((args.run/'settings.json').read_text())
    allowed = ['original','previous_linear_r3.5']+locked['selected']
    if args.variant not in allowed:
        raise ValueError('Variant was not selected before test evaluation')
    output = args.run/'isolated_timing'/f'{args.variant}.json'
    if output.exists():
        raise ValueError('Preserve existing timing records; choose a fresh run directory')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    model,_ = official_model(DEFAULT_ROOT/'DeepDFT','cuda')
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cuda',weights_only=False)['model'])
    model.eval().requires_grad_(False)
    core = FlatProbe(model).eval()
    config = candidates().get(args.variant)
    hashes = {s:sha256(s) for s in settings['source_hashes']}
    assert hashes == settings['source_hashes']
    result = dict(variant=args.variant,protocol='One variant per fresh process; eight structures in order, empty_cache between structures outside timing, two warmups then five synchronized wall-time repeats. No labels are evaluated. All previous records retained.',
        repeat=5,warmup=2,source_hashes=hashes,runner_sha256=sha256(__file__),
        checkpoint_sha256=sha256(CHECKPOINT),records=[])
    for sid in settings['samples']:
        torch.cuda.empty_cache()
        sample = load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        if args.variant=='original':
            fn = lambda:flat_density(model,core,sample,chunk=settings['chunk'])
        elif args.variant=='previous_linear_r3.5':
            fn = lambda:near_far_density(model,core,sample,radius=3.5,transition=1.,stride=2,chunk=settings['chunk'])
        else:
            fn = lambda:expanded_far_density(model,core,sample,chunk=settings['chunk'],**config)
        for _ in range(result['warmup']):
            out = fn()
            del out
        row = dict(id=sid,seconds=[],peak_increment_mib=0.,reserved_before_mib=[],peak_reserved_mib=[])
        for _ in range(result['repeat']):
            row['reserved_before_mib'].append(torch.cuda.memory_reserved()/2**20)
            out,stat = timed(fn)
            row['seconds'].append(stat['seconds'])
            row['peak_increment_mib'] = max(row['peak_increment_mib'],stat['peak_increment_mib'])
            row['peak_reserved_mib'].append(torch.cuda.max_memory_reserved()/2**20)
            del out
        row['median_seconds'] = float(np.median(row['seconds']))
        result['records'].append(row)
        save(output,result)
        print('ISOLATED '+json.dumps(dict(variant=args.variant,**row)),flush=True)
        del sample,fn
    result['source_hashes_match'] = all(sha256(s)==digest for s,digest in hashes.items())
    result['checkpoint_hash_matches'] = sha256(CHECKPOINT)==settings['checkpoint_sha256']
    save(output,result)
    print('COMPLETE '+args.variant,flush=True)


if __name__=='__main__':
    main()
