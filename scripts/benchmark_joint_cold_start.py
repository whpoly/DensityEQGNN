"""Supplementary fresh-process latency; does not change model selection.

Run only after the locked serial benchmark has completed. This is process-cold,
not OS-cache-cold, and excludes raw CHGCAR parsing and VASP-file serialization.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('runs/joint_density_benchmark_2026-10-30')


def child(variant):
    started = time.perf_counter()
    import torch
    from scripts.joint_density_predictor import JointPredictor

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    imported = time.perf_counter()
    sample = torch.load(ROOT/'cold_geometry.pt', map_location='cpu', weights_only=False)
    sample = {k:v.cuda() if torch.is_tensor(v) else v for k,v in sample.items()}
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    predictor = JointPredictor(variant)
    torch.cuda.synchronize()
    initialized = time.perf_counter()
    with torch.no_grad():
        output = predictor.predict(sample)
    torch.cuda.synchronize()
    predicted = time.perf_counter()
    output = {k:v.cpu() for k,v in output.items()}
    copied = time.perf_counter()
    assert all(torch.isfinite(v).all() for v in output.values())
    print('COLD_RESULT '+json.dumps(dict(
        variant=variant,imports_seconds=imported-started,
        cached_input_and_cuda_context_seconds=loaded-imported,
        predictor_setup_seconds=initialized-loaded,
        first_forward_seconds=predicted-initialized,
        output_to_cpu_seconds=copied-predicted,
        child_start_to_cpu_output_seconds=copied-started,
        density_points=output['density'].numel(),
        augmentation_shape=list(output['augmentation'].shape))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--child',choices=('deep_reference','deep_fused','eden'))
    args = parser.parse_args()
    if args.child:
        child(args.child)
        return
    from scripts.benchmark_joint_density import check
    import torch

    check(ROOT)
    assert json.loads((ROOT/'output_verification.json').read_text())['passed']
    destination = ROOT/'cold_start.json'
    assert not destination.exists(), 'Preserve existing results'
    source = torch.load(ROOT/'data/cache_0.pt',map_location='cpu',weights_only=False)
    torch.save({k:source[k] for k in ('positions','numbers','cell','probes','metadata')},
        ROOT/'cold_geometry.pt')
    # Fixed round-robin order reduces systematic run-order differences. No warmup.
    result = dict(completed=False,structure='cache_0',repetitions=3,
        supplementary_post_selection=True,
        scope='New Python process through CPU outputs; cached geometry input; no OS cache flush; '
              'no raw CHGCAR parsing, output serialization or SCF',records=[],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for repeat in range(3):
        for variant in ('deep_reference','deep_fused','eden'):
            started = time.perf_counter()
            completed = subprocess.run([sys.executable,'-m','scripts.benchmark_joint_cold_start',
                '--child',variant],text=True,capture_output=True,check=True)
            wall = time.perf_counter()-started
            (ROOT/f'cold_{variant}_{repeat}.log').write_text(
                completed.stdout+'\n'+completed.stderr,encoding='utf-8')
            matches = [line for line in completed.stdout.splitlines() if line.startswith('COLD_RESULT ')]
            assert len(matches)==1
            row = json.loads(matches[0].removeprefix('COLD_RESULT '))
            row.update(repeat=repeat,process_launch_to_exit_seconds=wall)
            result['records'].append(row)
            destination.write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)
    check(ROOT)
    result['medians'] = {variant:{key:statistics.median(r[key] for r in result['records']
        if r['variant']==variant) for key in (
            'imports_seconds','cached_input_and_cuda_context_seconds','predictor_setup_seconds',
            'first_forward_seconds','output_to_cpu_seconds','child_start_to_cpu_output_seconds',
            'process_launch_to_exit_seconds')}
        for variant in ('deep_reference','deep_fused','eden')}
    result['completed'] = True
    destination.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
