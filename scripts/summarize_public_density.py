"""Summarize observed public benchmark records without extrapolating missing cases."""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path('runs/public_benchmark_2026-10-30')
REPORT=Path('reports/public_benchmark_2026-10-30')


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None


def distribution(values):
    x=np.asarray(values,dtype=float)
    return dict(mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),
                maximum=float(x.max()),count=len(x))


def interval(values):
    x=np.asarray(values,dtype=float);rng=np.random.default_rng(20261030)
    means=[float(rng.choice(x,len(x),replace=True).mean()) for _ in range(2000)]
    return np.quantile(means,[.025,.975]).tolist()


def summarize(dataset):
    REPORT.mkdir(parents=True,exist_ok=True)
    expected=500 if dataset=='nmc' else 4000
    cfg=read(ROOT/dataset/'settings.json')
    accuracy=read(ROOT/dataset/'accuracy.json')
    timings={v:read(ROOT/dataset/f'timing_{v}.json') for v in ('reference','fused_same_chunk','fused')}
    result=dict(dataset=dataset,expected_test_structures=expected,
                accuracy_completed=bool(accuracy and accuracy['completed']),
                timing_completed={k:bool(v and v['completed']) for k,v in timings.items()})
    lines=[f'# {dataset.upper()} public pretrained benchmark',
           '', 'Results are generated from saved observations. Missing cases are not extrapolated.',
           'Protocol: [protocol.md](protocol.md). Hardware: RTX 5060 Ti 16 GB; FP32, TF32 off.',
           'Reference = official network algebra through the local FlatProbe adapter; not the unmodified upstream CLI.',
           'Warm seconds are per full material structure, excluding file parsing and one-time model/kernel setup.', '']
    preflight=read(ROOT/f'{dataset}_preflight.json')
    if preflight:
        lines+=['## Training-frame preflight (not official test results)', '',
            '| Variant | Chunk | NMAE (%) | Median seconds | Relative L1 vs reference |',
            '|---|---:|---:|---:|---:|']
        for r in preflight['records']:
            lines.append(f"| {r['name']} | {r['chunk']} | {100*r['nmae']:.7f} | {np.median(r['seconds']):.4f} | {r['relative_l1']:.3g} |")
        lines+=['']
    rows=accuracy['records'] if accuracy else []
    good=[r for r in rows if 'density' in r]
    result.update(attempted=len(rows),successful=len(good),errors=[r for r in rows if 'error' in r])
    lines+=['## Official test-set density accuracy', '',
        f"Attempted {len(rows)}/{expected}; successful {len(good)}; errors {len(result['errors'])}; loop completed: {result['accuracy_completed']}.", '']
    if good:
        result['nmae_percent']={}
        lines+=['| Variant | Mean NMAE (%) | Median (%) | p95 (%) | Maximum (%) |', '|---|---:|---:|---:|---:|']
        for v in ('reference','fused'):
            stats=distribution([100*r['density'][v]['nmae'] for r in good])
            stats['mean_bootstrap_95_percent']=interval([100*r['density'][v]['nmae'] for r in good])
            stats['pooled_nmae_percent']=100*sum(r['density'][v]['abs_error_sum'] for r in good)/sum(r['density'][v]['target_abs_sum'] for r in good)
            stats['worst_indices']=[r['index'] for r in sorted(good,key=lambda r:r['density'][v]['nmae'],reverse=True)[:10]]
            result['nmae_percent'][v]=stats
            lines.append(f"| {v} | {stats['mean']:.7f} | {stats['median']:.7f} | {stats['p95']:.7f} | {stats['maximum']:.7f} |")
        delta=[100*(r['density']['fused']['nmae']-r['density']['reference']['nmae']) for r in good]
        result['paired_delta_percentage_points']=dict(mean=float(np.mean(delta)),bootstrap_95=interval(delta))
        result['agreement_failures']=[r['index'] for r in good if not r['agreement']['passed']]
        result['max_relative_l1']=max(r['agreement']['relative_l1'] for r in good)
        lines+=['',f"Numerical-agreement failures: {len(result['agreement_failures'])}; maximum relative L1: {result['max_relative_l1']:.4g}.",
            f"Mean fused minus reference NMAE: {np.mean(delta):.4g} percentage points; descriptive bootstrap 95% interval: {result['paired_delta_percentage_points']['bootstrap_95']}.",
            'EC frames are correlated trajectory samples; bootstrap intervals do not establish independent-material generalization.', '']
    else:
        lines+=['No official test-set accuracy results yet.', '']
    lines+=['## Fixed 50-structure timing subset', '',
        '| Variant | Successful / attempted | Median of medians (s) | Mean (s) | p95 (s) | Maximum allocated GiB |',
        '|---|---:|---:|---:|---:|---:|']
    result['timing']={};indexed={}
    for v,data in timings.items():
        records=data['records'] if data else []
        ok=[r for r in records if 'median_seconds' in r]
        indexed[v]={r['index']:r['median_seconds'] for r in ok}
        stats=dict(attempted=len(records),successful=len(ok),errors=[r for r in records if 'error' in r])
        if ok:
            stats.update(distribution([r['median_seconds'] for r in ok]))
            stats['max_allocated_gib']=max(r['peak_absolute_gib'] for r in ok)
            lines.append(f"| {v} | {len(ok)} / {len(records)} | {stats['median']:.4f} | {stats['mean']:.4f} | {stats['p95']:.4f} | {stats['max_allocated_gib']:.3f} |")
        else:lines.append(f'| {v} | 0 / {len(records)} | pending | pending | pending | pending |')
        result['timing'][v]=stats
    result['paired_speedups']={}
    for v in ('fused_same_chunk','fused'):
        common=sorted(set(indexed['reference'])&set(indexed[v]))
        if common:
            stats=distribution([indexed['reference'][i]/indexed[v][i] for i in common])
            stats['ratio_of_total_seconds']=sum(indexed['reference'][i] for i in common)/sum(indexed[v][i] for i in common)
            result['paired_speedups'][v]=stats
            lines+=['',f"{v}: {len(common)} paired timing structures; median speedup {stats['median']:.3f}x; ratio of summed times {stats['ratio_of_total_seconds']:.3f}x."]
    result['all_pass']=bool(result['accuracy_completed'] and len(good)==expected and not result.get('agreement_failures',[]) and
        all(result['timing_completed'].values()) and all(len(indexed[v])==50 for v in indexed))
    lines+=['',f"Complete and all checks passed: **{result['all_pass']}**.",
        'This is a density-only inference benchmark; it does not measure PAW occupation accuracy, SCF iterations or total DFT runtime.',
        'A matched public pretrained EdenGNN NMC/EC baseline is not available in the inspected release.',
        'See per-structure JSON for failures, worst IDs, pooled errors, memory and all timing repetitions.']
    if good:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for v in ('reference','fused'):
            values=np.sort([100*r['density'][v]['nmae'] for r in good])
            axes[0].plot(values,np.arange(1,len(values)+1)/len(values),label=v)
        axes[0].set(xlabel='Per-structure NMAE (%)',ylabel='Empirical cumulative fraction',title=f'{dataset.upper()}: {len(good)}/{expected} test structures')
        axes[0].legend()
        for v,data in timings.items():
            ok=[r for r in (data['records'] if data else []) if 'median_seconds' in r]
            if ok:axes[1].scatter([r['metadata']['points']/1e6 for r in ok],[r['median_seconds'] for r in ok],s=16,label=v)
        axes[1].set(xlabel='Grid points (millions)',ylabel='Warm seconds per structure',title='Fixed timing subset')
        if any(indexed.values()):axes[1].legend()
        for ax in axes:ax.grid(alpha=.25)
        fig.savefig(REPORT/f'{dataset}_benchmark.png',dpi=180);plt.close(fig)
        lines+=['',f'![Accuracy and speed]({dataset}_benchmark.png)']
    (REPORT/f'{dataset}_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (REPORT/f'{dataset}_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(dataset=dataset,attempted=len(rows),expected=expected,all_pass=result['all_pass'])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=('nmc','ec'),required=True)
    summarize(p.parse_args().dataset)
