"""Paired local/global query-compression experiment on existing sealed splits."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.compare_local_nmc_oracles import load_sample
from scripts.nmc_probe_holdout import DEFAULT_ROOT, official_model, spatial_split, sha256, atom_batch
from scripts.deepdft_inference_fast import ProbeEdges, FlatProbe, flat_density
from scripts.deepdft_mixed_embedding import MixedProbe, TrainingEdges, descriptor, mixed_density

MODES = ('local', 'constant', 'global')
CHECKPOINT = DEFAULT_ROOT/'deepdft_eight/best.pt'


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


@torch.no_grad()
def evaluate(head, samples, partition='selection', chunk=4096, swap=False):
    head.eval()
    rows = []
    for k, s in enumerate(samples):
        context = descriptor(samples[(k+1) % len(samples)]['rep']) if swap else None
        states = head.atoms(s['rep'], context)
        sums = torch.zeros(2, dtype=torch.float64, device='cuda')
        for ids in s['split'][partition].split(chunk):
            out = head(*s['edges'].select(s, ids), *states, len(ids)).double()
            target = s['target'][ids]
            sums += torch.stack(((out-target).abs().sum(), target.abs().sum()))
        rows.append({'id': s['id'], 'nmae': float(sums[0]/sums[1])})
    return {'records': rows, 'macro_nmae': float(np.mean([r['nmae'] for r in rows]))}


def prepare(backbone, ids):
    samples = []
    for sid in ids:
        s = load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'), 'cuda')
        s['id'] = sid
        s['split'] = {k: torch.as_tensor(v, device='cuda')
                      for k, v in spatial_split(len(s['target']), sid).items()}
        s['normalizer'] = s['target'][s['split']['train']].abs().mean()
        with torch.no_grad():
            s['rep'] = backbone.atom_model(atom_batch(s))
        graph = ProbeEdges.build(s)
        s['edges'] = TrainingEdges(graph)
        del graph
        samples.append(s)
        print(f'PREPARED {sid} allocated_mib={torch.cuda.memory_allocated()/2**20:.0f}', flush=True)
    return samples


def train(args, backbone, samples):
    descriptors = torch.stack([descriptor(s['rep']) for s in samples])
    mean = descriptors.mean(0)
    scale = descriptors.std(0, unbiased=False).clamp_min(.01)
    for seed in args.seeds:
        folder = args.output/f'seed_{seed}'
        if (folder/'complete.json').exists():
            print(f'SKIP completed seed {seed}', flush=True)
            continue
        torch.manual_seed(seed)
        initial = MixedProbe('global', args.width).cuda()
        initial.global_mean.copy_(mean)
        initial.global_scale.copy_(scale)
        heads = {}
        for mode in MODES:
            head = MixedProbe(mode, args.width).cuda()
            head.load_state_dict(initial.state_dict())
            heads[mode] = head
        del initial
        opts = {m: torch.optim.Adam([p for p in h.parameters() if p.requires_grad], lr=args.lr)
                for m, h in heads.items()}
        sched = {m: torch.optim.lr_scheduler.CosineAnnealingLR(o, args.steps, eta_min=1e-5)
                 for m, o in opts.items()}
        rng = torch.Generator(device='cuda').manual_seed(seed+10000)
        order_rng = np.random.default_rng(seed+20000)
        order, history, best = [], [], {m: float('inf') for m in MODES}
        best_steps = {}
        started = time.perf_counter()
        counts = {m: {'allocated': sum(p.numel() for p in h.parameters()),
                       'trainable': sum(p.numel() for p in h.parameters() if p.requires_grad)}
                  for m,h in heads.items()}
        save(folder/'parameters.json', counts)
        for step in range(1, args.steps+1):
            if not order:
                order = order_rng.permutation(len(samples)).tolist()
            s = samples[order.pop()]
            allowed = s['split']['train']
            ids = allowed[torch.randint(len(allowed), (args.probes,), device='cuda', generator=rng)]
            geometry = s['edges'].select(s, ids)
            losses = {}
            for mode, head in heads.items():
                head.train()
                opts[mode].zero_grad(set_to_none=True)
                output = head(*geometry, *head.atoms(s['rep']), len(ids))
                loss = ((output.double()-s['target'][ids])/s['normalizer']).abs().mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite {seed} {step} {mode}')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True)
                opts[mode].step()
                sched[mode].step()
                losses[mode] = float(loss.detach())
            if step == 1 or step % 100 == 0:
                print(json.dumps({'seed':seed, 'step':step, 'loss':losses,
                                  'seconds':time.perf_counter()-started}), flush=True)
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                metrics = {m:evaluate(h, samples) for m,h in heads.items()}
                history.append({'step':step, 'seconds':time.perf_counter()-started, 'metrics':metrics})
                for mode, head in heads.items():
                    if metrics[mode]['macro_nmae'] < best[mode]:
                        best[mode] = metrics[mode]['macro_nmae']
                        best_steps[mode] = step
                        folder.mkdir(parents=True, exist_ok=True)
                        torch.save({'model':head.state_dict(), 'mode':mode, 'width':args.width,
                                    'step':step, 'seed':seed, 'selection':metrics[mode]}, folder/f'{mode}_best.pt')
                save(folder/'history.json', history)
                print('VALIDATION '+json.dumps({'seed':seed,'step':step,'nmae':{
                    m:r['macro_nmae'] for m,r in metrics.items()}}), flush=True)
        save(folder/'complete.json', {'best':best, 'best_steps':best_steps,
                                     'seconds':time.perf_counter()-started})


@torch.no_grad()
def final_evaluation(args, backbone, samples):
    # Only executed after every seed finishes and checkpoint selection is closed.
    results = {'accuracy': [], 'timing': [], 'checkpoint_sha256':sha256(CHECKPOINT)}
    heads = {}
    for seed in args.seeds:
        for mode in MODES:
            path = args.output/f'seed_{seed}'/f'{mode}_best.pt'
            state = torch.load(path, map_location='cuda', weights_only=False)
            h = MixedProbe(mode, args.width).cuda().eval()
            h.load_state_dict(state['model'])
            row = {'seed':seed, 'mode':mode, 'best_step':state['step'],
                   'checkpoint_sha256':sha256(path), 'selection':state['selection'],
                   'test':evaluate(h, samples, 'test'), 'validation':evaluate(h, samples, 'validation')}
            if mode == 'global':
                row['test_swapped_global'] = evaluate(h, samples, 'test', swap=True)
            results['accuracy'].append(row)
            if seed == args.seeds[0]:
                heads[mode] = h
            print('FINAL '+json.dumps(row), flush=True)
            save(args.output/'result.json', results)
    # Use the first predeclared seed for speed, never the lowest test error seed.
    core = FlatProbe(backbone).eval()
    for s in samples:
        functions = {'original_flat':lambda:flat_density(backbone, core, s, None, args.chunk)}
        functions.update({m:(lambda h=h:mixed_density(backbone, h, s, args.chunk)) for m,h in heads.items()})
        row = {'id':s['id'], 'variants':{}, 'test_nmae':{}}
        for name, fn in functions.items():
            out = fn()
            ids = s['split']['test']
            row['test_nmae'][name] = float((out[ids].double()-s['target'][ids]).abs().sum()/s['target'][ids].abs().sum())
            if name != 'original_flat':
                cached = next(r for r in results['accuracy'] if r['seed']==args.seeds[0] and r['mode']==name)['test']['records']
                expected = next(r['nmae'] for r in cached if r['id']==s['id'])
                assert abs(row['test_nmae'][name]-expected) < 1e-6
            del out
        names = list(functions)
        for repeat in range(args.repeat):
            for name in names[repeat % len(names):]+names[:repeat % len(names)]:
                torch.cuda.synchronize()
                base = torch.cuda.memory_allocated()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                out = functions[name]()
                torch.cuda.synchronize()
                seconds = time.perf_counter()-start
                entry = row['variants'].setdefault(name, {'seconds':[], 'peak_increment_mib':0})
                entry['seconds'].append(seconds)
                entry['peak_increment_mib'] = max(entry['peak_increment_mib'], (torch.cuda.max_memory_allocated()-base)/2**20)
                del out
        for item in row['variants'].values():
            item['median_seconds'] = float(np.median(item['seconds']))
        results['timing'].append(row)
        save(args.output/'result.json', results)
        print('TIMING '+json.dumps(row), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=Path('runs/mixed_embedding_2026-10-29'))
    p.add_argument('--samples', type=int, nargs='+', default=list(range(8)))
    p.add_argument('--seeds', type=int, nargs='+', default=[42,43])
    p.add_argument('--steps', type=int, default=6000)
    p.add_argument('--probes', type=int, default=1024)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--chunk', type=int, default=4096)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--skip-final', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    backbone, provenance = official_model(DEFAULT_ROOT/'DeepDFT', 'cuda')
    backbone.load_state_dict(torch.load(CHECKPOINT, map_location='cuda', weights_only=False)['model'])
    backbone.eval().requires_grad_(False)
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='skip_final'}
    settings.update(backbone_sha256=sha256(CHECKPOINT), backbone_provenance=provenance,
                    torch=torch.__version__, device=torch.cuda.get_device_name(),
                    source_hashes={str(p):sha256(p) for p in [Path(__file__),Path('scripts/deepdft_mixed_embedding.py')]})
    path = args.output/'settings.json'
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError('Settings/source changed; use a new output directory')
    save(path, settings)
    samples = prepare(backbone, args.samples)
    save(args.output/'sources.json', [{'id':s['id'],**s['metadata']} for s in samples])
    train(args, backbone, samples)
    if not args.skip_final:
        final_evaluation(args, backbone, samples)


if __name__ == '__main__':
    main()
