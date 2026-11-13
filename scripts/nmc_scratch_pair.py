"""Train the two occupation pipelines from scratch on the official NMC split."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from scripts.benchmark_public_density import save
from scripts.deepdft_paw import PAWReadout, geometry_features, training_statistics
from scripts.edengnn_paw import EdenPAW, inputs
from scripts.finetune_public_eden import geom, pack, write_pt
from scripts.finalize_public_paw import add_details
from scripts.nmc_probe_holdout import DEFAULT_ROOT, atom_batch, official_model, sha256
from scripts.public_paw_experiment import ROOT as DATA, structure_loss
from scripts.train_deepdft_paw import summarize

ROOT = Path('runs/nmc_scratch_pair')
REPORT = Path('reports/nmc_scratch_pair')
SPLIT = Path('runs/public_benchmark_2026-10-30/discovery/nmc_split.json')
ARMS = ('eden', 'deep')
SEEDS = (42, 43)


class DeepOccupation(torch.nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        model, _ = official_model(DEFAULT_ROOT / 'DeepDFT', 'cuda')
        self.encoder = model.atom_model
        self.head = PAWReadout(mean, scale, lmax=2)

    def forward(self, batch, geometry):
        scalars, vectors = self.encoder(batch)
        return self.head(torch.cat(scalars, -1), torch.cat(vectors, -1),
                         torch.cat([row[:n] for row, n in zip(batch['nodes'], batch['num_nodes'].tolist())]), geometry)


def build_model(arm, mean, scale):
    model = DeepOccupation(mean, scale) if arm == 'deep' else EdenPAW(mean, scale, pretrained=False)
    return model.cuda()


def graph(arm, row):
    if arm == 'eden':
        return inputs(geom(row))
    batch = atom_batch(geom(row))
    return batch, geometry_features(batch)


def batch_for(arm, ids, rows, graphs):
    if arm == 'eden':
        return pack(ids, rows, graphs)
    batches, geometries = zip(*(graphs[i] for i in ids))
    lengths = [len(rows[i]['numbers']) for i in ids]
    batch = {}
    for key in batches[0]:
        parts = [b[key] for b in batches]
        if key in ('num_atom_edges', 'num_nodes', 'cell'):
            batch[key] = torch.cat(parts)
        else:
            batch[key] = pad_sequence([p[0] for p in parts], batch_first=True)
    offsets = np.cumsum([0] + lengths[:-1]).tolist()
    geometry = {k: torch.cat([g[k] + offset if k in ('source', 'receiver') else g[k]
                              for g, offset in zip(geometries, offsets)]) for k in geometries[0]}
    labels = {k: torch.cat([rows[i][k] for i in ids]) for k in ('numbers', 'target_aug', 'aug_mask')}
    labels.update(lengths=lengths, groups=torch.repeat_interleave(
        torch.arange(len(ids), device='cuda'), batch['num_nodes'], output_size=sum(lengths)))
    return (batch, geometry), labels


@torch.no_grad()
def eden_initial_scale(model, graphs, ids):
    from e3nn import o3
    irreps = o3.Irreps(model.encoder_config['conv']['irreps_node_features'])
    x = torch.cat([model.encode(*graphs[i]) for i in ids[:64]])
    rms = [x[:, s].reshape(len(x), mul, ir.dim).square().mean((0, 2)).sqrt()
           .clamp_min(1e-4).repeat_interleave(ir.dim) for (mul, ir), s in zip(irreps, irreps.slices())]
    model.feature_rms.copy_(torch.cat(rms))


@torch.no_grad()
def evaluate(model, arm, ids, rows, graphs, details=False):
    model.eval()
    records = []
    for start in range(0, len(ids), 8):
        group = ids[start:start + 8]
        args, labels = batch_for(arm, group, rows, graphs)
        for i, pred in zip(group, model(*args).split(labels['lengths'])):
            row = rows[i]
            if details:
                records.append(add_details(pred, row))
            else:
                records.append(float((pred - row['target_aug']).abs()[row['aug_mask']].mean()))
    return records if details else float(np.mean(records))


def train(arm, seed, rows, split, mean, scale, lock):
    folder = ROOT / f'{arm}_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'selection.json').exists():
        assert json.loads((folder / 'selection.json').read_text())['lock'] == lock
        return
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va = split['train'], split['validation']
    print('START', arm, seed, flush=True)
    model = build_model(arm, mean, scale)
    graphs = {i: graph(arm, rows[i]) for i in tr + va}
    if arm == 'eden':
        eden_initial_scale(model, graphs, tr)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=2, threshold=.005, min_lr=1e-5)
    history, best, reference, stale = [], float('inf'), float('inf'), 0
    last_path = folder / 'last.pt'
    if last_path.exists():
        last = torch.load(last_path, weights_only=False)
        assert last['lock'] == lock
        model.load_state_dict(last['model'])
        opt.load_state_dict(last['opt'])
        sched.load_state_dict(last['sched'])
        history, best, reference, stale = (last[k] for k in ('history', 'best', 'reference', 'stale'))
        rng.bit_generator.state = last['rng']
        torch.set_rng_state(last['cpu_rng'].cpu())
        torch.cuda.set_rng_state(last['cuda_rng'].cpu())
        del last
    for epoch in range(len(history) + 1, 31):
        if stale >= 5:
            break
        model.train()
        order = rng.permutation(tr)
        first = time.perf_counter()
        for j in range(0, len(order), 8):
            args, labels = batch_for(arm, order[j:j+8], rows, graphs)
            opt.zero_grad(set_to_none=True)
            loss = structure_loss(model(*args), labels, scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step()
            if j == 8 and epoch == 1:
                assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.encoder.parameters())
                with torch.no_grad():
                    pair = tr[:2]
                    args, _ = batch_for(arm, pair, rows, graphs)
                    torch.testing.assert_close(model(*args), torch.cat([model(*graphs[i]) for i in pair]), atol=5e-5, rtol=5e-5)
            if torch.cuda.memory_reserved() > 10 * 2**30:
                torch.cuda.empty_cache()
            if j % 320 == 0:
                print('BATCH', arm, seed, epoch, j, flush=True)
        val = evaluate(model, arm, va, rows, graphs)
        if not np.isfinite(val):
            raise ValueError('Nonfinite validation MAE')
        lr = opt.param_groups[0]['lr']
        sched.step(val)
        if val < reference * .995:
            reference, stale = val, 0
        else:
            stale += 1
        if val < best:
            best = val
            write_pt(folder / 'best.pt', dict(model=model.state_dict(), epoch=epoch, lock=lock))
        history.append(dict(epoch=epoch, val=val, seconds=time.perf_counter()-first, lr=lr))
        write_pt(last_path, dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
            history=history, best=best, reference=reference, stale=stale, rng=rng.bit_generator.state,
            cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), lock=lock))
        save(folder / 'history.json', history)
        print('EPOCH', arm, seed, history[-1], flush=True)
    save(folder / 'selection.json', dict(lock=lock, best=min(history, key=lambda r:r['val']), epochs=len(history),
         seconds=sum(r['seconds'] for r in history), parameters=sum(p.numel() for p in model.parameters()),
         checkpoint_sha256=sha256(folder / 'best.pt')))


@torch.no_grad()
def final(rows, split, mean, scale, lock):
    ids = split['test']
    models, scores, detail = {}, {}, {}
    for arm in ARMS:
        graphs = {i: graph(arm, rows[i]) for i in ids}
        for seed in SEEDS:
            folder = ROOT / f'{arm}_{seed}'
            selected = json.loads((folder / 'selection.json').read_text())
            assert selected['lock'] == lock and selected['checkpoint_sha256'] == sha256(folder / 'best.pt')
            model = build_model(arm, mean, scale).eval()
            model.load_state_dict(torch.load(folder / 'best.pt', weights_only=False)['model'])
            records = evaluate(model, arm, ids, rows, graphs, details=True)
            assert [r['id'] for r in records] == ids
            save(folder / 'test.json', records)
            detail[arm, seed] = records
            scores[f'{arm}_{seed}'] = summarize(records)
            if seed == 42:
                models[arm] = model
        del graphs
    timings = []
    for j, i in enumerate(ids[:50]):
        row = geom(rows[i])
        times = dict(id=i)
        for arm in (ARMS if j % 2 == 0 else ARMS[::-1]):
            samples = []
            for k in range(7):
                torch.cuda.synchronize()
                start = time.perf_counter()
                models[arm](*graph(arm, row))
                torch.cuda.synchronize()
                if k >= 2:
                    samples.append(1000*(time.perf_counter()-start))
            times[arm] = samples
        timings.append(times)
    save(ROOT / 'timings_ms.json', timings)
    diff = np.mean([[a['mae']-b['mae'] for a,b in zip(detail['eden', s], detail['deep', s])] for s in SEEDS], axis=0)
    rng = np.random.default_rng(100)
    draws = rng.choice(diff, (10000, len(ids))).mean(axis=1)
    save(REPORT / 'results.json', dict(scores=scores, eden_minus_deep=float(diff.mean()),
         paired_structure_ci95=np.quantile(draws, [.025,.975]).tolist(),
         mean_median_ms={a:float(np.mean([np.median(r[a]) for r in timings])) for a in ARMS}))


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    split = json.loads(SPLIT.read_text())
    rows = torch.load(DATA / 'structures.pt', weights_only=False)
    audit = json.loads((DATA / 'audit.json').read_text())
    assert split == audit['split'] and not audit['exact_geometry_duplicates']
    assert [len(split[k]) for k in ('train','validation','test')] == [1450,50,500]
    assert sorted(sum(split.values(), [])) == list(range(2000)) == [r['id'] for r in rows]
    mean, scale = (x.cuda() for x in training_statistics([rows[i] for i in split['train']]))
    for row in rows:
        for k in ('numbers','target_aug','aug_mask'):
            row[k] = row[k].cuda()
    lock = {str(p):sha256(p) for p in (SPLIT, REPORT / 'protocol.md', DATA / 'structures.pt', Path(__file__))}
    for seed in SEEDS:
        for arm in ARMS:
            train(arm, seed, rows, split, mean, scale, lock)
            gc.collect()
            torch.cuda.empty_cache()
    final(rows, split, mean, scale, lock)
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    main()
