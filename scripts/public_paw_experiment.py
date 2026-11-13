"""Full official-split NMC PAW experiment; separate from historical experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from scripts.benchmark_public_density import model_for, save
from scripts.deepdft_paw import (BLOCKS, WIDTH, potcar_specs, read_structure_and_aug,
                                features, training_statistics, PAWReadout)
from scripts.nmc_probe_holdout import sha256

ROOT = Path('runs/public_paw_2026-11-02')
SOURCE_MANIFEST = Path('runs/public_benchmark_2026-10-30/nmc/settings.json')
PROTOCOL = Path('reports/public_paw_2026-11-02/protocol.md')
ARMS = ('deep_frozen', 'eden_frozen')
SEEDS = (42, 43)


def cpu_tree(x):
    if torch.is_tensor(x): return x.detach().cpu()
    if isinstance(x, dict): return {k:cpu_tree(v) for k,v in x.items()}
    return x


def gpu_tree(x):
    if torch.is_tensor(x): return x.cuda()
    if isinstance(x, dict): return {k:gpu_tree(v) for k,v in x.items()}
    return x


def payload(entry):
    with Path(entry['archive']).open('rb') as f:
        f.seek(entry['offset'])
        data = f.read(entry['size'])
    assert len(data) == entry['size']
    return data


def audit():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'audit.json').exists():
        raise ValueError('Audit already exists; preserve original data')
    cfg = json.loads(SOURCE_MANIFEST.read_text())
    specs = potcar_specs()
    rows, samples = [], []
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='nmc_paw_audit_') as tmp:
        path = Path(tmp)/'CHGCAR.lz4'
        for entry in cfg['entries']:
            data = payload(entry)
            path.write_bytes(data)
            s = read_structure_and_aug(path, specs, lmax=4)
            maxima = {str(l): float(s['target_aug'][:, [i for b in BLOCKS if b['l']==l
                        for i in range(b['start'], b['stop'])]].abs().max()) for l in range(5)}
            row = dict(index=entry['index'], member=entry['name'],
                       payload_sha256=hashlib.sha256(data).hexdigest(),
                       geometry_sha256=s['metadata']['geometry_sha256'],
                       atoms=len(s['numbers']), rank_max_abs=maxima)
            rows.append(row)
            samples.append(dict(id=entry['index'], **{k:s[k] for k in
                ('positions','cell','numbers','target_aug','aug_mask','raw_mapping')}, metadata=row))
            if len(rows)%100==0:
                print('AUDIT',len(rows),'/2000',round(time.perf_counter()-start,1),flush=True)
    assert len(rows)==2000
    rank_max = {str(l):max(r['rank_max_abs'][str(l)] for r in rows) for l in range(5)}
    # Missing high ranks are never inferred from zeros as physical ground truth.
    if rank_max['3']!=0 or rank_max['4']!=0:
        raise ValueError('New high-rank availability: inspect schema before locking training')
    for s in samples:
        for b in BLOCKS:
            if b['l']>2:
                s['aug_mask'][:,b['start']:b['stop']]=False
    local_matches = []
    local_hashes = {sha256(Path(f'data/nmc_cache/{i}.CHGCAR.lz4')):i for i in range(8)}
    for r in rows:
        if r['payload_sha256'] in local_hashes:
            local_matches.append(dict(local_id=local_hashes[r['payload_sha256']],official_index=r['index']))
    by_geometry = {}
    for r in rows:
        by_geometry.setdefault(r['geometry_sha256'],[]).append(r['index'])
    duplicates = [v for v in by_geometry.values() if len(v)>1]
    result = dict(completed=True,counts={k:len(v) for k,v in cfg['split'].items()},
        split=cfg['split'],rank_max_abs=rank_max,supported_lmax=2,
        potcar_specs=specs,local_pretraining_matches=local_matches,
        exact_geometry_duplicates=duplicates,records=rows,seconds=time.perf_counter()-start,
        source_manifest_sha256=sha256(SOURCE_MANIFEST),
        limitation='L3/L4 all zero: unavailable, not physical zero. Historical exact POTCAR identity unconfirmed.')
    torch.save(samples,ROOT/'structures.pt')
    save(ROOT/'audit.json',result)
    print('AUDIT_COMPLETE',json.dumps({k:result[k] for k in ('counts','rank_max_abs','local_pretraining_matches','exact_geometry_duplicates')}),flush=True)


@torch.no_grad()
def prepare(arm):
    from scripts.edengnn_paw import EdenPAW, inputs, CHECKPOINT as EDEN_CHECKPOINT
    from e3nn import o3
    audit_data=json.loads((ROOT/'audit.json').read_text())
    cfg=json.loads(SOURCE_MANIFEST.read_text())
    if not (ROOT/'settings.json').exists():
        overlaps=sorted(set(m['official_index'] for m in audit_data['local_pretraining_matches']) & set(cfg['split']['test']))
        settings=dict(split=cfg['split'],paired_test=[i for i in cfg['split']['test'] if i not in overlaps],
            eden_excluded_test=overlaps,epochs=100,batch=16,eval_every=5,seeds=list(SEEDS),arms=list(ARMS),
            lr=1e-3,min_lr=1e-5,clip=5.,lmax=2,deep_checkpoint=cfg['checkpoint'],
            deep_checkpoint_sha256=sha256(cfg['checkpoint']),eden_checkpoint=str(EDEN_CHECKPOINT),
            eden_checkpoint_sha256=sha256(EDEN_CHECKPOINT),audit_sha256=sha256(ROOT/'audit.json'),
            protocol_sha256=sha256(PROTOCOL),sources={p:sha256(p) for p in
                ('scripts/deepdft_paw.py','scripts/edengnn_paw.py','scripts/deepdft_radial_table.py','scripts/radial_message_kernel.cu')},
            torch=torch.__version__,gpu=torch.cuda.get_device_name(),
            pretraining='DeepDFT official NMC density training; Eden local 8 density structures. Different pretraining, not architecture ranking.')
        save(ROOT/'settings.json',settings)
    settings=check()
    target=ROOT/f'{arm}_features.pt'
    if target.exists(): raise ValueError('Preserve features')
    structures=torch.load(ROOT/'structures.pt',weights_only=False)
    train=[structures[i] for i in settings['split']['train']]
    mean,scale=training_statistics(train)
    if arm=='deep_frozen': model,provenance=model_for('nmc')
    else:
        model=EdenPAW(mean,scale).cuda().eval().requires_grad_(False)
        provenance=model.provenance
    rows=[];start=time.perf_counter()
    for s in structures:
        base={k:s[k] for k in ('id','numbers','target_aug','aug_mask')}
        if arm=='deep_frozen': base.update(cpu_tree(features(model,s)))
        else: base['eden_features']=model.encode(*inputs(s)).cpu()
        rows.append(base)
        if len(rows)%100==0: print('FEATURES',arm,len(rows),round(time.perf_counter()-start,1),flush=True)
    rms=None
    if arm=='eden_frozen':
        x=torch.cat([rows[i]['eden_features'] for i in settings['split']['train']])
        irreps=o3.Irreps(model.encoder_config['conv']['irreps_node_features'])
        rms=torch.cat([x[:,section].reshape(len(x),mul,ir.dim).square().mean((0,2)).sqrt().clamp_min(1e-4).repeat_interleave(ir.dim)
                       for (mul,ir),section in zip(irreps,irreps.slices())])
    torch.save(dict(samples=rows,mean=mean,scale=scale,feature_rms=rms),target)
    save(ROOT/f'{arm}_features.json',dict(sha256=sha256(target),seconds=time.perf_counter()-start,provenance=provenance))
    print('FEATURES_COMPLETE',arm,flush=True)


def check():
    cfg=json.loads((ROOT/'settings.json').read_text())
    assert sha256(PROTOCOL)==cfg['protocol_sha256']
    assert sha256(cfg['deep_checkpoint'])==cfg['deep_checkpoint_sha256']
    assert sha256(cfg['eden_checkpoint'])==cfg['eden_checkpoint_sha256']
    assert all(sha256(p)==h for p,h in cfg['sources'].items())
    return cfg


def collate(rows,arm):
    result={k:torch.cat([s[k] for s in rows]) for k in ('numbers','target_aug','aug_mask')}
    device=result['numbers'].device
    result['lengths']=[len(s['numbers']) for s in rows]
    result['groups']=torch.repeat_interleave(torch.arange(len(rows),device=device),
        torch.tensor(result['lengths'],device=device))
    if arm=='eden_frozen': result['eden_features']=torch.cat([s['eden_features'] for s in rows])
    else:
        for k in ('scalars','vectors'): result[k]=torch.cat([s[k] for s in rows])
        offsets=np.cumsum([0]+result['lengths'][:-1]).tolist()
        result['geometry']={k:torch.cat([s['geometry'][k]+offset if k in ('source','receiver') else s['geometry'][k]
            for s,offset in zip(rows,offsets)]) for k in rows[0]['geometry']}
    return result


def make_head(data,arm):
    if arm=='deep_frozen': return PAWReadout(data['mean'],data['scale'],lmax=2).cuda()
    from scripts.edengnn_paw import EdenPAW
    model=EdenPAW(data['mean'],data['scale'],pretrained=True,feature_rms=data['feature_rms']).cuda()
    model.encoder.requires_grad_(False)
    return model


def predict(head,batch,arm):
    if arm=='eden_frozen': return head.readout(batch['eden_features'],batch['numbers'])
    return head(batch['scalars'],batch['vectors'],batch['numbers'],batch['geometry'])


def structure_loss(out,batch,scale):
    error=((out-batch['target_aug'])/scale[batch['numbers']]).square()*batch['aug_mask']
    groups=batch['groups'];n=len(batch['lengths'])
    sums=out.new_zeros(n).index_add(0,groups,error.sum(1))
    counts=out.new_zeros(n).index_add(0,groups,batch['aug_mask'].sum(1).to(out.dtype))
    return (sums/counts).mean()


@torch.no_grad()
def evaluate(head,samples,indices,arm,details=False):
    from scripts.train_deepdft_paw import measures,summarize
    head.eval();records=[]
    for start in range(0,len(indices),16):
        ids=indices[start:start+16]
        b=collate([samples[i] for i in ids],arm)
        outputs=predict(head,b,arm).split(b['lengths'])
        for i,out in zip(ids,outputs):
            s=samples[i]
            row=dict(id=i,**measures(out,s['target_aug'],s['aug_mask']))
            if details:
                row['by_l']={}
                for l in range(3):
                    mask=torch.zeros_like(s['aug_mask'])
                    for block in BLOCKS:
                        if block['l']==l: mask[:,block['start']:block['stop']]=s['aug_mask'][:,block['start']:block['stop']]
                    row['by_l'][str(l)]=measures(out,s['target_aug'],mask)
                row['by_element']={str(int(z)):measures(out,s['target_aug'],s['aug_mask']&(s['numbers']==z)[:,None])
                    for z in s['numbers'].unique()}
            records.append(row)
    return dict(**summarize(records),records=records)


def train(arm,seed):
    cfg=check();folder=ROOT/f'{arm}_{seed}';folder.mkdir(exist_ok=True)
    if (folder/'selection.json').exists(): raise ValueError('Preserve completed training')
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    assert sha256(ROOT/f'{arm}_features.pt')==json.loads((ROOT/f'{arm}_features.json').read_text())['sha256']
    source_hash=sha256(__file__)
    data=torch.load(ROOT/f'{arm}_features.pt',weights_only=False)
    head=make_head(data,arm)
    samples={i:gpu_tree(data['samples'][i]) for split in ('train','validation') for i in cfg['split'][split]}
    del data['samples']
    parameters=[p for p in head.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=cfg['lr'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['epochs'],eta_min=cfg['min_lr'])
    best=float('inf');history=[];first=1;elapsed=0.
    if (folder/'last.pt').exists():
        last=torch.load(folder/'last.pt',weights_only=False)
        head.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer']);scheduler.load_state_dict(last['scheduler'])
        rng.bit_generator.state=last['rng'];first=last['epoch']+1;best=last['best'];history=last['history'];elapsed=last['elapsed']
    start=time.perf_counter()
    for epoch in range(first,cfg['epochs']+1):
        head.train();order=rng.permutation(cfg['split']['train']);losses=[]
        for j in range(0,len(order),cfg['batch']):
            b=collate([samples[int(i)] for i in order[j:j+cfg['batch']]],arm)
            optimizer.zero_grad(set_to_none=True)
            loss=structure_loss(predict(head,b,arm),b,head.scale)
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,cfg['clip']);optimizer.step()
            losses.append(float(loss))
        scheduler.step()
        if epoch%cfg['eval_every']==0:
            val=evaluate(head,samples,cfg['split']['validation'],arm)
            row=dict(epoch=epoch,train_standardized_mse=float(np.mean(losses)),validation_mae=val['macro_mae'],
                     elapsed_seconds=elapsed+time.perf_counter()-start)
            history.append(row)
            if val['macro_mae']<best:
                best=val['macro_mae']
                torch.save(dict(model=cpu_tree(head.state_dict()),arm=arm,seed=seed,epoch=epoch,validation=val,
                    settings_sha256=sha256(ROOT/'settings.json')),folder/'best.pt')
            save(folder/'history.json',history)
            torch.save(dict(model=cpu_tree(head.state_dict()),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                rng=rng.bit_generator.state,epoch=epoch,best=best,history=history,elapsed=row['elapsed_seconds']),folder/'last.pt')
            print('TRAIN',arm,seed,json.dumps(row),flush=True)
    assert sha256(__file__)==source_hash,'Training source changed during execution'
    saved=torch.load(folder/'best.pt',weights_only=False)
    save(folder/'selection.json',dict(seed=seed,arm=arm,epoch=saved['epoch'],validation_mae=best,
        training_seconds=elapsed+time.perf_counter()-start,parameters=sum(p.numel() for p in parameters),
        checkpoint_sha256=sha256(folder/'best.pt'),settings_sha256=sha256(ROOT/'settings.json'),
        experiment_source_sha256=sha256(__file__)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['audit','prepare','train'],required=True)
    p.add_argument('--arm',choices=ARMS)
    p.add_argument('--seed',type=int,choices=SEEDS)
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    if args.stage=='audit': audit()
    elif args.stage=='prepare': prepare(args.arm)
    elif args.stage=='train': train(args.arm,args.seed)


if __name__=='__main__':
    main()
