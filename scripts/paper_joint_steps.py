"""Paired step-budget training for the joint NMC and EC benchmark."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch

from scripts.run_joint_pair import (
    ROOT as DATA, ARMS, SEEDS, SPECIES, JointModel, audit, geometry, get_points,
    manifest, pt_save, read, read_form_factors, save, sha256, training_statistics,
)

ROOT = Path('runs/joint_paper_steps')
REPORT = Path('reports/joint_paper_steps')
BLOCK = 25000


def sequence(ids, seed, n):
    """Replayable implementation of a rotating 20-slot pool."""
    rng = np.random.default_rng(seed)
    pool = rng.choice(ids, 20)
    schedule = np.empty((n,2), dtype=np.int32)
    for i in range(n):
        if i % len(ids) == 0:
            incoming = rng.permutation(ids)
        pool[i % 20] = incoming[i % len(ids)]
        if i % 10 == 0:
            slots = rng.permutation(20).reshape(10,2)
        schedule[i] = pool[slots[i % 10]]
    return schedule


def forward(ds, model, row, rng, count):
    ids = rng.integers(np.prod(row['metadata']['grid_shape']),size=count)
    target = get_points(ds,row,ids,'rho')
    sad = get_points(ds,row,ids,'sad') if model.arm=='eden' else None
    pred,paw = model.sampled(geometry(row),ids,sad)
    truth = row['target_aug'].cuda()
    mask = row['aug_mask'].cuda()
    return pred-target,target,(paw-truth),truth,mask


@torch.no_grad()
def validate(ds,model,rows,ids):
    model.eval()
    metrics=[]
    for i in ids:
        err,rho,aug,target,mask=forward(ds,model,rows[i],np.random.default_rng([75113,i]),5000)
        diff=aug[mask].abs()
        metrics.append(torch.stack((err.abs().mean(),err.abs().sum()/rho.abs().sum(),
                                    diff.mean(), diff.sum()/target[mask].abs().sum())).tolist())
    return dict(zip(('density_mae','density_nmae','occupation_mae','occupation_relative_l1'),
                    np.mean(metrics,axis=0).tolist()))


def setup(ds):
    cfg=manifest(ds)
    split=cfg['split']
    rows={i:torch.load(DATA/ds/'data'/str(i)/'sample.pt',weights_only=False)
          for i in split['train']+split['validation']}
    stats=training_statistics([rows[i] for i in split['train']],elements=SPECIES[ds])
    factors=read_form_factors(Path('potpaw54.zip'),SPECIES[ds])
    return cfg,rows,stats,factors


def lock_for(ds):
    return {str(p):sha256(p) for p in (Path(__file__), REPORT/'protocol.md',
        Path('scripts/run_joint_pair.py'),Path('scripts/joint_pair_model.py'),
        DATA/ds/'fit_prepared.json',Path(f'data/public_density/{ds}/split.json'))}


def train(ds,arm,seed,limit):
    folder=ROOT/ds/f'{arm}_{seed}'
    folder.mkdir(exist_ok=True,parents=True)
    cfg,rows,stats,factors=setup(ds)
    lock=lock_for(ds)
    torch.manual_seed(seed)
    model=JointModel(arm,*stats).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    history=[]
    start=0
    last=folder/'last.pt'
    if last.exists():
        state=torch.load(last,weights_only=False)
        assert state['lock']==lock
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        history,start=state['history'],state['step']
        torch.set_rng_state(state['cpu_rng'].cpu())
        torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        del state
    else:
        audit(ds,model,rows[cfg['split']['train'][0]],factors)
    if start>=limit:
        return
    plan=sequence(cfg['split']['train'],seed,limit)
    print('TRAIN',ds,arm,seed,start,limit,flush=True)
    model.train()
    for step in range(start+1,limit+1):
        lr=1e-4*.96**((step-1)/100000)
        opt.param_groups[0]['lr']=lr
        opt.zero_grad(set_to_none=True)
        rng=np.random.default_rng([seed,step])
        for i in plan[step-1]:
            row=rows[int(i)]
            delta,_,aug,_,mask=forward(ds,model,row,rng,1000)
            loss=delta.square().mean()+.01*(aug/model.scale[row['numbers'].cuda()])[mask].square().mean()
            (loss*.5).backward()
        if not torch.isfinite(loss):
            raise FloatingPointError((ds,arm,seed,step))
        opt.step()
        if step==1 or step%200==0:
            print('STEP',ds,arm,seed,step,float(loss.detach()),flush=True)
        if step%5000==0:
            metrics=validate(ds,model,rows,cfg['split']['validation'])
            if not np.isfinite(list(metrics.values())).all():
                raise FloatingPointError(metrics)
            if not history or metrics['density_mae']<min(r['density_mae'] for r in history):
                pt_save(folder/'best.pt',dict(model=model.state_dict(),step=step,lock=lock))
            history.append(dict(step=step,lr=lr,**metrics))
            print('VAL',ds,arm,seed,history[-1],flush=True)
            model.train()
        if step%1000==0:
            pt_save(last,dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,
                     history=history,lock=lock,cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
            save(folder/'history.json',history)
            save(folder/'progress.json',dict(step=step,limit=limit,updated=time.time()))


def decide(ds,limit):
    gains={}
    for arm in ARMS:
        histories=[{r['step']:r for r in read(ROOT/ds/f'{arm}_{seed}'/'history.json')} for seed in SEEDS]
        for key in ('density_mae','occupation_relative_l1'):
            def best(end):
                return min(np.mean([h[t][key] for h in histories]) for t in range(end-BLOCK+5000,end+1,5000))
            gains[arm+'_'+key]=1-best(limit)/best(limit-BLOCK)
    decision=dict(step=limit,validation_gains=gains,extend=bool(max(gains.values())>=.01))
    save(ROOT/ds/f'decision_{limit}.json',decision)
    return decision['extend']


def queue():
    datasets=['nmc','ec']
    limit=BLOCK
    while datasets:
        active=[]
        for ds in datasets:
            for seed in SEEDS:
                for arm in ARMS:
                    folder=ROOT/ds/f'{arm}_{seed}'
                    folder.mkdir(exist_ok=True,parents=True)
                    status=folder/'progress.json'
                    if status.exists() and read(status)['step']>=limit:
                        continue
                    with (folder/'train.log').open('a') as f:
                        subprocess.run([sys.executable,'-u','-m','scripts.paper_joint_steps',ds,arm,
                                        str(seed),str(limit)],stdout=f,stderr=f,check=True)
            if limit<100000 or decide(ds,limit):
                active.append(ds)
            else:
                subprocess.run([sys.executable,'-u','-m','scripts.paper_joint_test',ds],check=True)
        datasets=active
        limit+=BLOCK


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('args',nargs='*')
    args=parser.parse_args().args
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if args:
        ds,arm,seed,limit=args
        train(ds,arm,int(seed),int(limit))
    else:
        queue()
