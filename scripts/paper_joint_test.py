"""Freeze a shared checkpoint per arm and score both outputs on official grids."""
import argparse
import time
import numpy as np
import torch
from scripts.paper_joint_steps import ARMS, SEEDS, DATA, REPORT, ROOT, lock_for, setup
from scripts.run_joint_pair import (
    JointModel, RadialTableProbe, geometry, metric, measures, prepare, read, save, sha256,
)


@torch.no_grad()
def evaluate(ds,single=False):
    cfg,_,stats,factors=setup(ds)
    run=ROOT/ds
    arms=('deep',) if single else ARMS
    seeds=(42,) if single else SEEDS
    output=run/'single_deep_42_4965000' if single else run
    output.mkdir(exist_ok=True)
    if single:
        assert ds=='nmc' and read(run/'deep_42'/'progress.json')['step']==4965000
    lock=lock_for(ds)
    models={}
    selections={}
    for arm in arms:
        for seed in seeds:
            name=f'{arm}_{seed}'
            folder=run/name
            path=folder/'best.pt'
            state=torch.load(path,weights_only=False,map_location='cpu')
            chosen=min(read(folder/'history.json'),key=lambda r:r['density_mae'])
            assert state['lock']==lock and state['step']==chosen['step']
            model=JointModel(arm,*stats).cuda().eval().requires_grad_(False)
            model.load_state_dict(state['model'])
            core=RadialTableProbe(model.net,fused=True).eval() if arm=='deep' else None
            models[name]=(model,core)
            selections[name]=dict(checkpoint_sha256=sha256(path),validation=chosen)
    freeze=output/'frozen.json'
    if freeze.exists():
        assert read(freeze)==selections
    else:
        save(freeze,selections)
    if not (DATA/ds/'test_prepared.json').exists():
        prepare(ds,'test')
    folder=output/'test_records'
    folder.mkdir(exist_ok=True)
    records=[]
    for j,i in enumerate(cfg['split']['test']):
        dest=folder/f'{i}.json'
        if dest.exists():
            records.append(read(dest))
            continue
        src=DATA/ds/'data'/str(i)
        row=torch.load(src/'sample.pt',weights_only=False)
        assert row['id']==i
        s=geometry(row,full=True)
        target=torch.tensor(np.load(src/'rho.npy'),device='cuda',dtype=torch.float64)
        truth,mask=row['target_aug'].cuda(),row['aug_mask'].cuda()
        result=dict(id=i,points=len(target),scores={},seconds={})
        for arm in (arms if j%2==0 else arms[::-1]):
            for seed in seeds:
                name=f'{arm}_{seed}'
                model,core=models[name]
                density,paw=model.full(s,factors,core)
                assert density.shape==target.shape and torch.isfinite(density).all() and torch.isfinite(paw).all()
                result['scores'][name]=dict(density=metric(density,target,s),occupation=measures(paw,truth,mask))
                del density,paw
                if seed==42 and i in cfg['timing_indices']:
                    times=[]
                    for repeat in range(7):
                        torch.cuda.synchronize()
                        start=time.perf_counter()
                        model.full(s,factors,core)
                        torch.cuda.synchronize()
                        if repeat>=2:
                            times.append(time.perf_counter()-start)
                    result['seconds'][arm]=times
        save(dest,result)
        records.append(result)
    assert [r['id'] for r in records]==cfg['split']['test']
    summary={}
    for name in models:
        summary[name]={}
        for task,key in [('density','nmae'),('density','mae'),('occupation','mae'),('occupation','relative_l1')]:
            summary[name][task+'_'+key]=float(np.mean([r['scores'][name][task][key] for r in records]))
    for arm in arms:
        times=[np.median(r['seconds'][arm]) for r in records if arm in r['seconds']]
        assert len(times)==50
        summary[arm+'_seconds']=float(np.mean(times))
    filename=f'{ds}_single_deep_42_4965000.json' if single else f'{ds}_summary.json'
    save(REPORT/filename,dict(test_count=len(records),summary=summary,selections=selections))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset',choices=('nmc','ec'))
    parser.add_argument('--single',action='store_true')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    args=parser.parse_args()
    evaluate(args.dataset,args.single)
