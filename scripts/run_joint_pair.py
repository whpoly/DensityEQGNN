"""Joint from-scratch training, validation selection, and full-grid paired test."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from density_eqgnn.chgcar import vasp_grid
from scripts.benchmark_public_density import metric, save
from scripts.deepdft_radial_table import RadialTableProbe
from scripts.joint_pair_data import ROOT, SPECIES, manifest, prepare
from scripts.joint_pair_model import JointModel
from scripts.joint_paw_tables import read_form_factors, training_statistics
from scripts.nmc_probe_holdout import sha256
from scripts.train_deepdft_paw import measures

REPORT = Path('reports/joint_nmc_ec')
ARMS = ('deep','eden')
SEEDS = (42,43)


def read(p):
    return json.loads(p.read_text())


def pt_save(path, obj):
    tmp = path.with_suffix('.tmp')
    torch.save(obj,tmp)
    tmp.replace(path)


def geometry(row, full=False):
    s = {k:row[k].cuda() for k in ('numbers','positions','cell')}
    s['metadata'] = row['metadata']
    if full:
        s['probes'] = torch.tensor(vasp_grid(row['cell'].numpy(),row['metadata']['grid_shape']),device='cuda')
    return s


def get_points(ds, row, ids, name):
    array = np.load(ROOT/ds/'data'/str(row['id'])/(name+'.npy'),mmap_mode='r')
    return torch.tensor(array[ids],device='cuda')


def predict_points(ds, model, row, seed, epoch, count):
    rng = np.random.default_rng([seed,epoch,row['id']])
    ids = rng.choice(np.prod(row['metadata']['grid_shape']),count,replace=False)
    sad = get_points(ds,row,ids,'sad') if model.arm=='eden' else None
    rho, paw = model.sampled(geometry(row),ids,sad)
    target = get_points(ds,row,ids,'rho')
    truth, mask = row['target_aug'].cuda(), row['aug_mask'].cuda()
    err = paw-truth
    return (rho-target).abs().mean(), target.abs().mean(), err, truth, mask


@torch.no_grad()
def validate(ds, model, rows, ids):
    model.eval()
    scores=[]
    for i in ids:
        err, den, aug, truth, mask = predict_points(ds,model,rows[i],20261103,0,4096)
        scores.append([float(err/den),float(aug[mask].abs().sum()/truth[mask].abs().sum())])
    return np.mean(scores,axis=0).tolist()


def audit(ds, model, row, factors):
    # Check the fresh whole-grid path against sparse training at the same points.
    ids = np.random.default_rng(13).choice(np.prod(row['metadata']['grid_shape']),1024,replace=False)
    with torch.no_grad():
        model.eval()
        sad = get_points(ds,row,ids,'sad') if model.arm=='eden' else None
        sparse, a = model.sampled(geometry(row),ids,sad)
        core = RadialTableProbe(model.net,fused=True).eval() if model.arm=='deep' else None
        full, b = model.full(geometry(row,True),factors,core)
        torch.testing.assert_close(full[ids], sparse, rtol=3e-4,atol=5e-5)
        torch.testing.assert_close(a,b,rtol=5e-5,atol=5e-5)


def train(ds, arm, seed, split, rows, stats, lock, factors):
    folder=ROOT/ds/f'{arm}_{seed}'
    folder.mkdir(exist_ok=True)
    if (folder/'selection.json').exists():
        assert read(folder/'selection.json')['lock']==lock
        return
    torch.manual_seed(seed)
    model = JointModel(arm,*stats).cuda()
    opt = torch.optim.Adam(model.parameters(),lr=3e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5,patience=2,threshold=.005,min_lr=1e-5)
    history, best, reference, stale = [], float('inf'), float('inf'), 0
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',weights_only=False)
        assert state['lock']==lock
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        sched.load_state_dict(state['scheduler'])
        history,best,reference,stale=[state[k] for k in ('history','best','reference','stale')]
        del state
    audit(ds,model,rows[split['train'][0]],factors)
    print('AUDIT_OK',ds,arm,seed,flush=True)
    for ep in range(len(history)+1,31):
        if stale>=5:
            break
        model.train()
        start=time.perf_counter()
        order = np.random.default_rng([seed,ep]).permutation(split['train'])
        for j,i in enumerate(order):
            row=rows[i]
            err,_,aug,_,mask=predict_points(ds,model,row,seed,ep,1024)
            loss=err/row['norm'] + ((aug/model.scale[row['numbers'].cuda()])[mask]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            opt.step()
            if j%500==0:
                print('STEP',ds,arm,seed,ep,j,float(loss.detach()),flush=True)
        v=validate(ds,model,rows,split['validation'])
        score=sum(v)
        assert np.isfinite(score)
        sched.step(score)
        if score<reference*.995:
            reference,stale=score,0
        else:
            stale+=1
        if score<best:
            best=score
            pt_save(folder/'best.pt',dict(model=model.state_dict(),epoch=ep,lock=lock,score=score))
        history.append(dict(epoch=ep,density_nmae=v[0],occupation_rel_l1=v[1],score=score,seconds=time.perf_counter()-start))
        save(folder/'history.json',history)
        pt_save(folder/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict(),
                lock=lock,history=history,best=best,reference=reference,stale=stale))
        print('EPOCH',ds,arm,seed,history[-1],flush=True)
    save(folder/'selection.json',dict(lock=lock,epochs=len(history),best=min(history,key=lambda r:r['score']),
         checkpoint_sha256=sha256(folder/'best.pt'),parameters=sum(p.numel() for p in model.parameters())))


@torch.no_grad()
def test(ds,cfg,stats,lock,factors):
    folder=ROOT/ds
    models={}
    cores={}
    for arm in ARMS:
        for seed in SEEDS:
            path=folder/f'{arm}_{seed}'
            sel=read(path/'selection.json')
            assert sel['lock']==lock and sel['checkpoint_sha256']==sha256(path/'best.pt')
            model=JointModel(arm,*stats).cuda().eval().requires_grad_(False)
            model.load_state_dict(torch.load(path/'best.pt',weights_only=False)['model'])
            models[arm,seed]=model
            cores[arm,seed]=RadialTableProbe(model.net,fused=True).eval() if arm=='deep' else None
    if not (folder/'test_prepared.json').exists():
        prepare(ds,'test')
    outdir=folder/'test_records'
    outdir.mkdir(exist_ok=True)
    records=[]
    for j,i in enumerate(cfg['split']['test']):
        path=outdir/f'{i}.json'
        if path.exists():
            records.append(read(path))
            continue
        row=torch.load(folder/'data'/str(i)/'sample.pt',weights_only=False)
        s=geometry(row,True)
        rho=torch.tensor(np.load(folder/'data'/str(i)/'rho.npy'),device='cuda',dtype=torch.float64)
        rr=dict(id=i,points=len(rho),arms={},seconds={})
        for arm in (ARMS if j%2==0 else ARMS[::-1]):
            for seed in SEEDS:
                model=models[arm,seed]
                def call():
                    return model.full(s,factors,cores[arm,seed])
                pred,aug=call()
                assert pred.shape==rho.shape and torch.isfinite(pred).all() and torch.isfinite(aug).all()
                rr['arms'][f'{arm}_{seed}']=dict(density=metric(pred,rho,s),
                    occupation=measures(aug,row['target_aug'].cuda(),row['aug_mask'].cuda()))
                del pred,aug
                if seed==42 and i in cfg['timing_indices']:
                    for _ in range(2):
                        call()
                    times=[]
                    for _ in range(5):
                        torch.cuda.synchronize()
                        start=time.perf_counter()
                        call()
                        torch.cuda.synchronize()
                        times.append(time.perf_counter()-start)
                    rr['seconds'][arm]=times
        save(path,rr)
        records.append(rr)
        if j%50==0:
            print('TEST',ds,j+1,len(cfg['split']['test']),flush=True)
    assert [r['id'] for r in records]==cfg['split']['test']
    summary={}
    for arm in ARMS:
        times=[np.median(r['seconds'][arm]) for r in records if arm in r['seconds']]
        assert len(times)==50
        for seed in SEEDS:
            name=f'{arm}_{seed}'
            results=[r['arms'][name] for r in records]
            fields=(('density','nmae'),('density','mae'),('occupation','mae'),('occupation','relative_l1'))
            summary[name]={k+'_'+v:float(np.mean([r[k][v] for r in results])) for k,v in fields}
        summary[arm+'_mean_seconds']=float(np.mean(times))
    save(REPORT/f'{ds}_summary.json',dict(count=len(records),summary=summary))


def main(ds):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg=manifest(ds)
    if not (ROOT/ds/'fit_prepared.json').exists():
        raise RuntimeError('Training data preparation has not finished')
    split=cfg['split']
    ids=split['train']+split['validation']
    rows={i:torch.load(ROOT/ds/'data'/str(i)/'sample.pt',weights_only=False) for i in ids}
    assert all(row['id']==i for i,row in rows.items())
    stats=training_statistics([rows[i] for i in split['train']],elements=SPECIES[ds])
    factors=read_form_factors(Path('potpaw54.zip'),SPECIES[ds])
    lock={str(p):sha256(p) for p in (Path(__file__),Path('scripts/joint_pair_model.py'),
        Path('scripts/joint_pair_data.py'),Path('scripts/joint_paw_tables.py'),REPORT/'protocol.md',
        Path(f'data/public_density/{ds}/split.json'))}
    for seed in SEEDS:
        for arm in ARMS:
            train(ds,arm,seed,split,rows,stats,lock,factors)
            torch.cuda.empty_cache()
    test(ds,cfg,stats,lock,factors)
    print('COMPLETE',ds,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dataset',choices=SPECIES)
    main(parser.parse_args().dataset)
