"""Fixed two-stage R16 coefficient pretraining; no test evaluation here."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from density_eqgnn.models import build_compact_coefficient_model
from scripts.fit_nmc_holdout_oracle import BASIS
from scripts.nmc_edengnn import write_json
from scripts.nmc_probe_holdout import DEFAULT_ROOT,compact_prediction,sha256
from scripts.prepare_nmc_coefficient_teachers import ROOT
from scripts.r16_decoder_diagnostic import decode
from scripts.train_local_nmc_overfit import state_digest
from scripts.train_nmc_probe_holdout import prepare,representation,evaluate


def coefficient_loss(coeff,target):
    return (coeff.double()-target).square().mean()/target.square().mean().clamp_min(1e-30)


def fast_prediction(sample,coeff,indices):
    source,receiver,shift=sample['graph'].select(indices)
    delta=sample['probes'][indices][receiver]-sample['positions'][source]-shift.double()@sample['cell']
    return decode(coeff,delta,torch.zeros_like(source),source,receiver,len(indices),BASIS)


def initialize(device):
    torch.manual_seed(42)
    np.random.seed(42)
    original=json.loads((DEFAULT_ROOT/'r16_eight/settings.json').read_text())
    model=build_compact_coefficient_model(original['model_config']).to(device)
    assert state_digest(model.state_dict())==original['initial_state_sha256']
    return model,original


def prepare_data(model,device):
    assert (ROOT/'teachers_complete.json').exists()
    samples=prepare(list(range(8)),Path('data/nmc_cache'),DEFAULT_ROOT,model,'r16',device)
    for sample in samples:
        folder=ROOT/'teachers'/f'sample_{sample["id"]:04d}'
        teacher=torch.load(folder/'teacher.pt',map_location=device,weights_only=False)
        complete=json.loads((folder/'complete.json').read_text())
        assert sha256(folder/'teacher.pt')==complete['checkpoint_sha256']
        assert teacher['split']==sample['split_metadata'] and teacher['basis_config']==asdict(BASIS)
        assert teacher['fit_points']==len(sample['split']['train'])
        assert teacher['validation_points_used_for_fit']==teacher['test_points_used_for_fit']==0
        sample['teacher']=teacher['coefficients']
        sample['teacher_sha256']=complete['checkpoint_sha256']
    return samples


@torch.no_grad()
def validation(model,samples):
    result=evaluate(model,samples,'r16',BASIS,2048)
    errors=[]
    for sample,row in zip(samples,result['records']):
        coeff=representation(model,sample,'r16')
        row['coefficient_relative_mse']=float(coefficient_loss(coeff,sample['teacher']))
        row['coefficient_rms']=float(coeff.double().square().mean().sqrt())
        row['degree_relative_mse']=[]
        c=coeff.double().reshape(1,len(sample['numbers']),16,25)
        t=sample['teacher'].reshape_as(c)
        for degree in range(5):
            block=slice(degree**2,(degree+1)**2)
            row['degree_relative_mse'].append(float(coefficient_loss(c[...,block],t[...,block])))
        errors.append(row['coefficient_relative_mse'])
    result['coefficient_relative_mse']=float(np.mean(errors))
    return result


def audit_decoder(model,sample):
    ids=sample['split']['train'][::997][:256]
    coeff=representation(model,sample,'r16').detach().requires_grad_()
    direct=compact_prediction(sample,coeff,ids,BASIS)
    fast=fast_prediction(sample,coeff,ids)
    torch.testing.assert_close(fast,direct,rtol=3e-5,atol=5e-5)
    weight=torch.linspace(.2,1.2,len(ids),device=ids.device)
    a=torch.autograd.grad((direct*weight).sum(),coeff)[0]
    b=torch.autograd.grad((fast*weight).sum(),coeff)[0]
    relative=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a))
    assert relative<2e-5
    return {'sample_id':sample['id'],'partition':'train','points':len(ids),
            'max_abs_difference':float((fast-direct).detach().abs().max()),
            'coefficient_gradient_relative_l2':relative}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=['coefficient_mse','density_metric'],required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda')
    model,original=initialize(device)
    folder=ROOT/args.arm
    folder.mkdir(parents=True,exist_ok=True)
    source_paths=[Path(__file__),Path('scripts/prepare_nmc_coefficient_teachers.py'),
                  Path('scripts/r16_decoder_diagnostic.py'),Path('scripts/train_nmc_probe_holdout.py'),
                  Path('scripts/nmc_probe_holdout.py'),Path('src/density_eqgnn/compact_basis.py')]
    settings={'arm':args.arm,'samples':list(range(8)),'pretrain_steps':8000,'finetune_steps':8000,
              'probes':1024,'eval_every':500,'seed':42,'pretrain_lr':[1e-3,1e-5],'finetune_lr':[1e-4,1e-6],
              'model_config':original['model_config'],'basis_config':asdict(BASIS),
              'initial_state_sha256':state_digest(model.state_dict()),
              'teacher_sha256':{str(i):sha256(ROOT/'teachers'/f'sample_{i:04d}'/'teacher.pt') for i in range(8)},
              'optimizer':'Adam, reset between phases; cosine; no decay; grad clip 5',
              'selection':'minimum original 8192 validation-point macro density NMAE; test excluded',
              'source_sha256':{str(p):sha256(p) for p in source_paths}}
    if (folder/'settings.json').exists():
        assert args.resume and json.loads((folder/'settings.json').read_text())==settings
    else:
        write_json(folder/'settings.json',settings)
    if (folder/'selection_frozen.json').exists():
        print(f'SKIP completed {args.arm}',flush=True)
        return
    samples=prepare_data(model,device)
    if not (folder/'decoder_audit.json').exists():
        audits=[audit_decoder(model,s) for s in samples]
        write_json(folder/'decoder_audit.json',audits)
    if args.audit_only:
        return
    rng=torch.Generator(device=device).manual_seed(10042)
    order_rng=np.random.default_rng(20042)
    order=[]
    history=[]
    seen=[torch.zeros(len(s['target']),dtype=torch.bool,device=device) for s in samples]
    saved=None
    if args.resume and (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model'])
        rng.set_state(saved['rng'].cpu())
        order_rng.bit_generator.state=saved['order_rng']
        order,history=saved['order'],saved['history']
        seen=[x.to(device) for x in saved['seen']]
    started=time.perf_counter()
    step_times=[]
    for phase in ['pretrain','finetune']:
        if saved is not None and saved['phase']=='finetune' and phase=='pretrain':
            continue
        phase_dir=folder/phase
        phase_dir.mkdir(exist_ok=True)
        lr,minimum=settings[phase+'_lr']
        continuing=saved is not None and saved['phase']==phase
        if phase=='finetune' and not continuing:
            pre=torch.load(folder/'pretrain/best.pt',map_location=device,weights_only=False)
            model.load_state_dict(pre['model'])
            del pre
        optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=8000,eta_min=minimum)
        best,best_step,start_step=float('inf'),0,1
        if continuing:
            optimizer.load_state_dict(saved['optimizer'])
            scheduler.load_state_dict(saved['scheduler'])
            best,best_step,start_step=saved['best'],saved['best_step'],saved['step']+1
        def checkpoint(step):
            return {'model':model.state_dict(),'model_config':settings['model_config'],'basis_config':asdict(BASIS),
                    'phase':phase,'step':step,'best':best,'best_step':best_step,'history':history,'settings':settings,
                    'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'rng':rng.get_state(),'order_rng':order_rng.bit_generator.state,'order':order,
                    'seen':[x.cpu() for x in seen]}
        def save(path,value):
            tmp=path.with_suffix('.tmp')
            torch.save(value,tmp)
            tmp.replace(path)
        def assess(step):
            nonlocal best,best_step
            metrics=validation(model,samples)
            row={'phase':phase,'step':step,'elapsed_this_process':time.perf_counter()-started,**metrics}
            history.append(row)
            if metrics['selection_macro_nmae']<best:
                best,best_step=metrics['selection_macro_nmae'],step
                save(phase_dir/'best.pt',checkpoint(step))
            save(folder/'last.pt',checkpoint(step))
            write_json(folder/'history.json',history)
            print('EVAL '+json.dumps({k:v for k,v in row.items() if k!='records'}),flush=True)
        if not continuing:
            assess(0)
        for step in range(start_step,8001):
            model.train()
            if not order:
                order=order_rng.permutation(8).tolist()
            si=order.pop()
            sample=samples[si]
            allowed=sample['split']['train']
            ids=allowed[torch.randint(len(allowed),(1024,),device=device,generator=rng)]
            # Records actual density-supervision points; coefficient-MSE uses
            # teachers fitted on all train points, not these unused draws.
            if phase=='finetune' or args.arm=='density_metric':
                seen[si][ids]=True
            optimizer.zero_grad(set_to_none=True)
            tick=time.perf_counter()
            coeff=representation(model,sample,'r16')
            if phase=='pretrain' and args.arm=='coefficient_mse':
                loss=coefficient_loss(coeff,sample['teacher'])
            elif phase=='pretrain':
                loss=(fast_prediction(sample,coeff.double()-sample['teacher'],ids)/sample['normalizer']).square().mean()
            else:
                loss=((fast_prediction(sample,coeff,ids).double()-sample['target'][ids])/sample['normalizer']).abs().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss')
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter()-tick)
            if step%100==0:
                status={'arm':args.arm,'phase':phase,'step':step,'phase_steps':8000,'loss':float(loss.detach()),
                        'grad':float(grad),'recent_step_seconds':float(np.median(step_times[-100:])),
                        'best_validation_nmae':best,'updated_utc':datetime.now(timezone.utc).isoformat()}
                write_json(folder/'status.json',status)
                write_json(ROOT/'status.json',status)
                print('TRAIN '+json.dumps(status),flush=True)
            if step%500==0:
                assess(step)
        write_json(phase_dir/'complete.json',{'best_step':best_step,'best_validation_nmae':best,
                   'checkpoint_sha256':sha256(phase_dir/'best.pt')})
        saved=None
    audit=[]
    for sample,visited in zip(samples,seen):
        assert int(visited[sample['split']['validation']].sum())==int(visited[sample['split']['test']].sum())==0
        audit.append({'sample_id':sample['id'],'teacher_fit_points':len(sample['split']['train']),
                      'unique_density_gradient_points':int(visited.sum()),'validation_gradient_points':0,'test_gradient_points':0,
                      'teacher_sha256':sample['teacher_sha256'],'split':sample['split_metadata']})
    write_json(folder/'sampling_audit.json',audit)
    frozen={'arm':args.arm,'training_complete':True,'pretrain_steps':8000,'finetune_steps':8000,
            'frozen_before_test_utc':datetime.now(timezone.utc).isoformat(),
            'checkpoints':{phase:json.loads((folder/phase/'complete.json').read_text()) for phase in ['pretrain','finetune']}}
    write_json(folder/'selection_frozen.json',frozen)
    write_json(folder/'status.json',{'phase':'complete',**frozen})
    print('ARM_COMPLETE '+json.dumps(frozen),flush=True)


if __name__=='__main__':
    main()
