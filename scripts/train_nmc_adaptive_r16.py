"""Paired fixed-budget continuation; test labels are never used here."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.nmc_adaptive_r16 import ROOT, ARMS, STEPS, BASE_SHA, build
from scripts.nmc_eden_r16 import prepare, prediction, evaluate
from scripts.nmc_edengnn import write_json
from scripts.nmc_probe_holdout import sha256
from scripts.train_nmc_eden_r16 import decoder_audit, graph_audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    folder=ROOT/args.arm
    folder.mkdir(parents=True,exist_ok=True)
    model=build(args.arm)
    paths=[Path(__file__),Path('scripts/nmc_adaptive_r16.py'),Path('scripts/nmc_eden_r16.py'),
           Path('scripts/train_nmc_eden_r16.py'),Path('scripts/nmc_edengnn.py'),
           Path('scripts/r16_decoder_diagnostic.py'),Path('scripts/r16_grid_diagnostic.py'),
           Path('scripts/finalize_nmc_adaptive_r16.py'),Path('scripts/run_nmc_adaptive_r16.py'),
           Path('reports/nmc_adaptive_r16_protocol_2026-10-29.md')]
    settings={'arm':args.arm,'steps':STEPS,'probes':1024,'lr':1e-4,'min_lr':1e-6,
              'probe_seed':11042,'order_seed':21042,'base_sha256':BASE_SHA,
              'parameter_count':sum(p.numel() for p in model.parameters()),
              'source_sha256':{str(p):sha256(p) for p in paths},
              'optimizer':'reset Adam, cosine LR, clip 5, no weight decay',
              'torch':torch.__version__,'gpu':torch.cuda.get_device_name()}
    if (folder/'settings.json').exists():
        assert args.resume and json.loads((folder/'settings.json').read_text())==settings
    else:
        write_json(folder/'settings.json',settings)
    if (folder/'selection_frozen.json').exists():
        return
    samples=prepare(model,'eden_r16_sad')
    write_json(folder/'sources.json',[{'sample_id':s['id'],'source':s['metadata'],'split':s['split_metadata']} for s in samples])
    write_json(folder/'decoder_audit.json',decoder_audit(model,samples,'eden_r16_sad'))
    model.capture(samples)
    write_json(folder/'graph_audit.json',graph_audit(model,samples))
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=STEPS,eta_min=1e-6)
    rng=torch.Generator(device='cuda').manual_seed(11042)
    order_rng=np.random.default_rng(21042)
    order,history=[],[]
    seen=[torch.zeros(len(s['target']),device='cuda',dtype=torch.bool) for s in samples]
    best,best_step,start=float('inf'),0,1
    if args.resume and (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        rng.set_state(saved['rng'].cpu())
        order_rng.bit_generator.state=saved['order_rng']
        order,history=saved['order'],saved['history']
        seen=[x.to('cuda') for x in saved['seen']]
        best,best_step,start=saved['best'],saved['best_step'],saved['step']+1
        del saved
    started=time.perf_counter()
    timings=[]
    def save(name,step):
        state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
               'step':step,'best':best,'best_step':best_step,'settings':settings,'history':history,
               'rng':rng.get_state(),'order_rng':order_rng.bit_generator.state,'order':order,
               'seen':[s.cpu() for s in seen]}
        temporary=folder/(name+'.tmp')
        torch.save(state,temporary)
        temporary.replace(folder/(name+'.pt'))
    def assess(step):
        nonlocal best,best_step
        metrics=evaluate(model,samples,'eden_r16_sad')
        row={'step':step,'elapsed_this_process':time.perf_counter()-started,**metrics}
        history.append(row)
        if metrics['selection_macro_nmae']<best:
            best,best_step=metrics['selection_macro_nmae'],step
            save('best',step)
        save('last',step)
        write_json(folder/'history.json',history)
        print('EVAL '+json.dumps({k:v for k,v in row.items() if k!='records'}),flush=True)
    if start==1:
        assess(0)
        # Both arms must begin at the verified pretrained validation score.
        assert abs(best-0.0034417719262500855)<2e-8
    for step in range(start,STEPS+1):
        model.train()
        if not order:
            order=order_rng.permutation(8).tolist()
        si=order.pop()
        s=samples[si]
        allowed=s['split']['train']
        ids=allowed[torch.randint(len(allowed),(1024,),device='cuda',generator=rng)]
        seen[si][ids]=True
        optimizer.zero_grad(set_to_none=True)
        tick=time.perf_counter()
        coeff=model.coefficients(s['atom_data'])
        pred=prediction(s,coeff,ids,True)
        loss=((pred-s['target'][ids])/s['normalizer']).abs().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite training loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-tick)
        if step%100==0:
            status={'arm':args.arm,'phase':'training','step':step,'total_steps':STEPS,
                    'loss':float(loss.detach()),'grad':float(grad),'best_validation_nmae':best,
                    'recent_step_seconds':float(np.median(timings[-100:])),
                    'updated_utc':datetime.now(timezone.utc).isoformat()}
            write_json(folder/'status.json',status)
            print('TRAIN '+json.dumps(status),flush=True)
        if step%500==0:
            assess(step)
    audit=[]
    for s,visited in zip(samples,seen):
        val=int(visited[s['split']['validation']].sum())
        test=int(visited[s['split']['test']].sum())
        assert val==test==0
        audit.append({'sample_id':s['id'],'unique_training_points':int(visited.sum()),
                      'validation_points_used_for_gradients':val,'test_points_used_for_gradients':test})
    if args.arm=='adaptive':
        assert audit==json.loads((ROOT/'continued/sampling_audit.json').read_text())
    write_json(folder/'sampling_audit.json',audit)
    frozen={'arm':args.arm,'training_complete':True,'best_step':best_step,'best_validation_nmae':best,
            'checkpoint_sha256':sha256(folder/'best.pt'),'frozen_before_test_utc':datetime.now(timezone.utc).isoformat()}
    write_json(folder/'selection_frozen.json',frozen)
    write_json(folder/'status.json',{'phase':'complete',**frozen})
    print('COMPLETE '+json.dumps(frozen),flush=True)


if __name__=='__main__':
    main()
