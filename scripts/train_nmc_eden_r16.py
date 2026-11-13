"""Fixed 16k-update SAD and official-Eden-encoder/R16 experiments."""
from dataclasses import asdict
from datetime import datetime,timezone
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from scripts.nmc_eden_r16 import ROOT,ARMS,build,prepare,coefficients,prediction,evaluate
from scripts.nmc_edengnn import write_json
from scripts.fit_nmc_holdout_oracle import BASIS
from scripts.nmc_probe_holdout import DEFAULT_ROOT,sha256
from scripts.train_local_nmc_overfit import state_digest


def decoder_audit(model,samples,arm):
    rows=[]
    for sample in samples:
        ids=sample['split']['train'][::997][:256]
        coeff=coefficients(model,sample,arm).detach().requires_grad_()
        normal=prediction(sample,coeff,ids)
        fast=prediction(sample,coeff,ids,True)
        torch.testing.assert_close(normal,fast,rtol=3e-5,atol=5e-5)
        a=torch.autograd.grad(normal.square().mean(),coeff)[0]
        b=torch.autograd.grad(fast.square().mean(),coeff)[0]
        relative=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-30))
        assert relative<2e-5
        rows.append({'sample_id':sample['id'],'train_points':len(ids),
                     'max_abs':float((normal-fast).detach().abs().max()),'gradient_relative_l2':relative})
    return rows


def graph_audit(model,samples):
    rows=[]
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
    # Restore BOTH parameters and buffers afterwards; audit never contributes training updates.
    original={k:v.detach().clone() for k,v in model.state_dict().items()}
    for iteration in range(16):
        s=samples[iteration%8]
        grads=[]
        values=[]
        for enabled in [False,True]:
            model.use_graphs=enabled
            model.zero_grad(set_to_none=True)
            result=model.coefficients(s['atom_data'])
            result.square().mean().backward()
            values.append(result.detach().clone())
            grads.append({n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None})
            del result
        assert grads[0].keys()==grads[1].keys()
        num=sum(float((grads[0][k]-grads[1][k]).double().square().sum()) for k in grads[0])
        den=sum(float(grads[0][k].double().square().sum()) for k in grads[0])
        relative=(num/max(den,1e-30))**.5
        maximum=float((values[0]-values[1]).abs().max())
        assert relative<2e-5
        torch.testing.assert_close(values[0],values[1],rtol=2e-5,atol=2e-5)
        rows.append({'iteration':iteration,'sample_id':s['id'],'max_abs':maximum,'gradient_relative_l2':relative})
        optimizer.step()
    model.load_state_dict(original)
    model.zero_grad(set_to_none=True)
    model.use_graphs=True
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    model=build(args.arm,'cuda')
    folder=ROOT/args.arm
    folder.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),Path('scripts/nmc_eden_r16.py'),Path('scripts/nmc_edengnn.py'),
           Path('scripts/r16_decoder_diagnostic.py'),Path('scripts/train_nmc_coefficient_pretrain.py'),
           Path('src/density_eqgnn/models/mace_like_model.py'),Path('scripts/edengnn_graph_training.py')]
    settings={'arm':args.arm,'samples':list(range(8)),'steps':16000,'probes':1024,'seed':42,'eval_every':500,
              'lr':1e-3,'min_lr':1e-5,'basis_config':asdict(BASIS),'initial_state_sha256':state_digest(model.state_dict()),
              'parameter_count':sum(p.numel() for p in model.parameters()),
              'optimizer':'Adam, cosine LR, no weight decay, grad clip 5',
              'loss':'L1(SAD + decoded residual - DFT) / training mean(abs(DFT))',
              'selection':'original fixed 8192 validation points, no test decisions',
              'prior':'same POTCAR-derived SAD as Eden baseline; not verified against native VASP zero-step density',
              'source_sha256':{str(x):sha256(x) for x in paths},
              'encoder_provenance':getattr(model,'provenance',None),
              'torch':torch.__version__,'gpu':torch.cuda.get_device_name()}
    if (folder/'settings.json').exists():
        assert args.resume and json.loads((folder/'settings.json').read_text())==settings
    else:
        write_json(folder/'settings.json',settings)
    if (folder/'selection_frozen.json').exists():
        return
    samples=prepare(model,args.arm)
    write_json(folder/'sources.json',[{'sample_id':s['id'],'source':s['metadata'],'split':s['split_metadata'],
               'training_normalizer':float(s['normalizer'])} for s in samples])
    if not (folder/'decoder_audit.json').exists():
        write_json(folder/'decoder_audit.json',decoder_audit(model,samples,args.arm))
    if args.arm=='eden_r16_sad':
        model.capture(samples)
        if not (folder/'graph_audit.json').exists():
            write_json(folder/'graph_audit.json',graph_audit(model,samples))
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=16000,eta_min=.00001)
    rng=torch.Generator(device='cuda').manual_seed(10042)
    order_rng=np.random.default_rng(20042)
    order=[]
    history=[]
    seen=[torch.zeros(len(s['target']),device='cuda',dtype=torch.bool) for s in samples]
    best,best_step,start_step=float('inf'),0,1
    if args.resume and (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        rng.set_state(saved['rng'].cpu())
        order_rng.bit_generator.state=saved['order_rng']
        order,history=saved['order'],saved['history']
        seen=[x.to('cuda') for x in saved['seen']]
        best,best_step,start_step=saved['best'],saved['best_step'],saved['step']+1
        del saved
    started=time.perf_counter()
    timings=[]
    def checkpoint(step):
        return {'model':model.state_dict(),'step':step,'best':best,'best_step':best_step,'history':history,
                'settings':settings,'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                'rng':rng.get_state(),'order_rng':order_rng.bit_generator.state,'order':order,
                'seen':[v.cpu() for v in seen]}
    def save(name,step):
        tmp=folder/(name+'.tmp')
        torch.save(checkpoint(step),tmp)
        tmp.replace(folder/(name+'.pt'))
    def assess(step):
        nonlocal best,best_step
        metrics=evaluate(model,samples,args.arm)
        row={'step':step,'elapsed_this_process':time.perf_counter()-started,**metrics}
        history.append(row)
        if metrics['selection_macro_nmae']<best:
            best,best_step=metrics['selection_macro_nmae'],step
            save('best',step)
        save('last',step)
        write_json(folder/'history.json',history)
        print('EVAL '+json.dumps({k:v for k,v in row.items() if k!='records'}),flush=True)
    if start_step==1:
        write_json(folder/'sad_validation.json',evaluate(model,samples,args.arm,sad_only=True))
        assess(0)
    for step in range(start_step,16001):
        model.train()
        if not order:
            order=order_rng.permutation(8).tolist()
        si=order.pop()
        sample=samples[si]
        allowed=sample['split']['train']
        ids=allowed[torch.randint(len(allowed),(1024,),device='cuda',generator=rng)]
        seen[si][ids]=True
        optimizer.zero_grad(set_to_none=True)
        tick=time.perf_counter()
        coeff=coefficients(model,sample,args.arm)
        pred=prediction(sample,coeff,ids,True)
        loss=((pred-sample['target'][ids])/sample['normalizer']).abs().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-tick)
        if step%100==0:
            status={'arm':args.arm,'phase':'training','step':step,'total_steps':16000,
                    'loss':float(loss.detach()),'grad':float(grad),'recent_step_seconds':float(np.median(timings[-100:])),
                    'best_validation_nmae':best,'updated_utc':datetime.now(timezone.utc).isoformat()}
            write_json(folder/'status.json',status)
            write_json(ROOT/'status.json',status)
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
    assert audit==json.loads((DEFAULT_ROOT/'r16_eight/sampling_audit.json').read_text())
    write_json(folder/'sampling_audit.json',audit)
    frozen={'arm':args.arm,'training_complete':True,'best_step':best_step,'best_validation_nmae':best,
            'checkpoint_sha256':sha256(folder/'best.pt'),'frozen_before_test_utc':datetime.now(timezone.utc).isoformat()}
    write_json(folder/'selection_frozen.json',frozen)
    write_json(folder/'status.json',{'phase':'complete',**frozen})
    print('COMPLETE '+json.dumps(frozen),flush=True)


if __name__=='__main__':
    main()
