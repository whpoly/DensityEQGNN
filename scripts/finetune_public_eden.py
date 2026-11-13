"""Full-split, end-to-end Eden occupation fine-tuning; see locked protocol."""
import gc, json, math, time, hashlib
from pathlib import Path
import numpy as np
import torch
from scripts.public_paw_experiment import ROOT as BASE, check, structure_loss
from scripts.benchmark_public_density import save
from scripts.edengnn_paw import EdenPAW,inputs
from scripts.finalize_public_paw import add_details
from scripts.train_deepdft_paw import summarize
from scripts.public_paw_predictor import PublicPAWPredictor
from scripts.nmc_probe_holdout import sha256

ROOT=Path('runs/eden_full_finetune_2026-11-02')
REPORT=Path('reports/eden_full_finetune_2026-11-02')
def read(p):return json.loads(Path(p).read_text())
def write_pt(p,x):
    tmp=p.with_suffix('.tmp');torch.save(x,tmp);tmp.replace(p)
def geom(s):return {k:s[k].cuda() for k in ('positions','numbers','cell')}
def model(seed):
    folder=BASE/f'eden_frozen_{seed}'
    assert sha256(folder/'best.pt')==read(folder/'selection.json')['checkpoint_sha256']
    state=torch.load(folder/'best.pt',weights_only=False,map_location='cpu')['model']
    m=EdenPAW(state['mean'],state['scale'],pretrained=False,feature_rms=state['feature_rms']).cuda()
    m.load_state_dict(state);m.requires_grad_(True)
    return m
def pack(ids,rows,graphs):
    g=[graphs[i] for i in ids]; ns=[len(a[1]) for a in g]
    args=(torch.cat([a[0] for a in g]),torch.cat([a[1] for a in g]),
          torch.cat([a[2]+int(n) for a,n in zip(g,np.cumsum([0]+ns[:-1]))],1),torch.cat([a[3] for a in g]))
    b={k:torch.cat([rows[i][k] for i in ids]) for k in ('target_aug','aug_mask')}
    b.update(lengths=ns,numbers=args[1],groups=torch.repeat_interleave(torch.arange(len(ns),device='cuda'),torch.tensor(ns,device='cuda'),output_size=sum(ns)))
    return args,b
@torch.no_grad()
def validate(m,ids,rows,graphs):
    m.eval();maes=[]
    for j in range(0,len(ids),8):
        a,b=pack(ids[j:j+8],rows,graphs);p=m(*a)
        for err,mask in zip((p-b['target_aug']).abs().split(b['lengths']),b['aug_mask'].split(b['lengths'])):
            maes.append(float(err[mask].mean()))
    return float(np.mean(maes))
def train(seed,rows,cfg):
    folder=ROOT/f'seed_{seed}';folder.mkdir(exist_ok=True)
    if (folder/'selection.json').exists():return
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    tr,va=cfg['split']['train'],cfg['split']['validation']
    m=model(seed)
    graphs={i:inputs(geom(rows[i])) for i in tr+va}
    for i in tr+va:
        for k in ('target_aug','aug_mask'):rows[i][k]=rows[i][k].cuda()
    ref={n:p.detach().clone() for n,p in m.named_parameters()}
    opt=torch.optim.Adam([dict(params=m.encoder.parameters(),lr=3e-5),dict(params=m.aug.parameters(),lr=3e-4)])
    steps=math.ceil(len(tr)/8)*4
    history=[];best=float('inf');start_ep=0
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda t:.1+.9*(1+math.cos(math.pi*t/steps))/2)
    a,b=pack(tr[:2],rows,graphs)
    with torch.no_grad():
        p=m(*a);single=torch.cat([m(*graphs[i]) for i in tr[:2]])
        torch.testing.assert_close(p,single,rtol=5e-5,atol=5e-5)
    v0=validate(m,va,rows,graphs)
    assert abs(v0-read(BASE/f'eden_frozen_{seed}/selection.json')['validation_mae'])<1e-5
    if (folder/'last.pt').exists():
        last=torch.load(folder/'last.pt',weights_only=False)
        assert last['protocol']==sha256(REPORT/'protocol.md')
        m.load_state_dict(last['state']);opt.load_state_dict(last['opt']);scheduler.load_state_dict(last['scheduler'])
        history=last['history'];start_ep=last['epoch'];best=last['best'];rng.bit_generator.state=last['rng'];del last
    for ep in range(start_ep+1,5):
        m.train();order=rng.permutation(tr);losses=[];start=time.perf_counter()
        for j in range(0,len(order),8):
            a,b=pack(order[j:j+8],rows,graphs);opt.zero_grad(set_to_none=True)
            loss=structure_loss(m(*a),b,m.scale)
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),5.,error_if_nonfinite=True);opt.step();scheduler.step()
            losses.append(loss.detach().item())
            # WDDM can page fragmented reservations into shared memory even
            # though this batch needs < 10 GiB live. Release unused blocks.
            if torch.cuda.memory_reserved()>10*2**30:
                torch.cuda.empty_cache()
            if (j//8)%20==0: print('TRAIN',seed,ep,j+len(b['lengths']),round(time.perf_counter()-start,1),flush=True)
        val=validate(m,va,rows,graphs)
        updates={key:float(sum((p.detach()-ref[n]).square().sum() for n,p in m.named_parameters() if n.startswith(key)).sqrt()) for key in ('encoder','aug')}
        assert all(v>0 and np.isfinite(v) for v in updates.values())
        row=dict(epoch=ep,validation_mae=val,standardized_loss=float(np.mean(losses)),update_l2=updates,seconds=time.perf_counter()-start)
        if val<best:
            best=val;write_pt(folder/'best.pt',dict(model=m.state_dict(),epoch=ep,validation_mae=val))
        history.append(row);save(folder/'history.json',history)
        write_pt(folder/'last.pt',dict(state=m.state_dict(),opt=opt.state_dict(),scheduler=scheduler.state_dict(),history=history,epoch=ep,best=best,rng=rng.bit_generator.state,protocol=sha256(REPORT/'protocol.md')))
        print('EPOCH',seed,row,flush=True)
    save(folder/'selection.json',dict(epoch=min(history,key=lambda r:r['validation_mae'])['epoch'],validation_mae=best,initial_validation_mae=v0,sha256=sha256(folder/'best.pt'),parameters=sum(p.numel() for p in m.parameters())))
def selected(seed):
    folder=ROOT/f'seed_{seed}';assert sha256(folder/'best.pt')==read(folder/'selection.json')['sha256']
    m=model(seed);m.load_state_dict(torch.load(folder/'best.pt',weights_only=False)['model']);return m.eval().requires_grad_(False)
@torch.no_grad()
def final(rows,cfg):
    if (ROOT/'accuracy.json').exists():return
    models={s:selected(s) for s in (42,43)}; records={s:[] for s in models}
    ids=cfg['split']['test'];graphs={i:inputs(geom(rows[i])) for i in ids}
    for j in range(0,len(ids),8):
        group=ids[j:j+8]
        for i in group:
            for k in ('target_aug','aug_mask','numbers'):rows[i][k]=rows[i][k].cuda()
        a,b=pack(group,rows,graphs)
        for seed,m in models.items():
            for i,p in zip(group,m(*a).split(b['lengths'])):records[seed].append(add_details(p,rows[i]))
    data={}
    for seed,rr in records.items():
        common=[r for r in rr if r['id'] in cfg['paired_test']]
        assert len(common)==497 and len(rr)==500
        data[str(seed)]=dict(all500=summarize(rr),paired497=summarize(common),records=rr)
    save(ROOT/'accuracy.json',data)
@torch.no_grad()
def timing(rows):
    if (ROOT/'timing.json').exists():return
    deep=PublicPAWPredictor(seed=42); eden=selected(42)
    ids=[r['id'] for r in read(BASE/'timing_deep_occupation.json')['records']]
    assert len(ids)==50
    calls={'DeepDFT':lambda s:deep.predict(s,density=False)['augmentation'],'Eden':lambda s:eden(*inputs(s))}
    records=[]
    for j,i in enumerate(ids):
        s=geom(rows[i]); row=dict(id=i)
        for name in (list(calls) if j%2==0 else list(reversed(calls))):
            call=calls[name]
            for _ in range(2):out=call(s);del out
            tt=[];peaks=[]
            for _ in range(5):
                torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                t=time.perf_counter();out=call(s);torch.cuda.synchronize();tt.append(time.perf_counter()-t)
                peaks.append((torch.cuda.max_memory_allocated()-base)/2**30);del out
            row[name]=dict(seconds=tt,median=float(np.median(tt)),peak_gib=max(peaks))
        records.append(row)
    save(ROOT/'timing.json',records)
def report():
    rows=[];results=read(ROOT/'accuracy.json')
    for arm in ('deep_frozen','eden_frozen','eden_finetuned'):
        for seed in ('42','43'):
            x=results[seed]['paired497'] if arm=='eden_finetuned' else read(BASE/f'{arm}_final.json')['seeds'][seed]['paired_test']
            rows.append((arm,seed,x['macro_mae'],x['macro_relative_l1']*100))
    tt=read(ROOT/'timing.json');times={m:float(np.mean([r[m]['median'] for r in tt])) for m in ('Eden','DeepDFT')}
    save(REPORT/'summary.json',dict(primary497=rows,seconds=times,diagnostic500={s:results[s]['all500'] for s in results},selections={s:read(ROOT/f'seed_{s}/selection.json') for s in results}))
    (REPORT/'report.md').write_text('# 完整 NMC 划分：Eden 编码器与 occupation 头微调\n\n官方划分 1450/50/500；下表剔除 Eden 密度预训练见过的 3 个结构，仅比较相同 497 个测试结构。角阶 L≤2，先逐结构再平均。\n\n| 模型 | seed | MAE | 相对 L1 (%) |\n|---|---:|---:|---:|\n'+'\n'.join('| %s | %s | %.6f | %.3f |'%r for r in rows)+f'\n\n50 结构配对重测，含构图和编码的 occupation 单独推理：DeepDFT {times["DeepDFT"]:.5f} 秒，微调 Eden {times["Eden"]:.5f} 秒。两者预训练与参数量不同；微调从同种子的冻结头结果继续 4 个完整 epoch。没有测试密度分支替换、SCF 或 DFT 总耗时。\n',encoding='utf8')
def main():
    ROOT.mkdir(exist_ok=True);cfg=check();torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    rows=torch.load(BASE/'structures.pt',weights_only=False)
    for seed in (42,43):train(seed,rows,cfg);gc.collect();torch.cuda.empty_cache()
    final(rows,cfg);gc.collect();torch.cuda.empty_cache();timing(rows);report()
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
