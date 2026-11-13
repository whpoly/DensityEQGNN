"""Frozen/fine-tuned official Eden PAW heads, matched to DeepDFT pilot."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from e3nn import o3

from scripts.benchmark_deepdft_reuse import save
from scripts.deepdft_paw import BLOCKS,ELEMENTS,IRREPS
from scripts.edengnn_paw import EdenPAW,CHECKPOINT,inputs,capture
from scripts.nmc_probe_holdout import sha256
from scripts.train_deepdft_paw import ROOT as BASE,measures,summarize
from scripts.train_local_nmc_overfit import state_digest

ROOT = Path('runs/edengnn_paw_scaled_2026-10-30')
SOURCES = ['scripts/edengnn_paw.py','scripts/benchmark_edengnn_paw.py',
    'scripts/deepdft_paw.py','scripts/train_deepdft_paw.py','scripts/nmc_edengnn.py']


def load_data(root,split_names=('train','validation','test')):
    stored = torch.load(root/'dataset.pt',weights_only=False)
    result = {}
    for split in split_names:
        result[split] = []
        for row in stored['samples'][split]:
            s = {k:v.cuda() if torch.is_tensor(v) else v for k,v in row.items()}
            s['inputs'] = inputs(s)
            result[split].append(s)
    return stored,result


def prediction(model,s,arm):
    if arm=='frozen':
        return model.readout(s['eden_features'],s['numbers'])
    return model(*s['inputs'])


@torch.no_grad()
def evaluate(model,samples,arm,details=False):
    model.eval()
    records = []
    for s in samples:
        out = prediction(model,s,arm)
        row = dict(id=s['id'],**measures(out,s['target_aug'],s['aug_mask']))
        if details:
            row['by_l'] = {}
            anisotropic = s['aug_mask'].clone()
            for l in range(3):
                mask = torch.zeros_like(s['aug_mask'])
                for b in BLOCKS:
                    if b['l']==l:
                        mask[:,b['start']:b['stop']] = s['aug_mask'][:,b['start']:b['stop']]
                    if b['l']==0:
                        anisotropic[:,b['start']:b['stop']] = False
                row['by_l'][str(l)] = measures(out,s['target_aug'],mask)
            row['anisotropic'] = measures(out,s['target_aug'],anisotropic)
            row['by_element'] = {symbol:measures(out,s['target_aug'],s['aug_mask']&(s['numbers']==z)[:,None])
                for symbol,z in ELEMENTS.items() if (s['numbers']==z).any()}
        records.append(row)
    return dict(**summarize(records),records=records)


@torch.no_grad()
def prepare(root):
    if (root/'dataset.pt').exists():
        raise ValueError('Preserve existing data')
    original = torch.load(BASE/'dataset.pt',weights_only=False)
    base_settings = json.loads((BASE/'settings.json').read_text())
    assert all(sha256(s)==h for s,h in base_settings['source_hashes'].items())
    model = EdenPAW(original['mean'],original['scale']).cuda().eval()
    data = dict(mean=original['mean'],scale=original['scale'],samples={})
    for split,rows in original['samples'].items():
        data['samples'][split] = []
        for s in rows:
            row = {k:s[k] for k in ('id','positions','cell','numbers','target_aug','aug_mask','metadata')}
            row['eden_features'] = model.encode(*inputs(row)).cpu()
            data['samples'][split].append(row)
            print('PREPARED',split,row['id'],flush=True)
    train_features = torch.cat([s['eden_features'] for s in data['samples']['train']])
    irreps = o3.Irreps(model.encoder_config['conv']['irreps_node_features'])
    rms = []
    for (mul,ir),section in zip(irreps,irreps.slices()):
        values = train_features[:,section].reshape(len(train_features),mul,ir.dim)
        rms.append(values.square().mean((0,2)).sqrt().clamp_min(1e-4).repeat_interleave(ir.dim))
    data['feature_rms'] = torch.cat(rms)
    settings = dict(base_dataset_sha256=sha256(BASE/'dataset.pt'),base_settings_sha256=sha256(BASE/'settings.json'),
        base_results_sha256=sha256(BASE/'results.json'),eden_checkpoint_sha256=sha256(CHECKPOINT),
        source_hashes={s:sha256(s) for s in SOURCES},encoder_config=model.encoder_config,provenance=model.provenance,
        initial_encoder_state=state_digest(model.encoder.state_dict()),
        arms=['frozen','finetuned'],seeds=[42,43],steps=6000,batch=2,eval_every=200,lr=1e-3,min_lr=1e-5,
        selection='Validation macro raw MAE. All arm/seed selections frozen before new test metrics.',
        pretrained_density_encoder='Same eight training structures, 16000 density updates; no occupation labels in source checkpoint.',
        caveats='Known small test set from DeepDFT pilot; only L<=2; not equal parameter count/FLOPs; finetuned arm has dedicated occupation encoder.',
        feature_conditioning='Divide each initial encoder irrep channel by training-only RMS, shared across m, floor 1e-4. Fixed in both arms. Preliminary unscaled frozen seed 42 failed to learn at matched LR; preserved separately, no test evaluation.',
        torch=torch.__version__,device=torch.cuda.get_device_name(),
        head_parameters=sum(p.numel() for p in model.aug.parameters()),encoder_parameters=sum(p.numel() for p in model.encoder.parameters()))
    save(root/'settings.json',settings)
    torch.save(data,root/'dataset.pt')


def check_sources(root):
    cfg = json.loads((root/'settings.json').read_text())
    assert all(sha256(s)==h for s,h in cfg['source_hashes'].items())
    assert sha256(CHECKPOINT)==cfg['eden_checkpoint_sha256']
    return cfg


def audit_graphs(root):
    check_sources(root)
    if (root/'gradient_audit.json').exists():
        raise ValueError('Preserve audit')
    torch.manual_seed(181)
    data,samples = load_data(root,('train',))
    model = EdenPAW(data['mean'],data['scale'],feature_rms=data['feature_rms']).cuda().train()
    # A zero augmentation head would make encoder gradients trivially zero.
    for p in model.aug.parameters():
        torch.nn.init.normal_(p,std=.02)
    graphs = capture(model,samples['train'])
    opt = torch.optim.Adam(model.parameters(),lr=1e-4)
    records = []
    for step in range(8):
        pair = [samples['train'][(2*step+i)%8] for i in (0,1)]
        outputs,gradients,times = [],[],[]
        for mode in ('eager','graph'):
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            values = [model(*s['inputs']) if mode=='eager' else graphs[s['id']](*s['inputs']) for s in pair]
            loss = torch.stack([((out-s['target_aug'])/model.scale[s['numbers']])[s['aug_mask']].square().mean()
                for out,s in zip(values,pair)]).mean()
            loss.backward()
            torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
            outputs.append([v.detach().clone() for v in values])
            gradients.append({name:p.grad.detach().clone() for name,p in model.named_parameters() if p.grad is not None})
            del loss,values
        assert gradients[0].keys()==gradients[1].keys()==dict(model.named_parameters()).keys()
        numerator=denominator=0.
        for name in gradients[0]:
            numerator += float((gradients[0][name]-gradients[1][name]).double().square().sum())
            denominator += float(gradients[0][name].double().square().sum())
        relative = (numerator/max(denominator,1e-30))**.5
        max_abs = max(float((a-b).abs().max()) for a,b in zip(*outputs))
        assert denominator>0 and relative<2e-4 and max_abs<1e-4,(relative,max_abs)
        records.append(dict(step=step,ids=[s['id'] for s in pair],max_abs=max_abs,gradient_relative_l2=relative,
            eager_seconds=times[0],graph_seconds=times[1]))
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        opt.step()
        print('AUDIT '+json.dumps(records[-1]),flush=True)
        del outputs,gradients
    save(root/'gradient_audit.json',dict(passed=True,records=records,
        description='All eight training structures twice; two-structure accumulation; all parameter gradients; optimizer updates between comparisons.'))


def train(root,arm,seed):
    cfg = check_sources(root)
    folder = root/f'{arm}_{seed}'
    if folder.exists():
        raise ValueError('Preserve existing training')
    folder.mkdir()
    torch.manual_seed(seed)
    data,samples = load_data(root,('train','validation'))
    model = EdenPAW(data['mean'],data['scale'],feature_rms=data['feature_rms']).cuda()
    initial_encoder = state_digest(model.encoder.state_dict())
    assert initial_encoder==cfg['initial_encoder_state']
    graphs = None
    if arm=='frozen':
        model.encoder.requires_grad_(False)
    else:
        assert json.loads((root/'gradient_audit.json').read_text())['passed']
        graphs = capture(model,samples['train'])
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters,lr=cfg['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['steps'],eta_min=cfg['min_lr'])
    rng = np.random.default_rng(seed)
    best = float('inf')
    history = []
    start = time.perf_counter()
    for step in range(1,cfg['steps']+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index in rng.choice(len(samples['train']),size=cfg['batch'],replace=False):
            s = samples['train'][index]
            out = prediction(model,s,arm) if graphs is None else graphs[s['id']](*s['inputs'])
            losses.append(((out-s['target_aug'])/model.scale[s['numbers']])[s['aug_mask']].square().mean())
        loss = torch.stack(losses).mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters,5.)
        optimizer.step()
        scheduler.step()
        if step%cfg['eval_every']==0:
            val = evaluate(model,samples['validation'],arm)
            fit = evaluate(model,samples['train'],arm)
            record = dict(step=step,loss=float(loss),validation_mae=val['macro_mae'],train_mae=fit['macro_mae'],
                elapsed_seconds=time.perf_counter()-start)
            if val['macro_mae']<best:
                best = val['macro_mae']
                torch.save(dict(model=model.state_dict(),step=step,validation=val,arm=arm,seed=seed),folder/'best.pt')
            history.append(record)
            save(folder/'history.json',history)
            print('TRAIN '+json.dumps(record),flush=True)
    saved = torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    save(folder/'selection.json',dict(arm=arm,seed=seed,step=saved['step'],validation_mae=saved['validation']['macro_mae'],
        best_checkpoint_sha256=sha256(folder/'best.pt'),training_seconds=time.perf_counter()-start,
        trainable_parameters=sum(p.numel() for p in parameters),
        frozen_encoder_unchanged=state_digest(model.encoder.state_dict())==initial_encoder,
        source_hashes_match=all(sha256(s)==h for s,h in cfg['source_hashes'].items())))


@torch.no_grad()
def final(root):
    cfg = check_sources(root)
    if (root/'results.json').exists():
        raise ValueError('Preserve final results')
    selected = {f'{arm}_{seed}':json.loads((root/f'{arm}_{seed}/selection.json').read_text())
        for arm in cfg['arms'] for seed in cfg['seeds']}
    data,samples = load_data(root)
    results = dict(records={})
    for name,selection in selected.items():
        assert sha256(root/name/'best.pt')==selection['best_checkpoint_sha256']
        model = EdenPAW(data['mean'],data['scale'],feature_rms=data['feature_rms']).cuda().eval()
        model.load_state_dict(torch.load(root/name/'best.pt',map_location='cuda',weights_only=False)['model'])
        # All final predictions use freshly encoded features, even frozen arm.
        row = dict(selection=selection,metrics={split:evaluate(model,ss,'finetuned',details=True) for split,ss in samples.items()})
        row['equivariance'],row['timing'] = [],[]
        for s in samples['test']:
            out = model(*s['inputs'])
            if selection['arm']=='frozen':
                torch.testing.assert_close(out,prediction(model,s,'frozen'),rtol=2e-4,atol=2e-5)
            np.savez_compressed(root/name/(s['id']+'_prediction.npz'),prediction=out.cpu().numpy(),
                target=s['target_aug'].cpu().numpy(),mask=s['aug_mask'].cpu().numpy())
            torch.manual_seed(981)
            rotation = -o3.rand_matrix().double()
            q = torch.eye(3,dtype=torch.float64)[[1,2,0]]
            rotated = dict(s,positions=s['positions']@rotation.cuda().T+.07,cell=s['cell']@rotation.cuda().T)
            changed = model(*inputs(rotated))
            expected = out@IRREPS.D_from_matrix(q@rotation@q.T).float().cuda().T
            torch.testing.assert_close(changed,expected,rtol=3e-3,atol=3e-4)
            row['equivariance'].append(dict(id=s['id'],max_abs=float((changed-expected).abs().max()),
                relative_l1=float((changed-expected).abs().sum()/expected.abs().sum())))
            f = model.encode(*s['inputs'])
            times = dict(head=[],encoder_graph_and_head=[])
            for _ in range(3):
                model(*inputs(s)); model.readout(f,s['numbers'])
            for _ in range(10):
                torch.cuda.synchronize()
                started = time.perf_counter()
                model.readout(f,s['numbers'])
                torch.cuda.synchronize()
                times['head'].append(time.perf_counter()-started)
                started = time.perf_counter()
                value = model(*inputs(s))
                torch.cuda.synchronize()
                times['encoder_graph_and_head'].append(time.perf_counter()-started)
                torch.testing.assert_close(value,out,rtol=2e-4,atol=2e-5)
            row['timing'].append(dict(id=s['id'],seconds=times))
        results['records'][name] = row
        save(root/'results_partial.json',results)
        print('FINAL '+json.dumps(dict(name=name,mae=row['metrics']['test']['macro_mae'],relative_l1=row['metrics']['test']['macro_relative_l1'])),flush=True)
        del model
        torch.cuda.empty_cache()
    results['audit'] = dict(source_hashes_match=all(sha256(s)==h for s,h in cfg['source_hashes'].items()),
        original_eden_checkpoint_unchanged=sha256(CHECKPOINT)==cfg['eden_checkpoint_sha256'],
        deepdft_dataset_unchanged=sha256(BASE/'dataset.pt')==cfg['base_dataset_sha256'],
        deepdft_results_unchanged=sha256(BASE/'results.json')==cfg['base_results_sha256'])
    save(root/'results.json',results)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage',choices=('prepare','audit','train','final'),required=True)
    p.add_argument('--arm',choices=('frozen','finetuned'),default='frozen')
    p.add_argument('--seed',type=int,choices=(42,43),default=42)
    p.add_argument('--root',type=Path,default=ROOT)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    args.root.mkdir(parents=True,exist_ok=True)
    if args.stage=='prepare': prepare(args.root)
    elif args.stage=='audit': audit_graphs(args.root)
    elif args.stage=='train': train(args.root,args.arm,args.seed)
    else: final(args.root)


if __name__=='__main__':
    main()
