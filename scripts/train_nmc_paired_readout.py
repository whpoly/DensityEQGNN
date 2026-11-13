"""Train a coefficient/query readout with identical atom backbone and sampling."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from density_eqgnn.compact_basis import basis_checkpoint_metadata
from density_eqgnn.models.paired_readout import PairedReadoutModel
from density_eqgnn.training.config import load_config
from scripts.fit_nmc_holdout_oracle import BASIS, ROOT, record
from scripts.nmc_probe_holdout import DEFAULT_ROOT, compact_prediction, sha256
from scripts.train_local_nmc_overfit import state_digest
from scripts.train_nmc_probe_holdout import prepare


def predict(model,sample,ids,rep=None):
    rep=model.representation(sample) if rep is None else rep
    if model.kind=="coefficient":
        return compact_prediction(sample,rep,ids,BASIS)
    return model.query(sample,ids,rep)


@torch.no_grad()
def evaluate(model,samples,chunk,final=False):
    model.eval()
    rows=[]
    for sample in samples:
        rep=model.representation(sample)
        if final:
            ids=torch.arange(len(sample["target"]),device=sample["target"].device)
            labels=torch.zeros_like(ids)
            labels[sample["split"]["validation"]]=1
            labels[sample["split"]["test"]]=2
            sums=torch.zeros(3,2,dtype=torch.float64,device=ids.device)
        else:
            ids=sample["split"]["selection"]
            sums=torch.zeros(1,2,dtype=torch.float64,device=ids.device)
        for selected in ids.split(chunk):
            output=predict(model,sample,selected,rep).double()
            target=sample["target"][selected]
            values=torch.stack(((output-target).abs(),target.abs()),1)
            if not torch.isfinite(values).all(): raise RuntimeError("Nonfinite predictions")
            sums.index_add_(0,labels[selected] if final else torch.zeros_like(selected),values)
        row={"sample_id":sample["id"]}
        for name,parts in zip(("train","validation","test") if final else ("selection",),sums):
            row[name+"_nmae"]=float(parts[0]/parts[1])
        if final:
            total=sums.sum(0)
            row["full_grid_nmae"]=float(total[0]/total[1])
        rows.append(row)
    names=("train","validation","test","full_grid") if final else ("selection",)
    return {"records":rows,**{name+"_macro_nmae":float(np.mean([r[name+"_nmae"] for r in rows])) for name in names}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--readout",choices=("coefficient","query"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--samples",type=int,nargs="+",default=list(range(8)))
    p.add_argument("--steps",type=int,default=16000)
    p.add_argument("--probes",type=int,default=1024)
    p.add_argument("--eval-every",type=int,default=500)
    p.add_argument("--chunk",type=int,default=2048)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--min-lr",type=float,default=1e-5)
    p.add_argument("--query-radial-hidden",type=int,default=25)
    p.add_argument("--parent",type=Path,help="Optional validation-selected checkpoint for a separately recorded fine-tuning run")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--final-eval",action="store_true")
    p.add_argument("--cpu",action="store_true")
    args=p.parse_args()
    if min(args.steps,args.probes,args.eval_every,args.chunk)<1 or len(set(args.samples))!=len(args.samples):
        p.error("Invalid settings")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model_config=dict(load_config("configs/nmc_full_l2_to_l4_mace_pbc.yaml")["model"],num_radial=16)
    model=PairedReadoutModel(model_config,args.readout,args.query_radial_hidden).to(device)
    parent_state=None
    if args.parent:
        parent_state=torch.load(args.parent,map_location=device,weights_only=False)
        previous=parent_state["settings"]
        if previous["model_config"]!=model_config or previous["readout"]!=args.readout:
            raise ValueError("Parent architecture differs")
        if previous["query_radial_hidden"]!=args.query_radial_hidden:
            raise ValueError("Parent query width differs")
        model.load_state_dict(parent_state["model"])
    trunk_initial=state_digest(model.atom_model.state_dict(),True)
    source_files=[Path(__file__),Path("src/density_eqgnn/models/paired_readout.py"),
        Path("src/density_eqgnn/models/mace_like_model.py"),Path("scripts/nmc_probe_holdout.py"),
        Path("scripts/train_nmc_probe_holdout.py"),Path("scripts/fit_nmc_holdout_oracle.py")]
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ("resume","final_eval")}
    settings.update(model_config=model_config,basis_config=asdict(BASIS),
        parameter_count=sum(p.numel() for p in model.parameters()),trunk_initial_sha256=trunk_initial,
        trunk_parameters=sum(p.numel() for n,p in model.atom_model.named_parameters() if n.startswith(("embedding.","interactions.","edge_basis."))),
        source_hashes={str(path.resolve()):sha256(path) for path in source_files},
        loss="training-only normalized L1",optimizer="Adam; cosine lr; no weight decay; clip 5",
        selection="fixed 8192 validation points per structure; tests sealed until separate final evaluation",
        query_cutoff=3.,query_features="final atom backbone state only",
        initialization="validation-selected parent; optimizer reset; no oracle labels" if args.parent else
                       "from scratch; identical atom backbone seed; no oracle labels",
        parent_sha256=sha256(args.parent) if args.parent else None,
        torch_version=torch.__version__,device=str(device))
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/"settings.json").exists():
        if not (args.resume or args.final_eval) or json.loads((args.output/"settings.json").read_text())!=settings:
            raise ValueError("Existing output requires identical settings/source and resume/final-eval")
    else:
        record(args.output/"settings.json",settings)
    print("SETTINGS "+json.dumps(settings),flush=True)
    samples=prepare(args.samples,Path("data/nmc_cache"),DEFAULT_ROOT,model.atom_model,"r16",device)
    record(args.output/"sources.json",[{"sample_id":s["id"],**s["metadata"],"split":s["split_metadata"],
        "training_normalizer":float(s["normalizer"])} for s in samples])
    if args.final_eval:
        if not (args.output/"fit_complete.json").exists(): raise ValueError("Train first")
        state=torch.load(args.output/"best.pt",map_location=device,weights_only=False)
        model.load_state_dict(state["model"])
        result=evaluate(model,samples,args.chunk,final=True)
        result.update(best_step=state["step"],checkpoint_sha256=sha256(args.output/"best.pt"),
                      sampling_audit=json.loads((args.output/"sampling_audit.json").read_text()))
        record(args.output/"result.json",result)
        print("RESULT "+json.dumps(result),flush=True)
        return
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.steps,eta_min=args.min_lr)
    rng=torch.Generator(device=device).manual_seed(args.seed+10000)
    order_rng=np.random.default_rng(args.seed+20000)
    seen=[torch.zeros(len(s["target"]),dtype=torch.bool,device=device) for s in samples]
    if parent_state is not None:
        if parent_state["settings"]["samples"]!=args.samples: raise ValueError("Parent samples differ")
        seen=[x.to(device) for x in parent_state["seen"]]
    order,history=[],[]
    best=float("inf")
    best_step,start_step=0,1
    started=time.perf_counter()
    def checkpoint(step):
        return {"model":model.state_dict(),"step":step,"settings":settings,
                "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                "rng":rng.get_state(),"order_rng":order_rng.bit_generator.state,"order":order,
                "seen":[x.cpu() for x in seen],"history":history,"best":best,"best_step":best_step,
                **basis_checkpoint_metadata(BASIS)}
    if args.resume:
        state=torch.load(args.output/"last.pt",map_location=device,weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        rng.set_state(state["rng"].cpu())
        order_rng.bit_generator.state=state["order_rng"]
        order,history,best,best_step=state["order"],state["history"],state["best"],state["best_step"]
        seen=[x.to(device) for x in state["seen"]]
        start_step=state["step"]+1
    for step in range(start_step,args.steps+1):
        model.train()
        if not order: order=order_rng.permutation(len(samples)).tolist()
        si=order.pop()
        sample=samples[si]
        train=sample["split"]["train"]
        ids=train[torch.randint(len(train),(args.probes,),device=device,generator=rng)]
        seen[si][ids]=True
        optimizer.zero_grad(set_to_none=True)
        residual=(predict(model,sample,ids).double()-sample["target"][ids])/sample["normalizer"]
        loss=residual.abs().mean()
        if not torch.isfinite(loss): raise RuntimeError("Nonfinite training loss")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step==1 or step%100==0:
            print(f"TRAIN step={step} loss={float(loss.detach()):.7f} grad={float(grad):.5g} seconds={time.perf_counter()-started:.1f}",flush=True)
        if step==1 or step%args.eval_every==0 or step==args.steps:
            row={"step":step,"elapsed_seconds":time.perf_counter()-started,**evaluate(model,samples,args.chunk)}
            history.append(row)
            if row["selection_macro_nmae"]<best:
                best,best_step=row["selection_macro_nmae"],step
                torch.save(checkpoint(step),args.output/"best.pt")
            torch.save(checkpoint(step),args.output/"last.pt")
            record(args.output/"history.json",history)
            print("EVAL "+json.dumps({k:v for k,v in row.items() if k!="records"}),flush=True)
    audit=[]
    for sample,visited in zip(samples,seen):
        row={"sample_id":sample["id"],"unique_training_points":int(visited.sum()),
             "validation_points_used_for_gradients":int(visited[sample["split"]["validation"]].sum()),
             "test_points_used_for_gradients":int(visited[sample["split"]["test"]].sum())}
        assert row["validation_points_used_for_gradients"]==row["test_points_used_for_gradients"]==0
        audit.append(row)
    record(args.output/"sampling_audit.json",audit)
    record(args.output/"fit_complete.json",{"best_step":best_step,"selection_macro_nmae":best,
        "elapsed_seconds":time.perf_counter()-started,"test_evaluated":False})
    print("FIT_COMPLETE test sealed",flush=True)


if __name__=="__main__":
    main()
