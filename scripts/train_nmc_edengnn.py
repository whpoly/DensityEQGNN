"""From-scratch official EdenGNN on the existing eight-structure spatial split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.nmc_edengnn import ROOT, EdenDensity, prepare_samples, evaluate, write_json
from scripts.nmc_probe_holdout import sha256
from scripts.train_local_nmc_overfit import state_digest


def audit(model,sample):
    """Compare values and all parameter gradients against unchanged native forward."""
    ids=sample["split"]["train"][::991][:64]
    _,_,offset=sample["stencil"].select(ids)
    chosen=offset.unique(sorted=True)
    reduced=dict(sample)
    native=sample["native_data"].clone()
    native["map_probe"]=native["map_probe"][chosen]
    native["edge_vec_probes"]=native["edge_vec_probes"][chosen]
    native["npb"]=len(chosen)
    native["npb_total"]=len(chosen)*len(sample["numbers"])
    reduced["native_data"]=native
    model.eval()
    model.zero_grad(set_to_none=True)
    sparse=model.sampled(sample,ids)
    weight=torch.linspace(.2,1.2,len(ids),device=ids.device)
    (sparse*weight).mean().backward()
    gradients={name:p.grad.detach().clone() for name,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    reference=model.native(reduced)[ids]
    (reference*weight).mean().backward()
    numerator=denominator=0.
    max_absolute=0.
    for name,p in model.named_parameters():
        if name not in gradients:
            if p.grad is not None:
                raise RuntimeError("Gradient graph differs")
            continue
        difference=(p.grad-gradients[name]).double()
        numerator+=float(difference.square().sum())
        denominator+=float(gradients[name].double().square().sum())
        max_absolute=max(max_absolute,float(difference.abs().max()))
    relative=(numerator/max(denominator,1e-30))**.5
    error=float((reference-sparse).detach().abs().max())
    if relative>2e-4 or not torch.allclose(reference,sparse,rtol=2e-5,atol=3e-5):
        raise RuntimeError(f"Official adapter mismatch values={error} gradient={relative}")
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        native_full=model.native(sample)
        full_difference=float((native_full[ids]-sparse.detach()).abs().max())
        if not torch.allclose(native_full[ids],sparse,rtol=2e-5,atol=3e-5):
            raise RuntimeError("Full native stencil mismatch")
    return {"sample_id":sample["id"],"partition":"train only", "points":len(ids),
            "native_value_max_abs":error,"native_full_value_max_abs":full_difference,
            "all_parameter_gradient_relative_l2":relative,"gradient_max_abs":max_absolute,
            "upstream_class_source_unmodified":True,"native_stencil_preserved":True,
            "note":"Reduced stencil includes every native contribution to selected points; full native forward separately checked."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"eden_eight")
    parser.add_argument("--steps",type=int,default=16000)
    parser.add_argument("--eval-every",type=int,default=500)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--audit-only",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(42)
    np.random.seed(42)
    device=torch.device("cuda")
    started=time.perf_counter()
    model=EdenDensity().to(device)
    settings={"samples":list(range(8)),"steps":args.steps,"probes":1024,"seed":42,
              "eval_every":args.eval_every,"lr":.001,"min_lr":.00001,
              "optimizer":"Adam; cosine decay; zero weight decay; grad clip 5",
              "loss":"training-only normalized density L1", "model":model.config,
              "provenance":model.provenance,"parameter_count":sum(p.numel() for p in model.parameters()),
              "initial_state_sha256":state_digest(model.state_dict()),
              "selection":"original fixed 8192 validation points; no test decisions",
              "baseline":"POTCAR-derived SAD computed without density targets; not verified against native VASP zero-step output",
              "torch_version":torch.__version__,"gpu":torch.cuda.get_device_name(),
              "source_sha256":{str(p):sha256(p) for p in (Path(__file__),Path("scripts/nmc_edengnn.py"))}}
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/"settings.json"
    if path.exists():
        if json.loads(path.read_text())!=settings or not args.resume:
            raise ValueError("Use a new run or identical --resume")
    else:
        write_json(path,settings)
    print("SETTINGS "+json.dumps(settings),flush=True)
    samples=prepare_samples([0] if args.audit_only else range(8),device)
    if not (args.output/"adapter_audit.json").exists():
        result=audit(model,samples[0])
        write_json(args.output/"adapter_audit.json",result)
        print("AUDIT "+json.dumps(result),flush=True)
    if args.audit_only:
        return
    write_json(args.output/"sources.json",[{"sample_id":s["id"],**s["metadata"],"split":s["split_metadata"],
                "training_normalizer":float(s["normalizer"])} for s in samples])
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.steps,eta_min=.00001)
    rng=torch.Generator(device=device).manual_seed(10042)
    order_rng=np.random.default_rng(20042)
    order,history=[],[]
    seen=[torch.zeros(len(s["target"]),dtype=torch.bool,device=device) for s in samples]
    best,best_step,start_step=float("inf"),0,1
    def state(step):
        return {"model":model.state_dict(),"step":step,"best":best,"best_step":best_step,"history":history,
                "optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"rng":rng.get_state(),
                "order_rng":order_rng.bit_generator.state,"order":order,"seen":[x.cpu() for x in seen],"settings":settings}
    if args.resume and (args.output/"last.pt").exists():
        saved=torch.load(args.output/"last.pt",map_location=device,weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        rng.set_state(saved["rng"].cpu())
        order_rng.bit_generator.state=saved["order_rng"]
        order,history=saved["order"],saved["history"]
        best,best_step=saved["best"],saved["best_step"]
        seen=[v.to(device) for v in saved["seen"]]
        start_step=saved["step"]+1
    train_times=[]
    for step in range(start_step,args.steps+1):
        model.train()
        if not order:
            order=order_rng.permutation(len(samples)).tolist()
        si=order.pop()
        sample=samples[si]
        allowed=sample["split"]["train"]
        ids=allowed[torch.randint(len(allowed),(1024,),device=device,generator=rng)]
        seen[si][ids]=True
        tick=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        pred=model.sampled(sample,ids)
        loss=((pred.double()-sample["target"][ids])/sample["normalizer"]).abs().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        train_times.append(time.perf_counter()-tick)
        if step==1 or step%100==0:
            status={"phase":"training","step":step,"total_steps":args.steps,"loss":float(loss),
                    "grad":float(grad),"recent_step_seconds":float(np.median(train_times[-100:])),
                    "best_validation_nmae":best,"elapsed_seconds_this_process":time.perf_counter()-started,
                    "updated_utc":datetime.now(timezone.utc).isoformat()}
            write_json(args.output/"status.json",status)
            print("TRAIN "+json.dumps(status),flush=True)
        if step==1 or step%args.eval_every==0 or step==args.steps:
            metrics=evaluate(model,samples)
            row={"step":step,"elapsed_seconds":time.perf_counter()-started,**metrics}
            history.append(row)
            if metrics["selection_macro_nmae"]<best:
                best,best_step=metrics["selection_macro_nmae"],step
                torch.save(state(step),args.output/"best.pt")
            torch.save(state(step),args.output/"last.pt")
            write_json(args.output/"history.json",history)
            print("EVAL "+json.dumps({k:v for k,v in row.items() if k!="records"}),flush=True)
    audit_rows=[]
    for sample,visited in zip(samples,seen):
        val=int(visited[sample["split"]["validation"]].sum())
        test=int(visited[sample["split"]["test"]].sum())
        if val or test:
            raise RuntimeError("Holdout leakage")
        audit_rows.append({"sample_id":sample["id"],"unique_training_points":int(visited.sum()),
                           "validation_points_used_for_gradients":val,"test_points_used_for_gradients":test})
    previous=json.loads(Path("runs/nmc_probe_holdout_2026-10-26/r16_eight/sampling_audit.json").read_text())
    if args.steps==16000 and audit_rows!=previous:
        raise RuntimeError("Matched sampling coverage differs from R16")
    write_json(args.output/"sampling_audit.json",audit_rows)
    frozen={"best_step":best_step,"best_validation_nmae":best,"checkpoint_sha256":sha256(args.output/"best.pt"),
            "frozen_before_test_utc":datetime.now(timezone.utc).isoformat(),"training_complete":True}
    write_json(args.output/"selection_frozen.json",frozen)
    saved=torch.load(args.output/"best.pt",map_location=device,weights_only=False)
    model.load_state_dict(saved["model"])
    result=evaluate(model,samples,final=True)
    result.update(frozen)
    write_json(args.output/"result.json",result)
    write_json(args.output/"status.json",{"phase":"complete",**frozen})
    print("RESULT "+json.dumps(result),flush=True)


if __name__=="__main__":
    main()
