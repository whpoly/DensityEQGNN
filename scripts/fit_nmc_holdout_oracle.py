"""Fit free R16 coefficients using the existing TRAIN partition only.

Validation selects a coefficient checkpoint. Test evaluation is a separate
command, after fitting is finished; no earlier full-grid oracle is imported.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from density_eqgnn.compact_basis import CompactBasisConfig, basis_checkpoint_metadata
from density_eqgnn.oracle_basis import FullGridDesign, OracleBasis
from scripts.compare_local_nmc_oracles import load_sample
from scripts.nmc_probe_holdout import DEFAULT_ROOT, PeriodicQueryGraph, compact_prediction, save_split, sha256

ROOT = Path("runs/nmc_bottleneck_2026-10-27")
BASIS = CompactBasisConfig(num_radial=16,lmax=4,cutoff=3.,pbc=True,
                          regular_radial=True,radial_whitening=True,radial_minimum=3/32)


def degree_to_radial(coefficients, atoms, basis=BASIS):
    shaped = coefficients.reshape(atoms,-1)
    blocks = [shaped[:,basis.num_radial*l*l:basis.num_radial*(l+1)**2]
              .reshape(atoms,basis.num_radial,2*l+1) for l in range(basis.lmax+1)]
    return torch.cat(blocks,-1).flatten(1)[None]


def record(path, value):
    path.write_text(json.dumps(value,indent=2),encoding="utf-8")


@torch.no_grad()
def independent_metrics(sample, coefficients, split, names, chunk):
    sample["graph"] = PeriodicQueryGraph(sample["positions"],sample["probes"],sample["cell"],3.)
    radial = degree_to_radial(coefficients,len(sample["positions"]))
    metrics = {}
    for name in names:
        ids = torch.as_tensor(split[name],device=coefficients.device)
        sums = torch.zeros(2,dtype=torch.float64,device=coefficients.device)
        for selected in ids.split(chunk):
            pred = compact_prediction(sample,radial,selected,BASIS)
            target = sample["target"][selected]
            sums += torch.stack(((pred-target).abs().sum(),target.abs().sum()))
        metrics[name+"_nmae"] = float(sums[0]/sums[1])
    return metrics


def fit_one(sample_id,args):
    out = args.output/f"sample_{sample_id:04d}"
    out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample=load_sample(args.cache/f"{sample_id}.CHGCAR.lz4",device)
    split,metadata=save_split(DEFAULT_ROOT/"spatial_splits",len(sample["target"]),sample_id,sample["metadata"]["sha256"])
    settings={"sample_id":sample_id,"source":sample["metadata"],"split":metadata,
              "basis":asdict(BASIS),"steps_per_phase":args.steps_per_phase,"eval_every":args.eval_every,
              "gram_ridge":args.gram_ridge,"phases":[1e-3,1e-4,1e-5],
              "objective":"training-only normalized smooth L1; Gram ridge only preconditions coordinates",
              "selection":"minimum NMAE on original fixed 8192 validation points",
              "source_hashes":{str(Path(__file__).resolve()):sha256(Path(__file__)),
                               "src/density_eqgnn/oracle_basis.py":sha256("src/density_eqgnn/oracle_basis.py")}}
    if args.final_eval:
        if not (out/"fit_complete.json").exists():
            raise ValueError("Fitting must finish before test evaluation")
        if json.loads((out/"settings.json").read_text())!=settings:
            raise ValueError("Settings/source changed")
        state=torch.load(out/"best.pt",map_location=device,weights_only=False)
        metrics=independent_metrics(sample,state["coefficients"],split,("train","validation","test"),args.chunk)
        result={"sample_id":sample_id,"best_step":state["step"],**metrics,
                "checkpoint_sha256":sha256(out/"best.pt"),"split":metadata,
                "fit_points":len(split["train"]),"validation_points_used_for_fit":0,"test_points_used_for_fit":0}
        record(out/"result.json",result)
        print("RESULT "+json.dumps(result),flush=True)
        return
    if (out/"fit_complete.json").exists():
        if json.loads((out/"settings.json").read_text())!=settings:
            raise ValueError("Completed run differs")
        print(f"SKIP completed sample={sample_id}",flush=True)
        return
    if (out/"settings.json").exists() and json.loads((out/"settings.json").read_text())!=settings:
        raise ValueError("Existing settings/source differ")
    record(out/"settings.json",settings)
    train=torch.as_tensor(split["train"],device=device)
    validation=torch.as_tensor(split["selection"],device=device)
    target=sample["target"][train]
    normalizer=target.abs().mean()
    y=target/normalizer
    basis=OracleBasis([(sample["positions"],BASIS.radial_spec)],sample["cell"])
    design=FullGridDesign(basis,sample["probes"][train])
    scales=design.normalize_columns().double()
    print(f"DESIGN sample={sample_id} rows={len(train)} columns={basis.size} seconds={time.perf_counter()-started:.1f}",flush=True)
    factor=design.gram_cholesky(args.gram_ridge)
    rhs=design.precise_transpose(y)
    initial=torch.cholesky_solve(rhs[:,None],factor).squeeze(1)
    parameter=torch.nn.Parameter(factor.T@initial)
    validation_design=FullGridDesign(basis,sample["probes"][validation])
    validation_target=sample["target"][validation]
    history=[]
    best=float("inf")
    step=0
    def physical():
        return torch.linalg.solve_triangular(factor.T,parameter[:,None],upper=True).squeeze(1)
    def check(epsilon,closures=0):
        nonlocal best
        with torch.no_grad():
            coeff=physical().detach()/scales*normalizer
            val=validation_design.precise_matvec(coeff)
            score=float((val-validation_target).abs().sum()/validation_target.abs().sum())
            residual=design.precise_matvec(physical())-y
            row={"step":step,"epsilon":epsilon,"train_nmae":float(residual.abs().mean()),
                 "selection_nmae":score,"coefficient_rms":float(coeff.square().mean().sqrt()),
                 "closures":closures,"elapsed_seconds":time.perf_counter()-started}
            history.append(row)
            if score<best:
                best=score
                torch.save({"coefficients":coeff.cpu(),"step":step,"selection_nmae":score,
                            "layout":"atom,degree,radial,m",**basis_checkpoint_metadata(BASIS),
                            "settings":settings},out/"best.pt")
            record(out/"history.json",history)
            print("EVAL "+json.dumps({"sample_id":sample_id,**row}),flush=True)
    check(None)
    for epsilon in settings["phases"]:
        optimizer=torch.optim.LBFGS([parameter],lr=1.,max_iter=args.eval_every,max_eval=2*args.eval_every,
            tolerance_grad=1e-10,tolerance_change=1e-12,history_size=30,line_search_fn="strong_wolfe")
        for _ in range(args.steps_per_phase//args.eval_every):
            calls=0
            def closure():
                nonlocal calls
                optimizer.zero_grad(set_to_none=True)
                residual=design.precise_matvec(physical())-y
                loss=torch.sqrt(residual.square()+epsilon**2).mean()
                loss.backward()
                calls+=1
                return loss
            optimizer.step(closure)
            step+=args.eval_every
            check(epsilon,calls)
    # Free the fit matrix before the independent production decoder audit.
    del design,validation_design,factor,optimizer,parameter
    gc.collect()
    if device.type=="cuda": torch.cuda.empty_cache()
    state=torch.load(out/"best.pt",map_location=device,weights_only=False)
    metrics=independent_metrics(sample,state["coefficients"],split,("selection","validation"),args.chunk)
    difference=abs(metrics["selection_nmae"]-state["selection_nmae"])
    if difference>2e-5:
        raise RuntimeError(f"Cached/production NMAE mismatch {difference}")
    record(out/"fit_complete.json",{"sample_id":sample_id,"best_step":state["step"],**metrics,
        "cache_vs_production_selection_nmae_difference":difference,
        "checkpoint_sha256":sha256(out/"best.pt"),"fit_points":len(train),
        "validation_points_used_for_fit":0,"test_points_used_for_fit":0,
        "test_evaluated":False,"elapsed_seconds":time.perf_counter()-started})
    print(f"FIT_COMPLETE sample={sample_id} validation_nmae={metrics['validation_nmae']:.8f} test sealed",flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=ROOT/"oracle")
    p.add_argument("--cache",type=Path,default=Path("data/nmc_cache"))
    p.add_argument("--samples",type=int,nargs="+",default=[1,0,2,3,4,5,6,7])
    p.add_argument("--steps-per-phase",type=int,default=75)
    p.add_argument("--eval-every",type=int,default=25)
    p.add_argument("--gram-ridge",type=float,default=1e-9)
    p.add_argument("--chunk",type=int,default=2048)
    p.add_argument("--final-eval",action="store_true")
    args=p.parse_args()
    if args.steps_per_phase<1 or args.eval_every<1 or args.steps_per_phase%args.eval_every:
        p.error("Positive phase steps must be a multiple of eval-every")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    for sample_id in args.samples:
        fit_one(sample_id,args)
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()


if __name__=="__main__":
    main()
