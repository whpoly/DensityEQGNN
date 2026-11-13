"""Matched full-batch L-BFGS polishing of locally trained R12/R16 GNNs.

Each round freezes a fresh uniform grid sample per structure. All checkpoint
selection and final metrics use the complete grid. Architecture is unchanged.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import types

import numpy as np
import torch

from density_eqgnn.compact_basis import (
    CompactBasisConfig, check_basis_checkpoint, basis_checkpoint_metadata, radial_to_degree_major,
)
from density_eqgnn.fixed_grid import FixedGridDensity
from density_eqgnn.models import build_compact_coefficient_model
from density_eqgnn.oracle_basis import FullGridDesign, OracleBasis
from density_eqgnn.periodic import PERIODIC_GEOMETRY_VERSION, check_periodic_checkpoint
from scripts.train_local_nmc_overfit import load_sample, oracle_record, predict, evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--parent",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--iterations",type=int,default=40)
    parser.add_argument("--probes",type=int,default=16384)
    parser.add_argument("--epsilon",type=float,default=1e-5)
    parser.add_argument("--chunk",type=int,default=16384)
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parent=json.loads((args.parent/"settings.json").read_text())
    parent_result=json.loads((args.parent/"result.json").read_text())
    state=torch.load(args.parent/"best.pt",map_location=device,weights_only=False)
    check_periodic_checkpoint(state,pbc=True,resume=True)
    basis=CompactBasisConfig(**state["basis_config"])
    check_basis_checkpoint(state,basis)
    model=build_compact_coefficient_model(state["model_config"]).to(device)
    model.load_state_dict(state["model"])
    settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings.update(parent_checkpoint_sha256=hashlib.sha256((args.parent/"best.pt").read_bytes()).hexdigest(),
                    seed=parent["seed"],variant=parent["variant"],basis_config=asdict(basis),
                    objective="macro smooth relative L1; all structures per gradient; fresh uniform grid subset each round",
                    selection="minimum complete-grid training NMAE including parent; no oracle coefficient targets")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if (args.output_dir/"settings.json").exists():
        raise ValueError("Polishing output exists; use a new directory")
    (args.output_dir/"settings.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")
    started=time.perf_counter()
    samples=[]
    for i in parent["samples"]:
        s=load_sample(Path(parent["cache"])/f"{i}.CHGCAR.lz4",device)
        s["id"]=i
        s["model_positions"]=s["positions"].float()[None]
        s["model_cell"]=s["cell"].float()[None]
        s["atom_mask"]=torch.ones(1,len(s["positions"]),device=device,dtype=torch.bool)
        _,s["oracle"]=oracle_record(i,parent["variant"])
        if s["oracle"]["sha256"]!=s["metadata"]["sha256"]:
            raise ValueError("Data differ from oracle")
        s["normalizer"]=s["target"].abs().mean()
        s["decoder"]=FixedGridDensity(s["positions"],s["probes"],s["cell"],basis)
        with torch.no_grad():
            s["geometry"]=model._geometry(s["model_positions"],s["atom_mask"],s["model_cell"])
        samples.append(s)
    original_geometry=model._geometry
    geometries={s["model_positions"].data_ptr():s["geometry"] for s in samples}
    def geometry(self,positions,mask,cell):
        cached=geometries.get(positions.data_ptr())
        return cached if cached is not None else original_geometry(positions,mask,cell)
    model._geometry=types.MethodType(geometry,model)
    history=[{"round":0,**evaluate(model,samples,basis,args.chunk)}]
    best=history[0]["macro_nmae"]
    best_round=0
    def save(round_index,path):
        torch.save({"periodic_geometry_version":PERIODIC_GEOMETRY_VERSION,
                    "model":model.state_dict(),"model_config":state["model_config"],
                    **basis_checkpoint_metadata(basis),"coefficient_scale":torch.ones(basis.num_radial,basis.angular_channels),
                    "round":round_index,"history":history,"parent":str(args.parent)},path)
    save(0,args.output_dir/"best.pt")
    rng=torch.Generator(device=device).manual_seed(parent["seed"]+30000)
    for round_index in range(1,args.rounds+1):
        matrices=[]
        targets=[]
        for s in samples:
            ids=torch.randperm(len(s["target"]),device=device,generator=rng)[:args.probes]
            basis_operator=OracleBasis([(s["positions"],basis.radial_spec)],s["cell"])
            matrices.append(FullGridDesign(basis_operator,s["probes"][ids]))
            targets.append(s["target"][ids])
        print(f"prepared round={round_index} probes_per_structure={args.probes}",flush=True)
        optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=args.iterations,
                                    max_eval=args.iterations*2,history_size=20,line_search_fn="strong_wolfe",
                                    tolerance_grad=1e-9,tolerance_change=1e-11)
        calls=0
        model.train()
        def closure():
            nonlocal calls
            optimizer.zero_grad(set_to_none=True)
            total=torch.zeros((),device=device,dtype=torch.float64)
            for s,a,target in zip(samples,matrices,targets):
                c=radial_to_degree_major(predict(model,s),basis).flatten()
                residual=(a@c).double()-target
                residual=residual/s["normalizer"]
                loss=torch.sqrt(residual.square()+args.epsilon**2).mean()/len(samples)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite polishing loss")
                loss.backward()
                total+=loss.detach()
            calls+=1
            if calls==1 or calls%10==0:
                print(f"POLISH round={round_index} closure={calls} sampled_loss={float(total):.8f}",flush=True)
            return total
        optimizer.step(closure)
        del matrices,targets,optimizer
        torch.cuda.empty_cache()
        metrics=evaluate(model,samples,basis,args.chunk)
        history.append({"round":round_index,"closures":calls,**metrics})
        if metrics["macro_nmae"]<best:
            best=metrics["macro_nmae"]
            best_round=round_index
            save(round_index,args.output_dir/"best.pt")
        save(round_index,args.output_dir/"last.pt")
        (args.output_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        print("EVAL "+json.dumps({"round":round_index,"macro_nmae":metrics["macro_nmae"],"best_round":best_round}),flush=True)
    best_state=torch.load(args.output_dir/"best.pt",map_location=device,weights_only=False)
    model.load_state_dict(best_state["model"])
    model._geometry=original_geometry
    final=evaluate(model,samples,basis,args.chunk,independent=True)
    final.update(variant=parent["variant"],seed=parent["seed"],best_round=best_round,
                 parent_adam_macro_nmae=parent_result["macro_nmae"],initialization=parent["initialization"],
                 basis_config=asdict(basis),model_config=state["model_config"],
                 interpretation="training-set overfit after matched Adam + full-batch sampled L-BFGS",
                 elapsed_seconds=time.perf_counter()-started)
    (args.output_dir/"result.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
    print("RESULT "+json.dumps({k:v for k,v in final.items() if k!="records"}),flush=True)


if __name__=="__main__":
    main()
