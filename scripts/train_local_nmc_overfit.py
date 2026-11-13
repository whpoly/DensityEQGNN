"""From-scratch paired overfit experiments on the eight cached NMC structures.

Train on fresh uniform samples of the entire DFT grid. Checkpoint selection and
reported metrics use all grid points. No oracle coefficients supervise the GNN.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import types

import numpy as np
import torch

from density_eqgnn.compact_basis import CompactBasisConfig, check_basis_checkpoint, reconstruct_compact_density
from density_eqgnn.fixed_grid import FixedGridDensity
from density_eqgnn.models import build_compact_coefficient_model
from density_eqgnn.periodic import PERIODIC_GEOMETRY_VERSION, check_periodic_checkpoint
from density_eqgnn.radial_basis import radial_transform
from density_eqgnn.training.config import load_config
from scripts.compare_local_nmc_oracles import load_sample


def state_digest(state, trunk_only=False):
    digest = hashlib.sha256()
    for key,value in sorted(state.items()):
        if trunk_only and not key.startswith(("embedding.","interactions.","edge_basis.")):
            continue
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def oracle_record(sample_id,variant):
    folder = Path("runs/nmc_oracle_pbc_sweep")
    name = "current" if variant == "r12" else "regular_capacity"
    if variant == "r16" and sample_id in (4,6):
        folder = Path("runs/nmc_oracle_pbc_fixed_r16")
        name = "regular_r16"
    path = folder/f"sample_{sample_id:04d}_{name}.json"
    return path,json.loads(path.read_text())


def predict(model,sample):
    return model(sample["numbers"][None],sample["model_positions"],sample["atom_mask"],
                 sample["model_cell"])["coefficients"]


@torch.no_grad()
def evaluate(model,samples,basis,chunk,*,independent=False):
    model.eval()
    records = []
    for sample in samples:
        coefficients = predict(model,sample).double()
        error = reference = charge = negative = 0.
        max_cache_difference = 0.
        radial_error = np.zeros(6)
        boundaries = (0.,.1,.25,.5,1.,2.,float("inf"))
        for start in range(0,len(sample["target"]),chunk):
            stop = min(start+chunk,len(sample["target"]))
            ids = torch.arange(start,stop,device=sample["target"].device)
            prediction = sample["decoder"](coefficients,ids)
            if independent:
                direct = reconstruct_compact_density(
                    coefficients,sample["positions"][None],sample["probes"][ids][None],
                    sample["atom_mask"],torch.ones(1,len(ids),dtype=torch.bool,device=ids.device),
                    sample["cell"][None],basis,
                )[0]
                max_cache_difference = max(max_cache_difference,float((prediction-direct).abs().max()))
                prediction = direct
            target = sample["target"][ids]
            error += float((prediction-target).abs().sum())
            reference += float(target.abs().sum())
            charge += float((prediction-target).sum())
            negative += float((-prediction).clamp_min(0).sum())
            radii = sample["decoder"].nearest_atom[ids]
            for j,(low,high) in enumerate(zip(boundaries[:-1],boundaries[1:])):
                selected = (radii>=low)&(radii<high)
                if high == float("inf"):
                    selected |= torch.isinf(radii)
                radial_error[j] += float((prediction[selected]-target[selected]).abs().sum())
        oracle = sample["oracle"]["metrics"]["oracle"]["full_grid_nmae"]
        records.append({"sample_id":sample["id"],"formula":sample["metadata"]["formula"],
                        "full_grid_nmae":error/reference,"oracle_nmae":oracle,
                        "excess_nmae":error/reference-oracle,"ratio_to_oracle":error/reference/oracle,
                        "relative_signed_charge_error":charge/reference,
                        "negative_density_integral_fraction":negative/reference,
                        "coefficient_rms":float(coefficients.square().mean().sqrt()),
                        "radial_error_contributions":(radial_error/reference).tolist(),
                        "max_cached_vs_direct_difference":max_cache_difference if independent else None})
    scores = np.array([r["full_grid_nmae"] for r in records])
    oracle_mean = float(np.mean([r["oracle_nmae"] for r in records]))
    return {"macro_nmae":float(scores.mean()),"p90_nmae":float(np.quantile(scores,.9)),
            "oracle_macro_nmae":oracle_mean,"macro_excess_nmae":float(scores.mean()-oracle_mean),
            "ratio_to_oracle_macro":float(scores.mean()/oracle_mean),"records":records}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",choices=("r12","r16"),required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--cache",type=Path,default=Path("data/nmc_cache"))
    parser.add_argument("--samples",type=int,nargs="+",default=list(range(8)))
    parser.add_argument("--steps",type=int,default=4000)
    parser.add_argument("--probes",type=int,default=8192)
    parser.add_argument("--eval-every",type=int,default=500)
    parser.add_argument("--chunk",type=int,default=16384)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--lr",type=float,default=1e-3)
    parser.add_argument("--min-lr",type=float,default=1e-5)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--independent-final",action="store_true")
    args = parser.parse_args()
    if any(i not in range(8) for i in args.samples) or len(set(args.samples)) != len(args.samples):
        parser.error("This diagnostic only supports distinct locally cached samples 0..7")
    if args.steps < 1 or args.probes < 1 or args.eval_every < 1:
        parser.error("steps, probes and eval-every must be positive")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = load_config("configs/nmc_full_l2_to_l4_mace_pbc.yaml")
    nradial = 12 if args.variant == "r12" else 16
    model_config = dict(base["model"],num_radial=nradial)
    basis = CompactBasisConfig(num_radial=nradial,lmax=4,cutoff=3.,pbc=True,
                               regular_radial=args.variant=="r16",radial_whitening=True,
                               radial_minimum=3/32)
    model = build_compact_coefficient_model(model_config).to(device)
    initialization = {"trunk_sha256":state_digest(model.state_dict(),True),
                      "full_state_sha256":state_digest(model.state_dict()),
                      "parameter_count":sum(p.numel() for p in model.parameters()),
                      "source":"random initialization; no pretrained weights or oracle coefficient targets"}
    transforms = [radial_transform(basis.radial_spec,l)[0] for l in range(basis.lmax+1)]
    whitening_hash = hashlib.sha256(np.stack(transforms).tobytes()).hexdigest()
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!="resume"}
    settings.update(model_config=model_config,basis_config=asdict(basis),initialization=initialization,
                    periodic_geometry_version=PERIODIC_GEOMETRY_VERSION,whitening_sha256=whitening_hash,
                    objective="macro relative L1; fresh uniform full-grid probe samples; exact full-grid normalizers",
                    checkpoint_selection="minimum full-grid training macro NMAE; not a validation/generalization score",
                    optimizer="Adam, zero weight decay, gradient norm clipping at 5; cosine learning rate",
                    device=str(device),torch_version=torch.__version__)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    settings_path = args.output_dir/"settings.json"
    if settings_path.exists():
        previous = json.loads(settings_path.read_text())
        if previous != settings:
            raise ValueError("Experiment settings differ; use a separate output directory")
        if not args.resume:
            raise ValueError("Existing experiment requires --resume; refusing to overwrite")
    settings_path.write_text(json.dumps(settings,indent=2),encoding="utf-8")
    samples = []
    started = time.perf_counter()
    for sample_id in args.samples:
        sample = load_sample(args.cache/f"{sample_id}.CHGCAR.lz4",device)
        sample["id"] = sample_id
        sample["model_positions"] = sample["positions"].float()[None]
        sample["model_cell"] = sample["cell"].float()[None]
        sample["atom_mask"] = torch.ones(1,len(sample["positions"]),device=device,dtype=torch.bool)
        oracle_path,oracle = oracle_record(sample_id,args.variant)
        if oracle["sha256"] != sample["metadata"]["sha256"]:
            raise ValueError(f"Sample {sample_id} differs from the oracle source")
        sample["oracle"] = oracle
        sample["oracle_path"] = str(oracle_path)
        sample["normalizer"] = sample["target"].abs().mean()
        sample["decoder"] = FixedGridDensity(sample["positions"],sample["probes"],sample["cell"],basis)
        with torch.no_grad():
            sample["geometry"] = model._geometry(sample["model_positions"],sample["atom_mask"],sample["model_cell"])
        samples.append(sample)
        print(f"prepared sample={sample_id} edges={len(sample['decoder'].atoms)} elapsed={time.perf_counter()-started:.1f}s",flush=True)
    # Only fixed inputs are cached. The graph, backbone, readout and learnable
    # atomic baseline are otherwise identical to the production model.
    original_geometry = model._geometry
    geometries = {s["model_positions"].data_ptr():s["geometry"] for s in samples}
    def geometry(self,positions,mask,cell):
        cached = geometries.get(positions.data_ptr())
        return cached if cached is not None else original_geometry(positions,mask,cell)
    model._geometry = types.MethodType(geometry,model)
    sources = [{"sample_id":s["id"],**s["metadata"],"oracle_file":s["oracle_path"]} for s in samples]
    (args.output_dir/"sources.json").write_text(json.dumps(sources,indent=2),encoding="utf-8")
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.steps,eta_min=args.min_lr)
    rng = torch.Generator(device=device).manual_seed(args.seed+10000)
    order_rng = np.random.default_rng(args.seed+20000)
    order = []
    history = []
    best = float("inf")
    best_step = 0
    start_step = 1
    def checkpoint(step):
        return {"periodic_geometry_version":PERIODIC_GEOMETRY_VERSION,"model_config":model_config,
                "basis_config":asdict(basis),"whitening_sha256":whitening_hash,
                "radial_whitening_matrices":torch.from_numpy(np.stack(transforms)),
                "coefficient_scale":torch.ones(basis.num_radial,basis.angular_channels),
                "model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),
                "step":step,"best_step":best_step,"best":best,"history":history,
                "probe_rng":rng.get_state(),"order_rng":order_rng.bit_generator.state,"order":order,
                "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if device.type=="cuda" else None}
    if args.resume:
        state = torch.load(args.output_dir/"last.pt",map_location=device,weights_only=False)
        check_periodic_checkpoint(state,pbc=True,resume=True)
        check_basis_checkpoint(state,basis)
        if state["whitening_sha256"] != whitening_hash:
            raise ValueError("Radial whitening coordinates changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = state["step"]+1
        history,best,best_step = state["history"],state["best"],state["best_step"]
        rng.set_state(state["probe_rng"].cpu())
        order_rng.bit_generator.state = state["order_rng"]
        order = state["order"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([r.cpu() for r in state["cuda_rng"]])
    else:
        metrics = evaluate(model,samples,basis,args.chunk)
        history.append({"step":0,**metrics})
        best = metrics["macro_nmae"]
        torch.save(checkpoint(0),args.output_dir/"best.pt")
        torch.save(checkpoint(0),args.output_dir/"last.pt")
        print("EVAL "+json.dumps({"step":0,"macro_nmae":best}),flush=True)
    losses = []
    training_start = time.perf_counter()
    for step in range(start_step,args.steps+1):
        model.train()
        if not order:
            order = order_rng.permutation(len(samples)).tolist()
        sample = samples[order.pop()]
        ids = torch.randint(len(sample["target"]),(args.probes,),device=device,generator=rng)
        optimizer.zero_grad(set_to_none=True)
        coefficients = predict(model,sample)
        prediction = sample["decoder"](coefficients,ids)
        loss = (prediction.double()-sample["target"][ids]).abs().mean()/sample["normalizer"]
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise RuntimeError(f"Nonfinite loss or gradient at step {step}")
        optimizer.step()
        scheduler.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0 or step == start_step:
            print(f"TRAIN step={step} sampled_nmae={np.mean(losses):.7f} lr={scheduler.get_last_lr()[0]:.3g} elapsed={time.perf_counter()-training_start:.1f}s",flush=True)
            losses = []
        if step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model,samples,basis,args.chunk)
            history.append({"step":step,"learning_rate":scheduler.get_last_lr()[0],**metrics})
            improved = metrics["macro_nmae"] < best
            if improved:
                best,best_step = metrics["macro_nmae"],step
            state = checkpoint(step)
            if improved:
                torch.save(state,args.output_dir/"best.pt")
            torch.save(state,args.output_dir/"last.pt")
            (args.output_dir/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
            print("EVAL "+json.dumps({"step":step,"macro_nmae":metrics["macro_nmae"],"ratio_to_oracle":metrics["ratio_to_oracle_macro"],"best_step":best_step}),flush=True)
    state = torch.load(args.output_dir/"best.pt",map_location=device,weights_only=False)
    model.load_state_dict(state["model"])
    # Recompute model geometry in the final audit, independently of the cache.
    model._geometry = original_geometry
    final = evaluate(model,samples,basis,args.chunk,independent=args.independent_final)
    final.update(variant=args.variant,best_step=best_step,completed_steps=args.steps,seed=args.seed,
                 interpretation="full-grid training-set overfit; no generalization claim",
                 initialization=initialization,basis_config=asdict(basis),model_config=model_config,
                 independent_final=args.independent_final,elapsed_seconds=time.perf_counter()-started)
    (args.output_dir/"result.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
    print("RESULT "+json.dumps({k:v for k,v in final.items() if k!="records"}),flush=True)


if __name__ == "__main__":
    main()
