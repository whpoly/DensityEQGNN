"""Paired local-NMC full-grid projection, with an optional sampled-grid mode.

The sampled mode keeps a grid-point holdout; --full-grid fits all grid points.
This is a representation diagnostic, not a new train/test benchmark. No GNN is
trained. Legacy operators below are isolated controls for the old checkpoint.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile
import time
import types

import lz4.frame
import numpy as np
from pymatgen.io.vasp.outputs import Chgcar
import torch
from e3nn import o3

from density_eqgnn.chgcar import vasp_grid
from density_eqgnn.models import build_compact_coefficient_model
from density_eqgnn.oracle_basis import (
    FullGridDesign, OracleBasis, RadialSpec, periodic_bond_centers, radial_transform, raw_to_whitened,
)
from density_eqgnn.periodic import PERIODIC_GEOMETRY_VERSION


def load_sample(path, device):
    with tempfile.TemporaryDirectory(prefix="local_nmc_oracle_") as tmp:
        decoded = Path(tmp) / "CHGCAR"
        decoded.write_bytes(lz4.frame.decompress(path.read_bytes()))
        chgcar = Chgcar.from_file(decoded)
    structure = chgcar.structure
    raw = np.asarray(chgcar.data["total"], dtype=np.float64)
    return {
        "positions": torch.tensor(structure.cart_coords, device=device, dtype=torch.float64),
        "numbers": torch.tensor(structure.atomic_numbers, device=device),
        "cell": torch.tensor(structure.lattice.matrix, device=device, dtype=torch.float64),
        "probes": torch.tensor(vasp_grid(structure.lattice.matrix, raw.shape), device=device),
        "target": torch.tensor(raw.reshape(-1)/structure.volume, device=device),
        "metadata": {"source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                     "formula": structure.formula, "atoms": len(structure),
                     "grid_shape": list(raw.shape), "volume": structure.volume},
    }


def legacy_geometry(self, positions, atom_mask, cell):
    """Historical fractional-rounding operator, ONLY for checkpoint attribution."""
    d = positions[:, None] - positions[:, :, None]
    fractional = torch.einsum("bijd,bdk->bijk", d, torch.linalg.inv(cell))
    d = torch.einsum("bijd,bdk->bijk", fractional-fractional.round(), cell)
    distances = d.norm(dim=-1)
    valid = atom_mask[:, :, None] & atom_mask[:, None, :]
    valid &= ~torch.eye(positions.shape[1], device=positions.device, dtype=torch.bool)[None]
    valid &= distances < self.config.cutoff
    source = torch.arange(positions.shape[1],device=positions.device)[None,None].expand_as(valid)
    return distances, d/distances[...,None].clamp_min(1e-12), valid, source


@torch.no_grad()
def checkpoint_coefficients(sample, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_compact_coefficient_model(state["model_config"]).double().to(sample["cell"].device).eval()
    model.load_state_dict(state["model"])
    positions = sample["positions"][None]
    mask = torch.ones(positions.shape[:2],device=positions.device,dtype=torch.bool)
    args = (sample["numbers"][None],positions,mask,sample["cell"][None])
    corrected = model(*args)["coefficients"][0]
    model._geometry = types.MethodType(legacy_geometry,model)
    historical = model(*args)["coefficients"][0]
    return historical.detach(), corrected.detach()


@torch.no_grad()
def legacy_density(coefficients, positions, probes, cell):
    d = probes[:,None]-positions[None]
    frac = d @ torch.linalg.inv(cell)
    d = (frac-frac.round()) @ cell
    r = d.norm(dim=-1)
    s = torch.as_tensor(np.geomspace(3/32,3,12),device=r.device,dtype=r.dtype)
    radial = torch.exp(-.5*(r[...,None]/s)**2)
    radial *= (.5*(1+torch.cos(torch.pi*(r/3).clamp(0,1))))[...,None]
    radial *= (r<3)[...,None]
    directions = d/r[...,None].clamp_min(1e-12)
    angular = o3.spherical_harmonics(list(range(5)),directions,normalize=True,normalization="integral")
    return torch.einsum("anm,pan,pam->p",coefficients.reshape(-1,12,25),radial,angular)


def cgls(a, y, iterations, tolerance=1e-7):
    """Least-squares initialization without forming normal equations."""
    x = torch.zeros(a.shape[1],device=a.device,dtype=a.dtype)
    r = y.clone()
    s = a.T @ r
    p = s.clone()
    gamma = torch.dot(s.double(),s.double())
    initial = gamma.sqrt()
    history = []
    for step in range(1, iterations+1):
        q = a @ p
        alpha = gamma / torch.dot(q.double(),q.double()).clamp_min(1e-30)
        x += alpha.to(x.dtype)*p
        r -= alpha.to(r.dtype)*q
        if step % 100 == 0:
            r = y-a@x
        s = a.T@r
        next_gamma = torch.dot(s.double(),s.double())
        if step % 100 == 0 or step == iterations:
            record = {"step":step,"fit_nmae":float(r.abs().sum()/y.abs().sum()),
                      "relative_normal_residual":float(next_gamma.sqrt()/initial)}
            history.append(record)
            print("cgls "+json.dumps(record),flush=True)
        if next_gamma.sqrt() < tolerance*initial:
            break
        p = s + (next_gamma/gamma).to(p.dtype)*p
        gamma = next_gamma
    return x, history


def fit(a, target, args):
    normalization = target.abs().mean()
    y = (target/normalization).float()
    if isinstance(a,FullGridDesign):
        column_scale = a.normalize_columns()
    else:
        column_scale = torch.linalg.vector_norm(a,dim=0).clamp_min(1e-12)
        a /= column_scale[None]
    factor = None
    if args.gram_precondition:
        started = time.perf_counter()
        factor = a.gram_cholesky(args.gram_ridge)
        rhs = a.precise_transpose(target/normalization)
        solution = torch.cholesky_solve(rhs[:,None],factor).squeeze(1)
        residual = a.precise_matvec(solution)-target/normalization
        initialization = [{"method":"full_grid_double_gram_cholesky",
                           "fit_nmae":float(residual.abs().mean()),
                           "ridge":args.gram_ridge,"seconds":time.perf_counter()-started}]
        print("initialization "+json.dumps(initialization),flush=True)
        x = factor.T@solution
    else:
        x, initialization = cgls(a,y,args.cg_steps)
    x = torch.nn.Parameter(x)
    precise = args.precise_steps < 0
    def physical_parameter():
        if factor is None:
            return x
        return torch.linalg.solve_triangular(factor.T,x[:,None],upper=True).squeeze(1)
    def prediction():
        if factor is not None and precise:
            return a.precise_matvec(physical_parameter())
        return a@physical_parameter().float()
    history = []
    total_closures = 0
    phases = [(epsilon,args.round_steps,args.rounds,precise) for epsilon in (1e-3,1e-4,1e-5)]
    if args.precise_steps > 0:
        phases.append((1e-5,args.precise_steps,1,True))
    for epsilon,max_steps,rounds,precise in phases:
        optimizer = torch.optim.LBFGS(
            [x],lr=1,max_iter=max_steps,max_eval=max_steps*2,
            tolerance_grad=1e-10,tolerance_change=1e-12,history_size=30,
            line_search_fn="strong_wolfe",
        )
        for round_index in range(rounds):
            calls = 0
            def closure():
                nonlocal calls,total_closures
                optimizer.zero_grad(set_to_none=True)
                residual = prediction().double()-target/normalization
                loss = torch.sqrt(residual.square()+epsilon**2).mean()
                loss.backward()
                calls += 1
                total_closures += 1
                return loss
            optimizer.step(closure)
            with torch.no_grad():
                residual = prediction().double()-target/normalization
                record = {"epsilon_relative_density":epsilon,"round":round_index+1,
                          "arithmetic":"float64" if precise and factor is not None else "float32",
                          "closures":calls,"fit_nmae":float(residual.abs().mean()),
                          "gradient_inf":float(x.grad.abs().max())}
                history.append(record)
                print("lbfgs "+json.dumps(record),flush=True)
    coefficients = (physical_parameter().detach().double()/column_scale.double())*normalization
    dual = None
    if args.dual_check and factor is not None:
        with torch.no_grad():
            residual = a.precise_matvec(physical_parameter().detach())-target/normalization
            u = residual/torch.sqrt(residual.square()+1e-10)
            projection_residuals = []
            for _ in range(12):
                constraint = a.precise_transpose(u)
                projection_residuals.append(float(constraint.abs().max()))
                if projection_residuals[-1] < 1e-10:
                    break
                correction = torch.cholesky_solve(constraint[:,None],factor).squeeze(1)
                u -= a.precise_matvec(correction)
            u /= u.abs().max().clamp_min(1)*(1+1e-12)
            value = float(-(u*target/normalization).mean())
            primal = float(residual.abs().mean())
            dual = {"numerical_dual_value":value,"primal_nmae":primal,
                    "numerical_primal_dual_gap":primal-value,
                    "max_abs_adjoint_constraint":float(a.precise_transpose(u).abs().max()),
                    "max_abs_dual":float(u.abs().max()),"projection_residuals":projection_residuals,
                    "status":"numerical LP dual diagnostic for stored operator; not an exact-arithmetic certificate"}
            print("dual "+json.dumps(dual),flush=True)
    return coefficients, {"cgls":initialization,"lbfgs":history,"closures":total_closures,
                          "dual_diagnostic":dual,
                          "column_scale_min":float(column_scale.min()),
                          "column_scale_max":float(column_scale.max())}


@torch.no_grad()
def evaluate(basis, coefficients, sample, fit_indices, chunk, historical=None, corrected=None):
    target = sample["target"]
    predictions = {"oracle":torch.empty_like(target)}
    if corrected is not None:
        for name in ("gnn_old_graph_old_decoder","gnn_old_graph_corrected_decoder","gnn_corrected_graph_corrected_decoder"):
            predictions[name] = torch.empty_like(target)
        spec = basis.blocks[0][1]
        historical_white = raw_to_whitened(historical,spec)
        corrected_white = raw_to_whitened(corrected,spec)
    for start in range(0,len(target),chunk):
        stop = min(start+chunk,len(target))
        probes = sample["probes"][start:stop]
        design = basis.design(probes,dtype=torch.float64)
        predictions["oracle"][start:stop] = design @ coefficients
        if corrected is not None:
            predictions["gnn_old_graph_old_decoder"][start:stop] = legacy_density(historical,sample["positions"],probes,sample["cell"])
            predictions["gnn_old_graph_corrected_decoder"][start:stop] = design @ historical_white
            predictions["gnn_corrected_graph_corrected_decoder"][start:stop] = design @ corrected_white
        if start % (chunk*16) == 0:
            print(f"full-grid {start}/{len(target)}",flush=True)
    holdout = torch.ones(len(target),device=target.device,dtype=torch.bool)
    holdout[fit_indices] = False
    result = {}
    for name,prediction in predictions.items():
        error = (prediction-target).abs()
        result[name] = {
            "full_grid_nmae":float(error.sum()/target.abs().sum()),
            "fit_grid_nmae":float(error[fit_indices].sum()/target[fit_indices].abs().sum()),
            "holdout_grid_nmae":float(error[holdout].sum()/target[holdout].abs().sum()) if holdout.any() else None,
            "relative_signed_charge_error":float((prediction.sum()-target.sum())/target.sum()),
            "negative_density_integral_fraction":float((-prediction).clamp_min(0).sum()/target.sum()),
        }
        if "nearest_atom" in sample:
            boundaries = (0.,.1,.25,.5,1.,2.,float("inf"))
            bins = []
            for lower,upper in zip(boundaries[:-1],boundaries[1:]):
                selected = (sample["nearest_atom"]>=lower)&(sample["nearest_atom"]<upper)
                if upper == float("inf"):
                    selected |= torch.isinf(sample["nearest_atom"])
                bins.append({"min_radius":lower,"max_radius":upper if np.isfinite(upper) else None,
                             "points":int(selected.sum()),
                             "density_mass_fraction":float(target[selected].abs().sum()/target.abs().sum()),
                             "absolute_error_fraction":float(error[selected].sum()/error.sum()),
                             "contribution_to_nmae":float(error[selected].sum()/target.abs().sum())})
            result[name]["nearest_atom_bins"] = bins
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache",type=Path,default=Path("data/nmc_cache"))
    parser.add_argument("--checkpoint",type=Path,default=Path("runs/nmc_full_l2_to_l4_mace/best.pt"))
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--samples",type=int,nargs="+",default=list(range(8)))
    parser.add_argument("--variants",nargs="+",default=["current","regular","regular_bonds","regular_capacity"])
    parser.add_argument("--fit-probes",type=int,default=131072)
    parser.add_argument("--full-grid",action="store_true")
    parser.add_argument("--gram-precondition",action="store_true")
    parser.add_argument("--gram-ridge",type=float,default=1e-9)
    parser.add_argument("--precise-steps",type=int,default=-1,
                        help="-1: all-float64 optimization; positive: float32 stages followed by this many float64 polishing steps")
    parser.add_argument("--dual-check",action="store_true")
    parser.add_argument("--chunk",type=int,default=2048)
    parser.add_argument("--cg-steps",type=int,default=600)
    parser.add_argument("--round-steps",type=int,default=100)
    parser.add_argument("--rounds",type=int,default=2)
    parser.add_argument("--seed",type=int,default=20261024)
    args = parser.parse_args()
    if args.gram_precondition and not args.full_grid:
        parser.error("--gram-precondition requires --full-grid")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    settings.update(periodic_geometry_version=PERIODIC_GEOMETRY_VERSION,device=str(device),
                    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                    basis_features="float64 construction/evaluation, float32 storage, float64 matvec/adjoint with Gram preconditioning; no radial rank truncation",
                    objective="full-grid smooth-L1 continuation" if args.full_grid else "uniform sampled-grid smooth-L1 continuation",
                    uncertainty="feasible oracle upper bounds, not certified projection minima; grid holdout is not a structure test split")
    settings_path = args.output_dir/"settings.json"
    if settings_path.exists():
        previous = json.loads(settings_path.read_text())
        for key in ("fit_probes","full_grid","cg_steps","round_steps","rounds","seed","checkpoint","checkpoint_sha256","periodic_geometry_version","cache","gram_precondition","gram_ridge","precise_steps","dual_check"):
            if previous.get(key) != settings[key]:
                raise ValueError(f"Existing experiment differs in {key}; use a new output directory")
    settings_path.write_text(json.dumps(settings,indent=2))
    for sample_id in args.samples:
        sample = load_sample(args.cache/f"{sample_id}.CHGCAR.lz4",device)
        centers = periodic_bond_centers(sample["positions"],sample["numbers"],sample["cell"])
        count = len(sample["target"]) if args.full_grid else min(args.fit_probes,len(sample["target"])-1)
        indices = np.sort(np.random.default_rng(args.seed+sample_id).permutation(len(sample["target"]))[:count])
        fit_indices = torch.tensor(indices,device=device)
        historical,corrected = checkpoint_coefficients(sample,args.checkpoint)
        torch.cuda.empty_cache()
        for variant in args.variants:
            destination = args.output_dir/f"sample_{sample_id:04d}_{variant}.json"
            if destination.exists():
                previous_record = json.loads(destination.read_text())
                if previous_record["sha256"] != sample["metadata"]["sha256"]:
                    raise ValueError(f"Sample {sample_id} data changed; use a new output directory")
                print(f"skip completed {destination}",flush=True)
                continue
            if variant not in ("current","regular","regular_bonds","regular_capacity","regular_r16"):
                raise ValueError(variant)
            nradial = 12
            if variant == "regular_capacity":
                nradial = round((len(sample["positions"])*300+len(centers)*36)/(len(sample["positions"])*25))
            elif variant == "regular_r16":
                nradial = 16
            spec = RadialSpec(num_radial=nradial,regular=variant!="current")
            blocks = [(sample["positions"],spec)]
            if variant == "regular_bonds":
                blocks.append((centers,RadialSpec(num_radial=4,lmax=2,cutoff=2.,minimum=.15,regular=True)))
            basis = OracleBasis(blocks,sample["cell"])
            print(f"sample={sample_id} variant={variant} columns={basis.size} fit_points={count}",flush=True)
            started = time.perf_counter()
            if args.full_grid:
                matrix = FullGridDesign(basis,sample["probes"])
            else:
                matrix = torch.empty(count,basis.size,device=device,dtype=torch.float32)
                for start in range(0,count,args.chunk):
                    stop = min(start+args.chunk,count)
                    matrix[start:stop] = basis.design(sample["probes"][fit_indices[start:stop]])
            print(f"design complete in {time.perf_counter()-started:.2f}s",flush=True)
            coeff,history = fit(matrix,sample["target"][fit_indices],args)
            if args.full_grid:
                sample["nearest_atom"] = matrix.nearest_atom
            del matrix
            torch.cuda.empty_cache()
            np.savez_compressed(
                destination.with_suffix(".npz"),coefficients=coeff.cpu().numpy(),fit_indices=indices,
                bond_centers=centers.cpu().numpy(),cell=sample["cell"].cpu().numpy(),
            )
            metrics = evaluate(basis,coeff,sample,fit_indices,args.chunk,
                               historical if variant=="current" else None,
                               corrected if variant=="current" else None)
            record = {"sample_id":sample_id,"variant":variant,**sample["metadata"],
                      "fit_probes":count,"holdout_probes":len(sample["target"])-count,
                      "num_bond_centers":len(centers),"num_coefficients":basis.size,
                      "basis_blocks":[{"centers":len(c),**asdict(s)} for c,s in blocks],
                      "radial_condition_numbers":[[float(radial_transform(s,l)[2][0]/radial_transform(s,l)[2][-1])
                                                    for l in range(s.lmax+1)] for _,s in blocks],
                      "metrics":metrics,"optimization":history,"seconds":time.perf_counter()-started}
            destination.write_text(json.dumps(record,indent=2),encoding="utf-8")
            brief_metrics = {name:{key:value for key,value in values.items() if key!="nearest_atom_bins"}
                             for name,values in metrics.items()}
            print("RESULT "+json.dumps({"sample":sample_id,"variant":variant,"metrics":brief_metrics,"seconds":record["seconds"]}),flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
