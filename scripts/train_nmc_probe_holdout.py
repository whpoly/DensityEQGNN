"""Matched from-scratch DeepDFT/R16 experiments with sealed spatial test points."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import types

import numpy as np
import torch

from scripts.nmc_probe_holdout import (DEFAULT_ROOT, PeriodicQueryGraph, atom_batch,
                                     compact_prediction, official_model, probe_batch,
                                     save_split, sha256)
from scripts.compare_local_nmc_oracles import load_sample
from scripts.train_local_nmc_overfit import state_digest
from density_eqgnn.compact_basis import CompactBasisConfig, basis_checkpoint_metadata
from density_eqgnn.models import build_compact_coefficient_model
from density_eqgnn.periodic import PERIODIC_GEOMETRY_VERSION
from density_eqgnn.training.config import load_config


def prepare(sample_ids, cache, root, model, name, device):
    samples = []
    for sample_id in sample_ids:
        started = time.perf_counter()
        sample = load_sample(cache / f"{sample_id}.CHGCAR.lz4", device)
        sample["id"] = sample_id
        split, metadata = save_split(root / "spatial_splits", len(sample["target"]), sample_id,
                                     sample["metadata"]["sha256"])
        sample["split"] = {k: torch.tensor(v, device=device) for k, v in split.items()}
        sample["split_metadata"] = metadata
        # ONLY training labels determine the objective scale.
        sample["normalizer"] = sample["target"][sample["split"]["train"]].abs().mean()
        sample["graph"] = PeriodicQueryGraph(sample["positions"], sample["probes"], sample["cell"],
                                             4. if name == "deepdft" else 3.)
        if name == "deepdft":
            sample["batch"] = atom_batch(sample)
        else:
            sample["model_positions"] = sample["positions"].float()[None]
            sample["model_cell"] = sample["cell"].float()[None]
            sample["mask"] = torch.ones(1, len(sample["numbers"]), dtype=torch.bool, device=device)
            with torch.no_grad():
                sample["geometry"] = model._geometry(sample["model_positions"], sample["mask"], sample["model_cell"])
        samples.append(sample)
        print(f"PREPARED sample={sample_id} edges={len(sample['graph'].atoms)} seconds={time.perf_counter()-started:.1f}", flush=True)
    if name != "deepdft":
        original = model._geometry
        cache_geometry = {s["model_positions"].data_ptr(): s["geometry"] for s in samples}
        def geometry(self, positions, mask, cell):
            return cache_geometry.get(positions.data_ptr()) if positions.data_ptr() in cache_geometry else original(positions, mask, cell)
        model._geometry = types.MethodType(geometry, model)
    return samples


def representation(model, sample, name):
    if name == "deepdft":
        return model.atom_model(sample["batch"])
    return model(sample["numbers"][None], sample["model_positions"], sample["mask"],
                 sample["model_cell"])["coefficients"]


def predict(model, sample, indices, name, basis, rep=None):
    if rep is None:
        rep = representation(model, sample, name)
    if name == "deepdft":
        return model.probe_model(probe_batch(sample, indices), *rep)[0]
    return compact_prediction(sample, rep, indices, basis)


@torch.no_grad()
def evaluate(model, samples, name, basis, chunk, *, final=False):
    model.eval()
    rows = []
    for sample in samples:
        rep = representation(model, sample, name)
        if final:
            indices = torch.arange(len(sample["target"]), device=sample["target"].device)
            labels = torch.zeros(len(indices), dtype=torch.long, device=indices.device)
            labels[sample["split"]["validation"]] = 1
            labels[sample["split"]["test"]] = 2
            accum = torch.zeros(3, 4, device=indices.device, dtype=torch.float64)
        else:
            indices = sample["split"]["selection"]
            accum = torch.zeros(1, 4, device=indices.device, dtype=torch.float64)
        for ids in indices.split(chunk):
            output = predict(model, sample, ids, name, basis, rep).double()
            target = sample["target"][ids]
            values = torch.stack(((output-target).abs(), target.abs(), output-target,
                                  (-output).clamp_min(0)), 1)
            if not torch.isfinite(values).all():
                raise RuntimeError("Nonfinite evaluation")
            accum.index_add_(0, labels[ids] if final else torch.zeros_like(ids), values)
        row = {"sample_id": sample["id"], "formula": sample["metadata"]["formula"]}
        for key, sums in zip(("train", "validation", "test") if final else ("selection",), accum.cpu().numpy()):
            row[key + "_nmae"] = float(sums[0] / sums[1])
            if final:
                row[key + "_charge_error"] = float(sums[2] / sums[1])
                row[key + "_negative_fraction"] = float(sums[3] / sums[1])
        if final:
            total = accum.sum(0)
            row["full_grid_nmae"] = float(total[0]/total[1])
        rows.append(row)
    keys = ("train", "validation", "test", "full_grid") if final else ("selection",)
    return {"records": rows, **{key + "_macro_nmae": float(np.mean([r[key+"_nmae"] for r in rows])) for key in keys}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=("deepdft", "r16"), required=True)
    p.add_argument("--samples", type=int, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--cache", type=Path, default=Path("data/nmc_cache"))
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--probes", type=int, default=1024)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--loss", choices=("l1", "mse"), default="l1")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--defer-final", action="store_true")
    args = p.parse_args()
    if min(args.steps, args.probes, args.eval_every, args.chunk) < 1 or len(set(args.samples)) != len(args.samples):
        p.error("Invalid training settings")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    basis = None
    if args.model == "deepdft":
        model, model_config = official_model(args.root / "DeepDFT", device)
    else:
        model_config = dict(load_config("configs/nmc_full_l2_to_l4_mace_pbc.yaml")["model"], num_radial=16)
        model = build_compact_coefficient_model(model_config).to(device)
        basis = CompactBasisConfig(num_radial=16, lmax=4, cutoff=3., pbc=True,
                                   regular_radial=True, radial_whitening=True, radial_minimum=3/32)
    settings = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items() if k not in ("resume", "defer_final")}
    settings.update(model_config=model_config, basis_config=asdict(basis) if basis else None,
                    initial_state_sha256=state_digest(model.state_dict()),
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    initialization="from scratch; no pretrained weights, no oracle labels",
                    objective="training-only normalized density loss; uniform training-partition samples",
                    selection="fixed 8192-point subset of validation partition; never test or full-grid",
                    optimizer="Adam; cosine LR; no weight decay; gradient norm clip 5",
                    note="Official architecture is unmodified; local paired training protocol differs from published full-data runner",
                    torch_version=torch.__version__, device=str(device),
                    source_hashes={str(path): sha256(path) for path in (Path(__file__), Path("scripts/nmc_probe_holdout.py"))})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    settings_file = args.output_dir / "settings.json"
    if settings_file.exists():
        if not args.resume or json.loads(settings_file.read_text()) != settings:
            raise ValueError("Existing run requires identical settings and --resume")
    else:
        settings_file.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print("SETTINGS " + json.dumps(settings), flush=True)
    samples = prepare(args.samples, args.cache, args.root, model, args.model, device)
    (args.output_dir / "sources.json").write_text(json.dumps([
        {"sample_id":s["id"], **s["metadata"], "split":s["split_metadata"],
         "training_normalizer":float(s["normalizer"])} for s in samples], indent=2), encoding="utf-8")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.min_lr)
    rng = torch.Generator(device=device).manual_seed(args.seed + 10000)
    order_rng = np.random.default_rng(args.seed + 20000)
    order, history = [], []
    best, best_step, start_step = float("inf"), 0, 1
    seen = [torch.zeros(len(s["target"]), dtype=torch.bool, device=device) for s in samples]
    def checkpoint(step):
        extra = basis_checkpoint_metadata(basis) if basis else {}
        return {"model":model.state_dict(), "model_config":model_config, **extra,
                "periodic_geometry_version":PERIODIC_GEOMETRY_VERSION,
                "step":step, "best":best, "best_step":best_step, "history":history,
                "optimizer":optimizer.state_dict(), "scheduler":scheduler.state_dict(),
                "rng":rng.get_state(), "order_rng":order_rng.bit_generator.state, "order":order,
                "seen":[x.cpu() for x in seen], "settings":settings}
    if args.resume:
        state = torch.load(args.output_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, best_step, history, order = state["best"], state["best_step"], state["history"], state["order"]
        rng.set_state(state["rng"].cpu())
        order_rng.bit_generator.state = state["order_rng"]
        seen = [x.to(device) for x in state["seen"]]
        start_step = state["step"] + 1
    for step in range(start_step, args.steps + 1):
        model.train()
        if not order:
            order = order_rng.permutation(len(samples)).tolist()
        si = order.pop()
        sample = samples[si]
        allowed = sample["split"]["train"]
        ids = allowed[torch.randint(len(allowed), (args.probes,), device=device, generator=rng)]
        seen[si][ids] = True
        optimizer.zero_grad(set_to_none=True)
        output = predict(model, sample, ids, args.model, basis)
        residual = (output.double() - sample["target"][ids]) / sample["normalizer"]
        loss = residual.abs().mean() if args.loss == "l1" else residual.square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 100 == 0:
            print(f"TRAIN step={step} loss={float(loss):.7f} grad={float(grad):.5g} seconds={time.perf_counter()-started:.1f}", flush=True)
        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            metrics = evaluate(model, samples, args.model, basis, args.chunk)
            row = {"step":step, "elapsed_seconds":time.perf_counter()-started,
                   "learning_rate":optimizer.param_groups[0]["lr"], **metrics}
            history.append(row)
            if metrics["selection_macro_nmae"] < best:
                best, best_step = metrics["selection_macro_nmae"], step
                torch.save(checkpoint(step), args.output_dir / "best.pt")
            torch.save(checkpoint(step), args.output_dir / "last.pt")
            (args.output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print("EVAL " + json.dumps({k:v for k,v in row.items() if k != "records"}), flush=True)
    audit = []
    for sample, visited in zip(samples, seen):
        validation_seen = int(visited[sample["split"]["validation"]].sum())
        test_seen = int(visited[sample["split"]["test"]].sum())
        if validation_seen or test_seen:
            raise RuntimeError("Spatial holdout leakage detected")
        audit.append({"sample_id":sample["id"], "unique_training_points":int(visited.sum()),
                      "validation_points_used_for_gradients":validation_seen,
                      "test_points_used_for_gradients":test_seen})
    (args.output_dir / "sampling_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if args.defer_final:
        print("TRAINING_COMPLETE test remains sealed", flush=True)
        return
    state = torch.load(args.output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    result = evaluate(model, samples, args.model, basis, args.chunk, final=True)
    result.update(model_name=args.model, best_step=best_step, seed=args.seed,
                  parameter_count=settings["parameter_count"], sampling_audit=audit,
                  elapsed_seconds=time.perf_counter()-started,
                  interpretation="unseen spatial grid points in seen structures; not unseen-structure generalization",
                  checkpoint_sha256=sha256(args.output_dir / "best.pt"))
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("RESULT " + json.dumps({k:v for k,v in result.items() if k not in ("records", "sampling_audit")}), flush=True)


if __name__ == "__main__":
    main()
