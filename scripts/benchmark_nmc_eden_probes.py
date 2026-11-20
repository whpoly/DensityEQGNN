"""Time full-grid and uniformly sampled joint EdenGNN gradients on training structures."""
import argparse
import copy
import hashlib
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.functional import l1_loss

from scripts import nmc_eden_fullgrid as run
from scripts.nmc_probe_holdout import PeriodicQueryGraph


def probe_density(model, data, ids, cell):
    """Evaluate the published snapped-grid stencil at the requested flat grid IDs."""
    shape = data.grid_shape.cpu()
    ijk = torch.stack(torch.unravel_index(ids, tuple(shape.tolist())), dim=1)
    grid = cell / shape[:, None]
    centres = data.pos_n.cpu()
    radius = model.probe.cutoff
    graph = PeriodicQueryGraph(centres.double() @ grid, ijk.double() @ grid, cell, radius + 1e-8)
    atoms, receivers, shifts = graph.select(torch.arange(len(ids)))
    offsets = ijk[receivers] - centres[atoms] - shifts.long() * shape
    vectors = offsets.double() @ grid
    inside = vectors.norm(dim=-1) <= radius
    device = data.pos.device
    atoms, receivers = atoms[inside].to(device), receivers[inside].to(device)
    vectors = vectors[inside].to(device=device, dtype=data.pos.dtype)
    residual = data.pos - data.pos_n.to(data.pos.dtype) @ data.grid
    layer = model.probe
    edges = (vectors - residual[atoms])[:, layer.coord_change][:, None, :]
    layer.conv_to_output_node(data)
    h = layer.reshape_irreps(data["node_charge"])
    z = layer.z_net(data["node_attrs"])
    values = layer.compute_chunk(edges, edges.norm(dim=-1), z[atoms, None], h[atoms]).flatten()
    delta = values.new_zeros(len(ids)).index_add(0, receivers, values)
    return delta + data.grid_func_in.flatten()[ids.to(device)]


def encode_structure(model, base):
    data = copy.copy(base)
    model.representation(data)
    return data


def run_loss(model, features, cell, ids, method):
    # The density layer adds fields in-place. Both heads consume the same encoder output.
    data = copy.copy(features)
    model.aug(data)
    occupation = data["node_aug"]
    if method == "sampled":
        density = probe_density(model, data, ids, cell)
    else:
        out = {}
        model._cal_density(data, out)
        density = out["grid_func_out"].flatten()
        if method == "full_at_probes":
            density = density[ids.to(density.device)]
    target = data.grid_func_out.flatten()
    if method != "full":
        target = target[ids.to(target.device)]
    paw_mask = data.aug_mask
    loss = (0.9 * l1_loss(density, target) +
            0.1 * l1_loss(occupation.flatten()[paw_mask], data.aug_tensor.flatten()[paw_mask]))
    return loss, density, occupation.flatten()[paw_mask]


def gradient_report(model, base, cell, ids):
    grads = {}
    features = encode_structure(model, base)
    for method in ("full_at_probes", "sampled"):
        model.zero_grad(set_to_none=True)
        loss, density, paw = run_loss(model, features, cell, ids, method)
        loss.backward(retain_graph=method == "full_at_probes")
        current = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.grad is not None}
        if method == "full_at_probes":
            grads = current
            ref_density, ref_paw = density.detach().cpu(), paw.detach().cpu()
        else:
            if current.keys() != grads.keys():
                raise ValueError("The sampled loss changes which parameters receive gradients")
            norms = {}
            for group in ("representation", "probe", "aug"):
                keys = [k for k in grads if k.startswith(group + ".")]
                norm = sum(float(grads[k].double().square().sum()) for k in keys) ** 0.5
                diff = sum(float((current[k].double() - grads[k]).square().sum()) for k in keys) ** 0.5
                if not norm > 0:
                    raise ValueError(f"Zero reference gradient in {group}")
                norms[group] = dict(reference_norm=norm, relative_l2=diff / norm)
            result = dict(density_max_abs=float((density.detach().cpu() - ref_density).abs().max()),
                          occupation_max_abs=float((paw.detach().cpu() - ref_paw).abs().max()),
                          parameter_gradients=norms)
        del loss, density, paw, current
    model.zero_grad(set_to_none=True)
    if (not np.isfinite([result["density_max_abs"], result["occupation_max_abs"]]).all()
            or result["density_max_abs"] > 1e-5 or result["occupation_max_abs"] > 1e-6
            or not all(v["relative_l2"] < 1e-3 for v in norms.values())):
        raise ValueError(f"Sparse evaluation differs from the dense model at the same probes: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("At least two repeats are needed to alternate timing order")
    folder = run.ROOT / "reconstructed_sacd_seed42"
    settings = run.read(folder / "settings.json")
    dll = run.enable_upstream()
    from edengnn.model.model import EfficientDensity

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("medium")
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.16)
    raw = (folder / "best.pt").read_bytes()
    state = torch.load(io.BytesIO(raw), weights_only=False, map_location="cpu")
    if state["lock"] != settings:
        raise ValueError("Benchmark weights and run settings differ")
    chunk = settings["model"]["chunk_size_train"]
    cfg = json.loads(json.dumps(settings["model"]), object_hook=lambda d: SimpleNamespace(**d))
    model = EfficientDensity(cfg).to(args.device).train()
    model.load_state_dict(state["model"])
    step, digest = state["step"], hashlib.sha256(raw).hexdigest()
    del state, raw
    examples = run.dataset(folder / "train.txt", run.NMCReader(Path(settings["sacd_directory"]), reconstructed=True))
    sync = torch.cuda.synchronize if args.device == "cuda" else lambda: None
    rows = []
    for pos in (0, len(examples) // 2, len(examples) - 1):
        data = examples[pos].to(args.device)
        cell = torch.load(run.DATA / str(data.name) / "sample.pt", map_location="cpu", weights_only=False)["cell"].double()
        seed = np.random.SeedSequence([run.SEED, int(data.name)])
        rng = np.random.default_rng(seed)
        size = data.grid_func_out.numel()
        ids = torch.from_numpy(rng.choice(size, 1000, replace=False))
        torch.set_float32_matmul_precision("highest")
        agreement = gradient_report(model, data, cell, ids)
        torch.set_float32_matmul_precision("medium")
        times = {"full": [], "sampled": []}
        for rep in range(args.repeats + 1):
            order = ("full", "sampled") if (pos + rep) % 2 == 0 else ("sampled", "full")
            for method in order:
                sync()
                start = time.perf_counter()
                model.zero_grad(set_to_none=True)
                if method == "sampled":
                    ids = torch.from_numpy(rng.choice(size, 1000, replace=False))
                features = encode_structure(model, data)
                loss, density, paw = run_loss(model, features, cell, ids, method)
                loss.backward()
                sync()
                elapsed = time.perf_counter() - start
                if rep:
                    times[method].append(elapsed)
                del loss, density, paw, features
        row = dict(id=int(data.name), atoms=int(data.nat), grid_points=size, valid_paw=int(data.aug_mask.sum()),
                   native_atom_grid_pairs=int(data.npb_total), agreement=agreement, seconds=times)
        rows.append(row)
        print("Checked structure", row["id"], {k: float(np.median(v)) for k, v in times.items()}, flush=True)
    medians = {k: float(np.median([v for row in rows for v in row["seconds"][k]])) for k in ("full","sampled")}
    output = Path("reports/nmc_eden_fullgrid") / f"probe_timing_{args.device}_step{step}.json"
    run.save(output, dict(device=args.device, device_name=torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
                          split="train", seed=run.SEED, probes=1000, repeats=args.repeats, chunk=chunk,
                          checkpoint_step=step, checkpoint_sha256=digest,
                          gradient_audit="same weights, encoder activations and probes, highest FP32 matmul precision",
                          timing_scope="joint forward and backward, medium matmul precision, inputs in memory, including probe selection and periodic neighbour lookup; no optimizer, file IO or validation",
                          median_seconds=medians, speedup=medians["full"]/medians["sampled"], structures=rows))
    print(output, medians, "speedup", medians["full"]/medians["sampled"], flush=True)
    if dll is not None:
        dll.close()


if __name__ == "__main__":
    main()
