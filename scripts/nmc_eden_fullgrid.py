"""Train and score the published EdenGNN NMC architecture on complete CHGCAR grids."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import torch
import yaml
from filelock import FileLock

from scripts.benchmark_public_density import metric, save
from scripts.joint_pair_data import ROOT as PREPARED, manifest
from scripts.nmc_probe_holdout import sha256
from scripts.run_joint_pair import pt_save, read
from scripts.train_deepdft_paw import measures


ROOT = Path("runs/nmc_eden_fullgrid")
UPSTREAM = ROOT / "EdenGNN"
REVISION = "f37c9a1864803bf04d75fa91c932474040220021"
PUBLIC = Path("data/public_density/eden")
DATA = PREPARED / "nmc" / "data"
REFERENCE = Path("runs/public_benchmark_2026-10-30/nmc/accuracy.json")
SEED = 42
CHECKPOINT_INTERVAL = 25


def enable_upstream():
    head = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True)
    if head.strip() != REVISION:
        raise ValueError(f"Expected EdenGNN {REVISION}, found {head.strip()}")
    subprocess.run(["git", "-C", str(UPSTREAM), "diff", "--quiet", "HEAD"], check=True)
    sys.path[:0] = [str((UPSTREAM / "src").resolve()), str((ROOT / "dependencies").resolve())]
    mingw = ROOT / "compiler/Library/mingw-w64/bin"
    if os.name == "nt" and mingw.is_dir():
        return os.add_dll_directory(str(mingw.resolve()))
    return None


def model_config(chunk):
    # Start from the released MD architecture and apply the two NMC overrides in SI §1.
    with ZipFile(PUBLIC / "model_weight.zip") as published:
        config = yaml.safe_load(published.read("model_weight/md/config.yaml"))["model"]
    config.update(task=2, use_sad=True, lmix_max=2, chunk_size_train=chunk, chunk_size_predict=chunk)
    config["probe"]["edge"]["cutoff"] = 3.0
    config["probe"]["conv"]["n_channels"] = 3
    return config


def build_model(chunk):
    from edengnn.model.model import EfficientDensity

    cfg = json.loads(json.dumps(model_config(chunk)), object_hook=lambda d: SimpleNamespace(**d))
    return EfficientDensity(cfg).cuda()


def potcar_files(species, table):
    # Only the five selected, hash-checked POTCARs are used for NMC.
    with ZipFile(PUBLIC / "data.zip") as published:
        records = json.loads(published.read(table))
    paths = {}
    with ZipFile("potpaw54.zip") as pots:
        for item in records:
            element, name = item["element"], item["pseudo_name"]
            if element not in species:
                continue
            path = ROOT / "potcar" / name / "POTCAR"
            if not path.exists():
                raw = pots.read(f"potpaw54/potpaw_PBE/PAW_GGA_PBE/{name}/POTCAR")
                if hashlib.sha256(raw).hexdigest() != item["potcar_hash"]:
                    raise ValueError(f"POTCAR hash mismatch for {element}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            if sha256(path) != item["potcar_hash"]:
                raise ValueError(f"Local POTCAR mismatch for {element}")
            paths[element] = {"path_pp": str(path.resolve())}
    if set(paths) != set(species):
        raise ValueError("Missing published POTCAR metadata")
    return paths


def filelist(folder, split, ids):
    path = folder / f"{split}.txt"
    content = "".join(f"{i}\n" for i in ids)
    if path.exists() and path.read_text() != content:
        raise ValueError(f"Existing split changed: {path}")
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    return path


class NMCReader:
    """Adapt audited NMC arrays to the author's DensityDataset.read_data contract."""

    lmix_max = 2

    def __init__(self, sad_root, reconstructed=False):
        from edengnn.data.io.vasp.basis import read_basis
        from edengnn.data.io.vasp.parse_density import aug2irreps
        from scripts.deepdft_paw import BLOCKS, ELEMENTS, blocks

        self.sad_root = sad_root
        self.reconstructed = reconstructed
        self.pp = potcar_files(ELEMENTS, "data/vasp/nmc/pseudo.json")
        self.basis = read_basis(self.pp)
        self.indices = {}
        compact = {block["key"]: block for block in BLOCKS}
        for z, spec in self.basis.items():
            # Native CHGCAR order -> cached compact order -> author's universal PAW layout.
            native = blocks(spec["angular_momenta"])
            raw_indices = np.concatenate([
                np.arange(compact[b["key"]]["start"], compact[b["key"]]["stop"]) for b in native
            ])
            indices, mask = aug2irreps(z, raw_indices, self.lmix_max, self.basis)
            self.indices[z] = indices.astype(int), mask.astype(bool)

    def read_data(self, path):
        from edengnn.data.io.vasp.parse_density import _read_chgcar

        index = int(path)
        folder = DATA / str(index)
        row = torch.load(folder / "sample.pt", weights_only=False, map_location="cpu")
        if row["id"] != index:
            raise ValueError(f"Geometry ID mismatch for {index}")
        cell, pos, z = [row[k].numpy() for k in ("cell", "positions", "numbers")]
        shape = tuple(row["metadata"]["grid_shape"])
        rho = np.load(folder / "rho.npy").reshape(shape)
        volume = row["metadata"]["volume"]
        if self.reconstructed:
            from scripts.nmc_eden_sacd import cached_sacd
            sad = cached_sacd(self.sad_root, index, z, pos, cell, shape, self.pp)
        else:
            source = self.sad_root / str(index) / "CHGCAR"
            if not source.is_file():
                raise FileNotFoundError(f"Missing native NMC SACD: {source}")
            nz, nc, np_, _, sad, _ = _read_chgcar(str(source))
            np.testing.assert_array_equal(nz, z)
            np.testing.assert_allclose(nc, cell, rtol=0, atol=2e-5)
            df = (np_ - pos) @ np.linalg.inv(cell)
            np.testing.assert_allclose((df - np.rint(df)) @ cell, 0, rtol=0, atol=2e-4)
            if sad.shape == rho.shape and np.array_equal(sad.astype(rho.dtype), rho):
                raise ValueError(f"The supplied SACD is identical to the converged target for {index}")
        if sad.shape != shape or not np.isfinite(sad).all():
            raise ValueError(f"Invalid SACD grid for {index}")
        nelec = sad.mean(dtype=np.float64) * volume
        if not np.isclose(nelec, rho.mean(dtype=np.float64) * volume, rtol=0, atol=3e-4):
            raise ValueError(f"SACD electron count differs from CHGCAR for {index}")
        aug, mask = self.occupations(row)
        return str(index), cell, z, pos, rho, sad, shape, nelec, volume, aug, mask

    def occupations(self, row):
        z = row["numbers"].numpy()
        old_values, old_mask = row["target_aug"].numpy(), row["aug_mask"].numpy()
        indices = np.stack([self.indices[k][0] for k in z])
        mask = np.stack([self.indices[k][1] for k in z])
        if not np.take_along_axis(old_mask, indices, axis=1)[mask].all() or mask.sum() != old_mask.sum():
            raise ValueError(f"PAW mask mapping differs for {row['id']}")
        aug = np.where(mask, np.take_along_axis(old_values, indices, axis=1), 0)
        return aug, mask


def dataset(list_path, reader):
    from edengnn.data.dataload import DensityDataset
    return DensityDataset(str(list_path), 4.0, 3.0, reader, stage="train", dft_software="vasp")


def step_loss(model, data):
    output = model(data)
    target = data.grid_func_out
    density = output["grid_func_out"]
    if tuple(density.shape) != tuple(target.shape) or target.numel() != math.prod(data.grid_shape.tolist()):
        raise ValueError("A structure did not produce its complete original grid")
    density_l1 = torch.nn.functional.l1_loss(density, target)
    mask = data.aug_mask
    aug_l1 = torch.nn.functional.l1_loss(output["aug_tensor"].flatten()[mask], data.aug_tensor.flatten()[mask])
    return 0.9 * density_l1 + 0.1 * aug_l1, density_l1.detach(), aug_l1.detach()


def optimizer_for(model):
    return torch.optim.AdamW(model.parameters(), lr=0.005, eps=1e-8, betas=(0.99, 0.999),
                             weight_decay=0, amsgrad=True)


@torch.no_grad()
def validation(model, examples):
    model.eval()
    rows = []
    for data in examples:
        data = data.cuda()
        pred = model(data)
        if pred["grid_func_out"].shape != data.grid_func_out.shape:
            raise ValueError(f"Invalid validation grid for {data.name}")
        delta = (pred["grid_func_out"].double() - data.grid_func_out).abs()
        mask = data.aug_mask
        occ = (pred["aug_tensor"].flatten()[mask] - data.aug_tensor.flatten()[mask]).abs().double().mean()
        native_norm = delta.sum() * data.dvolume / data.nelec
        rows.append([float(native_norm), float(delta.sum() / data.grid_func_out.double().abs().sum()), float(occ)])
    paper_density, nmae, occ_mae = np.mean(rows, axis=0).tolist()
    score = 0.9 * paper_density + 0.1 * occ_mae
    if not np.isfinite([score, nmae, occ_mae]).all():
        raise FloatingPointError("Nonfinite full-grid validation metric")
    return dict(paper_density=paper_density, density_nmae=nmae, occupation_mae=occ_mae, score=score)


def write_progress(folder, step, epoch, **fields):
    save(folder / "progress.json", dict(step=step, epoch=epoch, updated=time.time(), **fields))


def fit(args, reader, cfg, folder, lock):
    train = dataset(filelist(folder, "train", cfg["split"]["train"]), reader)
    valid = dataset(filelist(folder, "validation", cfg["split"]["validation"]), reader)
    torch.manual_seed(SEED)
    model = build_model(args.chunk)
    opt = optimizer_for(model)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=2e-5)
    last = folder / "last.pt"
    step, best, stale, history = 0, float("inf"), 0, []
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["lock"] != lock:
            raise ValueError("Training checkpoint configuration differs")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        step, best, stale, history = [state[k] for k in ("step", "best", "stale", "history")]
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        del state

    # The author's DataLoader uses filelist order with batch_size=1 and no shuffle.
    while step < args.max_steps and stale < 30:
        index = step % len(train)
        epoch = step // len(train) + 1
        model.train()
        data = train[index].cuda()
        opt.zero_grad(set_to_none=True)
        loss, density_mae, occ_mae = step_loss(model, data)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss for structure {data.name}")
        loss.backward()
        opt.step()
        step += 1
        if step <= 3 or step % 20 == 0:
            print("TRAIN", step, epoch, data.name, data.grid_func_out.numel(), float(density_mae),
                  float(occ_mae), flush=True)
            write_progress(folder, step, epoch, phase="train", structure=data.name,
                           full_grid_points=data.grid_func_out.numel(), density_mae=float(density_mae),
                           occupation_mae=float(occ_mae))
        del data, loss, density_mae, occ_mae
        end_epoch = step % len(train) == 0 or step == args.max_steps
        if end_epoch:
            with torch.no_grad():
                metrics = validation(model, valid)
            sched.step(metrics["score"])
            history.append(dict(step=step, epoch=epoch, lr=opt.param_groups[0]["lr"], **metrics))
            if metrics["score"] < best:
                best, stale = metrics["score"], 0
                pt_save(folder / "best.pt", dict(model=model.state_dict(), step=step, lock=lock))
            else:
                stale += 1
            save(folder / "validation.json", history)
            print("VALIDATION", history[-1], flush=True)
        if step % CHECKPOINT_INTERVAL == 0 or end_epoch or step == 1:
            pt_save(last, dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                               step=step, best=best, stale=stale, history=history, lock=lock,
                               cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
    chosen = min(history, key=lambda r: r["score"])
    selection = dict(reason="validation_plateau" if stale >= 30 else "step_limit", step=step,
                     validation=chosen, checkpoint_sha256=sha256(folder / "best.pt"), lock=lock)
    save(folder / "selection.json", selection)
    write_progress(folder, step, chosen["epoch"], phase="selected")


@torch.no_grad()
def test(args, reader, cfg, folder, lock):
    selection = read(folder / "selection.json")
    path = folder / "best.pt"
    if selection["lock"] != lock or sha256(path) != selection["checkpoint_sha256"]:
        raise ValueError("Frozen model differs from validation selection")
    state = torch.load(path, weights_only=False, map_location="cpu")
    if state["step"] != selection["validation"]["step"] or state["lock"] != lock:
        raise ValueError("Best checkpoint and validation history disagree")
    model = build_model(args.chunk).eval().requires_grad_(False)
    model.load_state_dict(state["model"])
    reference = read(REFERENCE)
    ids = cfg["split"]["test"]
    baseline = {r["index"]: r for r in reference["records"]}
    if (not reference["completed"] or list(baseline) != ids or len(reference["records"]) != len(ids)
            or reference["settings_sha256"] != sha256(REFERENCE.with_name("settings.json"))):
        raise ValueError("DeepDFT test coverage differs")
    examples = dataset(filelist(folder, "test", ids), reader)
    records = []
    for j, index in enumerate(ids):
        dest = folder / "test_records" / f"{index}.json"
        if dest.is_file():
            row = read(dest)
        else:
            data = examples[j].cuda()
            out = model(data)
            target = torch.from_numpy(np.load(DATA / str(index) / "rho.npy")).cuda().double()
            pred = out["grid_func_out"].flatten()
            target_info = {"metadata": {"volume": float(data.volume)}}
            if target.shape != pred.shape or not torch.isfinite(pred).all() or not torch.isfinite(out["aug_tensor"]).all():
                raise ValueError(f"Invalid prediction for {index}")
            own = metric(pred, target, target_info)
            deep = baseline[index]["density"]["reference"]
            if not math.isclose(own["target_abs_sum"], deep["target_abs_sum"], rel_tol=1e-9):
                raise ValueError(f"DeepDFT target grid differs for {index}")
            row = dict(id=index, checkpoint_sha256=selection["checkpoint_sha256"], points=target.numel(),
                       density=own, deepdft_nmae=deep["nmae"],
                       occupation=measures(out["aug_tensor"].flatten(), data.aug_tensor.flatten(), data.aug_mask))
            save(dest, row)
        if row["id"] != index or row["checkpoint_sha256"] != selection["checkpoint_sha256"]:
            raise ValueError(f"Stale test record for {index}")
        records.append(row)
    eden = np.array([r["density"]["nmae"] for r in records])
    deep = np.array([r["deepdft_nmae"] for r in records])
    delta = 100 * (eden - deep)
    bootstrap = np.random.default_rng(SEED).choice(delta, (10000, len(ids))).mean(axis=1)
    result = dict(count=len(ids), seed=SEED, selection=selection, eden_nmae_pct=100*eden.mean(),
                  deepdft_nmae_pct=100*deep.mean(), difference_percentage_points=delta.mean(),
                  paired_95pct_difference=np.quantile(bootstrap, [.025, .975]).tolist(),
                  eden_lower_error_count=int((delta < 0).sum()), reaches_deepdft_mean=bool(eden.mean() <= deep.mean()),
                  occupation_mae=np.mean([r["occupation"]["mae"] for r in records]))
    save(folder / "benchmark.json", result)
    write_progress(folder, selection["step"], selection["validation"]["epoch"], phase="complete")
    print("BENCHMARK", result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--native-sacd", type=Path, help="Native VASP zero-step CHGCAR under <root>/<id>/CHGCAR")
    source.add_argument("--potcar-sacd", action="store_true", help="Use the separately validated reconstruction")
    parser.add_argument("--chunk", type=int, default=150000, help="Atom-grid pairs per native memory chunk")
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--gpu-fraction", type=float, default=0.16, help="PyTorch allocation limit on the shared GPU")
    args = parser.parse_args()
    if args.chunk <= 0 or not 1 <= args.max_steps <= 1000000 or not 0 < args.gpu_fraction <= 1:
        parser.error("Require a positive chunk size, 1–1000000 steps and a GPU fraction in (0, 1]")
    dll = enable_upstream()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("medium")
    torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    mode = "reconstructed_sacd" if args.potcar_sacd else "native_sacd"
    folder = ROOT / f"{mode}_seed{SEED}"
    folder.mkdir(parents=True, exist_ok=True)
    cfg = manifest("nmc")
    ids = cfg["split"]
    if [len(ids[k]) for k in ("train","validation","test")] != [1450,50,500] or len(set(sum(ids.values(), []))) != 2000:
        raise ValueError("Unexpected NMC split")
    sad_root = (ROOT / "sacd_grids" / sha256(Path("scripts/nmc_eden_sacd.py"))[:16]
                if args.potcar_sacd else args.native_sacd.resolve())
    reader = NMCReader(sad_root, reconstructed=args.potcar_sacd)
    lock = dict(source_revision=REVISION, split_sha256=cfg["split_sha256"], seed=SEED,
                sacd_source=mode, sacd_directory=str(sad_root.resolve()), max_steps=args.max_steps,
                model=model_config(args.chunk), objective_weights=[0.9, 0.1], scheduler_patience=5,
                code_sha256=sha256(Path(__file__)))
    if args.potcar_sacd:
        lock["sacd_code_sha256"] = sha256(Path("scripts/nmc_eden_sacd.py"))
    config_file = folder / "settings.json"
    with FileLock(folder / "benchmark.lock", timeout=0):
        if config_file.exists() and read(config_file) != lock:
            raise ValueError("Existing run has a different configuration")
        save(config_file, lock)
        if not (folder / "selection.json").exists():
            fit(args, reader, cfg, folder, lock)
        test(args, reader, cfg, folder, lock)
    if dll is not None:
        dll.close()


if __name__ == "__main__":
    main()
