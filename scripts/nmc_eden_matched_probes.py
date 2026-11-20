"""Train NMC EdenGNN with its AdamW and L1 recipe on the DeepDFT+PAW probe stream."""
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from scripts import benchmark_nmc_eden_probes as probes
from scripts import nmc_eden_fullgrid as full
from scripts import paper_joint_steps as deep
from scripts.compare_nmc_joint_test import DEEP, DEEP_LIMIT, EDEN


RUN = EDEN
SEED = full.SEED
BATCH = 2
TRAIN_PROBES = 1000
VAL_PROBES = 5000
VAL_INTERVAL = 5000
CHUNK = 150000
METRICS = ("density_mae", "density_nmae", "occupation_mae", "occupation_relative_l1")


def settings(cfg):
    sacd_code = full.sha256(Path("scripts/nmc_eden_sacd.py"))
    return dict(seed=SEED, max_steps=DEEP_LIMIT, initialization="random_from_step_zero",
                full_grid_training_steps=0, batch_size=BATCH, probes_per_structure=TRAIN_PROBES,
                training_probes="uniform_with_replacement; default_rng([42,step])",
                structure_sampler="paper_joint_steps.sequence, same seed, split order and horizon",
                validation_interval=VAL_INTERVAL, validation_structures=50, validation_probes=VAL_PROBES,
                validation_sampler="uniform_with_replacement; default_rng([75113,structure_id])",
                optimizer="AdamW", initial_learning_rate=0.005, betas=[0.99, 0.999],
                eps=1e-8, weight_decay=0, amsgrad=True,
                loss="0.9 * sampled_density_mae + 0.1 * valid_paw_mae", objective_weights=[0.9, 0.1],
                scheduler="ReduceLROnPlateau", lr_factor=0.5, scheduler_patience=5, min_lr=2e-5,
                scheduler_metric="0.9 * density_mae / mean_reference_charge_density + 0.1 * occupation_mae, averaged over the validation structures",
                primary_selection="exact_terminal_step", secondary_selection="minimum_validation_density_mae",
                source_revision=full.REVISION, split_sha256=cfg["split_sha256"], model=full.model_config(CHUNK),
                sacd_source="reconstructed_sacd", sacd_code_sha256=sacd_code,
                sacd_directory=str((full.ROOT / "sacd_grids" / sacd_code[:16]).resolve()),
                code_sha256=full.sha256(Path(__file__)), native_code_sha256=full.sha256(Path(full.__file__)),
                probe_code_sha256=full.sha256(Path(probes.__file__)),
                geometry_code_sha256=full.sha256(Path("scripts/nmc_probe_holdout.py")),
                deep_protocol=deep.lock_for("nmc"))


class Inputs:
    def __init__(self, cfg, reader):
        ids = cfg["split"]["train"] + cfg["split"]["validation"]
        self.examples = full.dataset(full.filelist(RUN, "fit", ids), reader)
        self.positions = {i: j for j, i in enumerate(ids)}
        self.reader = reader
        for z, (indices, mask) in reader.indices.items():
            if len(np.unique(indices[mask])) != mask.sum():
                raise ValueError(f"Duplicate PAW slot for element {z}")

    @lru_cache(maxsize=32)
    def example(self, i):
        base = self.examples[self.positions[i]]
        if int(base.name) != i:
            raise ValueError(f"Structure schedule and input disagree for {i}")
        # These native dense-stencil arrays are only used by full-grid inference.
        del base["map_probe"], base["edge_vec_probes"]
        row = torch.load(full.DATA / str(i) / "sample.pt", map_location="cpu", weights_only=False)
        return base.cuda(), row

    def sampled(self, model, i, rng, count, audit=None):
        base, row = self.example(i)
        flat = rng.integers(base.grid_func_out.numel(), size=count)
        indices = torch.from_numpy(flat)
        target = base.grid_func_out.flatten()[indices.cuda()]
        truth = base.aug_tensor.flatten()[base.aug_mask]
        if audit is not None:
            if not torch.equal(deep.get_points("nmc", row, flat, "rho"), target):
                raise ValueError(f"DeepDFT and EdenGNN probe labels disagree on {i}")
            indices_by_atom = np.stack([self.reader.indices[z][0] for z in row["numbers"].tolist()])
            mapped = row["target_aug"].gather(1, torch.from_numpy(indices_by_atom))
            if (int(row["aug_mask"].sum()) != truth.numel()
                    or not torch.equal(mapped.cuda().flatten()[base.aug_mask], truth)):
                raise ValueError(f"DeepDFT and EdenGNN PAW labels disagree on {i}")
            audit.append(dict(id=i, density_probes=count, first_probe_indices=flat[:8].tolist(),
                              probe_indices_sha256=hashlib.sha256(flat.astype("<i8").tobytes()).hexdigest(),
                              valid_paw=truth.numel()))
        features = probes.encode_structure(model, base)
        model.aug(features)
        paw = features["node_aug"].flatten()[base.aug_mask]
        density = probes.probe_density(model, features, indices, row["cell"].double())
        return density - target, target, paw - truth, truth


@torch.no_grad()
def validate(model, inputs, ids):
    model.eval()
    rows = []
    for i in ids:
        rho_err, rho, paw_err, paw = inputs.sampled(model, i, np.random.default_rng([75113, i]), VAL_PROBES)
        ae, occ = rho_err.abs(), paw_err.abs()
        base, _ = inputs.example(i)
        # Uniform-grid estimate of the native integrated density L1 divided by electron count.
        charge_relative_l1 = ae.mean() * base.volume / base.nelec
        rows.append(torch.stack((ae.mean(), ae.sum()/rho.abs().sum(),
                                 occ.mean(), occ.sum()/paw.abs().sum(),
                                 0.9 * charge_relative_l1 + 0.1 * occ.mean())).tolist())
    if len(rows) != 50 or not np.isfinite(rows).all():
        raise ValueError("Incomplete or nonfinite validation")
    model.train()
    return dict(zip((*METRICS, "scheduler_score"), np.mean(rows, axis=0).tolist()))


def publish_validation(history):
    full.save(RUN / "validation.json", history)
    reference = {r["step"]: r for r in full.read(DEEP / "history.json")}
    paired = [dict(step=r["step"], edengnn={k: r[k] for k in METRICS},
                   deepdft_paw={k: reference[r["step"]][k] for k in METRICS})
              for r in history if r["step"] in reference]
    full.save(RUN / "paired_validation.json", dict(structures=50, density_probes_per_structure=VAL_PROBES,
              paw="all valid coefficients", units="ratios for both relative errors", records=paired))


def train(cfg, reader, lock):
    inputs = Inputs(cfg, reader)
    train_ids, val_ids = cfg["split"]["train"], cfg["split"]["validation"]
    torch.manual_seed(SEED)
    model = full.build_model(CHUNK)
    opt = full.optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=lock["lr_factor"], patience=lock["scheduler_patience"], min_lr=lock["min_lr"])
    last = RUN / "last.pt"
    start, best_model, history = 0, None, []
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["lock"] != lock:
            raise ValueError("Checkpoint belongs to a different training protocol")
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])
        start, best_model, history = state["step"], state["best_model"], state["history"]
        del state
    publish_validation(history)
    if start < DEEP_LIMIT:
        print("Preparing shared DeepDFT training sequence", start, DEEP_LIMIT, flush=True)
        schedule = deep.sequence(train_ids, SEED, DEEP_LIMIT)
        print("TRAIN", start, DEEP_LIMIT, flush=True)
        model.train()
    for step in range(start + 1, DEEP_LIMIT + 1):
        lr = opt.param_groups[0]["lr"]
        opt.zero_grad(set_to_none=True)
        rng = np.random.default_rng([SEED, step])
        batch_ids = schedule[step - 1].tolist()
        audit = [] if step == 1 else None
        losses = []
        for i in batch_ids:
            delta, _, aug, _ = inputs.sampled(model, i, rng, TRAIN_PROBES, audit)
            density_l1, occupation_l1 = delta.abs().mean(), aug.abs().mean()
            loss = 0.9 * density_l1 + 0.1 * occupation_l1
            (loss / BATCH).backward()
            losses.append(loss.detach())
            if audit is not None:
                audit[-1].update(density_mae=float(density_l1.detach()),
                                 occupation_mae=float(occupation_l1.detach()), weighted_l1=float(loss.detach()))
        value = float(torch.stack(losses).mean())
        if not np.isfinite(value):
            raise FloatingPointError((step, batch_ids, value))
        opt.step()
        if audit is not None:
            full.save(RUN / "first_update.json", dict(step=step, structures=audit, labels_verified=True,
                                                      learning_rate=lr, batch_mean_loss=value))
        if step % VAL_INTERVAL == 0:
            metrics = validate(model, inputs, val_ids)
            scheduler.step(metrics["scheduler_score"])
            if not history or metrics["density_mae"] < min(x["density_mae"] for x in history):
                best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            entry = dict(step=step, epoch=step*BATCH/len(train_ids), lr=lr,
                         next_lr=opt.param_groups[0]["lr"], **metrics)
            history.append(entry)
            print("VAL", entry, flush=True)
        if step in (1, 25) or step % 1000 == 0:
            full.pt_save(last, dict(model=model.state_dict(), optimizer=opt.state_dict(), lock=lock,
                                    scheduler=scheduler.state_dict(),
                                    step=step, history=history, best_model=best_model,
                                    cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))
        if step % VAL_INTERVAL == 0:
            publish_validation(history)
        if step in (1, 2, 3, 25) or step % 100 == 0:
            print("STEP", step, "structures", batch_ids, "mean_joint_loss", value, flush=True)
            full.write_progress(RUN, step, step*BATCH/len(train_ids), phase="training",
                                limit=DEEP_LIMIT, last_validation=history[-1]["step"] if history else None)
    inputs.example.cache_clear()


def main():
    dll = full.enable_upstream()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.16)
    RUN.mkdir(exist_ok=True, parents=True)
    cfg = full.manifest("nmc")
    split = cfg["split"]
    if ([len(split[k]) for k in ("train", "validation", "test")] != [1450, 50, 500]
            or len(set(split["train"] + split["validation"] + split["test"])) != 2000):
        raise ValueError("Unexpected official NMC split")
    lock = settings(cfg)
    reader = full.NMCReader(Path(lock["sacd_directory"]), reconstructed=True)
    path = RUN / "settings.json"
    with FileLock(RUN / "benchmark.lock", timeout=0):
        if path.exists() and full.read(path) != lock:
            raise ValueError("Existing run has a different configuration")
        full.save(path, lock)
        if not (RUN / "last.pt").exists():
            full.write_progress(RUN, 0, 0, phase="initializing", limit=DEEP_LIMIT)
        train(cfg, reader, lock)
        torch.cuda.empty_cache()
        from scripts.nmc_eden_equal_step_test import finish
        finish(cfg, reader, lock)
    if dll is not None:
        dll.close()


if __name__ == "__main__":
    main()
