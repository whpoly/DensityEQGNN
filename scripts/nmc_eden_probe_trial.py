"""Run a matched-exposure validation trial before choosing the NMC training protocol."""
import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from filelock import FileLock

from scripts import benchmark_nmc_eden_probes as probes
from scripts import nmc_eden_fullgrid as full


FULL = full.ROOT / "reconstructed_sacd_seed42"
TRIAL = full.ROOT / "probe_pair_trial_seed42"
PROBES = 1000
BATCH = 2
MARGIN = 0.01
METRICS = ("density_nmae", "occupation_relative_l1")


def filelists(cfg, reader):
    train_ids = cfg["split"]["train"]
    val_ids = cfg["split"]["validation"]
    if len(set(train_ids)) != 1450 or len(train_ids) != 1450 or len(val_ids) != 50 or set(train_ids) & set(val_ids):
        raise ValueError("Unexpected training/validation split")
    return (full.dataset(full.filelist(TRIAL, "train", train_ids), reader),
            full.dataset(full.filelist(TRIAL, "validation", val_ids), reader))


def locked_settings(origin, cfg):
    original = full.read(FULL / "settings.json")
    if (original["split_sha256"] != cfg["split_sha256"] or original["seed"] != full.SEED
            or original["sacd_source"] != "reconstructed_sacd"
            or original["code_sha256"] != full.sha256(Path(full.__file__))
            or original["sacd_code_sha256"] != full.sha256(Path("scripts/nmc_eden_sacd.py"))
            or full.sha256(TRIAL / "origin.pt") != origin["checkpoint_sha256"]):
        raise ValueError("Training source, inputs or fork checkpoint changed")
    if (origin["trial_structures"], origin["samples_per_update"], origin["probes_per_structure"]) != (1450, BATCH, PROBES):
        raise ValueError("Trial exposure differs")
    return dict(**original, fork_step=origin["origin_step"], fork_sha256=origin["checkpoint_sha256"],
                probe_batch=BATCH, probes_per_structure=PROBES, sampling="uniform_with_replacement",
                structure_order="original cyclic order from fork", relative_margin=MARGIN,
                compare_after_structures=1450, trial_code_sha256=full.sha256(Path(__file__)),
                probe_code_sha256=full.sha256(Path(probes.__file__)))


@torch.no_grad()
def evaluate(model, valid, ids):
    model.eval()
    rows = []
    for idx, example in enumerate(valid):
        data = example.cuda()
        pred = model(data)
        rho = data.grid_func_out.double()
        if int(data.name) != ids[idx] or rho.numel() != 439040 or pred["grid_func_out"].shape != rho.shape:
            raise ValueError(f"Unexpected validation structure/grid at {idx}")
        delta = (pred["grid_func_out"].double() - rho).abs().sum()
        paw = full.measures(pred["aug_tensor"].flatten(), data.aug_tensor.flatten(), data.aug_mask)
        row = dict(id=ids[idx], points=rho.numel(), valid_paw=paw["count"], paw_target_abs=paw["target_absolute_sum"],
                   density_target_abs=float(rho.abs().sum()), density_nmae=float(delta / rho.abs().sum()),
                   paper_density=float(delta * data.dvolume / data.nelec),
                   occupation_mae=paw["mae"], occupation_relative_l1=paw["relative_l1"])
        if not all(isinstance(x, (float, int)) and math.isfinite(x) for x in row.values()):
            raise ValueError(f"Invalid validation result for {data.name}")
        rows.append(row)
    if len(rows) != 50 or [r["id"] for r in rows] != ids:
        raise ValueError("Incomplete validation coverage")
    means = {key: float(np.mean([r[key] for r in rows]))
             for key in (*METRICS, "occupation_mae", "paper_density")}
    means["score"] = 0.9 * means["paper_density"] + 0.1 * means["occupation_mae"]
    return means, rows


def sample_update(model, opt, train, position, step):
    # Traverse exactly the same structures as the running full-grid control.
    rng = np.random.default_rng([full.SEED, step])
    opt.zero_grad(set_to_none=True)
    losses, names = [], []
    for shift in range(BATCH):
        data = train[(position + shift) % len(train)].cuda()
        names.append(int(data.name))
        cell = torch.load(full.DATA / data.name / "sample.pt", weights_only=False, map_location="cpu")["cell"].double()
        ids = torch.from_numpy(rng.integers(data.grid_func_out.numel(), size=PROBES))
        features = probes.encode_structure(model, data)
        loss, density, paw = probes.run_loss(model, features, cell, ids, "sampled")
        if density.numel() != PROBES or paw.numel() != int(data.aug_mask.sum()) or not torch.isfinite(loss):
            raise ValueError(f"Invalid probe update for {data.name}")
        (loss / BATCH).backward()
        losses.append(float(loss.detach()))
        del data, features, loss, density, paw
    opt.step()
    return names, sum(losses) / BATCH


def fit(cfg, reader, lock):
    train, valid = filelists(cfg, reader)
    fork = lock["fork_step"]
    folder = TRIAL
    model = full.build_model(lock["model"]["chunk_size_train"])
    opt = full.optimizer_for(model)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5, min_lr=2e-5)
    last = folder / "last.pt"
    resuming = last.is_file()
    state = torch.load(last if resuming else folder / "origin.pt", map_location="cpu", weights_only=False)
    if state["lock"] != (lock if resuming else full.read(FULL / "settings.json")):
        raise ValueError("Checkpoint belongs to a different protocol")
    if not resuming and state["step"] != fork:
        raise ValueError("Incorrect fork step")
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    sched.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    updates = state["sampled_updates"] if resuming else 0
    history = state["history"] if resuming else []
    stale = state["stale"] if resuming else 0
    best = min((r["score"] for r in history), default=float("inf"))
    del state

    def checkpoint():
        full.pt_save(last, dict(model=model.state_dict(), optimizer=opt.state_dict(), scheduler=sched.state_dict(),
                                step=fork + updates, sampled_updates=updates, history=history, stale=stale, lock=lock,
                                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()))

    def record_validation():
        nonlocal best, stale
        exposure = fork + BATCH * updates
        endpoint = exposure == fork + len(train)
        regular_check = exposure % len(train) == 0 or fork + updates == lock["max_steps"]
        metrics, rows = evaluate(model, valid, cfg["split"]["validation"])
        if endpoint:
            full.save(folder / "sampled_validation.json",
                      dict(fork_sha256=lock["fork_sha256"], step=fork + updates, structures_seen=exposure, rows=rows))
        if updates and not regular_check:
            checkpoint()
            return
        entry = dict(step=fork + updates, epoch=(exposure + len(train)-1) // len(train),
                     structures_seen=exposure, sampled_updates=updates, **metrics)
        if updates:
            sched.step(metrics["score"])
        entry["lr"] = opt.param_groups[0]["lr"]
        history.append(entry)
        if entry["score"] < best:
            best, stale = entry["score"], 0
            full.pt_save(folder / "best.pt", dict(model=model.state_dict(), step=entry["step"], lock=lock))
        else:
            stale += 1
        full.save(folder / "validation.json", history)
        print("VALIDATION", entry, flush=True)
        checkpoint()

    if not history:
        # The shared starting model remains a candidate for final validation-based selection.
        record_validation()
        full.write_progress(folder, fork, history[-1]["epoch"], phase="probe_training", sampled_updates=0, structures_seen=fork)
    while fork + updates < lock["max_steps"] and stale < 30:
        if updates >= len(train) // BATCH:
            decision_file = folder / "decision.json"
            if not decision_file.is_file() or not full.read(decision_file)["adopt_probes"]:
                phase = "awaiting_comparison" if not decision_file.is_file() else "full_grid_selected"
                full.write_progress(folder, fork + updates, history[-1]["epoch"], phase=phase,
                                    sampled_updates=updates, structures_seen=fork + BATCH * updates)
                return
        position = fork + updates * BATCH
        model.train()
        ids, loss = sample_update(model, opt, train, position, fork + updates + 1)
        updates += 1
        exposure = fork + BATCH * updates
        step = fork + updates
        if updates <= 3 or updates % 20 == 0:
            full.write_progress(folder, step, (exposure+len(train)-1)//len(train), phase="probe_training",
                                sampled_updates=updates, structures_seen=exposure, structures=ids,
                                probes_per_structure=PROBES, joint_loss=loss)
            print("PROBE_UPDATE", step, updates, ids, loss, flush=True)
        # Match the control's validation/LR schedule by structure exposures. Also check
        # the trial endpoint even though this checkpoint forked partway through an epoch.
        if exposure % len(train) == 0 or updates == len(train)//BATCH or step == lock["max_steps"]:
            record_validation()
        elif updates == 1 or updates % full.CHECKPOINT_INTERVAL == 0:
            checkpoint()
    chosen = min(history, key=lambda r: r["score"])
    selection = dict(reason="validation_plateau" if stale >= 30 else "step_limit", step=fork+updates,
                     validation=chosen, checkpoint_sha256=full.sha256(folder/"best.pt"), lock=lock)
    full.save(folder/"selection.json", selection)


def control_validation(cfg, reader, lock):
    dest = TRIAL / "control_validation.json"
    if dest.exists():
        return
    if not (TRIAL / "control.pt").is_file():
        raise FileNotFoundError("The matched live control checkpoint has not yet been captured")
    state = torch.load(TRIAL / "control.pt", map_location="cpu", weights_only=False)
    step = lock["fork_step"] + len(cfg["split"]["train"])
    if state["step"] != step or state["lock"] != full.read(FULL / "settings.json"):
        raise ValueError("Incorrect full-grid control checkpoint")
    model = full.build_model(lock["model"]["chunk_size_train"])
    model.load_state_dict(state["model"])
    del state
    _, valid = filelists(cfg, reader)
    _, rows = evaluate(model, valid, cfg["split"]["validation"])
    full.save(dest, dict(step=step, structures_seen=step, fork_sha256=lock["fork_sha256"],
                         checkpoint_sha256=full.sha256(TRIAL / "control.pt"), rows=rows))


def compare(cfg, lock):
    control = full.read(TRIAL / "control_validation.json")
    candidate = full.read(TRIAL / "sampled_validation.json")
    ids = cfg["split"]["validation"]
    if (control["step"] != lock["fork_step"] + 1450 or candidate["step"] != lock["fork_step"] + 725
            or control["structures_seen"] != candidate["structures_seen"] or len(ids) != 50
            or any(s["fork_sha256"] != lock["fork_sha256"] or [r["id"] for r in s["rows"]] != ids
                   for s in (control, candidate))):
        raise ValueError("Unmatched cohort, exposure, update count or initial weights")
    for a, b in zip(control["rows"], candidate["rows"]):
        if (a["points"] != b["points"] or a["points"] != 439040 or a["valid_paw"] != b["valid_paw"]
                or a["valid_paw"] <= 0 or any(not math.isclose(a[k], b[k], rel_tol=1e-9)
                                               for k in ("density_target_abs", "paw_target_abs"))):
            raise ValueError(f"Different reference labels for {a['id']}")
    draws = np.random.default_rng(full.SEED).integers(len(ids), size=(10_000, len(ids)))
    result = {}
    for key in METRICS:
        a, b = [np.array([r[key] for r in s["rows"]]) for s in (control, candidate)]
        if not (np.isfinite(a).all() and np.isfinite(b).all() and (a > 0).all() and (b >= 0).all()):
            raise ValueError(f"Invalid {key}")
        change = (b.mean()/a.mean()-1)*100
        boot = 100*(b[draws].mean(axis=1)/a[draws].mean(axis=1)-1)
        result[key] = dict(full_grid_pct=float(100*a.mean()), probe_pct=float(100*b.mean()),
                           relative_change_pct=float(change), paired_95pct_relative_change=np.quantile(boot,[.025,.975]).tolist(),
                           within_margin=bool(b.mean() <= a.mean() * (1+MARGIN)))
    return dict(fork_step=lock["fork_step"], fork_sha256=lock["fork_sha256"], full_step=control["step"],
                sampled_step=candidate["step"], additional_structures=1450, validation_count=50,
                relative_margin_pct=100*MARGIN, metrics=result,
                adopt_probes=all(v["within_margin"] for v in result.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("candidate", "control", "decide"))
    action = parser.parse_args().action
    cfg = full.manifest("nmc")
    origin = full.read(TRIAL / "origin.json")
    lock = locked_settings(origin, cfg)
    folder = TRIAL
    settings_path = folder/"settings.json"
    with FileLock(folder/"settings.lock", timeout=10):
        if settings_path.exists() and full.read(settings_path) != lock:
            raise ValueError("Locked probe trial configuration changed")
        if not settings_path.exists():
            full.save(settings_path, lock)
    if action == "decide":
        decision = compare(cfg, lock)
        path = folder/"decision.json"
        if path.exists() and full.read(path) != decision:
            raise ValueError("The saved training decision differs")
        full.save(path, decision)
        print("DECISION", decision, flush=True)
        return
    dll = full.enable_upstream()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("medium")
    torch.cuda.set_per_process_memory_fraction(.16)
    reader = full.NMCReader(Path(lock["sacd_directory"]), reconstructed=True)
    with FileLock(folder/f"{action}.lock", timeout=0):
        if action == "control":
            control_validation(cfg, reader, lock)
        else:
            if not (folder/"selection.json").is_file():
                fit(cfg, reader, lock)
            if (folder/"selection.json").is_file():
                full.test(SimpleNamespace(chunk=lock["model"]["chunk_size_train"]), reader, cfg, folder, lock)
    if dll is not None:
        dll.close()


if __name__ == "__main__":
    main()
