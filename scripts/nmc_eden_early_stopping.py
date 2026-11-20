"""Continue the locked pure-probe AdamW run with validation-based early stopping."""
import argparse
import datetime
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from scripts import nmc_eden_matched_probes as original
from scripts import nmc_eden_stop_policy as stopping
from scripts import nmc_eden_early_stop_test as testing


RUN = original.RUN
full, deep = original.full, original.deep
SEED, LIMIT = original.SEED, original.DEEP_LIMIT


def settings(cfg):
    lock = original.settings(cfg)
    lock.update(primary_selection="minimum_validation_density_mae",
                training_stop_rule="dual_validation_plateau_or_fixed_maximum",
                early_stopping=stopping.POLICY,
                early_stopping_code_sha256=full.sha256(Path(__file__)),
                early_stopping_policy_code_sha256=full.sha256(Path(stopping.__file__)),
                early_stopping_test_code_sha256=full.sha256(Path(testing.__file__)),
                paired_metrics_code_sha256=full.sha256(Path(testing.paired.__file__)))
    return lock


def load_or_adopt_checkpoint(cfg, lock, allow_adoption):
    """Explicit, restartable migration; preserve every training-state field."""
    path, source = RUN / "settings.json", RUN / "last.pt"
    saved = full.read(path)
    state = torch.load(source, map_location="cpu", weights_only=False)
    if saved == lock and state["lock"] == lock:
        return state
    if not allow_adoption:
        raise ValueError("Protocol adoption requires --adopt-early-stopping")
    previous = original.settings(cfg)
    prior_migration = RUN / "protocol_changes" / "early_stopping_v1" / "migration.json"
    if prior_migration.exists():
        prior = full.read(prior_migration)
        candidate = prior["new_lock"]
        if saved == candidate or state["lock"] == candidate:
            if prior["previous_lock"] != previous:
                raise ValueError("Original training recipe changed since the first migration")
            previous = candidate
    if saved not in (previous, lock) or state["lock"] not in (previous, lock):
        raise ValueError("Cannot migrate a checkpoint with different code, data or training settings")
    archive = RUN / "protocol_changes" / f"early_stopping_v{stopping.POLICY['version']}"
    record_path = archive / "migration.json"
    if not record_path.exists():
        if saved != previous or state["lock"] != previous:
            raise ValueError("Partial migration has no provenance record")
        archive.mkdir(parents=True, exist_ok=True)
        for name in ("last.pt", "settings.json", "launch.json", "validation.json", "paired_validation.json"):
            shutil.copy2(RUN / name, archive / name)
        shutil.copy2(original.__file__, archive / "trainer_before_early_stopping.py")
        record = dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      user_request="Train at least 1000000 steps; then stop when both relative improvements are below 0.5 percent",
                      adopted_at_step=state["step"], previous_lock=previous, new_lock=lock,
                      previous_checkpoint_sha256=full.sha256(archive / "last.pt"))
        full.save(record_path, record)
    else:
        record = full.read(record_path)
    if (record["previous_lock"] != previous or record["new_lock"] != lock
            or full.sha256(archive / "last.pt") != record["previous_checkpoint_sha256"]):
        raise ValueError("Migration archive or expected settings changed")
    if state["lock"] == previous:
        if full.sha256(source) != record["previous_checkpoint_sha256"]:
            raise ValueError("Source checkpoint advanced during protocol migration")
        state["lock"] = lock
        state["protocol_transition"] = dict(adopted_at_step=state["step"],
                                            record=str(record_path),
                                            previous_checkpoint_sha256=record["previous_checkpoint_sha256"])
        full.pt_save(source, state)
    full.save(path, lock)
    return state


def train(cfg, reader, lock, state):
    start, history = state["step"], state["history"]
    decision = stopping.evaluate(history, lock["early_stopping"])
    if state.get("stop_reason"):
        if state["stop_reason"] == "early_stopping":
            if not decision["stop"] or state["early_stopping_decision"] != decision:
                raise ValueError("Saved early-stop decision does not match validation history")
        elif state["stop_reason"] != "fixed_step_budget" or start != LIMIT:
            raise ValueError("Invalid saved stopping reason")
        return
    if start >= LIMIT:
        raise ValueError("A terminal checkpoint must record its stopping reason")
    if state["lock"] != lock:
        raise ValueError("Checkpoint belongs to a different training protocol")
    inputs = original.Inputs(cfg, reader)
    train_ids, val_ids = cfg["split"]["train"], cfg["split"]["validation"]
    torch.manual_seed(SEED)
    model = full.build_model(original.CHUNK)
    opt = full.optimizer_for(model)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=lock["lr_factor"], patience=lock["scheduler_patience"], min_lr=lock["min_lr"])
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    best_model, transition = state["best_model"], state["protocol_transition"]
    del state
    original.publish_validation(history)
    full.save(RUN / "early_stopping_status.json", decision)
    full.write_progress(RUN, start, start * original.BATCH / len(train_ids), phase="preparing_training",
                        limit=LIMIT, last_validation=history[-1]["step"],
                        minimum_steps_before_early_stopping=lock["early_stopping"]["minimum_training_steps"])
    print("Preparing shared DeepDFT training sequence", start, LIMIT, flush=True)
    schedule = deep.sequence(train_ids, SEED, LIMIT)
    print("TRAIN WITH EARLY STOPPING", start, LIMIT, json.dumps(lock["early_stopping"]), flush=True)
    model.train()
    for step in range(start + 1, LIMIT + 1):
        lr = opt.param_groups[0]["lr"]
        opt.zero_grad(set_to_none=True)
        rng = np.random.default_rng([SEED, step])
        batch_ids = schedule[step - 1].tolist()
        losses = []
        for i in batch_ids:
            delta, _, aug, _ = inputs.sampled(model, i, rng, original.TRAIN_PROBES)
            density_l1, occupation_l1 = delta.abs().mean(), aug.abs().mean()
            loss = 0.9 * density_l1 + 0.1 * occupation_l1
            (loss / original.BATCH).backward()
            losses.append(loss.detach())
        value = float(torch.stack(losses).mean())
        if not np.isfinite(value):
            raise FloatingPointError((step, batch_ids, value))
        opt.step()
        stop_reason = "fixed_step_budget" if step == LIMIT else None
        if step % original.VAL_INTERVAL == 0:
            metrics = original.validate(model, inputs, val_ids)
            scheduler.step(metrics["scheduler_score"])
            if metrics["density_mae"] < min(r["density_mae"] for r in history):
                best_model = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            entry = dict(step=step, epoch=step * original.BATCH / len(train_ids), lr=lr,
                         next_lr=opt.param_groups[0]["lr"], **metrics)
            history.append(entry)
            decision = stopping.evaluate(history, lock["early_stopping"])
            if decision["stop"]:
                stop_reason = "early_stopping"
            print("VAL", entry, flush=True)
            print("EARLY_STOP_CHECK", json.dumps(decision), flush=True)
        if step % 1000 == 0:
            full.pt_save(RUN / "last.pt", dict(model=model.state_dict(), optimizer=opt.state_dict(), lock=lock,
                         scheduler=scheduler.state_dict(), step=step, history=history, best_model=best_model,
                         cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(),
                         protocol_transition=transition, stop_reason=stop_reason,
                         early_stopping_decision=decision))
        if step % original.VAL_INTERVAL == 0:
            original.publish_validation(history)
            full.save(RUN / "early_stopping_status.json", decision)
        if step % 100 == 0:
            print("STEP", step, "structures", batch_ids, "mean_joint_loss", value, flush=True)
            full.write_progress(RUN, step, step * original.BATCH / len(train_ids),
                                phase="training_stopped" if stop_reason else "training",
                                limit=LIMIT, last_validation=history[-1]["step"],
                                minimum_steps_before_early_stopping=lock["early_stopping"]["minimum_training_steps"],
                                early_stopping_enabled=True, stop_reason=stop_reason)
        if stop_reason:
            print("TRAINING STOPPED", step, stop_reason, flush=True)
            break
    inputs.example.cache_clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adopt-early-stopping", action="store_true")
    args = parser.parse_args()
    dll = full.enable_upstream()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.16)
    cfg = full.manifest("nmc")
    split = cfg["split"]
    if ([len(split[k]) for k in ("train", "validation", "test")] != [1450, 50, 500]
            or len(set(split["train"] + split["validation"] + split["test"])) != 2000):
        raise ValueError("Unexpected official NMC split")
    lock = settings(cfg)
    with FileLock(RUN / "benchmark.lock", timeout=0):
        state = load_or_adopt_checkpoint(cfg, lock, args.adopt_early_stopping)
        reader = full.NMCReader(Path(lock["sacd_directory"]), reconstructed=True)
        train(cfg, reader, lock, state)
        del state
        torch.cuda.empty_cache()
        testing.finish(cfg, reader, lock)
    if dll is not None:
        dll.close()


if __name__ == "__main__":
    main()
