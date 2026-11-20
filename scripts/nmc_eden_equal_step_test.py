"""Freeze both terminal checkpoints and score them on the same 500 NMC structures."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts import compare_nmc_joint_test as paired
from scripts import nmc_eden_fullgrid as full
from scripts import paper_joint_steps as deep
from scripts.run_joint_pair import JointModel, RadialTableProbe, geometry, prepare


RUN = paired.EDEN
LIMIT = paired.DEEP_LIMIT
FIXED = RUN / f"fixed_step_{LIMIT}"
REPORT = paired.OUTPUT.with_name(f"paired_equal_step_{LIMIT}.json")


def freeze(source, dest, lock, validation, model_key, rule, filename):
    dest.mkdir(exist_ok=True, parents=True)
    marker = dest / "selection.json"
    path = dest / filename
    source_hash = full.sha256(source)
    if marker.exists():
        frozen = full.read(marker)
        if (frozen["source_sha256"] != source_hash or frozen["validation"] != validation
                or frozen["lock"] != lock or frozen["selection_rule"] != rule or frozen["step"] != LIMIT
                or full.sha256(path) != frozen["checkpoint_sha256"]):
            raise ValueError(f"Frozen checkpoint disagrees with {source}")
        return frozen
    state = torch.load(source, map_location="cpu", weights_only=False)
    if state["step"] != LIMIT or state["lock"] != lock:
        raise ValueError(f"Terminal step or provenance mismatch at {source}")
    expected_steps = list(range(5000, LIMIT + 1, 5000))
    if [r["step"] for r in state["history"]] != expected_steps:
        raise ValueError(f"Incomplete terminal validation history at {source}")
    chosen = state["history"][-1] if rule == "exact_terminal_step" else min(state["history"], key=lambda r: r["density_mae"])
    if chosen != validation:
        raise ValueError(f"Checkpoint selection differs from committed history at {source}")
    full.pt_save(path, dict(model=state[model_key], step=chosen["step"], lock=lock))
    selection = dict(reason="fixed_step_budget", step=LIMIT, selection_rule=rule, validation=chosen,
                     lock=lock, source_sha256=source_hash, checkpoint_sha256=full.sha256(path))
    full.save(marker, selection)
    return selection


@torch.no_grad()
def deep_terminal_test(cfg, selection, folder):
    _, _, stats, factors = deep.setup("nmc")
    model = JointModel("deep", *stats).cuda().eval().requires_grad_(False)
    state = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
    if state["step"] != LIMIT or state["lock"] != selection["lock"]:
        raise ValueError("Wrong DeepDFT+PAW terminal checkpoint")
    model.load_state_dict(state["model"])
    core = RadialTableProbe(model.net, fused=True).eval()
    for i in cfg["split"]["test"]:
        path = folder / "test_records" / f"{i}.json"
        if path.exists():
            row = full.read(path)
            if row["id"] != i or row["checkpoint_sha256"] != selection["checkpoint_sha256"]:
                raise ValueError(f"Stale DeepDFT+PAW record for structure {i}")
            continue
        src = full.DATA / str(i)
        sample = torch.load(src / "sample.pt", weights_only=False)
        if sample["id"] != i:
            raise ValueError(f"Wrong reference for structure {i}")
        geom = geometry(sample, full=True)
        target = torch.tensor(np.load(src / "rho.npy"), device="cuda", dtype=torch.float64)
        density, paw = model.full(geom, factors, core)
        if (density.shape != target.shape or not torch.isfinite(density).all()
                or not torch.isfinite(paw).all()):
            raise ValueError(f"Invalid DeepDFT+PAW prediction for structure {i}")
        metrics = dict(density=full.metric(density, target, geom),
                       occupation=full.measures(paw, sample["target_aug"].cuda(), sample["aug_mask"].cuda()))
        full.save(path, dict(id=i, points=target.numel(), checkpoint_sha256=selection["checkpoint_sha256"],
                             scores={"deep_42": metrics}))


def terminal_summary(cfg, eden, reference):
    ids = cfg["split"]["test"]
    if len(ids) != 500 or len(set(ids)) != len(ids):
        raise ValueError("Unexpected official test set")
    for model, selection in (("edengnn", eden), ("deepdft_paw", reference)):
        folder = FIXED / model
        if {p.stem for p in (folder/"test_records").glob("*.json")} != {str(i) for i in ids}:
            raise ValueError(f"Incomplete test records in {folder}")
        if selection["selection_rule"] != "exact_terminal_step" or selection["validation"]["step"] != LIMIT:
            raise ValueError(f"Wrong step for {model}")
    rows = []
    for i in ids:
        e = full.read(FIXED / "edengnn" / "test_records" / f"{i}.json")
        d = full.read(FIXED / "deepdft_paw" / "test_records" / f"{i}.json")
        if d["checkpoint_sha256"] != reference["checkpoint_sha256"]:
            raise ValueError(f"Wrong frozen DeepDFT+PAW weights for {i}")
        rows.append(paired.paired_record(e, d, i, eden["checkpoint_sha256"]))
    summary = paired.paired_metrics(rows)
    eden_check = full.read(FIXED / "edengnn" / "benchmark.json")
    if eden_check["count"] != len(rows) or eden_check["selection"] != eden:
        raise ValueError("EdenGNN terminal benchmark or coverage disagrees with selection")
    paired.require_close(eden_check["eden_nmae_pct"], summary["density_nmae"]["edengnn_pct"], "EdenGNN density mean")
    paired.require_close(eden_check["occupation_mae"], summary["occupation_mae"]["edengnn"], "EdenGNN PAW MAE")
    result = dict(seed=full.SEED, test_count=len(ids), split_sha256=cfg["split_sha256"],
                  training_steps_per_model=LIMIT, edengnn_full_grid_training_steps=0,
                  test_density="all original grid points", test_paw="all valid coefficients",
                  selection_rule="exact_terminal_step", stop_reason="fixed_step_budget",
                  edengnn_selection=eden, deepdft_paw_selection=reference,
                  records=rows, **summary)
    full.save(REPORT, result)
    print("FIXED-STEP TEST", REPORT, flush=True)


def finish(cfg, reader, lock):
    # Both joint outputs use the same weights. Commit both selection rules before test access.
    cfg_deep = full.read(paired.DEEP / "progress.json")
    if cfg_deep["step"] != LIMIT or cfg_deep["limit"] != LIMIT:
        full.write_progress(RUN, LIMIT, LIMIT*2/1450, phase="awaiting_deepdft_terminal", limit=LIMIT)
        return
    histories = [full.read(p) for p in (RUN / "validation.json", paired.DEEP / "history.json")]
    eden_hist, deep_hist = histories
    ed = freeze(RUN / "last.pt", FIXED / "edengnn", lock, eden_hist[-1], "model", "exact_terminal_step", "best.pt")
    ref = freeze(paired.DEEP / "last.pt", FIXED / "deepdft_paw", deep.lock_for("nmc"), deep_hist[-1],
                 "model", "exact_terminal_step", "model.pt")
    if ref["lock"] != lock["deep_protocol"]:
        raise ValueError("DeepDFT+PAW protocol differs from EdenGNN's frozen comparison settings")
    chosen = min(eden_hist, key=lambda r: r["density_mae"])
    freeze(RUN / "last.pt", RUN, lock, chosen, "best_model", "minimum_validation_density_mae", "best.pt")

    marker = full.DATA.parent / "test_prepared.json"
    if not marker.exists():
        prepare("nmc", "test")
    inputs = full.read(marker)
    if inputs["split_sha256"] != cfg["split_sha256"] or inputs["count"] != 500:
        raise ValueError("Prepared test set differs from the official 500 structures")
    args = SimpleNamespace(chunk=lock["model"]["chunk_size_predict"])
    full.write_progress(RUN, LIMIT, eden_hist[-1]["epoch"], phase="testing_exact_terminal_step", limit=LIMIT)
    full.test(args, reader, cfg, FIXED / "edengnn", lock)
    torch.cuda.empty_cache()
    deep_terminal_test(cfg, ref, FIXED / "deepdft_paw")
    terminal_summary(cfg, ed, ref)
    torch.cuda.empty_cache()

    full.test(args, reader, cfg, RUN, lock)
    if not paired.DEEP_SUMMARY.exists():
        full.write_progress(RUN, LIMIT, eden_hist[-1]["epoch"], phase="awaiting_deepdft_selected_test", limit=LIMIT)
        return
    paired.compare(RUN)
    full.write_progress(RUN, LIMIT, eden_hist[-1]["epoch"], phase="complete", limit=LIMIT,
                        exact_step_report=str(REPORT), validation_selected_report=str(paired.OUTPUT))
