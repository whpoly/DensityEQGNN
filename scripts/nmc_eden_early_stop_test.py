"""Freeze validation-selected weights after stopping and compare 500 full grids."""
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts import compare_nmc_joint_test as paired
from scripts import nmc_eden_fullgrid as full
from scripts import paper_joint_steps as deep
from scripts.nmc_eden_stop_policy import evaluate


RUN = paired.EDEN
TEST = RUN / "early_stop_test"
REPORT = paired.OUTPUT.with_name("paired_early_stop_seed42.json")


def freeze_stopped(state, lock, source, folder):
    reason, step, history = state.get("stop_reason"), state["step"], state["history"]
    if state["lock"] != lock or not history or history[-1]["step"] != step:
        raise ValueError("Stopping checkpoint does not end at its committed validation")
    decision = evaluate(history, lock["early_stopping"])
    if reason == "early_stopping":
        if not decision["stop"] or state["early_stopping_decision"] != decision:
            raise ValueError("Checkpoint does not satisfy its early-stopping rule")
    elif reason != "fixed_step_budget" or step != lock["max_steps"]:
        raise ValueError("Checkpoint has no valid stopping reason")
    chosen = min(history, key=lambda r: r["density_mae"])
    selection = dict(reason=reason, step=step, validation=chosen, lock=lock,
                     selection_rule="minimum_validation_density_mae",
                     source_sha256=full.sha256(source), stopping_decision=decision)
    folder.mkdir(parents=True, exist_ok=True)
    marker, checkpoint = folder / "selection.json", folder / "best.pt"
    if marker.exists():
        saved = full.read(marker)
        expected = dict(selection, checkpoint_sha256=full.sha256(checkpoint))
        if saved != expected:
            raise ValueError("Existing frozen weights or selection changed")
        return saved
    if state["best_model"] is None:
        raise ValueError("No validation-selected model was saved")
    full.pt_save(checkpoint, dict(model=state["best_model"], step=chosen["step"], lock=lock))
    selection["checkpoint_sha256"] = full.sha256(checkpoint)
    full.save(marker, selection)
    return selection


def compare_results(cfg, selection, lock):
    ids = cfg["split"]["test"]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError("Unexpected test split")
    single = full.read(paired.DEEP_SUMMARY)
    reference = full.read(paired.DEEP_TEST / "frozen.json")
    deep_progress = full.read(paired.DEEP / "progress.json")
    if (single["test_count"] != 500 or single["selections"] != reference
            or deep_progress["step"] != paired.DEEP_LIMIT
            or deep.lock_for("nmc") != lock["deep_protocol"]):
        raise ValueError("DeepDFT test provenance or training budget changed")
    benchmark = full.read(TEST / "benchmark.json")
    if benchmark["count"] != 500 or benchmark["selection"] != selection or benchmark["seed"] != full.SEED:
        raise ValueError("EdenGNN test report disagrees with selection")
    paired.verify_selection(TEST / "best.pt", selection, full.read(RUN / "validation.json"),
                            "density_mae", lock)
    paired.verify_selection(paired.DEEP / "best.pt", reference["deep_42"],
                            full.read(paired.DEEP / "history.json"), "density_mae", lock["deep_protocol"])
    for folder in (TEST, paired.DEEP_TEST):
        if {p.stem for p in (folder / "test_records").glob("*.json")} != {str(i) for i in ids}:
            raise ValueError(f"Incomplete official test IDs in {folder}")
    rows = [paired.paired_record(full.read(TEST / "test_records" / f"{i}.json"),
                                 full.read(paired.DEEP_TEST / "test_records" / f"{i}.json"),
                                 i, selection["checkpoint_sha256"]) for i in ids]
    metrics = paired.paired_metrics(rows)
    expected = single["summary"]["deep_42"]
    paired.require_close(metrics["density_nmae"]["edengnn_pct"], benchmark["eden_nmae_pct"], "Eden density")
    paired.require_close(metrics["occupation_mae"]["edengnn"], benchmark["occupation_mae"], "Eden PAW MAE")
    paired.require_close(metrics["density_nmae"]["deepdft_paw_pct"], 100 * expected["density_nmae"], "Deep density")
    paired.require_close(metrics["occupation_relative_l1"]["deepdft_paw_pct"],
                          100 * expected["occupation_relative_l1"], "Deep PAW")
    paired.require_close(metrics["occupation_mae"]["deepdft_paw"], expected["occupation_mae"], "Deep PAW MAE")
    result = dict(seed=full.SEED, test_count=500, split_sha256=cfg["split_sha256"],
                  selection_rule="minimum_validation_density_mae_for_each_model",
                  equal_training_steps=selection["step"] == deep_progress["step"],
                  comparison_scope="validation_selected_models_at_recorded_training_budgets",
                  test_density="all original grid points", test_paw="all valid coefficients",
                  edengnn=dict(stopped_at_step=selection["step"], stop_reason=selection["reason"],
                               selected_step=selection["validation"]["step"], full_grid_training_steps=0,
                               batch_size=lock["batch_size"], probes_per_structure=lock["probes_per_structure"],
                               checkpoint_sha256=selection["checkpoint_sha256"]),
                  deepdft_paw=dict(stopped_at_step=deep_progress["step"], stop_reason="fixed_step_budget",
                                   selected_step=reference["deep_42"]["validation"]["step"],
                                   checkpoint_sha256=reference["deep_42"]["checkpoint_sha256"]),
                  early_stopping_policy=lock["early_stopping"], stopping_decision=selection["stopping_decision"],
                  edengnn_selection=selection, deepdft_paw_selection=reference["deep_42"],
                  records=rows, **metrics)
    full.save(REPORT, result)
    return result


def finish(cfg, reader, lock):
    source = RUN / "last.pt"
    state = torch.load(source, map_location="cpu", weights_only=False)
    selection = freeze_stopped(state, lock, source, TEST)
    del state
    full.save(RUN / "stopping.json", selection)
    progress = dict(limit=lock["max_steps"], stop_reason=selection["reason"],
                    selected_step=selection["validation"]["step"])
    if not paired.DEEP_SUMMARY.exists():
        full.write_progress(RUN, selection["step"], selection["step"] * 2 / 1450,
                            phase="awaiting_deepdft_selected_test", **progress)
        return
    reference = full.read(paired.DEEP_TEST / "frozen.json")
    if deep.lock_for("nmc") != lock["deep_protocol"]:
        raise ValueError("DeepDFT protocol changed before test access")
    paired.verify_selection(paired.DEEP / "best.pt", reference["deep_42"],
                            full.read(paired.DEEP / "history.json"), "density_mae", lock["deep_protocol"])
    full.save(TEST / "deep_selection.json", reference)
    marker = full.DATA.parent / "test_prepared.json"
    if not marker.exists():
        from scripts.run_joint_pair import prepare
        prepare("nmc", "test")
    prepared = full.read(marker)
    if prepared["count"] != 500 or prepared["split_sha256"] != cfg["split_sha256"]:
        raise ValueError("Test preparation differs from the official split")
    full.write_progress(RUN, selection["step"], selection["step"] * 2 / 1450,
                        phase="testing_validation_selected", **progress)
    full.test(SimpleNamespace(chunk=lock["model"]["chunk_size_predict"]), reader, cfg, TEST, lock)
    compare_results(cfg, selection, lock)
    full.write_progress(RUN, selection["step"], selection["step"] * 2 / 1450,
                        phase="complete", report=str(REPORT), **progress)
    print("EARLY-STOP BENCHMARK COMPLETE", REPORT, flush=True)
