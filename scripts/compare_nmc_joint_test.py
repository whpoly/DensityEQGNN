"""Compare the two frozen seed-42 joint models on the official NMC test set."""
import math
from pathlib import Path

import numpy as np
import torch

from scripts.nmc_eden_fullgrid import ROOT as EDEN_ROOT, SEED
from scripts.paper_joint_steps import REPORT as DEEP_REPORT, ROOT as DEEP_ROOT, lock_for
from scripts.run_joint_pair import manifest, read, save, sha256


EDEN = EDEN_ROOT / "probe_adamw_l1_seed42"
DEEP = DEEP_ROOT / "nmc" / "deep_42"
DEEP_LIMIT = 4_965_000
DEEP_TEST = DEEP.parent / f"single_deep_42_{DEEP_LIMIT}"
DEEP_SUMMARY = DEEP_REPORT / f"nmc_single_deep_42_{DEEP_LIMIT}.json"
OUTPUT = Path("reports/nmc_eden_fullgrid/paired_test_seed42.json")


def verify_selection(checkpoint, selection, history, criterion, lock):
    best = min(history, key=lambda row: row[criterion])
    if selection["validation"] != best or sha256(checkpoint) != selection["checkpoint_sha256"]:
        raise ValueError(f"Frozen validation selection differs from {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["step"] != best["step"] or state["lock"] != lock:
        raise ValueError(f"Checkpoint step or training settings differ in {checkpoint}")


def require_close(a, b, label):
    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-14):
        raise ValueError(f"{label} differs: {a} versus {b}")


def paired_record(eden, deep, index, checkpoint):
    if (eden["id"] != index or deep["id"] != index or eden["checkpoint_sha256"] != checkpoint
            or eden["points"] != 439_040 or deep["points"] != eden["points"]):
        raise ValueError(f"Structure, grid or checkpoint mismatch for {index}")
    scores = {"edengnn": eden, "deepdft_paw": deep["scores"]["deep_42"]}
    result = {"id": index}
    for model, score in scores.items():
        rho, paw = score["density"], score["occupation"]
        if rho["target_abs_sum"] <= 0 or paw["target_absolute_sum"] <= 0 or paw["count"] <= 0:
            raise ValueError(f"Invalid reference labels for {model} on {index}")
        require_close(rho["nmae"], rho["abs_error_sum"] / rho["target_abs_sum"], f"{index} density")
        require_close(paw["relative_l1"], paw["absolute_error_sum"] / paw["target_absolute_sum"], f"{index} PAW")
        require_close(paw["mae"], paw["absolute_error_sum"] / paw["count"], f"{index} PAW MAE")
        values = (rho["nmae"], paw["relative_l1"], paw["mae"])
        if not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError(f"Invalid errors for {model} on {index}")
        result[model] = dict(density_nmae_pct=100 * values[0],
                             occupation_relative_l1_pct=100 * values[1], occupation_mae=values[2])
    other = scores["deepdft_paw"]
    require_close(eden["density"]["target_abs_sum"], other["density"]["target_abs_sum"], f"{index} density labels")
    require_close(eden["occupation"]["target_absolute_sum"], other["occupation"]["target_absolute_sum"],
                  f"{index} PAW labels")
    if eden["occupation"]["count"] != other["occupation"]["count"]:
        raise ValueError(f"Different valid PAW counts for {index}")
    return result


def paired_summary(rows, metric, resampled_ids):
    eden = np.array([r["edengnn"][metric] for r in rows])
    deep = np.array([r["deepdft_paw"][metric] for r in rows])
    delta = eden - deep
    boot = delta[resampled_ids].mean(axis=1)
    return dict(edengnn_pct=float(eden.mean()), deepdft_paw_pct=float(deep.mean()),
                difference_percentage_points=float(delta.mean()),
                paired_95pct_difference_points=np.quantile(boot, [0.025, 0.975]).tolist(),
                edengnn_at_or_below_deepdft_mean=bool(eden.mean() <= deep.mean()),
                edengnn_lower_count=int((delta < 0).sum()))


def paired_metrics(rows):
    draws = np.random.default_rng(SEED).integers(len(rows), size=(10_000, len(rows)), dtype=np.int32)
    density = paired_summary(rows, "density_nmae_pct", draws)
    occupation = paired_summary(rows, "occupation_relative_l1_pct", draws)
    mae = {name: float(np.mean([r[name]["occupation_mae"] for r in rows])) for name in ("edengnn", "deepdft_paw")}
    return dict(density_nmae=density, occupation_relative_l1=occupation, occupation_mae=mae,
                edengnn_at_or_below_both=all(m["edengnn_at_or_below_deepdft_mean"] for m in (density, occupation)),
                bootstrap=dict(unit="structure", draws=len(draws), seed=SEED, difference="edengnn_minus_deepdft_paw"))


def compare(eden=None):
    eden = EDEN if eden is None else eden
    cfg = manifest("nmc")
    ids = cfg["split"]["test"]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError("Unexpected official NMC test split")
    benchmark, single = read(eden / "benchmark.json"), read(DEEP_SUMMARY)
    selection = read(eden / "selection.json")
    deep_frozen = read(DEEP_TEST / "frozen.json")
    deep_progress = read(DEEP / "progress.json")
    settings = read(eden / "settings.json")
    if (benchmark["count"] != len(ids) or benchmark["selection"] != selection or benchmark["seed"] != SEED
            or selection["lock"] != settings or settings["split_sha256"] != cfg["split_sha256"]
            or single["test_count"] != len(ids) or single["selections"] != deep_frozen
            or deep_progress["step"] != DEEP_LIMIT or deep_progress["limit"] != DEEP_LIMIT):
        raise ValueError("Final reports, model selections or training budgets disagree")
    deep_selection = deep_frozen["deep_42"]
    criterion = "density_mae" if settings.get("secondary_selection") == "minimum_validation_density_mae" else "score"
    verify_selection(eden / "best.pt", selection, read(eden / "validation.json"), criterion, settings)
    verify_selection(DEEP / "best.pt", deep_selection, read(DEEP / "history.json"), "density_mae", lock_for("nmc"))
    for folder in (eden, DEEP_TEST):
        found = {p.stem for p in (folder / "test_records").glob("*.json")}
        if found != {str(i) for i in ids}:
            raise ValueError(f"Test record IDs differ from the official split in {folder}")
    rows = [paired_record(read(eden / "test_records" / f"{i}.json"),
                          read(DEEP_TEST / "test_records" / f"{i}.json"), i, selection["checkpoint_sha256"])
            for i in ids]
    metrics = paired_metrics(rows)
    density, occupation, mae = [metrics[k] for k in ("density_nmae", "occupation_relative_l1", "occupation_mae")]
    reference = single["summary"]["deep_42"]
    require_close(density["edengnn_pct"], benchmark["eden_nmae_pct"], "EdenGNN density mean")
    require_close(density["deepdft_paw_pct"], 100 * reference["density_nmae"], "DeepDFT+PAW density mean")
    require_close(occupation["deepdft_paw_pct"], 100 * reference["occupation_relative_l1"], "DeepDFT+PAW PAW mean")
    require_close(mae["edengnn"], benchmark["occupation_mae"], "EdenGNN occupation MAE")
    require_close(mae["deepdft_paw"], reference["occupation_mae"], "DeepDFT+PAW occupation MAE")
    result = dict(test_count=len(rows), seed=SEED, split_sha256=cfg["split_sha256"],
                  edengnn=dict(stopped_at_step=selection["step"], stop_reason=selection["reason"],
                               selected_step=selection["validation"]["step"], checkpoint_sha256=selection["checkpoint_sha256"]),
                  deepdft_paw=dict(stopped_at_step=deep_progress["step"], stop_reason="fixed_step_budget",
                                   selected_step=deep_selection["validation"]["step"], checkpoint_sha256=deep_selection["checkpoint_sha256"]),
                  selection_rule="validation_selected_after_training",
                  official_density_only_eqdeepdft_nmae_pct=benchmark["deepdft_nmae_pct"],
                  records=rows, **metrics)
    result["edengnn"]["training_directory"] = str(eden)
    if "fork_step" in settings:
        result["edengnn"].update(full_grid_warm_start_steps=settings["fork_step"],
                                 sampled_updates=selection["step"] - settings["fork_step"],
                                 probes_per_structure=settings["probes_per_structure"], batch_size=settings["probe_batch"])
    elif settings.get("initialization") == "random_from_step_zero":
        if selection["step"] != DEEP_LIMIT or settings["full_grid_training_steps"] != 0:
            raise ValueError("Matched EdenGNN budget or initialization differs")
        result["edengnn"].update(full_grid_training_steps=0, sampled_updates=selection["step"],
                                 probes_per_structure=settings["probes_per_structure"], batch_size=settings["batch_size"])
    save(OUTPUT, result)
    print(f"Wrote paired results for {len(rows)} test structures to {OUTPUT}")


if __name__ == "__main__":
    required = (EDEN / "benchmark.json", DEEP_SUMMARY)
    pending = [str(p) for p in required if not p.is_file()]
    if pending:
        print("Waiting for final test results: " + ", ".join(pending))
        raise SystemExit(2)
    compare()
