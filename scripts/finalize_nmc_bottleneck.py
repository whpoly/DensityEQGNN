"""Freeze validation choices before evaluating the diagnostic test partitions."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from scripts.fit_nmc_holdout_oracle import ROOT, record
from scripts.nmc_probe_holdout import sha256


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def freeze_selection():
    decision = read(ROOT / "refinement_decision.json")
    candidates = {}
    for name in ("query_eight", "query_eight_refine"):
        folder = ROOT / name
        complete = read(folder / "fit_complete.json")
        history = read(folder / "history.json")
        best = min(history, key=lambda r: r["selection_macro_nmae"])
        assert complete["best_step"] == best["step"]
        assert complete["selection_macro_nmae"] == best["selection_macro_nmae"]
        settings = read(folder / "settings.json")
        for path, digest in settings["source_hashes"].items():
            assert sha256(path) == digest, path
        candidates[name] = {
            "best_step": complete["best_step"],
            "selection_macro_nmae": complete["selection_macro_nmae"],
            "checkpoint_sha256": sha256(folder / "best.pt"),
        }
    refined = read(ROOT / "query_eight_refine" / "settings.json")
    for name in ("steps", "probes", "lr", "min_lr", "eval_every", "seed"):
        assert refined[name] == decision[name], name
    assert refined["parent_sha256"] == candidates["query_eight"]["checkpoint_sha256"]
    audit = read(ROOT / "paired_control_audit.json")
    for name in ("identical_backbone_initialization", "identical_settings_and_sources",
                 "identical_actual_sample_coverage"):
        assert audit[name]
    oracle = {}
    for i in range(8):
        folder = ROOT / "oracle" / f"sample_{i:04d}"
        complete = read(folder / "fit_complete.json")
        oracle[str(i)] = sha256(folder / "best.pt")
        assert oracle[str(i)] == complete["checkpoint_sha256"]
    payload = {
        "selection_basis": "Fixed validation selection points only; no test metrics read",
        "candidates": candidates,
        "selected_query": min(candidates, key=lambda k: candidates[k]["selection_macro_nmae"]),
        "oracle_checkpoint_sha256": oracle,
        "refinement_decision_sha256": sha256(ROOT / "refinement_decision.json"),
        "no_more_adaptive_runs_this_round": True,
    }
    path = ROOT / "frozen_selection.json"
    if path.exists():
        existing = read(path)
        assert {k: v for k, v in existing.items() if k != "frozen_at_utc"} == payload
    else:
        result_paths = [ROOT / name / "result.json" for name in candidates]
        result_paths += [ROOT / "oracle" / f"sample_{i:04d}" / "result.json" for i in range(8)]
        assert not any(p.exists() for p in result_paths), "Test was evaluated before freezing"
        record(path, {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), **payload})
    return payload


def run(module, arguments, log):
    print(f"START {module} {' '.join(arguments)}", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run([sys.executable, "-u", "-m", module, *arguments],
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    print(f"DONE {module}; log={log}", flush=True)


def main():
    frozen = freeze_selection()
    print("FROZEN " + json.dumps(frozen), flush=True)
    record(ROOT / "pipeline_status.json", {"phase": "final_evaluation", "status": "running"})
    run("scripts.fit_nmc_holdout_oracle", ["--final-eval"], ROOT / "oracle_final.log")
    run("scripts.train_nmc_paired_readout", ["--readout", "query", "--output",
        (ROOT / "query_eight").as_posix(), "--final-eval"], ROOT / "query_eight_final.log")
    run("scripts.train_nmc_paired_readout", ["--readout", "query", "--output",
        (ROOT / "query_eight_refine").as_posix(), "--parent", (ROOT / "query_eight/best.pt").as_posix(),
        "--steps", "4000", "--probes", "4096", "--lr", "0.0001", "--min-lr", "0.000001",
        "--final-eval"], ROOT / "query_eight_refine_final.log")
    run("scripts.summarize_nmc_bottleneck", [], ROOT / "summary.log")
    record(ROOT / "pipeline_status.json", {
        "phase": "artifact_review", "status": "evaluation_complete", "test_evaluated": True,
        "selected_query": frozen["selected_query"],
    })


if __name__ == "__main__":
    main()
