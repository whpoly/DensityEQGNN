"""Analyze paired VASP SCF-initialization benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


ELECTRONIC_STEP = re.compile(r"^\s*(?:DAV|RMM|CG)\s*:\s*(\d+)", re.MULTILINE)
TOTEN = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.Ee]+)")
ELAPSED = re.compile(r"Elapsed time \(sec\):\s*([-+0-9.Ee]+)")


def parse_case(case_dir: Path) -> dict[str, object]:
    oszicar_path = case_dir / "OSZICAR"
    outcar_path = case_dir / "OUTCAR"
    oszicar = oszicar_path.read_text(errors="replace") if oszicar_path.exists() else ""
    outcar = outcar_path.read_text(errors="replace") if outcar_path.exists() else ""
    step_numbers = [int(value) for value in ELECTRONIC_STEP.findall(oszicar)]
    energies = [float(value) for value in TOTEN.findall(outcar)]
    elapsed = [float(value) for value in ELAPSED.findall(outcar)]

    exit_status_path = case_dir / "exit_status.txt"
    wall_seconds_path = case_dir / "wall_seconds.txt"
    return {
        "completed": bool(outcar and "General timing and accounting informations" in outcar),
        "ediff_reached": "aborting loop because EDIFF is reached" in outcar,
        "electronic_steps": len(step_numbers),
        "last_electronic_index": step_numbers[-1] if step_numbers else None,
        "final_toten_ev": energies[-1] if energies else None,
        "vasp_elapsed_seconds": elapsed[-1] if elapsed else None,
        "wall_seconds": (
            float(wall_seconds_path.read_text().strip())
            if wall_seconds_path.exists()
            else None
        ),
        "exit_status": (
            int(exit_status_path.read_text().strip())
            if exit_status_path.exists()
            else None
        ),
    }


def paired_summary(records: list[dict[str, object]], case: str) -> dict[str, object]:
    by_sample: dict[int, dict[str, dict[str, object]]] = {}
    for record in records:
        sample_id = int(record["sample_id"])
        by_sample.setdefault(sample_id, {})[str(record["case"])] = record

    paired = []
    for sample_id, cases in sorted(by_sample.items()):
        if "sad" not in cases or case not in cases:
            continue
        sad = cases["sad"]
        candidate = cases[case]
        if not sad["ediff_reached"] or not candidate["ediff_reached"]:
            continue
        sad_steps = int(sad["electronic_steps"])
        candidate_steps = int(candidate["electronic_steps"])
        paired.append(
            {
                "sample_id": sample_id,
                "sad_steps": sad_steps,
                "candidate_steps": candidate_steps,
                "steps_saved": sad_steps - candidate_steps,
                "relative_reduction": (
                    (sad_steps - candidate_steps) / sad_steps if sad_steps else 0.0
                ),
                "energy_delta_ev": (
                    float(candidate["final_toten_ev"]) - float(sad["final_toten_ev"])
                    if candidate["final_toten_ev"] is not None
                    and sad["final_toten_ev"] is not None
                    else None
                ),
                "elapsed_seconds_saved": (
                    float(sad["vasp_elapsed_seconds"])
                    - float(candidate["vasp_elapsed_seconds"])
                    if candidate["vasp_elapsed_seconds"] is not None
                    and sad["vasp_elapsed_seconds"] is not None
                    else None
                ),
                "density_generation_seconds": candidate.get(
                    "density_generation_seconds"
                ),
                "net_seconds_saved": (
                    float(sad["vasp_elapsed_seconds"])
                    - float(candidate["vasp_elapsed_seconds"])
                    - float(candidate["density_generation_seconds"])
                    if candidate["vasp_elapsed_seconds"] is not None
                    and sad["vasp_elapsed_seconds"] is not None
                    and candidate.get("density_generation_seconds") is not None
                    else None
                ),
            }
        )

    if not paired:
        return {"case": case, "num_converged_pairs": 0, "pairs": []}
    savings = np.asarray([item["steps_saved"] for item in paired], dtype=float)
    reductions = np.asarray([item["relative_reduction"] for item in paired], dtype=float)
    energy_deltas = np.asarray(
        [
            abs(float(item["energy_delta_ev"]))
            for item in paired
            if item["energy_delta_ev"] is not None
        ],
        dtype=float,
    )
    elapsed_savings = np.asarray(
        [
            float(item["elapsed_seconds_saved"])
            for item in paired
            if item["elapsed_seconds_saved"] is not None
        ],
        dtype=float,
    )
    net_savings = np.asarray(
        [
            float(item["net_seconds_saved"])
            for item in paired
            if item["net_seconds_saved"] is not None
        ],
        dtype=float,
    )
    return {
        "case": case,
        "num_converged_pairs": len(paired),
        "mean_steps_saved": float(savings.mean()),
        "median_steps_saved": float(np.median(savings)),
        "mean_relative_reduction": float(reductions.mean()),
        "median_relative_reduction": float(np.median(reductions)),
        "fraction_improved": float(np.mean(savings > 0)),
        "fraction_worsened": float(np.mean(savings < 0)),
        "mean_absolute_energy_delta_ev": (
            float(energy_deltas.mean()) if len(energy_deltas) else None
        ),
        "max_absolute_energy_delta_ev": (
            float(energy_deltas.max()) if len(energy_deltas) else None
        ),
        "mean_elapsed_seconds_saved": (
            float(elapsed_savings.mean()) if len(elapsed_savings) else None
        ),
        "mean_net_seconds_saved_including_density_generation": (
            float(net_savings.mean()) if len(net_savings) else None
        ),
        "fraction_net_faster_including_density_generation": (
            float(np.mean(net_savings > 0)) if len(net_savings) else None
        ),
        "pairs": paired,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.benchmark_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest_records = {
        int(record["sample_id"]): record for record in manifest["records"]
    }
    records: list[dict[str, object]] = []
    for sample_id in manifest["sample_ids"]:
        for case in manifest["cases"]:
            case_dir = root / f"sample_{int(sample_id):04d}" / case
            record = {
                "sample_id": int(sample_id),
                "case": case,
                "case_dir": str(case_dir),
                **parse_case(case_dir),
            }
            metadata = manifest_records[int(sample_id)]
            if case == "ml":
                record["density_generation_seconds"] = metadata.get(
                    "prediction", {}
                ).get("generation_seconds")
            elif case == "charge3net":
                record["density_generation_seconds"] = metadata.get(
                    "charge3net_prediction", {}
                ).get("generation_seconds")
            else:
                record["density_generation_seconds"] = None
            records.append(record)

    comparison_cases = [
        case
        for case in ("ml", "charge3net", "oracle_smooth", "oracle_full")
        if case in manifest["cases"]
    ]
    summaries = {
        case: paired_summary(records, case) for case in comparison_cases
    }
    result = {
        "benchmark_dir": str(root),
        "records": records,
        "paired_vs_sad": summaries,
    }
    output = args.output or (root / "scf_results.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    compact = {
        case: {
            key: value
            for key, value in summary.items()
            if key not in {"pairs", "case"}
        }
        for case, summary in summaries.items()
    }
    print(json.dumps(compact, indent=2))
    print(f"Wrote {output} and {csv_path}")


if __name__ == "__main__":
    main()
