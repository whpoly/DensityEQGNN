"""Aggregate paired oracle diagnostics and export a scientific comparison plot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ("current","regular","regular_bonds","regular_capacity")
LABELS = ("Current atom R12","Regular atom R12","Regular + bond centers","Regular atom R15/16")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--fixed-r16-dir",type=Path,
                        help="Supplementary fixed-R16 fits for capacity controls that used R15")
    args = parser.parse_args()
    records = [json.loads(p.read_text()) for p in sorted(args.input_dir.glob("sample_*.json"))]
    grouped = {v:{r["sample_id"]:r for r in records if r["variant"]==v} for v in VARIANTS}
    ids = sorted(set.intersection(*(set(d) for d in grouped.values())))
    if not ids:
        raise ValueError("No complete paired samples")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    summary = {"input_directory":str(args.input_dir),"sample_ids":ids,"num_samples":len(ids),
               "metric":"macro mean of per-structure full-grid NMAE, stored as ratios",
               "interpretation":"per-structure fitted densities; not generalization scores or certified lower bounds",
               "variants":{},"gnn_controls":{}}
    scores = np.array([[grouped[v][i]["metrics"]["oracle"]["full_grid_nmae"]*100 for v in VARIANTS] for i in ids])
    for column,variant in enumerate(VARIANTS):
        selected = [grouped[variant][i] for i in ids]
        metrics = [r["metrics"]["oracle"] for r in selected]
        summary["variants"][variant] = {
            "mean_nmae":float(scores[:,column].mean()/100),
            "min_nmae":float(scores[:,column].min()/100),
            "max_nmae":float(scores[:,column].max()/100),
            "coefficients_min":min(r["num_coefficients"] for r in selected),
            "coefficients_max":max(r["num_coefficients"] for r in selected),
            "mean_absolute_charge_error":float(np.mean([abs(m["relative_signed_charge_error"]) for m in metrics])),
            "maximum_negative_fraction":max(m["negative_density_integral_fraction"] for m in metrics),
            "maximum_stored_matrix_vs_float64_eval_difference":max(abs(r["optimization"]["lbfgs"][-1]["fit_nmae"]-r["metrics"]["oracle"]["full_grid_nmae"]) for r in selected),
            "largest_final_polish_improvement":max(r["optimization"]["lbfgs"][-2]["fit_nmae"]-r["optimization"]["lbfgs"][-1]["fit_nmae"] for r in selected),
            "maximum_numerical_dual_feasibility_residual":max(r["optimization"]["dual_diagnostic"]["max_abs_adjoint_constraint"] for r in selected),
            "maximum_numerical_primal_dual_gap":max(r["optimization"]["dual_diagnostic"]["numerical_primal_dual_gap"] for r in selected),
            "mean_radial_error_contributions":[float(np.mean([m["nearest_atom_bins"][b]["contribution_to_nmae"] for m in metrics])) for b in range(6)],
        }
    control_names = ("gnn_old_graph_old_decoder","gnn_old_graph_corrected_decoder","gnn_corrected_graph_corrected_decoder")
    for name in control_names:
        values = [grouped["current"][i]["metrics"][name]["full_grid_nmae"] for i in ids]
        summary["gnn_controls"][name] = {"mean_nmae":float(np.mean(values)),"min_nmae":min(values),"max_nmae":max(values)}
    summary["paired_comparisons"] = {}
    for j,variant in enumerate(VARIANTS[1:],start=1):
        summary["paired_comparisons"][variant] = {
            "wins_over_current":int((scores[:,j]<scores[:,0]).sum()),
            "relative_reduction_of_macro_nmae":float(1-scores[:,j].mean()/scores[:,0].mean()),
        }
    summary["capacity_match_relative_count_differences"] = {
        str(i):grouped["regular_capacity"][i]["num_coefficients"]/grouped["regular_bonds"][i]["num_coefficients"]-1
        for i in ids
    }
    if args.fixed_r16_dir:
        fixed = []
        for i in ids:
            base = grouped["regular_capacity"][i]
            if base["basis_blocks"][0]["num_radial"] == 16:
                record = base
                source = args.input_dir/f"sample_{i:04d}_regular_capacity.json"
            else:
                source = args.fixed_r16_dir/f"sample_{i:04d}_regular_r16.json"
                record = json.loads(source.read_text())
                if record["sha256"] != base["sha256"] or record["fit_probes"] != base["fit_probes"]:
                    raise ValueError(f"Fixed-R16 comparison uses different data for sample {i}")
            fixed.append({"sample_id":i,"source":str(source),
                          "num_coefficients":record["num_coefficients"],
                          "full_grid_nmae":record["metrics"]["oracle"]["full_grid_nmae"]})
        values = [r["full_grid_nmae"] for r in fixed]
        summary["fixed_r16"] = {"records":fixed,"mean_nmae":float(np.mean(values)),
                                "min_nmae":min(values),"max_nmae":max(values)}
    table = ["| 样本 | 化学式 | 当前基 | 正则径向基 | 正则基＋键中心 | 增宽原子基 |",
             "|---|---|---:|---:|---:|---:|"]
    for row,i in enumerate(ids):
        table.append(f"| {i} | {grouped['current'][i]['formula']} | "+" | ".join(f"{value:.5f}%" for value in scores[row])+" |")
    table.append("| 平均 | | "+" | ".join(f"{value:.5f}%" for value in scores.mean(0))+" |")
    (args.output_dir/"paired_table.md").write_text("\n".join(table)+"\n",encoding="utf-8")
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    colors = ["#4B647A","#168F8C","#D78B22","#7658A7"]
    fig,axes = plt.subplots(1,2,figsize=(12,4.6),layout="constrained")
    for row in scores:
        axes[0].plot(range(4),row,color="#BBC2CB",alpha=.65,lw=.8,zorder=1)
    for j,color in enumerate(colors):
        axes[0].scatter(np.full(len(ids),j),scores[:,j],color=color,s=30,zorder=3)
        axes[0].scatter(j,scores[:,j].mean(),marker="D",s=75,color=color,edgecolor="black",zorder=4)
    axes[0].set_xticks(range(4),["Current\nR12","Regular\nR12","Regular +\nbond centers","Regular\nR15/16"])
    axes[0].set_ylabel("Full-grid NMAE (%)")
    axes[0].set_title(f"Paired representation oracles: {len(ids)} local NMC structures")
    axes[0].grid(axis="y",alpha=.2)
    error = np.array([summary["variants"][v]["mean_radial_error_contributions"] for v in VARIANTS])*100
    for j,color in enumerate(colors):
        axes[1].plot(range(6),error[j],marker="o",color=color,label=LABELS[j])
    axes[1].set_xticks(range(6),["0–0.1","0.1–0.25","0.25–0.5","0.5–1","1–2",">2"])
    axes[1].set_xlabel("Distance to nearest atom (Å)")
    axes[1].set_ylabel("Contribution to total NMAE (percentage points)")
    axes[1].set_title("Where the remaining fitting error occurs")
    axes[1].legend(fontsize=8,frameon=False)
    axes[1].grid(axis="y",alpha=.2)
    fig.savefig(args.output_dir/"oracle_comparison.png",dpi=180)
    fig.savefig(args.output_dir/"oracle_comparison.pdf")
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
