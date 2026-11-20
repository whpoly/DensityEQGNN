"""Reproduce the recent validation figure without loading or changing a model."""
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "runs/nmc_eden_fullgrid/probe_adamw_l1_seed42/validation.json"
END = 275_000
raw = SOURCE.read_bytes()
history = [r for r in json.loads(raw.decode("utf-8-sig")) if r["step"] <= END]
assert history[-1]["step"] == END
rows = [r for r in history if r["step"] >= 200_000]
before = [r for r in history if 225_000 <= r["step"] <= 240_000]
recent = [r for r in history if 260_000 <= r["step"] <= END]
summary = {}
for key in ("density_nmae", "occupation_relative_l1"):
    a, b = (statistics.mean(r[key] for r in window) for window in (before, recent))
    summary[key] = dict(before_mean_pct=100*a, recent_mean_pct=100*b,
                        relative_reduction_pct=100*(1-b/a))

font_manager.fontManager.addfont("C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                     "font.size": 10, "pdf.fonttype": 42, "savefig.facecolor": "#f8fafc"})
fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.9), facecolor="#f8fafc")
fig.subplots_adjust(left=.075, right=.975, top=.77, bottom=.23, wspace=.24)
fig.text(.065, .965, "纯探针 EdenGNN：最近验证变化", fontsize=18, weight="bold", va="top", color="#17243b")
fig.text(.065, .895, "同一批 50 个验证结构 · 密度：固定 5,000 探针/结构 · PAW：全部有效系数 · 越低越好",
         fontsize=10.5, color="#536173")
fig.text(.065, .84, "第 24 万步验证后，学习率由 0.005 降为 0.0025", fontsize=10.5, color="#a65d16")
for ax, key, title, color in zip(axes, ("density_nmae", "occupation_relative_l1"),
                                ("密度 NMAE（%）", "PAW 占据相对 L1（%）"),
                                ("#2666b0", "#168978")):
    x = [r["step"] / 10_000 for r in rows]
    y = [100*r[key] for r in rows]
    envelope = [100*min(r[key] for r in history if r["step"] <= end["step"]) for end in rows]
    ax.set_facecolor("white")
    ax.axvspan(24, 28.1, color="#fff0d9", alpha=.45, zorder=0)
    ax.axvline(24, color="#b97429", ls=":", lw=1.4)
    ax.plot(x, y, color=color, marker="o", ms=4, lw=1.7, label="每次验证", zorder=3)
    ax.step(x, envelope, where="post", color=color, alpha=.5, ls="--", lw=1.2, label="截至该步的历史最佳")
    best = min(history, key=lambda r:r[key])
    ax.scatter([best["step"]/10_000], [best[key]*100], marker="*", s=105, color=color,
               edgecolors="white", linewidths=.6, zorder=5)
    ax.text(.97, .95, f"最新：{y[-1]:.5f}%", transform=ax.transAxes, ha="right", va="top",
            fontsize=11, weight="bold", color=color)
    ax.annotate(f"最佳 {best[key]*100:.5f}%\n第 {best['step']/10_000:g} 万步",
                (best["step"]/10_000, best[key]*100), xytext=(-5, -18), textcoords="offset points",
                ha="center", va="top", fontsize=9, color=color)
    span = max(y)-min(y)
    ax.set_ylim(min(y)-span*.38, max(y)+span*.28)
    ax.set_xlim(19.7, 28.1)
    ax.set_title(title, loc="left", fontsize=12, weight="bold", color="#24364b", pad=12)
    ax.set_xlabel("训练步数（万步）", labelpad=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
    ax.grid(axis="y", color="#e2e8ef", lw=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#c9d3de")
    ax.tick_params(color="#a5b1bf", labelcolor="#536173")
    ax.legend(loc="lower left", fontsize=8.5, frameon=False)
fig.text(.065, .10, "降学习率前 4 次（22.5–24 万步）→最近 4 次（26–27.5 万步）的验证均值：", color="#536173")
fig.text(.065, .045, f"密度误差相对下降 {summary['density_nmae']['relative_reduction_pct']:.2f}%    ·    "
         f"PAW 误差相对下降 {summary['occupation_relative_l1']['relative_reduction_pct']:.2f}%    ·    此图仅为验证集结果",
         fontsize=10.5, color="#24364b")
prefix = Path(__file__).parent / f"recent_validation_step{END}"
fig.savefig(prefix.with_suffix(".png"), dpi=170)
fig.savefig(prefix.with_suffix(".pdf"))
plt.close(fig)
prefix.with_suffix(".json").write_text(json.dumps(dict(source=str(SOURCE), source_sha256=hashlib.sha256(raw).hexdigest(),
    endpoint_step=END, validation_structures=50, density_probes_per_structure=5000,
    relative_errors_in_rows="ratios; plotted values multiplied by 100", rows=rows,
    window_comparison=dict(before_steps=[r["step"] for r in before], recent_steps=[r["step"] for r in recent],
                           metrics=summary)), indent=2), encoding="utf-8")
print(json.dumps({"png": str(prefix.with_suffix('.png')), "pdf": str(prefix.with_suffix('.pdf')), "summary":summary}))
