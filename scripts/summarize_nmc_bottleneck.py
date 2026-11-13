"""Reconcile final held-out results, training provenance and the readout ablation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter, StrMethodFormatter
import numpy as np

from scripts.fit_nmc_holdout_oracle import ROOT, record
from scripts.nmc_probe_holdout import DEFAULT_ROOT, sha256


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--query",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/nmc_bottleneck_2026-10-27"))
    args=p.parse_args()
    query=args.query or args.root/"query_eight"
    oracle=[read(args.root/"oracle"/f"sample_{i:04d}"/"result.json") for i in range(8)]
    oracle_fits=[read(args.root/"oracle"/f"sample_{i:04d}"/"fit_complete.json") for i in range(8)]
    oracle_diagnostic=[]
    for i in range(8):
        h=read(args.root/"oracle"/f"sample_{i:04d}"/"history.json")
        best=min(h,key=lambda r:r["selection_nmae"])
        oracle_diagnostic.append({"sample_id":i,
            "ls_train_nmae":h[0]["train_nmae"],"ls_selection_nmae":h[0]["selection_nmae"],
            "ls_coefficient_rms":h[0]["coefficient_rms"],
            "best_selection_nmae":best["selection_nmae"],"best_coefficient_rms":best["coefficient_rms"]})
    record(args.root/"oracle_optimization_diagnostic.json",{
        "scope":"training and selection-validation only","records":oracle_diagnostic})
    control=DEFAULT_ROOT/"r16_eight"
    official=DEFAULT_ROOT/"deepdft_eight"
    assert read(control/"sources.json")==read(official/"sources.json")
    models={"coefficient":read(control/"result.json"),"query":read(query/"result.json"),
            "deepdft":read(official/"result.json")}
    refined=args.root/"query_eight_refine"
    models["refined"]=read(refined/"result.json")
    frozen=read(args.root/"frozen_selection.json")
    refinement_decision=read(args.root/"refinement_decision.json")
    selected="refined" if frozen["selected_query"]=="query_eight_refine" else "query"
    models["oracle"]={"records":oracle,**{key+"_macro_nmae":float(np.mean([r[key+"_nmae"] for r in oracle]))
                                          for key in ("train","validation","test")}}
    settings=read(query/"settings.json")
    audit=read(args.root/"paired_control_audit.json")
    assert audit["identical_actual_sample_coverage"]
    production_audit=read(args.root/"production_source_audit.json")
    assert production_audit["all_unchanged"]
    for row in production_audit["files"]: assert sha256(row["path"])==row["sha256"]
    for path,digest in settings["source_hashes"].items(): assert sha256(path)==digest,path
    assert sha256(query/"best.pt")==models["query"]["checkpoint_sha256"]
    assert sha256(control/"best.pt")==audit["control_checkpoint_sha256"]
    assert sha256(args.root/"refinement_decision.json")==frozen["refinement_decision_sha256"]
    refined_settings=read(refined/"settings.json")
    assert refined_settings["parent_sha256"]==sha256(query/"best.pt")
    for path,digest in refined_settings["source_hashes"].items(): assert sha256(path)==digest,path
    assert sha256(refined/"best.pt")==models["refined"]["checkpoint_sha256"]
    for name,folder in (("query",query),("refined",refined)):
        assert sha256(folder/"best.pt")==frozen["candidates"][folder.name]["checkpoint_sha256"]
        assert models[name]["best_step"]==frozen["candidates"][folder.name]["best_step"]
    for i,result in enumerate(oracle):
        source=read(args.root/"oracle"/f"sample_{i:04d}"/"settings.json")
        assert result["split"]==source["split"]
        assert result["validation_points_used_for_fit"]==result["test_points_used_for_fit"]==0
        assert result["checkpoint_sha256"]==sha256(args.root/"oracle"/f"sample_{i:04d}"/"best.pt")
    for name in ("query","refined"):
        for r in models[name]["sampling_audit"]:
            assert r["validation_points_used_for_gradients"]==r["test_points_used_for_gradients"]==0
    history=read(query/"history.json")
    assert models["query"]["best_step"]==min(history,key=lambda r:r["selection_macro_nmae"])["step"]
    args.output.mkdir(parents=True,exist_ok=True)
    order=("oracle","coefficient","query","refined","deepdft")
    labels={"oracle":"R16 自由系数（逐结构拟合）","coefficient":"共同骨干＋R16 系数读出",
            "query":"共同骨干＋查询点读出（16k，主对照）",
            "refined":"查询点读出＋额外精修（非等预算）","deepdft":"官方等变 DeepDFT（参照）"}
    table=["| 方案 | 训练点 NMAE | 验证点 NMAE | 测试点 NMAE |","|---|---:|---:|---:|"]
    for name in order:
        values=[f"{100*models[name][key+'_macro_nmae']:.5f}%" for key in ("train","validation","test")]
        table.append("| "+" | ".join([labels[name],*values])+" |")
    paired=["| 样本 | R16 自由系数 | 系数读出 | 查询点读出 | 查询点精修 | 官方 DeepDFT |",
            "|---|---:|---:|---:|---:|---:|"]
    for i in range(8):
        values=[]
        for name in order:
            row=next(r for r in models[name]["records"] if r["sample_id"]==i)
            values.append(f"{100*row['test_nmae']:.5f}%")
        paired.append("| "+" | ".join([str(i),*values])+" |")
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.8),constrained_layout=True)
    colors={"oracle":"#238b45","coefficient":"#c45122","query":"#1261a0",
            "refined":"#009e9a","deepdft":"#8b66b3"}
    english={"oracle":"Free R16 coefficients","coefficient":"Shared backbone / coefficient head",
             "query":"Shared backbone / query head (16k)",
             "refined":"Query + extra refinement (4k, 4x probes)","deepdft":"Official equivariant DeepDFT"}
    for name,folder in (("coefficient",control),("query",query),("deepdft",official)):
        h=[r for r in read(folder/"history.json") if r["step"]>1]
        axes[0].plot([r["step"] for r in h],[r["selection_macro_nmae"]*100 for r in h],
                     label=english[name],color=colors[name],linewidth=1.8)
    h=read(refined/"history.json")
    axes[0].plot([16000]+[16000+r["step"] for r in h],
                 [100*min(history,key=lambda r:r["selection_macro_nmae"])["selection_macro_nmae"]]+
                 [100*r["selection_macro_nmae"] for r in h],
                 label=english["refined"],color=colors["refined"],linestyle="--",linewidth=1.8)
    axes[0].axvline(16000,color="#888888",linestyle=":",alpha=.7)
    oracle_selection=100*np.mean([r["selection_nmae"] for r in oracle_fits])
    axes[0].axhline(oracle_selection,color=colors["oracle"],linestyle="--",label=english["oracle"]+" (validation)")
    axes[0].set(title="Readout comparison and separate refinement",xlabel="Optimizer steps (refinement uses 4x probes / step)",ylabel="Validation NMAE (%, log scale)",yscale="log")
    axes[0].set_xticks([0,4000,8000,12000,16000,20000],labels=["0","4k","8k","12k","16k","20k"])
    for name in order:
        rows=sorted(models[name]["records"],key=lambda r:r["sample_id"])
        axes[1].plot(range(8),[100*r["test_nmae"] for r in rows],"s--" if name=="refined" else "o-",
                     color=colors[name],label=english[name],linewidth=1.2,markersize=4)
    axes[1].axhline(.3,color="#777777",linestyle=":",label="0.3% diagnostic target")
    axes[1].set(title="Held-out spatial points",xlabel="NMC sample ID",ylabel="Test NMAE (%, log scale)",yscale="log",xticks=range(8))
    for ax in axes:
        ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,3,5)))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.2)
    for ax in axes:
        ax.legend(loc="upper center",bbox_to_anchor=(.5,-.18),ncol=2,
                  fontsize=7.3,frameon=False,columnspacing=1.)
    fig.savefig(args.output/"comparison.png",dpi=180)
    fig.savefig(args.output/"comparison.pdf")
    plt.close(fig)
    q=models["query"]["test_macro_nmae"]
    c=models["coefficient"]["test_macro_nmae"]
    o=models["oracle"]["test_macro_nmae"]
    final=models[selected]["test_macro_nmae"]
    refined_test=models["refined"]["test_macro_nmae"]
    reached="达到" if final<=.003 else "尚未达到"
    conclusion=("查询点读出在相同骨干与接近的参数预算下明显缩小了差距，支持优先研究系数输出与密度重构这条路径的学习瓶颈。"
                if q<c*.5 else "本轮查询头没有显著消除差距，需要保留骨干、读出结构和优化过程共同受限的解释，不能把它当作查询点方法本身的精度极限。")
    ls_selection=100*np.mean([r["ls_selection_nmae"] for r in oracle_diagnostic])
    selected_train=models[selected]["train_macro_nmae"]
    train_test_relative=abs(selected_train-final)/final
    fit_interpretation=("最终查询模型的训练与测试误差相近，当前差距更符合拟合能力或优化不足，而非明显的空间留出过拟合。"
        if train_test_relative<.1 else "训练与测试误差存在差距，需要结合更多划分与随机种子判断拟合不足和空间泛化各自的作用。")
    report=f"""# NMC 表示与读出瓶颈诊断

实验始于 2026-10-27，2026-10-28 完成最终测试评价。目录日期保留实验起始日期。

本轮沿用既有空间划分，先完成 8 个结构的 R16 自由系数拟合，再训练同骨干的查询点读出，
最后按验证结果安排一次有固定预算的精修。

{chr(10).join(table)}

指标是逐结构 `sum(abs(pred-target))/sum(abs(target))` 后宏平均，表中为百分数。
自由系数使用每个结构单独的一组系数，属于表示能力诊断，不是可从未知结构预测的模型。

## 结果解释

R16 自由系数测试误差为 **{100*o:.5f}%**；同骨干查询点模型为 **{100*q:.5f}%**，
相对于系数模型变化为 **{100*(1-q/c):.2f}% 的误差下降**。
{conclusion}

额外精修的测试误差为 **{100*refined_test:.5f}%**。仅按验证结果选定的最终查询模型为
“{labels[selected]}”，测试误差 **{100*final:.5f}%**，{reached}约 0.3% 的阶段目标。
精修在主实验之外增加 4,000 步，每步 4,096 个训练点，总采样预算增加 16,384,000 个点，
等于主实验的采样总量；不能将精修结果归入等训练预算对照。

{fit_interpretation}
这次实验仍不能把查询模型与 DeepDFT 的剩余差距单独归因于骨干：两者的通道数、
查询头和参数量也不同，需要进一步控制这些因素才能定位。

## 对系数路线的意义

R16 带 1e-9 稳定项的最小二乘初始化尚未做 L1 精修时，固定验证子集的宏平均误差已为 **{ls_selection:.5f}%**，
8 个结构均低于 0.3%。此时系数 RMS 为
{min(r['ls_coefficient_rms'] for r in oracle_diagnostic):.3f}–{max(r['ls_coefficient_rms'] for r in oracle_diagnostic):.3f}；
选择出的 L1 解系数 RMS 为
{min(r['best_coefficient_rms'] for r in oracle_diagnostic):.3f}–{max(r['best_coefficient_rms'] for r in oracle_diagnostic):.3f}。
因此，低于 0.3% 的表示能力并不依赖后期出现的大系数。

如果继续保留系数输出路线，下一项值得检验的是用仅由训练点拟合的较小范数最小二乘系数作预训练目标，
随后按重构密度误差微调，并比较密度度量下的系数监督与逐元素系数 MSE。
这是后续实验建议，本轮没有运行，也不保证能够达到 0.3%；单纯继续增加径向基数缺少本轮证据支持。

这些误差不能作为可加的“表示误差＋网络误差”分解。自由系数结果是实际找到的可行解，
不是最优误差的认证下界。查询模型是本项目新增的读出，官方 DeepDFT 仅作为参照。

![验证曲线与留出点结果](comparison.png)

下表均为最终测试分区：

{chr(10).join(paired)}

## 控制条件与泄漏检查

- 数据为原来的样本 0–7，每个 439,040 点；351,232 训练、43,904 验证、43,904 测试。
  原数据与划分 SHA256 保持一致，原固定的 8,192 个验证点用于选择模型。
- 自由系数的设计矩阵、归一化、Gram 预条件、最小二乘初始化及 L1 优化仅使用训练分区。
  依次使用 1e-3、1e-4、1e-5 平滑尺度，各最多 75 个 L-BFGS 内部迭代预算；每 25 步检查验证误差。
  没有导入历史完整网格 oracle 系数。Gram 中 1e-9 稳定项用于初始化求解和预条件，后续 L1 目标无 ridge 惩罚。
- 系数采用相同 R16、l≤4、3 Å、原点正则及径向白化。缓存用 float32 存储，Gram、优化变量和精确乘积用 float64；
  选出系数后通过独立的生产解码器重新评价验证点和最终三个分区。
- 两个读出的原子骨干均为原来 3 层、24/12/6 通道、6 Å 周期图，初始权重逐位相同，
  两者都只读取最后一层原子特征。端到端训练后骨干权重可以不同，并非冻结同一套已训练特征。
- 系数臂为 {audit['control_parameters']:,} 参数，查询臂为 {audit['query_parameters']:,} 参数，
  共同骨干为 {audit['backbone_parameters']:,} 参数。查询头为 3 层等变原子到查询点消息，
  采用 3 Å 平滑支持、不变标量/范数读出，不含查询点之间的消息。
- 两臂从头训练，均为 seed=42、16,000 步、每步 1,024 点、Adam、L1、学习率 1e-3 余弦到 1e-5、
  梯度范数截断 5、无 weight decay。结构顺序和训练点采样 RNG 相同，最终实际采样覆盖逐结构一致。
- 主实验查询验证误差从 14k 的 0.75474% 降至 16k 的 0.74369%，因此安排一次更大批量的低学习率精修。
  预算与选择规则在 `refinement_decision.json` 中提前固定：父模型为主查询臂的最佳权重，
  额外 4,000 步、每步 4,096 点、全参数 Adam、学习率 1e-4 余弦到 1e-6、seed=42。
  最佳精修步数为 {models['refined']['best_step']}；在父模型与精修检查点中仅按原验证子集选取。
  选择在 `{frozen['frozen_at_utc']}` 冻结，之后才评价本轮测试点，未按测试结果继续调整。
- 系数臂复用此前完全相同的 `r16_eight` 结果。新封装与生产前向/骨干梯度逐位一致，
  初始完整状态哈希与旧实验一致。`paired_control_audit.json` 记录数据、配置与权重检查。
  此前报告归档中的 {len(production_audit['files'])} 个生产源文件与当前文件逐字节一致。
- 官方 DeepDFT 为 1,487,233 参数、4 Å 邻域，骨干不同，未并入“只改变读出”的受控结论。
- 测试点没有参与拟合、归一化、检查点选择或本轮改动决策；先保存训练结束记录，再单独运行最终评价。
  同一测试分区已在前一轮公开过，因此本轮结果是已有基准上的描述性比较，尚需新独立数据确认。
- 所有结果不做电荷重归一化或负密度截断。这里只检验已知结构内随机散布网格点的插值，
  不代表新结构、连续离网格点或大块空间空缺的泛化。
- 本轮只有一个随机种子和一组空间划分，没有据此作统计显著性结论。

## 复现

使用本项目 `kan-cgcnn-cuda` Python 环境，运行 `scripts/run_nmc_bottleneck.ps1` 完成拟合与查询训练。
然后运行 `powershell -ExecutionPolicy Bypass -File scripts/run_nmc_bottleneck_refine.ps1` 完成额外精修。
两次训练均完成后运行 `python -m scripts.finalize_nmc_bottleneck`，先冻结验证选择与检查点哈希，
再分别进行自由系数、主查询模型和精修模型的最终评价，并生成本报告。

运行目录为 `{args.root.as_posix()}`。各样本保存系数、验证历史和哈希，查询臂保存最佳/最后权重、
完整训练设置、采样审计与结果。本目录保存源代码快照、文件清单、汇总 JSON 与 PNG/PDF 图件。
新增测试覆盖系数布局导出、生产骨干值与梯度一致、旋转不变、重复/分批查询、梯度传播和 cutoff 连续性。
训练前的完整测试套件为 **51 passed**，日志存于运行目录的 `pytest.log`；此后未改动模型或求解器代码。
"""
    (args.output/"report.md").write_text(report,encoding="utf-8")
    record(args.output/"summary.json",{"models":models,"query_settings":settings,"paired_control_audit":audit,
        "oracle_validation_audits":oracle_fits,"oracle_optimization_diagnostic":oracle_diagnostic,
        "primary_query_test_target_reached":q<=.003,
        "selected_query":selected,"selected_query_test_target_reached":final<=.003,
        "refinement_settings":refined_settings,"refinement_decision":refinement_decision,"frozen_selection":frozen})
    files=list(Path("src").rglob("*.py"))+[Path(p) for p in (
        "scripts/fit_nmc_holdout_oracle.py","scripts/train_nmc_paired_readout.py","scripts/audit_nmc_paired_control.py",
        "scripts/summarize_nmc_bottleneck.py","scripts/run_nmc_bottleneck.ps1","scripts/nmc_probe_holdout.py",
        "scripts/finalize_nmc_bottleneck.py","scripts/run_nmc_bottleneck_refine.ps1",
        "scripts/train_nmc_probe_holdout.py","scripts/compare_local_nmc_oracles.py","scripts/train_local_nmc_overfit.py",
        "tests/test_bottleneck.py","configs/nmc_full_l2_to_l4_mace_pbc.yaml","pyproject.toml",
        "reports/nmc_bottleneck_protocol_2026-10-27.md")]
    with zipfile.ZipFile(args.output/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as z:
        for path in files: z.write(path,path.as_posix())
    files+=list((args.root/"oracle").rglob("*.json"))+list((args.root/"oracle").rglob("*.pt"))
    for folder in (query,refined):
        files+=[folder/name for name in ("best.pt","settings.json","sources.json","history.json","result.json","sampling_audit.json")]
    files+=[args.root/"refinement_decision.json",args.root/"frozen_selection.json",args.root/"production_source_audit.json",
            args.root/"oracle_optimization_diagnostic.json"]
    for folder in (control,official):
        files+=[folder/name for name in ("best.pt","result.json","settings.json","sources.json","history.json","sampling_audit.json")]
    files+=[args.root/"paired_control_audit.json",args.root/"pytest.log",Path(production_audit["reference_snapshot"])]
    files+=[args.output/name for name in ("report.md","summary.json","comparison.png","comparison.pdf","source_snapshot.zip")]
    record(args.output/"manifest.json",{"files":[{"path":str(p.resolve()),"sha256":sha256(p),"bytes":p.stat().st_size} for p in files]})
    print("\n".join(table),flush=True)


if __name__=="__main__": main()
