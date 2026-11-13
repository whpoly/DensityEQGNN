"""Compare complete-grid GNN overfit results with the paired basis oracles."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r12",type=Path,required=True)
    parser.add_argument("--r16",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--polish-r12",type=Path)
    parser.add_argument("--polish-r16",type=Path)
    parser.add_argument("--report-date",help="Explicit user-local completion date")
    args = parser.parse_args()
    folders = [args.r12,args.r16]
    results = [json.loads((d/"result.json").read_text()) for d in folders]
    settings = [json.loads((d/"settings.json").read_text()) for d in folders]
    histories = [json.loads((d/"history.json").read_text()) for d in folders]
    adam_results = results
    polish_histories = None
    if bool(args.polish_r12) != bool(args.polish_r16):
        raise ValueError("Supply both polishing directories")
    if args.polish_r12:
        polish_folders = [args.polish_r12,args.polish_r16]
        polish_settings = [json.loads((d/"settings.json").read_text()) for d in polish_folders]
        for key in ("rounds","iterations","probes","epsilon","seed"):
            if polish_settings[0][key] != polish_settings[1][key]:
                raise ValueError(f"Polishing settings differ: {key}")
        results = [json.loads((d/"result.json").read_text()) for d in polish_folders]
        polish_histories = [json.loads((d/"history.json").read_text()) for d in polish_folders]
    if results[0]["initialization"]["trunk_sha256"] != results[1]["initialization"]["trunk_sha256"]:
        raise ValueError("Initial backbone weights differ")
    for key in ("steps","probes","seed","samples","lr","min_lr","eval_every"):
        if settings[0][key] != settings[1][key]:
            raise ValueError(f"Training settings differ: {key}")
    trunks = [{k:v for k,v in s["model_config"].items() if k!="num_radial"} for s in settings]
    if trunks[0] != trunks[1]:
        raise ValueError("Backbone or non-radial model configuration differs")
    sources = [{r["sample_id"]:r["sha256"] for r in json.loads((d/"sources.json").read_text())}
               for d in folders]
    if sources[0] != sources[1]:
        raise ValueError("Underlying DFT labels differ")
    records = [{r["sample_id"]:r for r in result["records"]} for result in results]
    if set(records[0]) != set(records[1]):
        raise ValueError("Comparison uses different structures")
    ids = sorted(records[0])
    output = args.output_dir
    output.mkdir(parents=True,exist_ok=True)
    summary = {"sample_ids":ids,"initial_backbone_identical":True,
               "interpretation":"Full-grid training-set overfit, not generalization; finite-budget GNN and oracle fits",
               "adam_results":{v:{k:x for k,x in r.items() if k!="records"}
                               for v,r in zip(("r12","r16"),adam_results)},
               "results":{variant:{k:v for k,v in result.items() if k!="records"}
                          for variant,result in zip(("r12","r16"),results)}}
    summary["r16_better_sample_count"] = sum(records[1][i]["full_grid_nmae"]<records[0][i]["full_grid_nmae"] for i in ids)
    summary["r16_relative_macro_improvement"] = 1-results[1]["macro_nmae"]/results[0]["macro_nmae"]
    (output/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    table = ["| 样本 | R12 GNN | R12 oracle | 倍数 | R16 GNN | R16 oracle | 倍数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for i in ids:
        values=[]
        for rows in records:
            r=rows[i]
            values += [f"{100*r['full_grid_nmae']:.5f}%",f"{100*r['oracle_nmae']:.5f}%",f"{r['ratio_to_oracle']:.2f}×"]
        table.append(f"| {i} | "+" | ".join(values)+" |")
    values=[]
    for result in results:
        values += [f"{100*result['macro_nmae']:.5f}%",f"{100*result['oracle_macro_nmae']:.5f}%",f"{result['ratio_to_oracle_macro']:.2f}×"]
    table.append("| 平均 | "+" | ".join(values)+" |")
    (output/"paired_table.md").write_text("\n".join(table)+"\n",encoding="utf-8")
    colors=["#336C94","#8559A5"]
    fig,axes=plt.subplots(1,3 if polish_histories else 2,
                          figsize=(16 if polish_histories else 12,4.7),layout="constrained")
    for variant,history,result,color in zip(("R12 current","R16 regular"),histories,results,colors):
        axes[0].plot([r["step"] for r in history],[r["macro_nmae"]*100 for r in history],
                     marker="o",ms=4,color=color,label=f"{variant} GNN")
        axes[0].axhline(result["oracle_macro_nmae"]*100,color=color,ls="--",label=f"{variant} oracle")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Optimizer steps")
    axes[0].set_ylabel("Full-grid training macro NMAE (%)")
    axes[0].set_title("Learning curves on 8 local NMC structures")
    axes[0].legend(fontsize=8,frameon=False)
    axes[0].grid(alpha=.2)
    if polish_histories:
        for label,history,result,color in zip(("R12 current","R16 regular"),polish_histories,results,colors):
            axes[1].plot([r["round"] for r in history],[r["macro_nmae"]*100 for r in history],
                         color=color,marker="o",label=label)
            axes[1].axhline(result["oracle_macro_nmae"]*100,color=color,ls="--")
        axes[1].set_xlabel("Full-batch L-BFGS round (0 = Adam best)")
        axes[1].set_ylabel("Full-grid training macro NMAE (%)")
        axes[1].set_yscale("log")
        axes[1].set_title("Effect of lower-noise optimization")
        axes[1].legend(fontsize=8,frameon=False)
        axes[1].grid(alpha=.2)
    for j,(rows,color,label) in enumerate(zip(records,colors,("R12 current","R16 regular"))):
        axes[-1].bar(np.arange(len(ids))+(j-.5)*.36,[rows[i]["ratio_to_oracle"] for i in ids],
                    width=.36,color=color,label=label)
    axes[-1].axhline(1,color="#333333",ls="--",lw=1,label="Oracle")
    axes[-1].set_xticks(range(len(ids)),ids)
    axes[-1].set_xlabel("Local NMC sample")
    axes[-1].set_ylabel("GNN NMAE / corresponding oracle NMAE")
    axes[-1].set_title("Remaining coefficient-prediction gap")
    axes[-1].legend(fontsize=8,frameon=False)
    axes[-1].grid(axis="y",alpha=.2)
    fig.savefig(output/"overfit_comparison.png",dpi=180)
    fig.savefig(output/"overfit_comparison.pdf")
    plt.close(fig)
    phase_table = ["| 方案 | Adam 后 | 精修后选出的模型 | 对应 oracle | 与 oracle 的差值 | 倍数 |",
                   "|---|---:|---:|---:|---:|---:|"]
    for label,adam,result in zip(("当前 R12＋白化","正则 R16＋白化"),adam_results,results):
        phase_table.append(f"| {label} | {100*adam['macro_nmae']:.5f}% | {100*result['macro_nmae']:.5f}% | "
                           f"{100*result['oracle_macro_nmae']:.5f}% | {100*result['macro_excess_nmae']:.5f} 个百分点 | "
                           f"{result['ratio_to_oracle_macro']:.2f}× |")
    physical_table = ["| 方案 | 平均绝对相对电子数误差 | 最大负密度积分比例 |",
                      "|---|---:|---:|"]
    for label,result in zip(("R12","R16"),results):
        physical_table.append(f"| {label} | {100*np.mean([abs(r['relative_signed_charge_error']) for r in result['records']]):.5f}% | "
                              f"{100*max(r['negative_density_integral_fraction'] for r in result['records']):.5f}% |")
    radial_table = ["| 最近原子距离（Å） | R12 对总 NMAE 的贡献（百分点） | R16 对总 NMAE 的贡献（百分点） |",
                    "|---|---:|---:|"]
    radial = [np.mean([r['radial_error_contributions'] for r in result['records']],axis=0)
              for result in results] if all('radial_error_contributions' in r for result in results for r in result['records']) else None
    if radial is not None:
        for i,label in enumerate(("0–0.1","0.1–0.25","0.25–0.5","0.5–1","1–2",">2")):
            radial_table.append(f"| {label} | {radial[0][i]*100:.5f} | {radial[1][i]*100:.5f} |")
    audit_path = output/"low_norm_oracle_audit.json"
    audit_text = ""
    if audit_path.exists():
        audit = json.loads(audit_path.read_text())
        audit_text = "还复核了样本 1 的较小系数范数解：\n\n| 表示 | 自由拟合系数 RMS | 新解码器完整网格 NMAE | 当前 GNN 系数 RMS | 当前 GNN NMAE |\n|---|---:|---:|---:|---:|\n"
        for label,a,rows in zip(("R12","R16"),audit,records):
            audit_text += f"| {label} | {a['coefficient_rms']:.5f} | {a['production_decoder_nmae']*100:.5f}% | {rows[1]['coefficient_rms']:.5f} | {rows[1]['full_grid_nmae']*100:.5f}% |\n"
        audit_text += "\n这些来自此前完整网格 CGLS/L-BFGS pilot 的可行系数，本次用新的生产解码器重新评价，未拿来监督 GNN。它们不是带范数约束的最优解，却足以说明样本 1 的大幅差距不能只用‘低误差需要巨大系数’解释。主 oracle 中进一步降低误差确实可能使用较大系数，但那不是当前百分之几误差的完整解释。\n"
    root = Path.cwd().as_posix()
    chart = (output/"overfit_comparison.png").resolve().as_posix()
    selected = [args.polish_r12 or args.r12,args.polish_r16 or args.r16]
    checkpoints = "\n".join(f"- [{label} 最佳 checkpoint](<{(folder/'best.pt').resolve().as_posix()}>)，[{label} 逐结构结果](<{(folder/'result.json').resolve().as_posix()}>)。" for label,folder in zip(("R12","R16"),selected))
    report = f"""# 本地 NMC：正则 R16 解码器与两组从头过拟合对照

{('完成日期：' + args.report_date + '。') if args.report_date else ''}

本轮只使用本地 8 个原始 NMC 结构，完成两组从头训练及相同的后续精修，没有启动完整 NMC 训练。稳定径向白化和正则 R16 已接入生产解码器；完整网格结果表明，两组 GNN 仍远高于各自已实现的 oracle 精度，尚未通过此前约 1.2×oracle 的小集合过拟合门槛。

**结果与判断**

{chr(10).join(phase_table)}

所有数值均为每结构完整网格 NMAE 的宏平均，单位为百分数。原始 JSON 保存比值。差值只是 GNN 超过已找到 oracle 的误差，不能当作严格可加的物理误差分解。oracle 是可行拟合误差上界，神经网络也只经过有限预算、单个种子的训练；这不是对任一模型理论误差下限的证明。

R16 的较低 oracle 误差尚未转化为接近对应 oracle 的 GNN 精度。当前不应仅依据 oracle 排名就确定完整训练采用 R16，也不应直接推断‘必须扩大骨干’。短程全结构梯度精修的收益有限，说明仅降低这部分更新噪声尚未弥合差距；更充分优化、输出参数化和骨干/读出表达能力仍需分开检验。

在本次单种子配对中，R16 的 GNN 宏平均误差比 R12 低 {100*summary['r16_relative_macro_improvement']:.2f}%，{summary['r16_better_sample_count']}/8 个结构更低；这远小于此前 oracle 宏平均约 45.55% 的下降，不能混用两者的改善幅度。

![完整网格学习曲线与 oracle 差距](<{chart}>)

左图为 Adam 的完整网格检查，中图为每轮 L-BFGS 后的完整网格检查；虚线是对应 oracle。右图使用经完整网格选择的最佳模型。

{chr(10).join(table)}

**配对实验如何控制**

- 数据为 `data/nmc_cache/0.CHGCAR.lz4` 至 `7.CHGCAR.lz4` 的原始 DFT `total` 密度。每个结构 56×56×140＝439,040 点，共 3,512,320 点；源文件 SHA256 与 oracle 逐个匹配。
- R12 保留原 Gaussian 径向函数空间，只做可逆白化。R16 使用 `(r/σ)^l Gaussian × cosine cutoff`，再做白化。两者 l≤4、密度 cutoff 3 Å、最小尺度 0.09375 Å；均累加全部周期像。
- 骨干完全相同：hidden l≤2，readout edge l≤2，3 个 interaction，scalar/vector/tensor 通道 24/12/6，图 cutoff 6 Å。初始骨干权重 SHA256 两组完全一致。参数数分别为 477,148 与 521,472，差异来自径向输出及相应读出/原子基线通道。
- seed＝42，从随机参数与零原子基线开始；没有加载旧模型、没有 oracle 系数 MSE，也没有把 oracle 系数用于初始化。两组使用相同的结构顺序和网格采样序列。不做旋转数据增强，模型的等变性由结构和测试保证。
- Adam 共 4,000 次更新，每次一个结构、从其全部网格均匀有放回抽 8,192 点；每个结构更新 500 次。学习率从 1e-3 余弦衰减至 1e-5，无 weight decay，梯度范数截断为 5。损失使用完整网格的密度均值作固定归一化，避免随机分母带来的目标偏差。
- 每 500 次更新完整评价全部 8 个结构，用训练集完整网格 macro NMAE 选择 checkpoint。之后两组都做 3 轮 L-BFGS，每轮最多 80 次迭代、每结构固定一组新的 16,384 点、8 个结构共同产生梯度，平滑尺度 1e-5。这里的“全结构梯度”不等于在全部网格上计算梯度；模型选择与最终比较才遍历全部网格。
- 后续精修保留 Adam 最佳模型作为候选。最终重新计算原子图，并绕过密度几何缓存，用生产解码器逐块评价所有网格；没有电荷重归一化或负密度截断。

精修的采样设计矩阵先在 float64 中构造已白化特征，再以 float32 缓存并参与网络优化；模型选择及最终结果均通过 float64 生产解码器复核。

本次是训练集过拟合诊断，不是泛化成绩。8 个样本在官方划分中的归属尚未核实，不用它们报告独立测试结果；当前只有配对的一个随机种子，不据此宣称稳健的架构优劣。

**残差位置与系数诊断**

{chr(10).join(radial_table) if radial is not None else '当前结果未包含距离分解。'}

{chr(10).join(physical_table)}

距离区间按最近原子划分，不等于化学键区间，也不能单独分离径向与角向误差。这组分解用来决定后续采样和监督诊断，避免仅凭近核直觉调整采样。

{audit_text}

**解码器实现与兼容性**

新增 `regular_radial`、`radial_whitening`、`radial_minimum` 三个基配置字段，默认保持旧函数和旧系数坐标。径向变换按 `r² dr` 度量做 SVD，保留所有方向；R16 原始径向矩阵条件数最高约 4.85×10⁸，因此禁用该步骤的 autocast，并在 float64 中计算变换和密度求和。网络仍可使用 float32。

正则版本用未归一化的 solid harmonics 与相应的径向因子相乘，既等价于 `r^l Y_lm`，又避免方向归一化 epsilon 破坏精确原点的一阶导数。白化只混合同一 l 的径向通道，对所有 m 使用相同变换，保持等变性。径向白化并不等于对跨原子重叠后的整个晶体基做全局白化。

系数布局继续使用 radial-major。checkpoint 保存完整基配置和实际白化矩阵；同形状但不同函数/坐标的 checkpoint 不允许静默续训，主要 compact 评估入口也校验变换。旧原始系数标签不会被当成新基的监督或 LS 参考。固定网格缓存只保存几何，用于本次位置固定的实验；需要位置/晶胞梯度时使用普通解码器。

测试覆盖新解码器与独立 oracle 的值及伴随一致、白化正交性和 float32 失精度对照、精确原点 l=1 导数、位置/晶胞梯度、mask/空邻居、单原子斜晶胞、原胞/超胞、旋转、重复采样点及 checkpoint 坐标保护。全套 **39 项测试通过**。样本 1 的较小范数 oracle 又经生产解码器完整网格复核，R12 与既有数值一致，R16 仅有浮点量级差别。

**下一步应先定位读出与优化，而非启动完整训练**

建议仍留在这 8 个结构上，冻结已学骨干与读出中的非线性特征，直接校准现有线性输出参数和元素球对称基线，并继续使用密度目标及完整网格评价。若这一步能明显逼近 oracle，应优先改初始化、分阶段训练和输出优化；若线性校准仍明显受限，再对骨干/读出特征做受控消融。需要把有限求解收敛程度一并记录，不能以一次校准失败证明架构不可行。

当前没有证据支持直接投入完整训练、提高 l 或开展效率比较。R16 的表示潜力保留，但它是否是更好的可学习参数化，尚未由这轮训练证明。后续优化方案通过小集合门槛后，再重复随机种子、核实正式数据划分。

**产物与复现**

{checkpoints}

实现为 [compact_basis.py](<{root}/src/density_eqgnn/compact_basis.py>)、[radial_basis.py](<{root}/src/density_eqgnn/radial_basis.py>)；训练、精修及汇总入口分别为 [train_local_nmc_overfit.py](<{root}/scripts/train_local_nmc_overfit.py>)、[polish_local_nmc_overfit.py](<{root}/scripts/polish_local_nmc_overfit.py>)、[summarize_local_nmc_overfit.py](<{root}/scripts/summarize_local_nmc_overfit.py>)。完整命令保存在 README，所有参数、数据 SHA256、逐次完整网格历史及 checkpoint 均保留在各实验目录。

报告目录另保存 `manifest.json`、`source_snapshot.zip` 与 `pytest.log`，记录交付时的源码、运行环境、结果及 checkpoint 哈希。源码快照是报告完成时的版本，并非各进程启动瞬间的逐位快照；可复现实验方法，不承诺 GPU 训练逐位相同。
"""
    (output/"report.md").write_text(report,encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
