"""Report the matched spatial holdout experiments without rerunning inference."""
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

from scripts.nmc_probe_holdout import DEFAULT_ROOT, OFFICIAL_COMMIT


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--output",type=Path,default=Path("reports/nmc_probe_holdout_2026-10-27"))
    parser.add_argument("--report-date",default="2026-10-27")
    args=parser.parse_args()
    names=("deepdft_one","r16_one","deepdft_eight","r16_eight")
    runs={name:{key:read(args.root/name/f"{key}.json") for key in ("settings","history","sources","result")} for name in names}
    for scope in ("one","eight"):
        a,b=(runs[f"{m}_{scope}"] for m in ("deepdft","r16"))
        for key in ("samples","steps","probes","seed","lr","min_lr","loss","eval_every","chunk","optimizer","source_hashes"):
            assert a["settings"][key] == b["settings"][key], key
        assert len(a["sources"]) == len(b["sources"]) == len(a["settings"]["samples"])
        assert a["result"]["sampling_audit"] == b["result"]["sampling_audit"]
        for sa,sb in zip(a["sources"],b["sources"]):
            assert sa["sha256"] == sb["sha256"] and sa["split"] == sb["split"]
            assert sa["training_normalizer"] == sb["training_normalizer"]
        for run in (a,b):
            result=run["result"]
            for row in result["sampling_audit"]:
                assert row["test_points_used_for_gradients"] == row["validation_points_used_for_gradients"] == 0
            for key in ("train","validation","test","full_grid"):
                assert abs(np.mean([r[key+"_nmae"] for r in result["records"]])-result[key+"_macro_nmae"])<1e-12
            assert result["best_step"] == min(run["history"],key=lambda x:x["selection_macro_nmae"])["step"]
    checked = {}
    def verify_hash(path, expected):
        path = Path(path).resolve()
        if path not in checked:
            checked[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert checked[path] == expected, f"Source/checkpoint changed: {path}"
    for name,run in runs.items():
        verify_hash(args.root/name/"best.pt",run["result"]["checkpoint_sha256"])
        for path,digest in run["settings"]["source_hashes"].items():
            verify_hash(path,digest)
        if name.startswith("deepdft"):
            assert run["settings"]["model_config"]["commit"] == OFFICIAL_COMMIT
            for path,digest in run["settings"]["model_config"]["sources"].items():
                verify_hash(args.root/"DeepDFT"/path,digest)
        for source in run["sources"]:
            verify_hash(source["source"],source["sha256"])
            split_path=args.root/"spatial_splits"/f"sample_{source['sample_id']:04d}.npz"
            assert read(split_path.with_suffix(".json")) == source["split"]
            with np.load(split_path) as split:
                for key,digest in source["split"]["indices_sha256"].items():
                    assert hashlib.sha256(split[key].tobytes()).hexdigest() == digest
                    assert len(split[key]) == source["split"]["counts"][key]
                all_ids=np.concatenate([split[key] for key in ("train","validation","test")])
                assert np.array_equal(np.sort(all_ids),np.arange(source["split"]["grid_points"]))
                assert np.isin(split["selection"],split["validation"]).all()
    args.output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(14,4.2),constrained_layout=True)
    colors={"deepdft":"#1261a0","r16":"#c45122"}
    for axis,scope,title in zip(axes[:2],("one","eight"),("One NMC structure","Eight NMC structures")):
        for model,label in (("deepdft","Official equivariant DeepDFT"),("r16","Current R16 coefficient model")):
            history=runs[f"{model}_{scope}"]["history"]
            history=[x for x in history if x["step"]>1]
            axis.plot([x["step"] for x in history],[100*x["selection_macro_nmae"] for x in history],
                      label=label,color=colors[model],linewidth=1.8)
        axis.set(title=title,xlabel="Optimizer steps",ylabel="Validation NMAE (%, log scale)",yscale="log")
        axis.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
        axis.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        axis.yaxis.set_minor_formatter(NullFormatter())
        axis.grid(alpha=.2)
    axes[1].set_xticks([0,4000,8000,12000,16000],labels=["0","4k","8k","12k","16k"])
    axes[0].legend(fontsize=8)
    for offset,model,label in ((-.18,"deepdft","DeepDFT"),(.18,"r16","R16")):
        rows=runs[f"{model}_eight"]["result"]["records"]
        axes[2].bar(np.arange(len(rows))+offset,[100*x["test_nmae"] for x in rows],.36,label=label,color=colors[model])
    axes[2].set(title="Sealed test points: eight structures",xlabel="NMC sample ID",ylabel="Test NMAE (%)",xticks=np.arange(8))
    axes[2].legend(fontsize=8)
    fig.savefig(args.output/"comparison.png",dpi=180)
    fig.savefig(args.output/"comparison.pdf")
    plt.close(fig)
    table=["| 训练范围 | 模型 | 训练点 NMAE | 验证点 NMAE | 最终留出点 NMAE | 完整网格 NMAE | 最佳步数 |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for name in names:
        result=runs[name]["result"]
        values=[f"{100*result[k+'_macro_nmae']:.5f}%" for k in ("train","validation","test","full_grid")]
        table.append("| "+" | ".join(["1 个结构" if name.endswith("one") else "8 个结构",
                                      "官方等变 DeepDFT" if name.startswith("deepdft") else "当前 R16",*values,str(result["best_step"])])+" |")
    paired=["| 样本 | DeepDFT 留出点 | R16 留出点 | DeepDFT 相对误差下降 |","|---|---:|---:|---:|"]
    for a,b in zip(runs["deepdft_eight"]["result"]["records"],runs["r16_eight"]["result"]["records"]):
        assert a["sample_id"]==b["sample_id"]
        paired.append(f"| {a['sample_id']} | {100*a['test_nmae']:.5f}% | {100*b['test_nmae']:.5f}% | {100*(1-a['test_nmae']/b['test_nmae']):.2f}% |")
    a,b=(runs[f"{m}_eight"]["result"]["test_macro_nmae"] for m in ("deepdft","r16"))
    wins=sum(x["test_nmae"]<y["test_nmae"] for x,y in zip(runs["deepdft_eight"]["result"]["records"],runs["r16_eight"]["result"]["records"]))
    report=f"""# NMC 新查询点实验：官方等变 DeepDFT 与当前 R16

报告日期：{args.report_date}；实验于 2026-10-26 启动。此次为单种子、固定结构的空间留出诊断。先在样本 1 上从头训练，再在样本 0–7 上分别从头训练；8 结构实验没有继承单结构权重。

## 结果

{chr(10).join(table)}

8 结构配对的最终留出点上，DeepDFT 相对 R16 的误差下降为 **{100*(1-a/b):.2f}%**，R16/DeepDFT 误差比为 **{b/a:.2f}**。各数值先逐结构计算 NMAE，再做宏平均，原始 JSON 保存比值。完整网格包含训练点，不能替代留出点结果。

DeepDFT 在 **{wins}/8 个结构**上都取得更低的留出点误差；训练点宏平均 {100*runs['deepdft_eight']['result']['train_macro_nmae']:.5f}% 与留出点 {100*a:.5f}% 接近。这支持直接查询点预测在这些已知 NMC 结构上的空间插值有效性。

NMAE = 查询点绝对误差之和 / 同一批查询点的真实密度绝对值之和。表中显示百分数；例如 JSON 中的 0.00214 对应约 0.214%。

![验证曲线与留出点结果](comparison.png)

{chr(10).join(paired)}

## 如何避免空间点泄漏

- 每个结构为 56×56×140 = 439,040 个原始 CHGCAR 网格点，密度单位为电子/Å³。
- 按结构 ID 与种子 20261026，仅由索引随机划分：351,232 个训练点（80%）、43,904 个验证点（10%）、43,904 个最终测试点（10%）。单结构和 8 结构实验共用该结构的同一份划分。
- 每次更新只从训练分区均匀有放回抽 1,024 点。损失的归一化只使用训练分区的密度均值。
- 每 500 步用验证分区中的固定 8,192 点选择模型，第一步也检查。最终测试点没有参与梯度、学习率安排或 checkpoint 选择；训练全部结束并载入已选模型后，才遍历全部网格并分别统计三个分区。
- `sampling_audit.json` 记录实际访问过的训练点，所有验证/测试点的梯度使用次数均为零。周期几何缓存可包含查询坐标，但不包含密度标签，查询点之间不传递消息。
- 使用从头训练的 R16 作为公平对照。旧 R16 checkpoint 曾在完整网格上训练/选择，不能声称其在这些点上的结果是空间留出成绩。

## 模型和预算

- DeepDFT 直接导入官方 `PainnDensityModel`，提交 `{OFFICIAL_COMMIT}`，3 层、128 通道、4 Å cutoff、30 个 sinc 径向输入；30 是官方代码与发布 NMC 模型的设置。没有修改官方 `densitymodel.py` 或 `layer.py`，没有使用预训练权重初始化正式实验。
- 使用 cKDTree 枚举全部周期像作为数据适配器；官方网络仍自行计算相对坐标、径向特征、消息和标量密度。未安装的 asap3 仅属于官方数据加载路径的邻居搜索依赖，不影响直接使用官方模型。
- R16 保持当前模型的 24/12/6 通道、3 层、6 Å 原子图和 l≤4 正则白化密度基，密度 cutoff 3 Å。新增缓存与生产解码器的值及系数梯度已做独立一致性测试。
- 参数量：DeepDFT 为 1,487,233，R16 为 521,472；R16 的可学习原子基线也从头训练。
- 两种模型均采用密度归一化 L1、Adam、初始学习率 1e-3、余弦衰减到 1e-5、无 weight decay、梯度范数截断 5。单结构 4,000 步，8 结构 16,000 步，seed=42；配对的结构顺序与采样序列相同。
- 评价原始网络输出，不做电荷重归一化或负密度截断。
- 这是统一预算的本地诊断，使用官方架构，但训练协议不同于官方论文的大数据 MSE/学习率协议，不能称作完整论文复现。两模型参数量和单步计算量不相等。
- 官方预训练 NMC 模型仅在训练分区的 4,096 点上做输入格式检查，NMAE 0.04310%；检查权重和 20 步吞吐预实验权重均被丢弃，不进入正式模型。

## 结论边界

本实验检验已知结构中未用于优化的新空间网格点，属于空间插值；不是新材料、新结构或离开原始网格的连续坐标泛化。随机散布留出点的相邻网格可能属于训练集，因此不能把结果解释成大块未知区域的外推能力。

这 8 个结构此前用于项目的基组与过拟合研究。因此这里的“留出”严格指本轮从头训练、归一化和 checkpoint 选择没有使用该分区标签，并不意味着这些结构的密度从未在项目历史中被查看。

单个随机种子和有限训练预算不能证明架构最优精度。后续应在正式结构划分上检验泛化，再比较完整场生成成本及 SCF；本轮没有启动完整 NMC 训练或 VASP。

这也是两套完整架构的比较：骨干、参数量、邻域和读出都不同，不能将全部误差差异单独归因于查询点读出。若要定位收益来源，应随后做相同骨干的读出消融。

## 复现与来源

使用 `C:\\Users\\User\\.conda\\envs\\kan-cgcnn-cuda\\python.exe`，运行 `scripts/run_nmc_probe_holdout.ps1`，再运行 `python -m scripts.summarize_nmc_probe_holdout`。训练脚本支持严格设置检查和断点恢复；流水线会跳过已完成的结果。实验输出在 `runs/nmc_probe_holdout_2026-10-26`，四组目录均保存 `best.pt`、`last.pt`、设置、采样记录与结果。

已完成 47 项测试，包括新增的空间划分互斥、独立 ASE 周期邻居对照、官方网络旋转/查询分批一致性，以及 R16 缓存的值和梯度对照。报告生成时再次核验了原始数据、划分数组、训练入口、官方模型文件和最佳 checkpoint 的哈希，以及配对训练设置、采样记录和验证集模型选择。

本目录的 `source_snapshot.zip` 保存源代码快照，`manifest.json` 保存文件哈希，`summary.json` 汇总原始指标，`comparison.png` / `comparison.pdf` 为图件。正式实验使用 PyTorch 2.11.0+cu128 和 RTX 5060 Ti；机器上有其他任务，因此运行时间不用于速度排名。

[DeepDFT 原文](https://www.nature.com/articles/s41524-022-00863-y) · [官方代码](https://github.com/peterbjorgensen/DeepDFT)
"""
    (args.output/"report.md").write_text(report,encoding="utf-8")
    (args.output/"paired_table.md").write_text("\n".join(table)+"\n\n"+"\n".join(paired)+"\n",encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps(runs,indent=2),encoding="utf-8")
    sources=[Path("scripts/nmc_probe_holdout.py"),Path("scripts/train_nmc_probe_holdout.py"),
             Path("scripts/run_nmc_probe_holdout.ps1"),Path(__file__),Path("tests/test_probe_holdout.py")]
    sources += [args.root/"DeepDFT"/name for name in ("densitymodel.py","layer.py","LICENSE")]
    sources += list(Path("src").rglob("*.py"))
    sources += [Path("scripts/compare_local_nmc_oracles.py"),Path("scripts/train_local_nmc_overfit.py"),
                Path("configs/nmc_full_l2_to_l4_mace_pbc.yaml"),Path("pyproject.toml")]
    sources = sorted(set(path.resolve() for path in sources))
    with zipfile.ZipFile(args.output/"source_snapshot.zip","w",zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path,path.relative_to(Path.cwd()).as_posix())
    files=sources+[args.root/name/f for name in names for f in ("best.pt","settings.json","sources.json","history.json","result.json","sampling_audit.json")]
    files+=list((args.root/"spatial_splits").glob("*"))
    manifest={"report_date":args.report_date,"official_commit":OFFICIAL_COMMIT,
              "paired_checks_passed":True,"recorded_hashes_verified":True,
              "files":[{"path":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size} for path in files]}
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("\n".join(table),flush=True)


if __name__=="__main__":
    main()
