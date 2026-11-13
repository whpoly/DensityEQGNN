"""Generate the full NMC PAW report from completed saved observations."""
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_public_density import save
from scripts.public_paw_experiment import ROOT,ARMS,SEEDS

REPORT=Path('reports/public_paw_2026-11-02')


def read(path): return json.loads(Path(path).read_text())


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg=read(ROOT/'settings.json')
    finals={a:read(ROOT/f'{a}_final.json') for a in ARMS}
    times={m:read(ROOT/f'timing_{m}.json') for m in ('density','joint','deep_occupation','eden_occupation')}
    verification=read(ROOT/'joint_verification.json')
    integrity=read(ROOT/'source_integrity.json')
    unit=read(ROOT/'unit_validation.json') if (ROOT/'unit_validation.json').exists() else None
    export=read(ROOT/'poscar_export_validation.json') if (ROOT/'poscar_export_validation.json').exists() else None
    assert all(f['completed'] for f in finals.values()) and all(t['completed'] for t in times.values())
    assert verification['passed'] and integrity['passed']
    summary=dict(settings=cfg,arms={},timing={m:{k:v for k,v in t.items() if k!='records'} for m,t in times.items()},
                 density_max_relative_l1=max(r['relative_l1'] for r in verification['records']),
                 density_mean_reference_nmae=np.mean([r['density_reference']['nmae'] for r in verification['records']]),
                 density_mean_joint_nmae=np.mean([r['density_joint']['nmae'] for r in verification['records']]),
                 complete_supported_experiment=True,complete_high_rank_paw=False,vasp_scf_run=False)
    summary.update(unit_validation=unit,poscar_export_validation=export)
    check_text=(f"合批梯度与 PAW/Eden 等变测试共 {unit['passed']} 项通过。" if unit else '单元测试记录尚未汇总。')
    check_text+=('POSCAR 独立入口已用测试结构 41 核对：密度、occupation 与直接几何调用在预设容差内一致，原生次序映射完全一致，有效掩码与源标签匹配。详情见 `poscar_export_validation.json`。'
                 if export and export['passed'] else 'POSCAR 独立入口的导出核对尚未记录。')
    table=['| 方法 | 训练 MAE | 验证 MAE | 测试 MAE | 测试相对 L1 (%) | 测试结构数 |',
           '|---|---:|---:|---:|---:|---:|']
    b=finals['deep_frozen']['baseline']
    table.append(f"| 元素均值基线 | — | — | {b['macro_mae']:.6f} | {100*b['macro_relative_l1']:.3f} | 500 |")
    for arm in ARMS:
        summary['arms'][arm]={}
        for seed in SEEDS:
            r=finals[arm]['seeds'][str(seed)]
            selection=read(ROOT/f'{arm}_{seed}/selection.json')
            summary['arms'][arm][str(seed)]={split:{k:v for k,v in metrics.items() if k!='records'} for split,metrics in r.items()}
            summary['arms'][arm][str(seed)]['selection']=selection
            table.append(f"| {arm} / {seed} | {r['train']['macro_mae']:.6f} | {r['validation']['macro_mae']:.6f} | {r['test']['macro_mae']:.6f} | {100*r['test']['macro_relative_l1']:.3f} | {len(r['test']['records'])} |")
    paired=['| 方法 | 共同测试 MAE（两种子均值） | 相对 L1 (%) |', '|---|---:|---:|']
    for arm in ARMS:
        vals=[finals[arm]['seeds'][str(s)]['paired_test'] for s in SEEDS]
        paired.append(f"| {arm} | {np.mean([v['macro_mae'] for v in vals]):.6f} | {100*np.mean([v['macro_relative_l1'] for v in vals]):.3f} |")
    rank=['| 方法 | L=0 MAE | L=1 MAE | L=2 MAE | L>0 相对 L1 (%) |','|---|---:|---:|---:|---:|']
    for arm in ARMS:
        values=[np.mean([r['by_l'][str(l)]['mae'] for seed in SEEDS for r in finals[arm]['seeds'][str(seed)]['paired_test']['records']]) for l in range(3)]
        anisotropic=[sum(r['by_l'][str(l)]['absolute_error_sum'] for l in (1,2))/sum(r['by_l'][str(l)]['target_absolute_sum'] for l in (1,2))
            for seed in SEEDS for r in finals[arm]['seeds'][str(seed)]['paired_test']['records']]
        rank.append('| '+arm+' | '+' | '.join(f'{x:.6f}' for x in values)+f' | {100*np.mean(anisotropic):.3f} |')
    elements=['| 元素 | DeepDFT pooled MAE | Eden pooled MAE |','|---|---:|---:|']
    for z,label in [('3','Li'),('8','O'),('25','Mn'),('27','Co'),('28','Ni')]:
        vals=[]
        for arm in ARMS:
            observations=[r['by_element'][z] for seed in SEEDS for r in finals[arm]['seeds'][str(seed)]['paired_test']['records'] if z in r['by_element']]
            vals.append(sum(v['absolute_error_sum'] for v in observations)/sum(v['count'] for v in observations))
        elements.append('| '+label+' | '+' | '.join(f'{v:.6f}' for v in vals)+' |')
    training=['| 模型 / 种子 | 验证选择 epoch | 可训练参数 | 训练秒数 |','|---|---:|---:|---:|']
    for arm in ARMS:
        for seed in SEEDS:
            s=summary['arms'][arm][str(seed)]['selection']
            training.append(f"| {arm} / {seed} | {s['epoch']} | {s['parameters']} | {s['training_seconds']:.1f} |")
    timing=['| 路径 | 平均秒/结构 | 结构中位数的中位数 (s) | 最大增量显存 (GiB) | 结构数 |',
            '|---|---:|---:|---:|---:|']
    for mode,t in times.items():
        timing.append(f"| {mode} | {t['mean_seconds']:.6f} | {t['median_seconds']:.6f} | {max(r['peak_increment_gib'] for r in t['records']):.3f} | {len(t['records'])} |")
    d=finals['deep_frozen'];mae=np.mean([d['seeds'][str(s)]['test']['macro_mae'] for s in SEEDS])
    relative=np.mean([d['seeds'][str(s)]['test']['macro_relative_l1'] for s in SEEDS])
    reduction=100*(1-mae/b['macro_mae'])
    summary.update(deep_mean_test_mae=float(mae),deep_mean_test_relative_l1=float(relative),baseline_mae_reduction_percent=float(reduction))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,ax=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for arm,color in [('deep_frozen','#167d9a'),('eden_frozen','#df8238')]:
        for seed in SEEDS:
            hist=read(ROOT/f'{arm}_{seed}/history.json')
            ax[0].plot([h['epoch'] for h in hist],[h['validation_mae'] for h in hist],
                       color=color,linestyle='-' if seed==42 else '--',label=f'{arm} {seed}')
    ax[0].set(xlabel='Epoch',ylabel='Validation coefficient MAE',yscale='log',title='Validation selection')
    ax[0].legend(fontsize=8)
    paired_b=finals['deep_frozen']['paired_baseline']['macro_mae']
    bar=[paired_b]+[np.mean([finals[a]['seeds'][str(s)]['paired_test']['macro_mae'] for s in SEEDS]) for a in ARMS]
    ax[1].bar(['Element mean','DeepDFT\nfrozen','Eden\nfrozen'],bar,color=['#9aa4ae','#167d9a','#df8238'])
    ax[1].set(ylabel='Coefficient MAE',title=f"Common {len(cfg['paired_test'])}-structure test")
    for j,v in enumerate(bar): ax[1].text(j,v,f'{v:.5f}',ha='center',va='bottom')
    modes=['density','joint','deep_occupation','eden_occupation']
    ax[2].bar(['Density','Density + PAW','Deep PAW','Eden PAW'],[times[m]['mean_seconds'] for m in modes],color=['#167d9a','#19485a','#5ab9a8','#df8238'])
    ax[2].set(ylabel='Seconds / structure',title='Fresh warm inference')
    ax[2].tick_params(axis='x',rotation=20)
    for a in ax: a.spines[['top','right']].set_visible(False)
    fig.get_layout_engine().set(rect=(0,.08,1,.92))
    fig.text(.01,.012,'L <= 2, nonmagnetic NMC. Different density pretraining and PAW heads; not an architecture ranking. No SCF speed claim.',fontsize=9)
    fig.savefig(REPORT/'benchmark.png',dpi=180);fig.savefig(REPORT/'benchmark.pdf');plt.close(fig)
    text=f'''# 官方预训练 DeepDFT 的完整 NMC occupation 实验

日期：2026-11-02。已完成可用标签范围内的训练、独立测试、联合推理验证与计时。
DeepDFT 两种子平均测试 MAE **{mae:.6f}**，相对 L1 **{100*relative:.3f}%**；
相对元素均值基线，MAE 降低 **{reduction:.2f}%**。
密度＋occupation 平均耗时 **{times['joint']['mean_seconds']:.4f} 秒/结构**。
这不是完整高阶 PAW 或 VASP SCF 加速的验证。

![精度、收敛与耗时](benchmark.png)

## 实验设计与数据

- 官方全部 2000 个 NMC 结构，1450 训练 / 50 验证 / 500 测试，按结构划分。
- 每个文件的原生 PAW 槽位与 POTCAR 投影子顺序完成解析；归档 MD5、划分及模型哈希复核通过。
- 全集 L=3、4 均为零，只将 L≤2 当作有效标签。未把高阶零输出计入精度。
- DeepDFT 使用官方 NMC 密度预训练模型；冻结密度网络，仅训练 49,899 参数的几何等变头。
- Eden 使用官方编码器/augmentation 架构的本地 8 结构密度预训练权重，冻结编码器并重新训练 occupation 头。它不是论文通用模型的复现。
- Eden 密度预训练见过官方测试结构 {cfg['eden_excluded_test']}，因此其测试剔除这些结构，两个模型的共同对比为 {len(cfg['paired_test'])} 个结构。预训练数据和输出头规模不同，不作架构优劣结论。
- 两臂各 seed 42/43，100 epoch，batch 16，Adam 1e-3→1e-5，裁剪 5；同种子同采样顺序。每 5 epoch 验证，全部四个检查点锁定后才打开新测试指标。
- 元素均值、张量缩放和 Eden 特征 RMS 仅来自训练集；每个张量块所有 m 共用缩放。训练可缓存冻结特征；正式测试重新计算原子特征。
- 该公开测试划分此前用于密度 benchmark，部分结构用于历史小规模方法开发；本轮按固定协议重新训练和评估，并非全新盲测。

## 完整测试结果

{chr(10).join(table)}

MAE 是有效原子中心 augmentation 系数的无量纲误差。相对 L1 的分母是这些系数绝对值之和；不是电子密度 NMAE。先逐结构计算再平均。两个种子均报告，不按测试集挑选。

## 排除预训练重叠后的共同测试集

{chr(10).join(paired)}

{chr(10).join(rank)}

{chr(10).join(elements)}

元素列按共同测试集有效系数汇总（pooled），合并两个种子的误差后除以系数数目；不与按结构平均的 MAE 混用。

完整逐结构、逐元素、逐角阶以及 pooled 指标在原始 final JSON 中。小规模历史实验的数值不与这次完整划分直接混合。

## 训练记录

{chr(10).join(training)}

每种子 100 个完整 epoch，共 9100 次参数更新、145000 个训练结构呈现。两个种子只提供本次训练重复性信息，不构成统计置信区间。
训练秒数不含数据解析、冻结特征准备或模型设置，受桌面环境状态影响，不能用它直接推断架构 FLOPs 优势。

## 联合推理性能

{chr(10).join(timing)}

RTX 5060 Ti，FP32，TF32 关闭；密度保留全部 4 Å 邻居、128 通道，融合分块 8192。
固定使用既有的 50 测试结构计时名单；occupation 对照只使用其共同测试子集。
每条路径独立进程，每结构预热 2 次、同步重复 5 次，先取中位数再平均。
每次重新构图、编码原子并预测；联合路径共享一次编码。不计文件读取、导入、模型加载、一次性表和 kernel 设置或写盘。
两条路径的独立中位数之差不是 occupation 的精确新增开销。设置时间保存在 timing JSON，不能把本表当作进程冷启动或完整 DFT 时间。

## 密度保持与正确性

50 个固定测试结构的完整网格逐点核对通过；联合融合输出相对原官方网络代数路径最大归一化 L1 差为 **{summary['density_max_relative_l1']:.3g}**。
这 50 个结构的平均密度 NMAE 为 reference **{100*summary['density_mean_reference_nmae']:.7f}%**、joint **{100*summary['density_mean_joint_nmae']:.7f}%**。
密度 checkpoint 未修改；这里是 50 结构新增联合核对，官方 500 结构密度测试见上一轮公开 benchmark，不能把本轮子集均值叫作 500 结构均值。
真实结构旋转/反演/平移 occupation 检查、缓存/新算特征输出一致性、不同大小结构合批预测与梯度等价性、原生槽位往返和有效掩码均已检查。
{check_text}

## 尚不能据此回答的问题

现有标签未提供可认证的 L=3、4 高阶系数；历史原始 POTCAR 完整身份不能从 CHGCAR 单独恢复；目前只支持非磁性 Li/O/Mn/Co/Ni。
当前环境 PATH 未发现可执行 VASP，WSL 也未安装。因此没有 SCF 步数、总 DFT 墙钟时间、最终能量/力一致性结果。
没有把真实 augmentation 拼入预测后称作完整模型，也没有把只保留真实密度的 occupation 替换文件称作完整 ML 初猜。
扩大到 EC、磁性材料或完整 LMAXMIX=4，需要分别准备元素/自旋/高阶映射和可靠标签。

## 使用与复现

`PublicPAWPredictor(seed=42).predict(geometry)` 返回 density、canonical augmentation、supported_lmax=2。
geometry 只包含 positions、numbers、cell、probes、metadata；输入来自当前 CUDA 上的完整规则周期网格。
占据分支单独调用可设置 `density=False`，只需 positions/numbers/cell。高阶占位不能当作完整 PAW 重启数据。

仅从 POSCAR 推理：`python -m scripts.predict_public_paw --structure <POSCAR> --grid 56 56 140 --output <prediction.npz>`。
输出包括密度、canonical 与本地 POTCAR 原生顺序的 occupation，以及有效掩码。该 NPZ 不是完整 VASP 重启 CHGCAR。

训练与准备：`python -m scripts.public_paw_experiment --stage audit/prepare/train`，配合 `--arm deep_frozen/eden_frozen --seed 42/43`。
最终评估：`python -m scripts.finalize_public_paw --stage final/verify/timing`；汇总：`python -m scripts.summarize_public_paw`。
已有数据与正式结果禁止覆盖；训练中断可从 last.pt 恢复。新实验应新建独立目录并锁定协议。
协议为 protocol.md；原始结果、审计、训练历史和模型见 `runs/public_paw_2026-11-02`。
'''
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'report.md').write_text(text,encoding='utf-8')
    save(REPORT/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('deep_mean_test_mae','deep_mean_test_relative_l1','baseline_mae_reduction_percent','density_max_relative_l1')}))


if __name__=='__main__': main()
