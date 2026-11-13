"""Report every locked Eden PAW arm/seed against the completed DeepDFT pilot."""
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_deepdft_reuse import save
from scripts.benchmark_edengnn_paw import ROOT, BASE
from scripts.train_deepdft_paw import summarize


def plot_comparison(output, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ('DeepDFT_frozen','EdenGNN_frozen','EdenGNN_finetuned')
    labels = ('DeepDFT: frozen','EdenGNN: frozen','EdenGNN: fine-tuned')
    colors = ('#167a9b','#d18621','#7b459e')
    fig, axes = plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for j,(name,label,color) in enumerate(zip(names,labels,colors)):
        for seed,style in ((42,'-'),(43,'--')):
            directory = BASE/f'seed_{seed}' if j==0 else ROOT/f'{name.removeprefix("EdenGNN_")}_{seed}'
            history = json.loads((directory/'history.json').read_text())
            axes[0].plot([r['step'] for r in history],[r['validation_mae'] for r in history],
                style,color=color,alpha=.9,label=label if seed==42 else None)
        selected = [r for r in rows if r['name'].startswith(name+'_')]
        values = [r['test_mae'] for r in selected]
        axes[1].bar(j,np.mean(values),color=color,alpha=.7,width=.55)
        axes[1].scatter([j-.08,j+.08],values,color=color,edgecolors='black',zorder=3)
    axes[0].set(xlabel='Optimizer updates',ylabel='Validation coefficient MAE',title='Validation: solid seed 42, dashed seed 43')
    axes[0].legend(fontsize=8)
    axes[1].axhline(rows[0]['test_mae'],color='gray',ls=':',label='Element mean baseline')
    axes[1].set(xticks=range(3),xticklabels=['DeepDFT\nfrozen','EdenGNN\nfrozen','EdenGNN\nfine-tuned'],
        ylabel='Test coefficient MAE',title='Test: bars = seed mean, dots = individual seeds')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('Matched NMC PAW pilot: 8 train / 2 validation / 5 test, L <= 2')
    fig.savefig(output/'comparison.png',dpi=180)
    plt.close(fig)


def metrics_row(name, metrics, selection=None, timing=None):
    test = metrics['test']
    row = dict(name=name, train_mae=metrics['train']['macro_mae'],
        validation_mae=metrics['validation']['macro_mae'], test_mae=test['macro_mae'],
        relative_l1_percent=100*test['macro_relative_l1'], pooled_mae=test['pooled_mae'],
        pooled_relative_l1_percent=100*test['pooled_relative_l1'],
        by_l={str(l):summarize([r['by_l'][str(l)] for r in test['records']]) for l in range(3)},
        anisotropic=summarize([r['anisotropic'] for r in test['records']]),
        per_structure={r['id']:r['mae'] for r in test['records']})
    if selection is not None:
        row['selection'] = selection
    if timing is not None:
        row['timing_ms'] = {key:1000*float(np.mean([np.median(r['seconds'][key]) for r in timing]))
            for key in ('head','encoder_graph_and_head')}
    assert all(np.isfinite(row[k]) for k in ('train_mae','validation_mae','test_mae','relative_l1_percent'))
    return row


def main():
    cfg = json.loads((ROOT/'settings.json').read_text())
    eden = json.loads((ROOT/'results.json').read_text())
    deep = json.loads((BASE/'results.json').read_text())
    graph_audit = json.loads((ROOT/'gradient_audit.json').read_text())
    assert all(eden['audit'].values()) and graph_audit['passed']
    assert deep['audit']['density_model_state_unchanged'] and deep['audit']['source_hashes_match']
    assert set(eden['records']) == {f'{a}_{s}' for a in cfg['arms'] for s in cfg['seeds']}
    rows = [metrics_row('element_mean',deep['baseline'])]
    for seed,r in deep['seeds'].items():
        rows.append(metrics_row(f'DeepDFT_frozen_{seed}',r['metrics'],r['selection'],r['timing']))
    for name,r in eden['records'].items():
        assert r['selection']['source_hashes_match']
        assert r['selection']['frozen_encoder_unchanged'] == (r['selection']['arm']=='frozen')
        rows.append(metrics_row(f'EdenGNN_{name}',r['metrics'],r['selection'],r['timing']))
    groups = {}
    for prefix in ('DeepDFT_frozen','EdenGNN_frozen','EdenGNN_finetuned'):
        selected = [r for r in rows if r['name'].startswith(prefix+'_')]
        assert len(selected)==2
        groups[prefix] = {k:dict(mean=float(np.mean([r[k] for r in selected])),
            minimum=min(r[k] for r in selected),maximum=max(r[k] for r in selected))
            for k in ('test_mae','relative_l1_percent')}
        groups[prefix]['timing_ms'] = {k:float(np.mean([r['timing_ms'][k] for r in selected]))
            for k in ('head','encoder_graph_and_head')}
    for r in rows:
        r['baseline_mae_reduction_percent'] = 100*(1-r['test_mae']/rows[0]['test_mae'])
    output = Path('reports/edengnn_paw_2026-10-30')
    save(output/'summary.json',dict(rows=rows,groups=groups,audit=eden['audit'],settings=cfg))
    plot_comparison(output,rows)
    lines = ['# EdenGNN 与 DeepDFT 的匹配 PAW occupation 对照', '',
        '本报告比较同一批 NMC 结构上的 VASP 原子中心 augmentation 实球张量系数。这里的 occupation 不是能带占据数。所有模型使用同一标签映射、有效分量掩码、训练统计和指标。', '',
        f"两种子平均测试 MAE：冻结 DeepDFT 为 {groups['DeepDFT_frozen']['test_mae']['mean']:.6f}，冻结 Eden 为 {groups['EdenGNN_frozen']['test_mae']['mean']:.6f}，整体微调 Eden 为 {groups['EdenGNN_finetuned']['test_mae']['mean']:.6f}。微调 Eden 比冻结 DeepDFT 降低 {100*(1-groups['EdenGNN_finetuned']['test_mae']['mean']/groups['DeepDFT_frozen']['test_mae']['mean']):.1f}% MAE。", '',
        '冻结 Eden 的简单线性头未追平 DeepDFT 的几何头；针对 occupation 调整整个 Eden 编码器后，测试精度提高。这个对照说明训练目标和可调整的表示有实际影响，但输出头容量、编码器规模和更新范围同时不同，不能据此断言 Eden 架构本身更优，或 DeepDFT 无法达到这一精度。', '',
        f"本地普通执行的完整 occupation 路径为 DeepDFT {groups['DeepDFT_frozen']['timing_ms']['encoder_graph_and_head']:.1f} ms/结构、微调 Eden {groups['EdenGNN_finetuned']['timing_ms']['encoder_graph_and_head']:.1f} ms/结构。Eden 的输出头虽小，编码器成本仍占主要部分；这个计时不评价完整网格密度任务的速度。", '',
        '![训练曲线及留出测试结果](comparison.png)', '',
        '## 测试结果', '',
        'MAE 无量纲：先对每个结构的有效系数取平均绝对误差，再对五个结构平均。相对 L1 的分母是系数绝对值之和，不能与电子密度 NMAE 直接比较。以下报告全部种子，没有按测试结果选种子。', '',
        '| 方法／种子 | 训练 MAE | 验证 MAE | 测试 MAE | 相对 L1 (%) | 比元素均值基线降低 MAE |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['train_mae']:.6f} | {r['validation_mae']:.6f} | {r['test_mae']:.6f} | {r['relative_l1_percent']:.3f} | {r['baseline_mae_reduction_percent']:.2f}% |")
    lines += ['', '两种子汇总仅表示本次训练重复性，不是统计置信区间。Eden 冻结组对应“直接复用密度表示”；微调组另行训练 occupation 编码器，参数更新范围比 DeepDFT 冻结组更大。不能把差异全部归因于架构。', '',
        '| 方法 | 两种子平均测试 MAE | 两种子平均相对 L1 (%) |', '|---|---:|---:|']
    for name,g in groups.items():
        lines.append(f"| {name} | {g['test_mae']['mean']:.6f} | {g['relative_l1_percent']['mean']:.3f} |")
    lines += ['', '## 角阶分解', '', '| 方法／种子 | L=0 MAE | L=1 MAE | L=2 MAE | L>0 MAE | L>0 相对 L1 (%) |', '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        ls = [r['by_l'][str(l)]['macro_mae'] for l in range(3)]
        lines.append(f"| {r['name']} | {ls[0]:.6f} | {ls[1]:.6f} | {ls[2]:.6f} | {r['anisotropic']['macro_mae']:.6f} | {100*r['anisotropic']['macro_relative_l1']:.3f} |")
    lines += ['', '## 逐结构测试 MAE', '',
        '| 结构 | '+ ' | '.join(r['name'] for r in rows)+' |',
        '|---|'+'---:|'*len(rows)]
    for sid in rows[0]['per_structure']:
        lines.append('| '+sid+' | '+' | '.join(f"{r['per_structure'][sid]:.6f}" for r in rows)+' |')
    lines += ['', '## 推理耗时', '',
        '| 方法／种子 | 仅 occupation 头 (ms/结构) | 新构图＋编码器＋occupation 头 (ms/结构) |', '|---|---:|---:|']
    for r in rows[1:]:
        t = r['timing_ms']
        lines.append(f"| {r['name']} | {t['head']:.3f} | {t['encoder_graph_and_head']:.3f} |")
    lines += ['',
        '在同一 RTX 5060 Ti 上，五个测试结构分别预热三次、CUDA 同步测量十次取中位数，再逐结构平均。完整 occupation 推理每次重新构图和计算原子特征；不借用其他结构的表示。均不包含文件／模型加载和完整网格密度解码。DeepDFT 时间来自前一轮已保存的相同计时协议，未在本轮交错重测，细微差异不宜过度解读。', '',
        'Eden 训练用 CUDA Graph，正式推理耗时全部用普通 eager 执行。不能把本地 Python 实现的计时当成论文优化部署性能。头本身计时假设已有特征；微调的独立 occupation 编码器不能与原密度编码器共用同一份特征，因此实际使用还要支付该编码器的成本。', '',
        '## 训练范围与成本', '',
        f"Eden 原子编码器含 {cfg['encoder_parameters']:,} 个参数；官方等变线性 occupation 头经 NMC 输出适配后含 {cfg['head_parameters']:,} 个参数。DeepDFT 自定义几何等变头含 49,899 个参数。不同编码器和输出头不做参数量或 FLOPs 匹配。", '',
        '| 方法／种子 | 验证选中步骤 | 可训练参数 | 训练秒数 |', '|---|---:|---:|---:|']
    for r in rows[1:]:
        s = r['selection']
        p = s.get('trainable_parameters',s.get('parameters'))
        lines.append(f"| {r['name']} | {s['step']} | {p:,} | {s['training_seconds']:.1f} |")
    lines += ['',
        '每组训练 6000 个优化步骤，batch=2 个不同结构，使用相同种子的同一采样序列；Adam 1e-3→1e-5 cosine、梯度裁剪 5、训练统计标准化后的 MSE，每 200 步用验证集原始系数 MAE 选模。匹配的是更新次数和样本预算，不是墙钟时间。训练秒数不含前置数据准备和 CUDA Graph 捕获。', '',
        '两种 Eden 模式都从原八结构密度检查点开始。冻结组只更新线性头，并可缓存不变的训练特征；微调组更新独立副本的整个编码器及头，每次重新计算训练特征。原密度检查点未改动；不把微调后的 occupation 编码器替换进原密度模型，也未验证替换后的密度精度。', '',
        '## 数值条件修正及记录', '',
        '预试验中，不缩放特征的冻结 seed 42 在 6000 步内训练 MAE 仅由约 0.03770 降至 0.03707。预训练编码器 L=1、L=2 特征 RMS 约为 0.0055、0.0025，新建零初始化线性头在既定学习率下学习缓慢。', '',
        '在查看本轮测试误差前修改了协议：用训练原子上的 RMS 缩放每个 irrep 通道，同一通道所有 m 共用一个标量，设下限 1e-4；固定用于两个正式组。它保持等变性和线性表达能力，但改变优化条件。原不缩放 seed 42 完整记录、被中断的 seed 43 和原源码快照均保留于 runs/edengnn_paw_2026-10-30。正式结果独立保存在 runs/edengnn_paw_scaled_2026-10-30，不混用预试验。', '',
        '## 数据与结论边界', '',
        '- 训练 8 个结构（cache 0–7），验证 2 个（SCF 170、314），测试 5 个（361、796、825、1056、1749）。与训练重复的 SCF 1、3、6 已剔除，其余通过严格周期结构匹配检查。',
        '- 本轮沿用 DeepDFT 已报告的小测试集，不是全新盲测。所有 Eden 模型均在测试前完成验证选模；未根据新测试结果改变结构、训练预算或种子。',
        '- 现有 CHGCAR 中 L=3、4 槽位均为零且历史 LMAXMIX 无法独立确认，因此仅训练和评价 L≤2。高阶槽位输出零仅作占位，不能声称完整 d 通道 PAW 初始化已验证。',
        '- 投影子次序按本地 POTCAR 建立映射，SCF 副本标题匹配；原 CHGCAR 不记录完整 POTCAR 身份，历史版本仍有溯源限制。',
        '- 本轮是官方 Eden 编码器和线性 augmentation 头的本地适配，不是论文通用模型、数据集或训练流程的复现。不能与论文的通用模型 occupation MAE 直接排名。',
        '- 尚未运行 VASP，不据此声称 SCF 迭代减少、总耗时下降或最终能量／力精度提高。', '',
        '## 核对与复现', '',
        f"CUDA Graph 审核覆盖全部八个训练结构、双结构梯度累积及更新后的参数；最大输出差 {max(r['max_abs'] for r in graph_audit['records']):.3g}，最大梯度相对 L2 差 {max(r['gradient_relative_l2'] for r in graph_audit['records']):.3g}。核对使用非零输出头，避免编码器梯度为零造成无效验证。", '',
        '正式评估重新计算所有模型的原子特征；冻结组额外验证缓存／新计算结果一致。所有四个检查点均通过真实测试结构上的旋转、反演和平移等变性检查。来源代码、原 Eden 检查点、DeepDFT 数据与结果哈希审核通过。', '',
        '本轮 13 项相关回归测试通过，涵盖 PAW 输出头、原生系数往返、Eden 适配器、DeepDFT 消息计算和推理；日志位于 runs/edengnn_paw_scaled_2026-10-30/tests.log。各组 best.pt、selection.json、history.json、逐结构 prediction.npz，以及原始误差、逐元素分解和计时均保存在正式 runs 目录。详细协议见同目录报告 protocol.md。', '',
        '复现：依次运行 `python -m scripts.benchmark_edengnn_paw --stage prepare`、`--stage audit`，分别运行 `--stage train --arm frozen/finetuned --seed 42/43`，全部完成后运行 `--stage final`，最后 `python -m scripts.summarize_edengnn_paw`。已有结果禁止覆盖，重跑用 `--root` 指向新目录并相应调整汇总入口。', '']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(groups,indent=2))


if __name__=='__main__':
    main()
