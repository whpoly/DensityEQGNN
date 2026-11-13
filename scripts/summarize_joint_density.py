"""Summarize the locked benchmark without any test-driven model selection."""
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_deepdft_reuse import save
from scripts.benchmark_joint_density import ROOT,check

OUTPUT = Path('reports/joint_density_benchmark_2026-10-30')
LABELS = {'deep_reference':'DeepDFT 完整基线','deep_factored':'DeepDFT 乘法重排',
    'deep_table':'DeepDFT 仅径向查表','deep_fused':'DeepDFT 全量融合',
    'fused_cubic_r3':'融合＋三次远场 R=3','fused_cubic_r3.25':'融合＋三次远场 R=3.25',
    'fused_cubic_r3.5':'融合＋三次远场 R=3.5','eden':'EdenGNN 密度＋独立 occupation'}


def aggregate(rows):
    valid = [r for r in rows if 'error' not in r]
    if len(valid)!=len(rows):
        return dict(error='One or more structures unavailable',successful=len(valid),total=len(rows))
    output = dict(structures=len(rows),
        density_nmae_percent=100*float(np.mean([r['density']['nmae'] for r in rows])),
        density_pooled_nmae_percent=100*sum(r['density']['absolute_error_sum'] for r in rows)/sum(r['density']['target_absolute_sum'] for r in rows),
        density_mae=float(np.mean([r['density']['mae'] for r in rows])),
        density_rmse=float(np.mean([r['density']['rmse'] for r in rows])),
        charge_abs_bias_percent=float(np.mean([r['density']['absolute_charge_bias_percent'] for r in rows])),
        max_charge_abs_bias_electrons=max(abs(r['density']['charge_bias_electrons']) for r in rows),
        negative_integral_percent=100*float(np.mean([r['density']['negative_integral_fraction'] for r in rows])),
        paw_by_seed={str(seed):dict(mae=float(np.mean([r['paw'][str(seed)]['mae'] for r in rows])),
            relative_l1_percent=100*float(np.mean([r['paw'][str(seed)]['relative_l1'] for r in rows])),
            by_l={str(l):float(np.mean([r['paw'][str(seed)]['by_l'][str(l)]['mae'] for r in rows])) for l in range(3)})
            for seed in (42,43)},timing={})
    output['paw_mae_seed_mean'] = float(np.mean([r['mae'] for r in output['paw_by_seed'].values()]))
    output['paw_relative_l1_seed_mean_percent'] = float(np.mean([r['relative_l1_percent'] for r in output['paw_by_seed'].values()]))
    for key in ('joint_timing','density_timing'):
        output['timing'][key] = dict(seconds=float(np.mean([r[key]['median_seconds'] for r in rows])),
            p90_all_repeats=float(np.percentile([t for r in rows for t in r[key]['seconds']],90)),
            max_peak_increment_gib=max(r[key]['peak_increment_mib'] for r in rows)/1024,
            max_peak_absolute_gib=max(r[key]['peak_absolute_mib'] for r in rows)/1024,
            max_peak_reserved_gib=max(r[key]['peak_reserved_mib'] for r in rows)/1024,
            median_within_structure_spread_percent=float(np.median([
                100*(max(r[key]['seconds'])-min(r[key]['seconds']))/r[key]['median_seconds'] for r in rows])))
    if all('seen_spatial_holdout' in r['density'] for r in rows):
        output['seen_spatial_test_nmae_percent'] = 100*float(np.mean([r['density']['seen_spatial_holdout']['test'] for r in rows]))
    output['distance_bands'] = {}
    bands = set().union(*(r['density'].get('distance_bands',{}) for r in rows))
    for band in sorted(bands,key=lambda b:float(b.split('-')[0])):
        values = [r['density']['distance_bands'][band] for r in rows if band in r['density']['distance_bands']]
        output['distance_bands'][band] = dict(structures=len(values),points=sum(r['count'] for r in values),
            mae=float(np.mean([r['mae'] for r in values])),
            nmae_percent=100*float(np.mean([r['nmae'] for r in values])))
    return output


def plot(summary,scaling,chosen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(dict.fromkeys(('deep_reference',chosen,'eden')))
    labels = ['DeepDFT','Fused DeepDFT','EdenGNN'] if chosen=='deep_fused' else names
    colors = ['#357ba4','#218c64','#b87931']
    fig,axes = plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,field,title in ((axes[0,0],'train','Seen structures: spatial holdout'),(axes[0,1],'test','Unseen structures: full grid')):
        values = [summary[n][field]['seen_spatial_test_nmae_percent'] if field=='train'
            else summary[n][field]['density_nmae_percent'] for n in names]
        bars = ax.bar(labels,values,color=colors)
        ax.bar_label(bars,fmt='%.3f',padding=3)
        ax.set(ylabel='Density NMAE (%)',title=title,ylim=(0,max(values)*1.2))
    values = [summary[n]['test']['timing']['joint_timing']['seconds'] for n in names]
    bars = axes[1,0].bar(labels,values,color=colors)
    axes[1,0].bar_label(bars,fmt='%.3f',padding=3)
    axes[1,0].set(ylabel='Seconds / structure',title='Fresh density + PAW inference',ylim=(0,max(values)*1.2))
    for name,label,color in zip(names,labels,colors):
        rows = [r for r in scaling[name]['records'] if 'error' not in r]
        axes[1,1].plot([r['atoms'] for r in rows],[r['joint_timing']['median_seconds'] for r in rows],
            'o-',label=label,color=color)
    axes[1,1].set(xlabel='Atoms in periodic replica',ylabel='Seconds / structure',title='Scaling diagnostic (fixed grid spacing)')
    axes[1,1].legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle('NMC benchmark: 8 train / 2 validation / 5 test; PAW L <= 2')
    fig.savefig(OUTPUT/'benchmark.png',dpi=180)
    plt.close(fig)


def main():
    cfg = check(ROOT)
    selected = json.loads((ROOT/'selection.json').read_text())
    verification = json.loads((ROOT/'output_verification.json').read_text())
    assert verification['passed']
    cold = json.loads((ROOT/'cold_start.json').read_text())
    assert cold['completed'] and len(cold['records'])==9
    results,summary = {},{}
    for name in cfg['variants']:
        result = json.loads((ROOT/'final'/f'{name}.json').read_text())
        assert result['completed'] and len(result['records'])==15
        results[name] = result
        summary[name] = {split:aggregate([r for r in result['records'] if r['split']==split])
            for split in ('train','validation','test')}
    chosen = selected['selected']['strict']['name']
    scaling = {name:json.loads((ROOT/'scaling'/f'{name}.json').read_text())
        for name in {'deep_reference','eden',chosen}}
    assert all(r['completed'] for r in scaling.values())
    test = {name:r['test'] for name,r in summary.items()}
    candidate,baseline,eden = test[chosen],test['deep_reference'],test['eden']
    assessment = dict(selected_on_validation=chosen,
        test_accuracy_ratio=candidate['density_nmae_percent']/baseline['density_nmae_percent'],
        test_speed_ratio_to_eden=candidate['timing']['joint_timing']['seconds']/eden['timing']['joint_timing']['seconds'],
        test_speedup_over_deep=baseline['timing']['joint_timing']['seconds']/candidate['timing']['joint_timing']['seconds'])
    assessment['strict_accuracy_passed'] = assessment['test_accuracy_ratio']<=1.01
    assessment['eden_speed_target_passed'] = assessment['test_speed_ratio_to_eden']<=1.10
    save(OUTPUT/'summary.json',dict(summary=summary,assessment=assessment,selection=selected,
        scaling=scaling,cold_start=cold,settings=cfg,verification=verification))
    plot(summary,scaling,chosen)
    lines = ['# DeepDFT 精度与 EdenGNN 速度：统一 NMC benchmark', '',
        f"验证集锁定的方案为 **{LABELS[chosen]}**。五个未见测试结构上，其密度 NMAE 为 **{candidate['density_nmae_percent']:.6f}%**，完整 DeepDFT 为 **{baseline['density_nmae_percent']:.6f}%**；联合密度＋occupation 推理为 **{candidate['timing']['joint_timing']['seconds']:.4f} 秒／结构**，DeepDFT 为 **{baseline['timing']['joint_timing']['seconds']:.4f} 秒**，Eden 为 **{eden['timing']['joint_timing']['seconds']:.4f} 秒**。", '',
        f"测试上的密度误差比值为 {assessment['test_accuracy_ratio']:.8f}，相对 DeepDFT 加速 {assessment['test_speedup_over_deep']:.2f} 倍；耗时为 Eden 的 {assessment['test_speed_ratio_to_eden']:.3f} 倍。预设的 ≤1% 相对误差增长门槛：{assessment['strict_accuracy_passed']}；≤1.1 倍 Eden 耗时门槛：{assessment['eden_speed_target_passed']}。这些门槛在验证／测试前写入协议，测试后没有换模型。", '',
        f"**拟合与泛化必须区分。** 已见结构的空间留出点上，DeepDFT NMAE 为 {summary['deep_reference']['train']['seen_spatial_test_nmae_percent']:.4f}%，Eden 为 {summary['eden']['train']['seen_spatial_test_nmae_percent']:.4f}%；真正未见结构的完整网格上，两者分别为 {baseline['density_nmae_percent']:.4f}% 和 {eden['density_nmae_percent']:.4f}%。不能用前一组结果推断 DeepDFT 的结构泛化一定更好。融合实现保留现有 DeepDFT 的函数，不能消除预训练模型的泛化误差。", '',
        f"![精度、联合推理与扩展性](<{(OUTPUT/'benchmark.png').resolve().as_posix()}>)", '',
        '## 统一设置与完成范围', '',
        '- 训练／已见结构：cache 0–7，共 8 个；验证：SCF 170、314；测试：361、796、825、1056、1749，共 5 个。重复的 SCF 1、3、6 已剔除，沿用严格周期结构匹配审核。',
        '- 全部原始结构均为 56×56×140 = 439,040 个网格点，36–48 个原子。已见结构空间留出、未见结构全网格和复制超胞分别报告，绝不混合其精度均值。',
        '- 原密度与 PAW 检查点不变，没有重新训练、压缩通道或降低计算精度。DeepDFT 保持 3×128、原始 4 Å 截断及原输出头。',
        '- 两个密度检查点均来自相同 8 个结构、seed 42、16,000 次更新、每步 1,024 个训练查询点；Adam＋余弦学习率 1e-3→1e-5，按空间验证点选择检查点。这是相同更新预算，不是相同训练时间或 FLOPs。',
        '- Eden 密度编码器为 3 层、最高角阶 l=6，密度读出为 4 通道、4 Å 截断；使用由 POTCAR form factor 重建的 SAD 加上学习到的密度修正。密度 l=6 与本轮 occupation 标签只覆盖 L≤2 是两件不同的事。',
        '- 上游版本固定为 [DeepDFT a6bab4d](https://github.com/peterbjorgensen/DeepDFT/tree/a6bab4deb5cf05d9b46ae397b72253d04ea3c694) 与 [EdenGNN 3be8c06](https://github.com/rubenlee11/EdenGNN/tree/3be8c06265c02ecf1ef0587500ee9b8e091186ef)。本地适配、checkpoint 和 CUDA 源码的哈希随结果保存。',
        '- 本轮是同机、同输入、同计时边界的本地 benchmark；只有 5 个已在历史试验中使用过的未见结构，不是大规模新盲测或论文通用模型复现。',
        '- 已完成密度精度、occupation 精度、联合／密度单独推理、显存、物理量偏差、数值一致性和 1／2／4 倍周期复制扩展性。尚未完成 VASP SCF 收敛、总 DFT 耗时、能量／力及跨材料数据集测试。', '',
        '## 未见结构：最终主表', '',
        '密度 NMAE = Σ|ρ预测−ρDFT|/Σ|ρDFT|；每结构先计算，再取五结构平均。occupation 是 VASP 原子中心 augmentation 系数，只评价 L≤2。PAW 列为两个种子的指标平均，预测本身没有集成；计时固定使用 seed 42，seed 43 在计时外独立核对。', '',
        '| 方法 | 密度 NMAE (%) | PAW MAE | PAW 相对 L1 (%) | 密度单独 (s) | 密度＋PAW (s) | 推理峰值增量 (GiB) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in test.items():
        if 'error' in r:
            lines.append(f"| {LABELS[name]} | 不可用 | — | — | — | — | — |")
        else:
            lines.append(f"| {LABELS[name]} | {r['density_nmae_percent']:.6f} | {r['paw_mae_seed_mean']:.6f} | {r['paw_relative_l1_seed_mean_percent']:.3f} | {r['timing']['density_timing']['seconds']:.4f} | {r['timing']['joint_timing']['seconds']:.4f} | {r['timing']['joint_timing']['max_peak_increment_gib']:.3f} |")
    lines += ['', 'DeepDFT 各加速方案共用相同冻结原子网络和 PAW 几何头。Eden 使用原密度模型与单独微调的 occupation 编码器，联合推理包含两次编码，几何可在当次调用内共用。两种架构／输出头参数量不同，不是参数匹配或训练 FLOPs 匹配实验。', '',
        '完整基线是现有 FlatProbe 推理路径：保持上游模型的完整计算，已经将仅依赖原子的 MLP 移出查询分块循环，并使用规则网格构图；不是未经适配的上游原始 Python 调用。基线／乘法重排／仅查表的查询 chunk 为 4096，融合系列为 8192，均在验证前固定。加速倍数比较的是这两套完整推理配置，不是固定 chunk 下单个 kernel 的独立加速倍数；未对 Eden 进行对等的底层 kernel 优化，因此也不代表两类架构各自的速度上限。', '',
        '## 验证集选择记录', '',
        '以下只使用 SCF 170、314；每路径预热 1 次、同步计时 3 次。严格精度门槛为完整 DeepDFT 的 1.01 倍，速度目标为 Eden 的 1.10 倍。全部候选的顺序和配置预先固定。', '',
        '| 方法 | 验证密度 NMAE (%) | 联合推理 (s/结构) |', '|---|---:|---:|']
    for name in cfg['variants']:
        rows = json.loads((ROOT/'validation'/f'{name}.json').read_text())['records']
        lines.append(f"| {LABELS[name]} | {100*np.mean([r['density']['nmae'] for r in rows]):.6f} | {np.mean([r['joint_timing']['median_seconds'] for r in rows]):.4f} |")
    lines += ['',
        '## 已见结构：原空间留出协议', '',
        '| 方法 | 空间测试 NMAE (%) | 完整网格 NMAE (%) | 完整联合推理 (s/结构) |', '|---|---:|---:|---:|']
    for name,r in summary.items():
        s = r['train']
        if 'error' not in s:
            lines.append(f"| {LABELS[name]} | {s['seen_spatial_test_nmae_percent']:.6f} | {s['density_nmae_percent']:.6f} | {s['timing']['joint_timing']['seconds']:.4f} |")
    lines += ['', '## 未见结构逐项结果', '',
        '| 结构 | 组成 | 原子数 | 体积 (Å³) | DeepDFT 密度 NMAE (%) | 所选方案 NMAE (%) | Eden NMAE (%) |', '|---|---|---:|---:|---:|---:|---:|']
    for r in results['deep_reference']['records']:
        if r['split']!='test': continue
        sid = r['id']
        values = [next(x for x in results[n]['records'] if x['id']==sid)['density']['nmae']*100
            for n in ('deep_reference',chosen,'eden')]
        lines.append(f"| {sid} | {r['metadata']['formula']} | {r['metadata']['atoms']} | {r['metadata']['volume']:.2f} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |")
    lines += ['', '本批训练晶胞体积为 359.39–376.18 Å³。测试中体积为 396.04 Å³ 的三个结构误差较高，而另外两个结构上 DeepDFT 比 Eden 更准。这是本小样本中的相关性：组成、原子排布也同时变化，不能据此断定误差由体积单独造成。总体平均不足以代表每个结构的表现，因此保留逐结构结果。', '',
        '## 物理量偏差与空间分解', '',
        '所有结果未经负值裁剪或电荷重归一化。以下电荷偏差来自完整晶胞网格；它不是能量／力误差，也不能直接换算 SCF 迭代数。', '',
        '| 方法 | 平均绝对电荷偏差 (%) | 最大绝对电荷偏差 (e) | 负密度积分比例 (%) |', '|---|---:|---:|---:|']
    for name in ('deep_reference',chosen,'eden'):
        r = test[name]
        lines.append(f"| {LABELS[name]} | {r['charge_abs_bias_percent']:.6f} | {r['max_charge_abs_bias_electrons']:.6f} | {r['negative_integral_percent']:.6f} |")
    lines += ['', '距离分带 MAE 先在各结构的非空带中计算，再对有点的结构取平均；空带不作为零误差加入。', '',
        '| 最近原子距离 (Å) | 参与结构数 | 总网格点数 | DeepDFT MAE (e/Å³) | 所选方案 MAE | Eden MAE |', '|---|---:|---:|---:|---:|---:|']
    for band in baseline['distance_bands']:
        values = [test[n]['distance_bands'][band]['mae'] for n in ('deep_reference',chosen,'eden')]
        info = baseline['distance_bands'][band]
        lines.append(f"| {band} | {info['structures']} | {info['points']:,} | {values[0]:.6g} | {values[1]:.6g} | {values[2]:.6g} |")
    lines += ['', '## Occupation 角阶和种子', '',
        '| 方法／种子 | 总 MAE | L=0 MAE | L=1 MAE | L=2 MAE |', '|---|---:|---:|---:|---:|']
    for name in ('deep_reference',chosen,'eden'):
        for seed,r in test[name]['paw_by_seed'].items():
            lines.append(f"| {LABELS[name]}／{seed} | {r['mae']:.6f} | {r['by_l']['0']:.6f} | {r['by_l']['1']:.6f} | {r['by_l']['2']:.6f} |")
    lines += ['', '现有文件的 L=3、4 标签不可用，因此这些槽位未进入指标，不能把输出占位零当成正确高阶预测。历史 POTCAR 完整版本也无法从 CHGCAR 单独恢复。这仍不是完整 LMAXMIX=4 的 PAW 初始化验证。', '',
        '## 计时与显存边界', '',
        '- 每种方法独立 Python/CUDA 进程、GPU 串行使用；每结构／路径预热 2 次、CUDA 同步测量 5 次，先取结构中位数再跨结构平均。保留全部重复，不报告最快一次。',
        '- 每次重新构造原子图和网格几何、重新编码原子、重新计算密度与 PAW；Eden 的 SAD 构造也在计时内。只复用权重和由权重决定的常量表，不跨结构缓存学习到的特征。',
        '- 稳态推理时间排除文件读取、模型／表／kernel 初始化、预热、指标计算及输出文件写盘。数据读取与模型设置时间另存；下表的设置时间不等于包含首次 forward/JIT 的完整冷启动时间。',
        '- 密度单独与联合路径分别测量；毫秒级差异可能受计时波动影响，不用两个独立中位数之差估计 occupation 头的精确新增成本。',
        '- 推理峰值增量以调用前 allocated 显存为基线，包含该次构图、临时张量和输出；绝对峰值还包括驻留模型、表、输入及计时外的参考标签。reserved 与 allocated 均保留，不能混用。',
        f"- 环境：{cfg['device']}，PyTorch {cfg['torch']}，CUDA {cfg['cuda']}，FP32，TF32 关闭。", '',
        '| 方法 | 设置时间 (s) | 密度参数 | occupation 参数 | 联合 P90 (s) | 最大 allocated (GiB) | 结构内计时波动中位数 (%) |', '|---|---:|---:|---:|---:|---:|---:|']
    for name,result in results.items():
        r = test[name]
        if 'error' not in r:
            t = r['timing']['joint_timing']
            lines.append(f"| {LABELS[name]} | {result['setup_seconds']:.2f} | {result['density_model_parameters']:,} | {result['occupation_model_parameters']:,} | {t['p90_all_repeats']:.4f} | {t['max_peak_absolute_gib']:.3f} | {t['median_within_structure_spread_percent']:.2f} |")
    lines += ['', 'P90 合并五个结构的全部重复，包含结构大小差异；“结构内波动”是每结构 (最大−最小)/中位数，再跨结构取中位数。', '',
        '## 补充诊断：新进程首次调用', '',
        '这项诊断在方案选择后补充，不参与选择。固定训练结构 cache_0，每种方法启动 3 个全新 Python 进程，轮流执行，完全不预热，表中为各段中位数。输入为不含标签的已解析几何缓存；没有清空操作系统文件缓存，因此这是进程冷启动，而非磁盘／驱动冷启动。', '',
        '| 方法 | 库导入 (s) | 缓存输入＋CUDA 上下文 (s) | 模型／表设置 (s) | 首次联合 forward (s) | 进程启动到退出 (s) |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('deep_reference',chosen,'eden'):
        r = cold['medians'][name]
        lines.append(f"| {LABELS[name]} | {r['imports_seconds']:.3f} | {r['cached_input_and_cuda_context_seconds']:.3f} | {r['predictor_setup_seconds']:.3f} | {r['first_forward_seconds']:.3f} | {r['process_launch_to_exit_seconds']:.3f} |")
    lines += ['', '进程总时间还包括启动、输出复制回 CPU、有限值检查及解释器退出；各段中位数之和不一定等于总时间中位数。没有包括原始 CHGCAR 解析、写出可用于 VASP 的完整初猜文件或 SCF。批量预测／常驻服务可摊薄导入与模型设置成本；每个结构单独启动 Python 时，应参考这张表，不能直接使用稳态 0.x 秒作为端到端时间。', '',
        '## 扩展性：周期复制诊断', '',
        '固定网格间距，复制 cache_0 为 1×1×1、2×1×1、2×2×1，即 47／94／188 原子及约 44／88／176 万网格点。这些是同一物理周期结构的复制，检验计算量、显存与周期一致性，不作为额外独立测试样本。', '',
        '| 方法 | 原子数 | 网格点 | 联合推理 (s) | 峰值增量 (GiB) | 密度与单胞复制预测的相对 L1 差 | PAW 最大绝对差 |', '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('deep_reference',chosen,'eden'):
        for r in scaling[name]['records']:
            if 'error' in r:
                lines.append(f"| {LABELS[name]} | {r['atoms']} | {r['points']:,} | OOM | — | — | — |")
            else:
                lines.append(f"| {LABELS[name]} | {r['atoms']} | {r['points']:,} | {r['joint_timing']['median_seconds']:.4f} | {r['joint_timing']['peak_increment_mib']/1024:.3f} | {r['density_periodic_relative_l1']:.3g} | {r['paw_periodic_max_abs']:.3g} |")
    lines += ['', '复制超胞不是 bitwise 相同输入：坐标、晶格平移和归约顺序的变化可能改变 FP32 舍入。PAW 最大绝对差在 DeepDFT 基线与融合版中都约为 7.7e-5，Eden 约为 3.7e-5；这些周期诊断差异与同一结构上新旧实现的数值一致性检查分开报告。', '',
        '## 加速来源与正确性', '',
        '径向部分是距离的确定函数：原 sinc 径向展开→线性滤波投影→截断包络。对每层生成 16,385 点 FP32 表，查询时线性插值。表只依赖已训练权重，不使用任何密度标签。单独查表仍会产生大量逐边张量，未必更快。', '',
        '融合 kernel 让每个查询点／通道直接遍历其 CSR 邻居，将查表、原子特征乘法、方向向量组合和求和在一次 kernel 中完成。它避免写出再读取逐边的 128 通道标量及三分量向量。全量融合方案仍使用所有原 4 Å 边，原查询更新门、PaiNN 更新和读出均保留。', '',
        '此前远场插值减少的是逐边计算量；融合后该部分已变便宜，粗网格场构造和插值自身会成为额外成本。因此需要重新测量全量与近／远场方案，不能沿用旧实现中的速度排序。', '',
        f"全部 15 个结构的输出核对通过。所选方案相对完整 DeepDFT 的最大结构归一化 L1 差为 {max(r['density']['relative_l1'] for r in verification['records'] if r['variant']==chosen):.3g}。所有 DeepDFT 加速方案、两个 PAW 种子均通过逐系数及 relative L1<1e-6 检查；三次方案还核对粗网格锚点。", '',
        '- 14 项相关测试通过：融合边界／空邻居／输入刷新、原子系数往返、等变输出头、原 Eden 适配器、消息重排、插值与原推理路径。',
        '- 所有原始 DFT 文件、模型检查点和已锁定实现哈希核对通过。表插值和 GPU 归约只保证已测容差，不宣称 bitwise 相同。',
        '- 初始 8193 点表有一个滤波器比较超出预设容差，验证集评估前提高至 16385 点，原日志和实现保留；详见 protocol.md。初始 bitwise PAW 比对也改为记录原 GPU 归约舍入差，未更改 PAW 模型。',
        '- 融合 kernel 当前仅用于 FP32 前向推理和内部排序 CSR 图，不支持反向训练。加载器针对本机 Windows＋PyTorch 所带 NVRTC；其他系统／GPU 尚未验证。CUDA 接口依据 [NVIDIA NVRTC 文档](https://docs.nvidia.com/cuda/nvrtc/index.html)，本轮性能数字全部来自本机测量。', '',
        '## 仍缺少的完整科学验证', '',
        '本机未配置可运行的 VASP 或 WSL 环境。本轮不能回答 ML 初猜最终减少多少 SCF 步、是否节约总 DFT 墙钟时间，或能量／力是否一致。要闭合这一部分，需要对应 POTCAR 的完整 occupation（含所需高阶与自旋分量）、一致的 INCAR/KPOINTS/收敛阈值及 VASP 运行环境，并把模型初始化、文件输出和 SCF 都计入总时间。', '',
        '目前只有 NMC、8 个密度训练结构和 5 个历史未见结构。大数据重训、多化学体系、真正新增盲测、不同网格间距、其他硬件以及训练种子对密度泛化的影响，都未由本轮证明。该优化解决当前实现的推理开销；提高未见结构精度仍需要独立的数据与训练工作。', '',
        '## 运行入口与产物', '',
        '`python -m scripts.benchmark_joint_density --stage prepare` → 对每个 variant 执行 `--stage validation` → `--stage select` → 各 variant 的 `--stage final` → reference／eden／所选方案的 `--stage scaling` → `python -m scripts.verify_joint_outputs` → `python -m scripts.benchmark_joint_cold_start` → `python -m scripts.summarize_joint_density`。数值预检入口为 `python -m scripts.audit_joint_density`，在 prepare 前执行。已有原始结果禁止覆盖；汇总报告可从原始结果重新生成。', '',
        '推理 API：`JointPredictor("deep_fused").predict(geometry_inputs)`，只接收 positions、numbers、cell、probes、metadata（含 grid_shape 和 volume），返回 density 与 canonical augmentation。DeepDFT weights、occupation seed 42 与表在创建时加载／构造一次；每次调用更新几何与原子表示。输入是当前 CUDA 上的规则周期网格；位置／晶胞／probes 为 float64，原子序数为 int64，网络为 FP32。当前模型及 PAW 头仅覆盖 Li/O/Mn/Co/Ni 体系。换权重后应重建 predictor，不能复用旧径向表。', '',
        '完整源数据清单和哈希见 runs/joint_density_benchmark_2026-10-30/settings.json；验证选择见 selection.json；所有逐结构原始计时、显存和误差见 final/；逐点密度和两个种子的 PAW 预测见 predictions/；扩展性见 scaling/；核对见 output_verification.json 和 tests.log；补充新进程诊断见 cold_start.json，入口为 scripts/benchmark_joint_cold_start.py。报告的机器可读汇总为同目录 summary.json。', '']
    (OUTPUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(assessment,indent=2))


if __name__=='__main__':
    main()
