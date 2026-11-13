"""Generate the paired benchmark report from completed, frozen artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT=Path('runs/edengnn_nmc_2026-10-28')
REPORT=Path('reports/nmc_edengnn_2026-10-28')


def main():
    benchmark=json.loads((ROOT/'benchmark.json').read_text())
    selection=json.loads((ROOT/'eden_eight/selection_frozen.json').read_text())
    settings=json.loads((ROOT/'eden_eight/settings.json').read_text())
    result=json.loads((ROOT/'eden_eight/result.json').read_text())
    records=benchmark['records']
    assert benchmark.get('complete') and not benchmark['smoke_only']
    assert [r['sample_id'] for r in records]==list(range(8))
    assert benchmark['checkpoint_sha256']['eden']==selection['checkpoint_sha256']
    sampled={r['sample_id']:r for r in result['records']}
    for row in records:
        for partition in ('train','validation','test'):
            assert abs(row['eden_accuracy'][partition+'_nmae']-sampled[row['sample_id']][partition+'_nmae'])<1e-5
    def mean(values):
        return float(np.mean(values))
    def stage(key):
        return mean([r['stages'][key]['median_seconds'] for r in records])
    summary={}
    for name in ('eden','r16'):
        observations=[t for r in records for t in r['fresh'][name]['seconds']]
        summary[name]={
            'test_macro_nmae':mean([r[name+'_accuracy']['test_nmae'] for r in records]),
            'fresh_mean_of_structure_medians_seconds':mean([r['fresh'][name]['median_seconds'] for r in records]),
            'fresh_pooled_p90_seconds':float(np.percentile(observations,90)),
            'warm_mean_of_structure_medians_seconds':stage(name+'_warm_total'),
            'fresh_peak_increment_mib':max(r['fresh'][name]['peak_increment_mib'] for r in records),
        }
    summary['r16_algebraic_diagnostic']={
        'decoder_seconds':stage('r16_reordered_decoder'),
        'fresh_seconds':stage('r16_reordered_fresh_total'),
        'test_macro_nmae':mean([r['r16_reordered_accuracy']['test_nmae'] for r in records]),
        'max_abs_difference_from_original':max(r['r16_reordered_max_abs'] for r in records),
        'max_relative_l1_difference_from_original':max(r['r16_reordered_relative_l1'] for r in records),
    }
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (REPORT/'per_structure.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=['sample_id','atoms','grid_points','eden_test_nmae_percent','r16_test_nmae_percent',
                'eden_fresh_seconds','r16_fresh_seconds','eden_warm_seconds','r16_warm_seconds']
        writer=csv.DictWriter(stream,fields)
        writer.writeheader()
        for r in records:
            writer.writerow({'sample_id':r['sample_id'],'atoms':r['atoms'],'grid_points':r['grid_points'],
                **{n+'_test_nmae_percent':100*r[n+'_accuracy']['test_nmae'] for n in ('eden','r16')},
                **{n+'_fresh_seconds':r['fresh'][n]['median_seconds'] for n in ('eden','r16')},
                **{n+'_warm_seconds':r['stages'][n+'_warm_total']['median_seconds'] for n in ('eden','r16')}})
    lines=['# EdenGNN / R16：8 个 NMC 结构的配对诊断','',
           '本轮使用相同的结构 0–7、每结构 439,040 个网格点、原 80/10/10 空间划分，'
           '以及同一 RTX 5060 Ti。空间测试点来自已见结构，不代表未知结构泛化。','',
           '## 主要结果','',
           '| 指标 | EdenGNN 密度分支 | 当前 R16 |','|---|---:|---:|']
    items=[('测试 NMAE，逐结构宏平均 (%)','test_macro_nmae',100),
           ('新结构完整网格推理 (s)','fresh_mean_of_structure_medians_seconds',1),
           ('新结构推理 P90，合并全部重复 (s)','fresh_pooled_p90_seconds',1),
           ('复用几何后的完整网格推理 (s)','warm_mean_of_structure_medians_seconds',1),
           ('新结构推理峰值增量显存 (MiB)','fresh_peak_increment_mib',1)]
    for label,key,scale in items:
        lines.append(f'| {label} | {summary["eden"][key]*scale:.4f} | {summary["r16"][key]*scale:.4f} |')
    ratio=summary['r16']['fresh_mean_of_structure_medians_seconds']/summary['eden']['fresh_mean_of_structure_medians_seconds']
    lines+=['',f'R16 / EdenGNN 新结构耗时比为 **{ratio:.2f}**。NMAE 定义为 Σ|预测−DFT| / Σ|DFT|；'
            '先在各结构测试点计算，再取算术平均。','',
            f'每个结构预热 {benchmark["warmup"]} 次，重复 {benchmark["repeats"]} 次。'
            '表中时间为“各结构重复中位数的平均值”；P90 是全部 40 次观测的分位数。'
            '新结构时间包含原子图、网格几何和 EdenGNN 的 SAD 构造；模型及 POTCAR 文件加载另列。'
            'CHGCAR 读取含真值标签，仅作数据管线诊断，不作为推理输入要求。','',
            '## 耗时拆分','',
            '| 阶段 | EdenGNN (s) | R16 (s) |','|---|---:|---:|']
    pairs=[('原子图','eden_atom_graph','r16_atom_graph'),
           ('网格几何','eden_grid_geometry','r16_grid_geometry_production'),
           ('GNN 主干','eden_encoder','r16_encoder'),
           ('系数输出头','eden_coefficient_head','r16_coefficient_head'),
           ('完整网格重构（Eden 含输出头）','eden_decoder_including_head','r16_decoder'),
           ('缓存几何、2048 查询点分块解码','eden_query_chunk_decoder','r16_query_chunk_decoder')]
    for label,e,r in pairs:
        lines.append(f'| {label} | {stage(e):.5f} | {stage(r):.5f} |')
    lines+=['',f'EdenGNN SAD 构造：{stage("eden_sad"):.5f} s。'
            f'R16 KD-tree 网格准备替代路径：{stage("r16_grid_geometry_kdtree"):.5f} s。'
            f'EdenGNN 查询 CSR 准备：{stage("eden_grid_geometry_csr"):.5f} s。',
            '各阶段分别预热并同步测量，阶段之和不要求与整段计时相等。查询分块采用各模型原有重构算子，'
            '仅统一查询批量；其缓存建立费用已单列。','',
            '### 瓶颈与后续优先级','',
            f'R16 的网络（含输出头）为 {stage("r16_network_including_head"):.5f} s，'
            f'网格几何为 {stage("r16_grid_geometry_production"):.5f} s，'
            f'重构为 {stage("r16_decoder"):.5f} s。'
            '应根据这些实测量级决定下一步，不能仅凭“每个原子输出少量系数”推断完整密度场会更快。','',
            '1. 首先优化 R16 的网格重构：减少小块循环和重复球谐/径向计算，缓存常量，'
            '核查白化矩阵能否先作用于每原子系数，再进行逐点收缩。代数重排必须比较密度与梯度误差；'
            '当前高条件数基组需要 float64，不能直接降到 float32 并假定精度不变。',
            '2. 针对规则周期网格，优化原子到局部网格的候选生成及周期折回，避免反复扫描全部探针。'
            '是否采用 KD-tree 要由本轮计时决定，不能把预缓存成本省略。',
            '3. 精度方面，将“系数预测误差”和“基组表示误差”分开。若更换数值更稳定的基组或引入 SAD 残差，'
            '需要独立训练和消融；这些属于下一轮模型工作，本轮没有用测试结果调参。',
            '4. 完整场密度预测达到精度与速度目标后，再进行 VASP 初始化/SCF 对照，'
            '检验密度质量是否实际减少迭代，以及总 DFT 时间是否下降。','',
            '### 重构内部诊断','',
            '下面的算子计时有额外同步或 CUDA event 开销，只用于归因，不能替代上面的无插桩总耗时。','',
            '| 模型 | 算子 | 平均诊断耗时 (s) |','|---|---|---:|']
    for name,key,inner in [('EdenGNN','eden_decoder_synchronized_diagnostic_seconds',None),
                           ('R16','r16_decoder_event_diagnostic','cuda_event_seconds')]:
        mappings=[r[key] if inner is None else r[key][inner] for r in records]
        for op in sorted(mappings[0],key=lambda op:mean([d[op] for d in mappings]),reverse=True):
            lines.append(f'| {name} | {op} | {mean([d[op] for d in mappings]):.5f} |')
    diagnostic=summary['r16_algebraic_diagnostic']
    lines+=['','## R16 代数重排诊断（主表以外）','',
            '保持检查点、基组和 float64 精度不变，把白化与 scale^(-l) 预先作用到每原子系数上，'
            '把 5 个角动量阶重复计算的高斯包络合并为一次，并使用固定 65,536 边块。'
            '这是组合工程诊断，尚未分别消融每项贡献。未替换生产代码，也未按 DFT 测试误差选择版本。','',
            '| 指标 | 当前 R16 | 重排诊断版本 |','|---|---:|---:|',
            f'| 重构 (s) | {stage("r16_decoder"):.5f} | {diagnostic["decoder_seconds"]:.5f} |',
            f'| 新结构完整推理 (s) | {summary["r16"]["fresh_mean_of_structure_medians_seconds"]:.5f} | {diagnostic["fresh_seconds"]:.5f} |',
            f'| 测试 NMAE (%) | {100*summary["r16"]["test_macro_nmae"]:.6f} | {100*diagnostic["test_macro_nmae"]:.6f} |','',
            f'相对原实现，8 个完整网格的最大逐点绝对差 {diagnostic["max_abs_difference_from_original"]:.3e} e/Å³，'
            f'最大相对 L1 差 {diagnostic["max_relative_l1_difference_from_original"]:.3e}。'
            '另有含重复目标索引和多块边的系数/坐标梯度测试。此结果检验实现层面的可优化空间，'
            '不解决当前模型的拟合精度差距。','',
            '## 逐结构结果','',
            '| 结构 | 原子数 | Eden 测试 NMAE (%) | R16 测试 NMAE (%) | Eden 新结构 (s) | R16 新结构 (s) |',
            '|---|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f'| {r["sample_id"]} | {r["atoms"]} | {100*r["eden_accuracy"]["test_nmae"]:.4f} | '
                     f'{100*r["r16_accuracy"]["test_nmae"]:.4f} | {r["fresh"]["eden"]["median_seconds"]:.4f} | '
                     f'{r["fresh"]["r16"]["median_seconds"]:.4f} |')
    lines+=['','## 精度与物理量检查','',
            '| 模型 | 训练 NMAE (%) | 验证 NMAE (%) | 测试 NMAE (%) | 全网格 NMAE (%) | 测试电荷偏差绝对值均值 (%) | 负密度积分比例 (%) |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for n in ('eden','r16'):
        values=[100*mean([r[n+'_accuracy'][k] for r in records]) for k in
                ('train_nmae','validation_nmae','test_nmae','full_grid_nmae')]
        values+=[100*mean([abs(r[n+'_accuracy']['test_charge_error']) for r in records]),
                 100*mean([r[n+'_accuracy']['negative_fraction'] for r in records])]
        lines.append('| '+n+' | '+' | '.join(f'{v:.6f}' for v in values)+' |')
    lines+=['','预测未裁剪，未做电荷重归一化。测试点电荷偏差是该子集的有符号密度积分偏差，'
            '不能直接当作整个晶胞电子数误差。负密度比例使用完整网格。','',
            '## 对照的模型差异','',
            '| 项目 | EdenGNN 密度分支 | 当前 R16 |','|---|---|---|',
            '| 参数量 | 6,194,410 | 521,472 |',
            '| 每原子密度输出 | 4 × 49 = 196 个系数 | 16 × 25 = 400 个系数 |',
            '| 径向表示 | 按元素学习的径向 MLP | 固定高斯径向基 + 分阶白化 |',
            '| 密度角动量截断 | l ≤ 6 | l ≤ 4 |',
            '| 密度邻域 | 最近网格点中心的 4 Å 原生 stencil | 真实原子中心 3 Å cutoff |',
            '| 起点 | POTCAR 推导的 SAD + 学习残差 | 学习的元素标量基线 + 系数预测 |',
            '| 解码浮点精度 | float32 | float64（最后输出 float32） |','',
            '因此本轮公平性是同数据、同划分、同硬件与同更新/采样预算；并未控制参数量、物理先验、'
            '函数空间或解码精度。结果可评价当前两个实现的实用表现，不能独立归因于某一架构选择。','',
            '## 可复核性与边界','',
            f'- EdenGNN 官方代码 commit `{settings["provenance"]["commit"]}`；'
            '原版 Encoder / DensityLayer，原版周期 stencil，未运行 PAW 分支。',
            f'- 参数量：EdenGNN {settings["parameter_count"]:,}；R16 521,472。两者均为 16,000 次更新、每步 1,024 训练点，'
            '训练顺序与采样 RNG 对齐。不是等参数量、等训练计算量或论文原始训练协议。',
            f'- Eden 最佳验证步骤 {selection["best_step"]}，固定验证子集 NMAE '
            f'{100*selection["best_validation_nmae"]:.5f}%。只依据验证集选择，随后冻结 checkpoint 再评估测试。',
            '- Eden 的 SAD 由本地 POTCAR 原子 form-factor 构造，不拟合目标密度；尚未与 VASP 原生零步密度逐点核验。'
            '这一适配限制意味着结果应称为“官方密度分支在本数据协议下的复现”。',
            '- 两个网络 float32，TF32 关闭；Eden 重构 float32，R16 白化重构 float64。此差异属于当前实现，'
            '不能把时间差全部归因于模型数学结构。',
            '- 训练结束后串行计时，CUDA 显式同步；显存增量排除已驻留模型和输入。绝对峰值包含两个驻留模型，原始 JSON 保留。',
            '- 训练中使用经全参数梯度审核的 CUDA Graph 执行加速；正式推理在独立进程使用原版 eager 前向。'
            '恢复步骤和审核记录见 execution_acceleration*.json；训练调度速度不作为本轮模型推理对比结果。',
            '- 本轮是单种子、8 个已见结构的空间插值诊断；既有测试结果已公开。不能据此声称跨结构泛化或统计显著性。',
            '- 本轮未测 VASP SCF 步数或总时间，尚不能据密度预测加速断言 DFT 整体加速。','',
            '### 文件','',
            '- 原始计时、逐次观测、峰值显存、硬件快照：`runs/edengnn_nmc_2026-10-28/benchmark.json`。',
            '- 训练配置、采样审计、冻结哈希、完整结果：`runs/edengnn_nmc_2026-10-28/eden_eight/`。',
            '- 配对逐结构表：`per_structure.csv`；汇总数值：`summary.json`。',
            '- 锁定协议：`reports/nmc_edengnn_protocol_2026-10-28.md`。','',
            '```powershell',
            '# 安装同一环境，并按协议克隆固定 commit；运行根目录为仓库根目录。',
            'python -m scripts.train_nmc_edengnn --resume',
            'python -m scripts.benchmark_nmc_edengnn --repeat 5',
            'python -m scripts.summarize_nmc_edengnn','```','']
    (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,key,label in [(axes[0],'test_macro_nmae','Test NMAE (%)'),
                         (axes[1],'fresh_mean_of_structure_medians_seconds','Full-grid fresh inference (s)')]:
        scale=100 if key=='test_macro_nmae' else 1
        values=[summary[n][key]*scale for n in ('eden','r16')]
        bars=ax.bar(['EdenGNN','R16'],values,color=['#287c8e','#c67a38'])
        ax.bar_label(bars,fmt='%.3f',padding=3)
        ax.set_ylabel(label)
        ax.set_ylim(0,max(values)*1.18)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('8 known NMC structures | fixed spatial split | RTX 5060 Ti')
    fig.savefig(REPORT/'comparison.png',dpi=180)
    plt.close(fig)
    print(json.dumps(summary,indent=2))
    print(REPORT/'report.md')


if __name__=='__main__':
    main()
