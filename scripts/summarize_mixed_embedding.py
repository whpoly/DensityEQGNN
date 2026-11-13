"""Report all predeclared local/global embedding results, without selecting seeds."""
import json
from pathlib import Path

import numpy as np


def main():
    root = Path('runs/mixed_embedding_2026-10-29')
    dest = Path('reports/mixed_embedding_2026-10-29')
    result = json.loads((root/'result.json').read_text())
    settings = json.loads((root/'settings.json').read_text())
    assert len(result['accuracy']) == len(settings['seeds'])*3
    assert len(result['timing']) == len(settings['samples'])
    counts = json.loads((root/f"seed_{settings['seeds'][0]}"/'parameters.json').read_text())
    names = {'original_flat':'原 DeepDFT（已优化推理）', 'local':'小 local',
             'constant':'小 local＋恒定条件', 'global':'小 local＋global embedding'}
    summary = {}
    for mode in names:
        times = [r['variants'][mode]['median_seconds'] for r in result['timing']]
        errors = ([r['test_nmae'][mode] for r in result['timing']] if mode=='original_flat'
                  else [r['test']['macro_nmae'] for r in result['accuracy'] if r['mode']==mode])
        summary[mode] = {'nmae_percent':float(np.mean(errors)*100),
                         'seconds':float(np.mean(times)),
                         ('structure_nmae_percent' if mode=='original_flat' else 'seed_nmae_percent'):[v*100 for v in errors]}
    for mode, value in summary.items():
        value['speedup_vs_optimized_original'] = summary['original_flat']['seconds']/value['seconds']
    comparisons = []
    for seed in settings['seeds']:
        rows = {r['mode']:r for r in result['accuracy'] if r['seed']==seed}
        comparisons.append({'seed':seed,
            'global_relative_reduction_vs_local':1-rows['global']['test']['macro_nmae']/rows['local']['test']['macro_nmae'],
            'global_relative_reduction_vs_constant':1-rows['global']['test']['macro_nmae']/rows['constant']['test']['macro_nmae'],
            'swapped_global_nmae_percent':rows['global']['test_swapped_global']['macro_nmae']*100})
    (dest/'summary.json').write_text(json.dumps({'models':summary,'paired_comparisons':comparisons},indent=2),encoding='utf-8')
    local_error = summary['local']['nmae_percent']
    global_error = summary['global']['nmae_percent']
    difference = (global_error/local_error-1)*100
    lines = ['# Local/global 混合 embedding：查询网络压缩验证', '',
        f"本轮混合版较已优化原模型加速 {summary['global']['speedup_vs_optimized_original']:.2f}×，"
        f"但测试 NMAE 为 {global_error:.6f}%，纯小 local 为 {local_error:.6f}%；"
        f"混合版相对误差变化 {difference:+.2f}%。"
        '是否值得采用须同时看速度和精度，不能将查询网络压缩的收益归因于 global 信息。', '',
        '## 实测结果', '',
        '误差为 8 个结构的空间测试 NMAE 宏平均；新训练模型再对两个随机种子取平均。'
        '时间为预先指定的种子 42，每结构三次中位数，再对 8 个结构取平均。', '',
        '| 方法 | 测试 NMAE (%) | 新输入完整网格 (s) | 相对优化原模型加速 |',
        '|---|---:|---:|---:|']
    for mode in names:
        row = summary[mode]
        lines.append(f"| {names[mode]} | {row['nmae_percent']:.6f} | {row['seconds']:.6f} | {row['speedup_vs_optimized_original']:.2f}× |")
    lines += ['', '原模型是既有 16000 步端到端训练检查点。小模型复用并冻结同一个原子编码器，'
              '从相同随机初始化训练查询头 6000 步；参照模型与新模型不是同训练预算比较。', '',
              '## 两个种子分别报告', '',
              '| 种子 | 方法 | 最优验证步 | selection NMAE (%) | test NMAE (%) |',
              '|---:|---|---:|---:|---:|']
    for r in result['accuracy']:
        lines.append(f"| {r['seed']} | {names[r['mode']]} | {r['best_step']} | {r['selection']['macro_nmae']*100:.6f} | {r['test']['macro_nmae']*100:.6f} |")
    lines += ['', '## Global 信息是否有贡献', '',
              '正的相对误差降幅表示 global 更好；负值表示更差。交换 global 为下一结构的 embedding '
              '只作依赖诊断，不能单独证明模型学到了可泛化的全局物理。', '',
              '| 种子 | 相对小 local 降幅 (%) | 相对恒定条件降幅 (%) | 交换 global 后测试 NMAE (%) |',
              '|---:|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['seed']} | {r['global_relative_reduction_vs_local']*100:.3f} | {r['global_relative_reduction_vs_constant']*100:.3f} | {r['swapped_global_nmae_percent']:.6f} |")
    lines += ['', '## 结构与计算', '',
              '- 原查询侧：3 层、128 通道。新查询侧：1 层、64 通道；4 Å 邻域和 30 维径向编码均保留。',
              '- 原原子编码器 594688 参数，原查询侧 892545 参数。混合版查询侧 146337 参数，'
              '加同一个原子编码器共 741025 参数；纯 local 的启用参数共 724481。',
              '- 三层原子标量/向量特征拼接，再分别投影到 64 通道；这些投影每结构只做一次。',
              '- Global 描述：最终原子层标量均值、标准差和向量模长均值，共 384 维；'
              '无标签特征标准化后，经 32 维 MLP 瓶颈输出 128 个 FiLM 参数，每结构只算一次。',
              '- 每点融合：s → s × (1 + 0.1 tanh(scale)) + shift，再由小 MLP 读出密度。'
              '无 SAD、密度插值或输出归一化。',
              '- 恒定条件版本与 global 版本网络相同，但输入为固定零向量；其中部分权重没有有效自由度，'
              '因此它控制了额外条件模块，不代表严格相同有效容量。', '',
              '| 新查询头 | 分配参数 | 可训练参数 |', '|---|---:|---:|']
    for mode,c in counts.items():
        lines.append(f"| {names[mode]} | {c['allocated']} | {c['trainable']} |")
    lines += ['', '## 计时与数据边界', '',
              '- RTX 5060 Ti，float32 网络、TF32 关闭；输入每结构 439040 点。',
              '- 各方法预热后，同进程交错计时并显式 CUDA 同步。'
              '每次重建查询邻域和原子图、重新运行原子网络及所有 embedding、重建完整网格。',
              '- 文件/检查点加载和已驻留输入传输不计入；训练时缓存的原子表示没有用于加速推理计时。',
              '- 三组每个种子使用完全相同的训练点序列、优化器、步数和验证选择规则。'
              '所有种子训练结束后才读取测试指标；原检查点和划分哈希保存在 runs 中。',
              '- 这里只测试 8 个已见结构的空间留出；global 表示可能编码结构身份。'
              '未验证未见结构、MD 新帧或 SCF 收敛，也未证明所有 local/global 架构的上限。',
              '- 冻结原子编码器用于隔离查询头差异；结果不等同于从头端到端训练或共同微调的结果。', '',
              '## 每结构测试误差', '',
              '| 种子 | 结构 | local (%) | 恒定条件 (%) | global (%) |', '|---:|---:|---:|---:|---:|']
    for seed in settings['seeds']:
        rows = {r['mode']:r for r in result['accuracy'] if r['seed']==seed}
        for k in range(len(settings['samples'])):
            sid = rows['local']['test']['records'][k]['id']
            values = [rows[m]['test']['records'][k]['nmae']*100 for m in ('local','constant','global')]
            lines.append(f'| {seed} | {sid} | '+ ' | '.join(f'{v:.6f}' for v in values)+' |')
    lines += ['', '## 验证与复现', '',
              '最终回归检查：5 项测试通过（tests/test_mixed_embedding.py 与 tests/test_deepdft_inference_fast.py）。', '',
              '测试覆盖：周期 CSR 重复/乱序查询、旋转不变性、原子排列不变性、global 条件确实影响输出、'
              '恒定条件不受结构描述变化影响、梯度可达、初始预测匹配及分块一致性。'
              '正式全网格预测还逐结构核对了缓存评估与新输入路径的测试误差一致性。', '',
              '```powershell',
              "& 'C:/Users/User/.conda/envs/kan-cgcnn-cuda/python.exe' -m scripts.verify_mixed_embedding",
              "& 'C:/Users/User/.conda/envs/kan-cgcnn-cuda/python.exe' -m scripts.summarize_mixed_embedding",
              '```', '', '训练曲线只使用验证 selection，不使用测试集调参。', '',
              '![Validation curves](validation_curves.png)', '']
    (dest/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,len(settings['seeds']),figsize=(11,4),sharey=True)
    colors = {'local':'#4271b5','constant':'#888888','global':'#d95c3a'}
    for ax,seed in zip(np.atleast_1d(axes),settings['seeds']):
        history = json.loads((root/f'seed_{seed}'/'history.json').read_text())
        for mode in ('local','constant','global'):
            ax.plot([r['step'] for r in history], [r['metrics'][mode]['macro_nmae']*100 for r in history],
                    label=mode,color=colors[mode],marker='o',markersize=3)
        ax.set(title=f'Seed {seed}',xlabel='Training step',yscale='log')
        ax.grid(alpha=.2)
        ax.legend()
    np.atleast_1d(axes)[0].set_ylabel('Validation selection NMAE (%)')
    fig.tight_layout()
    fig.savefig(dest/'validation_curves.png',dpi=160)
    print(json.dumps({'models':summary,'paired_comparisons':comparisons},indent=2))


if __name__=='__main__':
    main()
