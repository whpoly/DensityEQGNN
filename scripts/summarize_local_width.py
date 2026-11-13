"""Summarize the predeclared single-layer width sweep, including every seed."""
import json
from pathlib import Path

import numpy as np


def main():
    root = Path('runs/local_width_2026-10-29')
    dest = Path('reports/local_width_2026-10-29')
    settings = json.loads((root/'settings.json').read_text())
    result = json.loads((root/'result.json').read_text())
    widths = [64]+settings['widths']
    assert len(result['accuracy']) == len(widths)*len(settings['seeds'])
    assert len(result['timing']) == len(settings['samples'])
    summary = {}
    for name in ['original']+[f'local_{w}' for w in widths]:
        seconds = np.mean([r['variants'][name]['median_seconds'] for r in result['timing']])
        if name=='original':
            errors = [r['test_nmae'][name]*100 for r in result['timing']]
            parameters = 892545
        else:
            width = int(name.split('_')[1])
            rows = [r for r in result['accuracy'] if r['width']==width]
            errors = [r['test']['macro_nmae']*100 for r in rows]
            parameters = rows[0]['head_parameters']
        summary[name] = {'test_nmae_percent':float(np.mean(errors)), 'seconds':float(seconds),
                         'active_query_parameters':parameters}
        if name!='original':
            summary[name]['seed_nmae_percent'] = errors
    reference = summary['original']
    for item in summary.values():
        item['speedup'] = reference['seconds']/item['seconds']
        item['error_ratio_to_original'] = item['test_nmae_percent']/reference['test_nmae_percent']
    best_width = min(widths,key=lambda w:summary[f'local_{w}']['test_nmae_percent'])
    # This minimum is descriptive only: every predeclared model is reported.
    best = summary[f'local_{best_width}']
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines = ['# 单层 local 查询网络：增宽到 128 / 256 通道', '',
        f"本轮单层模型最低平均测试误差来自 {best_width} 通道，为 {best['test_nmae_percent']:.6f}%；"
        f"原三层 128 通道模型为 {reference['test_nmae_percent']:.6f}%。"
        f"该单层模型误差是原模型的 {best['error_ratio_to_original']:.3f} 倍，"
        f"新输入完整网格速度是原优化接口的 {best['speedup']:.2f} 倍。", '',
        '## 误差与速度', '',
        '误差为 8 个结构空间留出的测试 NMAE 宏平均，再对两个训练种子取平均。'
        '速度使用预先指定种子 42，各结构三次交错测量中位数再宏平均。', '',
        '| 查询结构 | 测试 NMAE (%) | 完整网格 (s) | 加速 | 相对原模型误差倍数 | 查询侧启用参数 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in summary.items():
        label = '原模型：3 层 × 128' if name=='original' else f"local：1 层 × {name.split('_')[1]}"
        lines.append(f"| {label} | {r['test_nmae_percent']:.6f} | {r['seconds']:.6f} | {r['speedup']:.2f}× | {r['error_ratio_to_original']:.3f} | {r['active_query_parameters']} |")
    lines += ['', '所有模型共用同一个 594688 参数原子编码器。单层模型查询侧计数不含已关闭、未使用的 global 模块。', '',
              '## 分种子结果', '',
              '| 通道 | 种子 | 最优验证步 | 验证 selection (%) | 独立空间测试 (%) |',
              '|---:|---:|---:|---:|---:|']
    for r in result['accuracy']:
        lines.append(f"| {r['width']} | {r['seed']} | {r['step']} | {r['selection']['macro_nmae']*100:.6f} | {r['test']['macro_nmae']*100:.6f} |")
    lines += ['', '## 设置与解释边界', '',
        '- 单层公式与上一轮完全一致：三层冻结原子特征拼接、一次原子投影、一次局域消息聚合、一次 PaiNN update 和标量读出。',
        '- 本轮只把单层通道从 64 扩到 128、256；邻域仍为 4 Å，径向编码仍为 30，无 global、SAD、密度插值或电荷后处理。',
        '- 128/256 每个宽度两个种子各训练 6000 步，每步 1024 个训练查询点；64 直接复用上一轮同预算检查点。'
        '训练结构顺序、查询点序列、优化器和验证选择规则一致。不同宽度的随机权重形状不同，不宣称初始函数相同。',
        '- 原 DeepDFT 参照是 16000 步端到端训练；本轮冻结原子编码器，只重训单层头。'
        '与原模型的差距不能全部归因于层数，也不代表单层架构在更多训练或联合微调后的上限。',
        '- 检查点仅由固定验证 selection 选择，全部宽度训练完才统一评估测试。'
        '测试集曾用于此前研究，非新的盲测数据集。所有预定宽度及种子均报告，不按测试挑选种子。',
        '- 8 个结构均被训练见过，只有网格点留出；没有测试结构外推或 SCF 收敛。',
        '- 本轮重新计时全部四种模型。每次重建原子图与查询图、重算原子编码与投影，输出每结构 439040 个点。'
        '缓存的训练特征没有进入计时路径；文件读取、检查点加载和驻留输入传输不计入。',
        '- RTX 5060 Ti，float32 网络、TF32 关闭；先预热，显式 CUDA 同步，同进程交错三轮。'
        '逐结构检查完整推理路径与缓存评估路径的测试误差一致。', '',
        '## 每结构误差', '',
        '| 通道 | 种子 | 结构 | 测试 NMAE (%) |', '|---:|---:|---:|---:|']
    for r in result['accuracy']:
        for sample in r['test']['records']:
            lines.append(f"| {r['width']} | {r['seed']} | {sample['id']} | {sample['nmae']*100:.6f} |")
    lines += ['', '## 复现', '',
        '最终回归检查：7 项测试通过。', '',
        '增宽测试检查反向梯度有限、global 分支关闭及分块一致性；此前的旋转/排列不变性、'
        '周期边读取和原推理等价性也一并回归。测试日志在 runs/local_width_2026-10-29/tests.log。', '',
        '```powershell',
        "& 'C:/Users/User/.conda/envs/kan-cgcnn-cuda/python.exe' -m scripts.verify_local_width",
        "& 'C:/Users/User/.conda/envs/kan-cgcnn-cuda/python.exe' -m scripts.summarize_local_width",
        '```', '', '![Validation curves](validation_curves.png)', '', '![Accuracy and speed](accuracy_speed.png)', '']
    (dest/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {64:'#4271b5',128:'#2c9477',256:'#d95c3a'}
    fig,axes = plt.subplots(1,len(settings['seeds']),figsize=(11,4),sharey=True)
    for ax,seed in zip(np.atleast_1d(axes),settings['seeds']):
        for width in widths:
            directory = (Path('runs/mixed_embedding_2026-10-29') if width==64 else root/f'width_{width}')/f'seed_{seed}'
            hist = json.loads((directory/'history.json').read_text())
            ax.plot([r['step'] for r in hist],[r['metrics']['local']['macro_nmae']*100 for r in hist],
                    label=f'1 layer, {width} channels',color=colors[width],marker='o',markersize=3)
        ax.set(title=f'Seed {seed}',xlabel='Training step',yscale='log')
        ax.grid(alpha=.2);ax.legend()
    np.atleast_1d(axes)[0].set_ylabel('Validation selection NMAE (%)')
    fig.tight_layout();fig.savefig(dest/'validation_curves.png',dpi=160)
    fig,ax = plt.subplots(figsize=(7,4.5))
    for name,r in summary.items():
        width = None if name=='original' else int(name.split('_')[1])
        label = 'Original (3 x 128)' if width is None else f'Local (1 x {width})'
        ax.scatter(r['seconds'],r['test_nmae_percent'],color='#333333' if width is None else colors[width],s=70)
        ax.annotate(label,(r['seconds'],r['test_nmae_percent']),xytext=(6,6),textcoords='offset points',fontsize=9)
    ax.set(xlabel='Full-grid inference time (s)',ylabel='Test NMAE (%)',
           title='Single-layer width sweep: lower and left are better')
    ax.margins(x=.25,y=.25);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(dest/'accuracy_speed.png',dpi=160)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
