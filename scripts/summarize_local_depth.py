"""Report every predeclared local-query depth result and its inference cost."""
import json
from pathlib import Path

import numpy as np


def main():
    root = Path('runs/local_depth_2026-10-30')
    dest = Path('reports/local_depth_2026-10-30')
    result = json.loads((root/'result.json').read_text())
    settings = json.loads((root/'settings.json').read_text())
    assert len(result['accuracy'])==(len(settings['depths'])+2)*len(settings['seeds'])
    assert len(result['timing'])==len(settings['samples'])
    names = ['original','local_1x128']+[f'local_{d}x128' for d in settings['depths']]+['local_1x256']
    summary = {}
    for name in names:
        seconds = np.mean([r['variants'][name]['median_seconds'] for r in result['timing']])
        if name=='original':
            errors = [r['test_nmae'][name]*100 for r in result['timing']]
            parameters = 892545
        else:
            rows = [r for r in result['accuracy'] if r['name']==name]
            errors = [r['test']['macro_nmae']*100 for r in rows]
            parameters = rows[0]['head_parameters']
        summary[name] = {'test_nmae_percent':float(np.mean(errors)),'seconds':float(seconds),
                         'active_query_parameters':parameters}
        if name!='original':
            summary[name]['seed_nmae_percent'] = errors
    for row in summary.values():
        row['speedup'] = summary['original']['seconds']/row['seconds']
        row['error_ratio_to_original'] = row['test_nmae_percent']/summary['original']['test_nmae_percent']
        row['relative_error_reduction_vs_1x128'] = 1-row['test_nmae_percent']/summary['local_1x128']['test_nmae_percent']
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    def label(name):
        return '原 DeepDFT：3×128' if name=='original' else 'local：'+name.split('_')[1].replace('x','×')
    lines = ['# Local 查询头增加层数：固定 128 通道', '',
        '新增 2×128、3×128 查询头，原子编码器冻结，与已有单层 128/256 和原 DeepDFT 同进程比较。'
        '每个新模型两个种子，各训练 6000 步。', '',
        '## 精度与新输入完整推理', '',
        '| 查询结构 | 测试 NMAE (%) | 完整网格 (s) | 相对原模型加速 | 查询侧启用参数 | 相对单层128误差降幅 (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for name,r in summary.items():
        lines.append(f"| {label(name)} | {r['test_nmae_percent']:.6f} | {r['seconds']:.6f} | {r['speedup']:.2f}× | {r['active_query_parameters']} | {r['relative_error_reduction_vs_1x128']*100:.2f} |")
    lines += ['', '误差为 8 个结构的空间测试 NMAE 宏平均，再对两个训练种子平均。'
        '耗时预先指定种子 42，每结构三次交错测量中位数，再对 8 个结构平均。'
        '所有模型另共用 594688 参数的原子编码器。', '',
        '## 分种子结果', '',
        '| 查询头 | 种子 | 最优验证步 | selection NMAE (%) | test NMAE (%) |',
        '|---|---:|---:|---:|---:|']
    for name in names[1:]:
        for r in [r for r in result['accuracy'] if r['name']==name]:
            lines.append(f"| {label(name)} | {r['seed']} | {r['step']} | {r['selection']['macro_nmae']*100:.6f} | {r['test']['macro_nmae']*100:.6f} |")
    lines += ['', '## 增加的具体计算', '',
        '- 原子编码器的三层标量/向量特征拼接，投影成 128 通道的原子表示，每结构只计算一次。',
        '- 各查询层使用同一原子表示，但有独立消息 MLP、径向滤波、门控和 PaiNN update。'
        '查询标量/向量状态逐层递推；每层重新计算边消息及聚合。几何、距离编码在层间共享。',
        '- 增加查询层不增加共享原子 embedding 的 128 通道容量。本轮没有测试逐层独立原子投影；'
        '共享表示是否限制最终精度，需要另行消融，不能仅由这组结果确定。',
        '- depth=1 与上一轮单层实现的输出和梯度一致；没有改变 4 Å 邻域或 30 维径向编码。无 global、SAD、插值和电荷后处理。',
        '- 单层 256 保留为宽度参照，以便对比更深和更宽的精度/时间成本。', '',
        '## 训练、计时和结论边界', '',
        '- 新头种子 42/43，6000 步，每步 1024 个训练点；原 80/10/10 空间划分、训练点顺序、Adam、cosine 学习率、归一化 L1、裁剪阈值均保持一致。',
        '- 仅按原验证 selection 选择检查点；所有新配置训练结束后统一测试。旧单层检查点直接复用，不重新选择。'
        '测试集已用于前序研究，不是新的盲测。',
        '- 新增层沿用旧单层第一层及投影的同种子初始权重，但额外层使初始函数不同；不宣称不同深度初始预测一致。',
        '- 原 DeepDFT 参照经过 16000 步端到端训练；本轮是冻结原子编码器、随机初始化查询头的 6000 步训练。',
        '- 新 3×128 与原 DeepDFT 不是完全相同的查询结构：本轮各层用共享投影后的拼接原子特征，原模型每层用对应原子层的特征。'
        '不能把差距全部归因于训练步数，也不能视作新结构充分训练后的上限。',
        '- 每次计时重新构建原子/查询图、运行原子网络、原子投影和全部 439040 点查询。'
        '未使用训练特征缓存；文件/检查点加载和驻留输入传输不计入。',
        '- RTX 5060 Ti、float32 网络、TF32 关闭，先预热，显式 CUDA 同步、同进程交错三轮。'
        '逐结构核对缓存评估与新输入完整推理的测试误差一致。',
        '- 只验证已见 8 个结构内空间留出，不证明未见结构泛化、MD 新帧表现或 SCF 收敛。', '',
        '## 每结构测试误差', '',
        '| 查询头 | 种子 | 结构 | NMAE (%) |','|---|---:|---:|---:|']
    for r in result['accuracy']:
        for s in r['test']['records']:
            lines.append(f"| {label(r['name'])} | {r['seed']} | {s['id']} | {s['nmae']*100:.6f} |")
    lines += ['', '## 检查与复现', '',
        '完整回归：10 项测试通过。', '',
        '新增检查覆盖 depth=1 与旧实现的输出/梯度一致性、多层旋转不变性、分块一致性和每层消息参数都有有限梯度。'
        '完整回归日志位于 runs/local_depth_2026-10-30/tests.log。', '',
        '```powershell',
        "& 'C:/Users/User/.conda/envs/kan-cgcnn-cuda/python.exe' -m scripts.verify_local_depth",
        "& 'C:/Users/User/.conda/envs/kan-cgcnn-cuda/python.exe' -m scripts.summarize_local_depth",
        '```', '', '![Validation curves](validation_curves.png)', '', '![Accuracy and speed](accuracy_speed.png)', '']
    (dest/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'original':'#333333','local_1x128':'#4271b5','local_2x128':'#279578',
              'local_3x128':'#d75c3a','local_1x256':'#9168b7'}
    fig,axes = plt.subplots(1,len(settings['seeds']),figsize=(11,4),sharey=True)
    for ax,seed in zip(np.atleast_1d(axes),settings['seeds']):
        for name in names[1:]:
            depth,width = map(int,name.split('_')[1].split('x'))
            folder = (Path('runs/local_width_2026-10-29')/f'width_{width}' if depth==1
                      else root/f'depth_{depth}')/f'seed_{seed}'
            hist = json.loads((folder/'history.json').read_text())
            ax.plot([r['step'] for r in hist],[r['metrics']['local']['macro_nmae']*100 for r in hist],
                    label=f'{depth} x {width}',color=colors[name],marker='o',markersize=3,
                    linestyle='--' if width==256 else '-')
        ax.set(title=f'Seed {seed}',xlabel='Training step',yscale='log')
        ax.grid(alpha=.2);ax.legend()
    np.atleast_1d(axes)[0].set_ylabel('Validation selection NMAE (%)')
    fig.tight_layout();fig.savefig(dest/'validation_curves.png',dpi=160)
    fig,ax = plt.subplots(figsize=(8,4.5))
    for name,r in summary.items():
        label_text = 'Original (3 x 128)' if name=='original' else name.split('_')[1].replace('x',' x ')
        ax.scatter(r['seconds'],r['test_nmae_percent'],color=colors[name],s=65)
        offset = (-8,8) if name=='original' else (7,7)
        ax.annotate(label_text,(r['seconds'],r['test_nmae_percent']),xytext=offset,
                    textcoords='offset points',ha='right' if name=='original' else 'left',fontsize=9)
    ax.set(xlabel='Full-grid inference time (s)',ylabel='Test NMAE (%)',
           title='Local query depth: lower and left are better')
    ax.margins(x=.25,y=.25);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(dest/'accuracy_speed.png',dpi=160)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
