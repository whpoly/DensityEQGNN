"""Report fixed-weight cubic far-message interpolation and its memory tradeoff."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_deepdft_reuse import save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run',type=Path,default=Path('runs/deepdft_cubic_far_2026-10-30'))
    p.add_argument('--output',type=Path,default=Path('reports/deepdft_cubic_far_2026-10-30'))
    args = p.parse_args()
    result = json.loads((args.run/'result.json').read_text())
    settings = json.loads((args.run/'settings.json').read_text())
    selection = result['selection']
    assert len(result['records'])==len(settings['samples'])
    assert result['source_hashes_match'] and result['checkpoint_hash_matches']
    rows = []
    for name in result['records'][0]['variants']:
        entries = [r['variants'][name] for r in result['records']]
        rows.append(dict(name=name,test_nmae_percent=100*float(np.mean([e['test_nmae'] for e in entries])),
            validation_nmae_percent=100*float(np.mean([e['validation_nmae'] for e in entries])),
            seconds=float(np.mean([e['median_seconds'] for e in entries])),
            max_peak_increment_mib=max(e['peak_increment_mib'] for e in entries),
            max_abs_vs_original=max(e['max_abs_vs_original'] for e in entries),
            max_within_structure_timing_spread_percent=max(100*(max(e['seconds'])-min(e['seconds']))/e['median_seconds'] for e in entries),
            median_within_structure_timing_spread_percent=float(np.median([100*(max(e['seconds'])-min(e['seconds']))/e['median_seconds'] for e in entries])),
            relative_l1_vs_original_percent=100*float(np.mean([e['relative_l1_vs_original'] for e in entries]))))
    baseline = rows[0]
    for row in rows:
        row['speedup'] = baseline['seconds']/row['seconds']
        row['relative_error_increase_percent'] = 100*(row['test_nmae_percent']/baseline['test_nmae_percent']-1)
    for row in rows:
        isolated = json.loads((args.run/'isolated_timing'/f"{row['name']}.json").read_text())
        assert len(isolated['records'])==len(settings['samples'])
        assert isolated['source_hashes_match'] and isolated['checkpoint_hash_matches']
        assert isolated['checkpoint_sha256']==settings['checkpoint_sha256']
        entries = isolated['records']
        row['isolated_seconds'] = float(np.mean([e['median_seconds'] for e in entries]))
        row['isolated_max_peak_increment_mib'] = max(e['peak_increment_mib'] for e in entries)
        row['isolated_median_timing_spread_percent'] = float(np.median([100*(max(e['seconds'])-min(e['seconds']))/e['median_seconds'] for e in entries]))
        row['isolated_max_timing_spread_percent'] = max(100*(max(e['seconds'])-min(e['seconds']))/e['median_seconds'] for e in entries)
    for row in rows:
        row['isolated_speedup'] = baseline['isolated_seconds']/row['isolated_seconds']
    by_name = {r['name']:r for r in rows}
    linear,cubic = by_name['linear_r2_w1_s2'],by_name['cubic_r2_w1_s2']
    improvement = 100*(1-cubic['test_nmae_percent']/linear['test_nmae_percent'])
    save(args.output/'summary.json',dict(rows=rows,selection=selection,
        r2_relative_error_reduction_percent=improvement,checkpoint_sha256=settings['checkpoint_sha256']))
    highlights = []
    for threshold,name in selection['fastest_within_percent'].items():
        if name:
            r = by_name[name]
            highlights.append(f"按验证集 ≤{threshold}% 误差增幅选出的 {name}：测试 NMAE {r['test_nmae_percent']:.6f}%（原模型 {baseline['test_nmae_percent']:.6f}%），独立进程复测耗时 {r['isolated_seconds']:.4f} 秒、加速 {r['isolated_speedup']:.2f}×；原交错计时为 {r['seconds']:.4f} 秒、{r['speedup']:.2f}×。")
    lines = ['# 固定权重 DeepDFT 高阶远场插值实验', '', *highlights, '',
        f"同样 2 Å 近场、1/8 粗网格及批量展开实现，三次插值将测试 NMAE 从 {linear['test_nmae_percent']:.6f}% 降为 {cubic['test_nmae_percent']:.6f}%（下降 {improvement:.2f}%），独立进程复测耗时分别为 {linear['isolated_seconds']:.4f}、{cubic['isolated_seconds']:.4f} 秒。", '',
        '本实验没有新训练；原始 3×128 通道、逐层原子特征、消息核、门控及读出全部保留。三次插值在笛卡尔标量／向量分量上沿晶胞分数坐标进行，允许负权重和过冲，不对预测做裁剪或电荷归一化。', '',
        '## 最终测试与独立进程端到端复测', '',
        '原交错实验部分路径波动较大，因此在不改模型、候选或实现的情况下追加独立进程复测。每种方法一个新进程，每结构在计时外释放空闲缓存，再预热 2 次、测 5 次；仍使用全部 8 个结构，按结构中位数取平均。补测不计算标签指标、不重新选择候选。以下采用整套独立复测结果，不在两组测量中逐项挑最快值；交错结果完整保留在后表。', '',
        '| 方法 | 测试 NMAE (%) | 耗时 (s) | 加速 | 相对原误差增加 | 峰值增量显存 (GiB) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['test_nmae_percent']:.6f} | {r['isolated_seconds']:.4f} | {r['isolated_speedup']:.3f}× | {r['relative_error_increase_percent']:.2f}% | {r['isolated_max_peak_increment_mib']/1024:.3f} |")
    lines += ['', 'previous_linear_r3.5 是上一轮的逐点邻点索引实现；linear_r2 使用与本次 cubic 相同的批量展开实现。因此 R2 线性／三次对照只改变插值阶数，旧 R3.5 与新方案的速度差同时包含插值实现及近场半径的影响。', '',
        '名称 rR_wW_sS 表示近场权重在 R−W 至 R Å 平滑降为零，粗网格每方向步长为 S。s2/s4 的粗网格点数分别是完整网格的 1/8、1/64。', '']
    for percent,name in selection['fastest_within_percent'].items():
        if name:
            r = by_name[name]
            lines += [f"验证阶段相对误差增加 ≤{percent}% 的最快候选为 **{name}**；最终测试相对增加 {r['relative_error_increase_percent']:.2f}%，独立复测加速 {r['isolated_speedup']:.2f}×。该门槛是宏平均误差门槛，不是逐点或逐结构的误差保证。", '']
        else:
            lines += [f'验证阶段无候选进入相对误差增加 ≤{percent}% 门槛。', '']
    lines += ['## 全部验证候选', '',
        f"原模型验证 NMAE 为 {100*selection['original_validation_nmae']:.6f}%。候选和筛选规则在 smoke 前写定；本轮 smoke 只读取验证指标。selection.json 写入后才计算本轮候选的测试指标。", '',
        '| 候选 | 验证 NMAE (%) | 单次平均预估耗时 (s) |',
        '|---|---:|---:|']
    for name,r in selection['candidates'].items():
        lines.append(f"| {name} | {100*r['validation_nmae']:.6f} | {r['provisional_seconds']:.4f} |")
    lines += ['', '单次预估耗时只用于验证阶段选择，正式速度结论使用最终重复计时。', '',
        '## 相对原模型输出的变化', '',
        '这里使用全网格，而非只看测试子集。归一化 L1 差以原模型输出绝对值之和为分母，再对结构取平均；最大绝对差取所有结构、所有网格点中的最大值。它们与对真实密度的测试 NMAE 是不同指标。', '',
        '| 方法 | 相对原输出的归一化 L1 差 (%) | 最大绝对差 (e/Å³) |',
        '|---|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['relative_l1_vs_original_percent']:.6f} | {r['max_abs_vs_original']:.6g} |")
    lines += ['', '## 原交错计时（保留，不覆盖）', '',
        '| 方法 | 交错时间 (s) | 交错加速 | 独立进程时间 (s) | 独立进程加速 |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['seconds']:.4f} | {r['speedup']:.3f}× | {r['isolated_seconds']:.4f} | {r['isolated_speedup']:.3f}× |")
    lines += ['',
        '跨轮披露：上一轮完整模型与旧 R3.5 线性方案分别为 3.5340、3.2535 秒；本轮独立复测分别约 3.6432、4.0925 秒。旧方法出现较大的跨轮时间变化，原因尚未定位。不能把这种劣化直接算作新算法的贡献；本文主要结论以同轮完整模型为对照，并同时保留交错与独立结果。', '',
        '## 计时波动', '',
        '用每个结构各次时间的 (最大−最小)/中位数 衡量波动，再列跨结构的中位数和最大值。交错实验测 3 次，独立复测测 5 次。结果是本机墙钟计时，具体波动原因未定位；不能直接归因于缓存、GPU 调度或理论 FLOPs。', '',
        '| 方法 | 交错波动中位数 | 交错最大波动 | 独立波动中位数 | 独立最大波动 |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['median_within_structure_timing_spread_percent']:.2f}% | {r['max_within_structure_timing_spread_percent']:.2f}% | {r['isolated_median_timing_spread_percent']:.2f}% | {r['isolated_max_timing_spread_percent']:.2f}% |")
    lines += ['',
        '## 改动与代价', '',
        '使用周期四点 Lagrange 三次插值。stride=2 时中点权重为 (-1,9,9,-1)/16；依次沿三个空间轴展开，数学上等于 64 项张量积，避免在每个查询块执行 64 次邻点读取。线性对照采用同一路径。', '',
        '三个消息层各有 128 个标量和 3×128 个向量分量。56×56×140 细网格上的三层完整远场缓存约 2.51 GiB；展开时还有中间缓冲，所以显存峰值高于旧的分块插值实现。报告的显存是每次推理相对调用前的 allocated 峰值增量，非整个 Python 进程占用或 GPU reserved 内存。', '',
        '新实现用额外显存减少反复索引和 kernel 调用；速度收益不能全部归因于三次多项式本身。同 R2、同实现线性／三次对照证明的是插值阶数对误差的作用。', '',
        '远场仍只是原始 4 Å 截断内较远部分的原子消息，不是全局 Fourier 神经算子，也没有在不同结构间直接复用旧特征。高阶插值依然是有损近似；NMAE 接近不代表逐点预测一致。', '',
        '## 验证与边界', '',
        '- 新旧数值与推理回归测试共 9 项通过。覆盖三次多项式再现、常量保持、直接张量积与可分离结果一致、周期边界、stride=1 原模型重构、粗网格锚点、同实现线性与旧线性一致。',
        '- 正式验证阶段逐结构检查所有候选锚点一致，并逐结构核对新旧线性完整输出一致。',
        '- 正式运行结束后确认实现文件及 checkpoint 哈希均未变化。',
        f"- GPU {settings['device']}；PyTorch {settings['torch']}；TF32 关闭；query chunk={settings['chunk']}。",
        f"- 初次交错实验每种方法、每结构测 {settings['repeat']} 次；独立进程复测预热 2 次、测 5 次。均按结构取中位时间，再对 {len(result['records'])} 个结构取平均。",
        '- 每次计时重新计算原子图与特征、近／远场图、远场特征与全网格展开、原门控／非线性更新及读出；不计文件读取、加载权重、误差计算。',
        '- 本次沿用历史 8 个结构及空间点 80/10/10 划分，不是未见结构泛化或全新独立测试集。没有新训练，也没有新增训练种子重复。', '',
        '## 复现', '',
        '- `python -m scripts.verify_deepdft_cubic_far`',
        '- `python -m scripts.benchmark_cubic_isolated --variant <locked-variant-name>`（各方法顺序独立启动）。',
        '- `python -m scripts.summarize_deepdft_cubic_far`',
        f"- Checkpoint SHA256: `{settings['checkpoint_sha256']}`。",
        f'- 配置、全部验证结果、锁定选择及正式记录：`{args.run.as_posix()}`。',
        '- 实现：`scripts/deepdft_cubic_far.py`；测试：`tests/test_deepdft_cubic_far.py`。', '']
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
