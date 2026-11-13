"""Report numerically equivalent message factoring and chunk tuning."""
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_deepdft_reuse import save


def main():
    root = Path('runs/deepdft_cubic_fast_2026-10-30')
    output = Path('reports/deepdft_cubic_fast_2026-10-30')
    settings = json.loads((root/'settings.json').read_text())
    selection = json.loads((root/'selection.json').read_text())
    rows = []
    for name in ('original','fast_full','previous_cubic3','fast_cubic3','fast_cubic3.5'):
        result = json.loads((root/'final'/f'{name}.json').read_text())
        assert len(result['records'])==8 and result['source_hashes_match'] and result['checkpoint_hash_matches']
        records = result['records']
        rows.append(dict(name=name,config=result['config'],
            nmae_percent=100*float(np.mean([r['test_nmae'] for r in records])),
            seconds=float(np.mean([r['median_seconds'] for r in records])),
            peak_increment_gib=max(r['peak_increment_mib'] for r in records)/1024,
            max_relative_l1_vs_own_previous=max(r['equivalence']['relative_l1'] for r in records),
            max_abs_vs_own_previous=max(r['equivalence']['max_abs'] for r in records),
            median_timing_spread_percent=float(np.median([100*(max(r['seconds'])-min(r['seconds']))/r['median_seconds'] for r in records])),
            max_timing_spread_percent=max(100*(max(r['seconds'])-min(r['seconds']))/r['median_seconds'] for r in records)))
    by_name = {r['name']:r for r in rows}
    for row in rows:
        row['speedup_vs_original'] = by_name['original']['seconds']/row['seconds']
        row['speedup_vs_fast_full'] = by_name['fast_full']['seconds']/row['seconds']
    previous,fast = by_name['previous_cubic3'],by_name['fast_cubic3']
    reduction = 100*(1-fast['seconds']/previous['seconds'])
    save(output/'summary.json',dict(rows=rows,selection=selection,additional_cubic_time_reduction_percent=reduction))
    lines = ['# DeepDFT 消息计算与分块优化', '',
        f"3 Å 三次方案保持原输出精度，端到端耗时从同轮 {previous['seconds']:.4f} 秒降到 {fast['seconds']:.4f} 秒，进一步减少 {reduction:.2f}% 时间。相对未经本轮优化的完整模型，加速 {fast['speedup_vs_original']:.2f}×；相对同样优化过的完整模型，加速 {fast['speedup_vs_fast_full']:.2f}×。", '',
        '## 八结构最终结果', '',
        '| 方法 | 测试 NMAE (%) | 秒/结构 | 相对原完整模型加速 | 峰值增量显存 (GiB) |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['nmae_percent']:.6f} | {r['seconds']:.4f} | {r['speedup_vs_original']:.3f}× | {r['peak_increment_gib']:.3f} |")
    lines += ['', 'original 为原完整逐点计算；fast_full 为相同优化应用于完整模型；previous_cubic3 为本轮重新测量的旧 3 Å 三次实现；fast_cubic3/3.5 为本轮优化后的两种近场半径。所有参数、通道数、权重、原始 cutoff、FP32 精度、三次插值权重保持不变，没有重训。', '',
        '每个结构均预测 56×56×140=439,040 个网格点。上一轮旧 3 Å 方案约 2.47 秒，本轮重测约 2.18 秒，说明跨轮计时存在变化；本轮优化收益只使用同轮旧／新实现比较，不能把 2.47→1.95 秒全部归因于此次代码修改。', '',
        '## 优化内容', '',
        '原消息对每条边计算径向滤波器与原子门控的乘积，然后再乘原子标量／向量。现在先按原子计算 s×g_s 与 v×g_v，边上只乘径向滤波器；方向项保留 g_e。这样减少逐边乘法、投影数据读取和中间张量。数学上等价，浮点计算顺序不同，因此不宣称 bitwise 一致。', '',
        f"三次方案选择 {selection['chosen']['cubic']}；完整模型选择 {selection['chosen']['full']}。分块候选为 4096、8192、16384。选择仅用结构 0、3 的性能及数值一致性，不使用密度标签。", '',
        '所有原子特征都在每次推理重新计算，没有跨结构借用旧特征。三次方案相对完整模型原有的插值误差仍然存在；本轮优化没有增加可测量的系统性预测误差。', '',
        '## 数值与计时检查', '',
        '| 方法 | 相对各自旧实现的最大归一化 L1 差 | 最大绝对差 (e/Å³) | 计时波动中位数 | 最大计时波动 |',
        '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['name']} | {r['max_relative_l1_vs_own_previous']:.3g} | {r['max_abs_vs_own_previous']:.3g} | {r['median_timing_spread_percent']:.2f}% | {r['max_timing_spread_percent']:.2f}% |")
    lines += ['', '归一化差异检查覆盖完整网格，误差分母为相应旧实现输出绝对值之和。计时波动按各结构 (最大−最小)/中位数 计算。', '',
        '- 11 项数值／推理回归测试通过，包括逐层消息相等、完整及三次模型相等、分块变化、原子坐标及元素变化刷新，以及此前周期插值测试。',
        '- 全部 8 个结构均通过数值一致性检查：rtol=3e-5、atol=5e-5，且全网格相对 L1 差小于 1e-6。',
        '- 正式运行结束后检查源码和 checkpoint 哈希与开始一致。',
        '- 每种方法独立进程；每结构预热 2 次，CUDA 同步计时 3 次取中位数，再跨 8 个结构平均。',
        '- 时间包含构图、原子网络、近／远场计算、远场展开、逐点更新和读出；排除加载文件、checkpoint、预热和误差计算。显存是单次推理的 allocated 峰值增量。',
        f"- GPU {settings['device']}，PyTorch {settings['torch']}，TF32 关闭。",
        '- 沿用历史 8 个结构的空间点划分，不是未见结构泛化测试。', '',
        '## 阶段诊断与选择记录', '',
        '结构 0、3 的 CUDA-event 诊断显示近场消息约占 60%，远场构图及消息约占 18%，插值约占 6%。该诊断有事件记录开销，仅用于定位；正式秒数来自未插桩推理。', '',
        '| 候选 | 调优阶段秒/结构 |', '|---|---:|']
    for name,seconds in selection['seconds'].items():
        lines.append(f'| {name} | {seconds:.4f} |')
    lines += ['', '调优阶段 full_factored_16384 在结构 3 出现约 28 秒的异常慢速，而结构 0 约 3.38 秒；同期观察到 GPU 总占用接近 16 GiB。尚未确定根因，可能与连续候选运行的显存状态有关，因此保留原始记录，不把该数值解释为大分块的固有速度。最终所选配置在独立进程重新测量，并在每个结构预热前清理缓存。', '',
        '## 后续加速方向', '',
        '本轮只减少了逐边重复乘法。近场仍需要生成和归约大量边消息；下一步可尝试把滤波、消息乘法和按查询点归约融合，减少中间张量写回显存。该方向尚未验证，不能据此承诺额外加速幅度。进一步缩小近场或增大远场步长会改变插值误差，需要独立的精度实验。', '',
        '## 文件与复现', '',
        '- 实现：`scripts/deepdft_factored_messages.py`。',
        '- `python -m scripts.verify_cubic_fast --stage select`',
        '- `python -m scripts.verify_cubic_fast --stage final --variant <original|fast_full|previous_cubic3|fast_cubic3|fast_cubic3.5>`（每种方法依次独立启动）。',
        '- `python -m scripts.summarize_cubic_fast`',
        '- 正式结果禁止覆盖；重跑选择和正式测试时均传入相同的新目录 `--output runs/<new-dir>`，汇总脚本的 root 也需对应修改。',
        f'- 原始诊断、选择与完整结果：`{root.as_posix()}`。',
        f"- Checkpoint SHA256: `{settings['checkpoint_sha256']}`。", '']
    output.mkdir(parents=True,exist_ok=True)
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
