"""Summarize the locked-validation near/far inference experiment."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.benchmark_deepdft_reuse import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=Path('runs/deepdft_near_far_2026-10-30'))
    parser.add_argument('--output',type=Path,default=Path('reports/deepdft_near_far_2026-10-30'))
    args = parser.parse_args()
    result = json.loads((args.run/'result.json').read_text())
    settings = json.loads((args.run/'settings.json').read_text())
    selection = result['selection']
    assert len(result['records']) == len(settings['samples'])
    assert result['source_hashes_match'] and result['checkpoint_hash_matches']
    rows = []
    for name in result['records'][0]['variants']:
        records = [r['variants'][name] for r in result['records']]
        rows.append(dict(name=name,
            test_nmae_percent=100*float(np.mean([r['test_nmae'] for r in records])),
            validation_nmae_percent=100*float(np.mean([r['validation_nmae'] for r in records])),
            seconds=float(np.mean([r['median_seconds'] for r in records])),
            max_peak_increment_mib=max(r['peak_increment_mib'] for r in records),
            relative_l1_vs_original_percent=100*float(np.mean([r['relative_l1_vs_original'] for r in records]))))
    baseline = rows[0]
    for row in rows:
        row['speedup'] = baseline['seconds']/row['seconds']
        row['relative_error_increase_percent'] = (row['test_nmae_percent']/baseline['test_nmae_percent']-1)*100
    save(args.output/'summary.json',dict(rows=rows,selection=selection,
        checkpoint_sha256=settings['checkpoint_sha256']))
    chosen = next((r for r in rows if r['name']==selection['fastest_within_5percent']),None)
    fast = next(r for r in rows if r['name']=='near2_blend1_stride2')
    conclusion = (f"固定 3 层、128 通道及所有权重时，R=2 Å 的空间消息复用得到 {fast['speedup']:.2f}× 加速，"
                  f"但测试 NMAE 从 {baseline['test_nmae_percent']:.6f}% 增至 {fast['test_nmae_percent']:.6f}%。")
    if chosen:
        conclusion += (f"验证集选出的近精度方案 {chosen['name']}，测试 NMAE 为 {chosen['test_nmae_percent']:.6f}%"
                       f"（相对原模型增加 {chosen['relative_error_increase_percent']:.2f}%），"
                       f"实际加速 {chosen['speedup']:.2f}×。")
        conclusion += (f"其全网格输出相对原模型的平均归一化 L1 差为 "
                       f"{chosen['relative_l1_vs_original_percent']:.6f}%，因此属于有损近似；"
                       '平均测试误差接近不代表每个点都与原模型等同。')
    text = ['# DeepDFT 固定权重近场／远场消息复用测试', '',
        conclusion, '',
        '本次保留原模型全部 3×128 参数，直接复用每层远场聚合消息。没有新训练、通道压缩、原子层融合、标签拟合或输出密度插值。', '',
        '## 最终测试及端到端性能', '',
        '| 方法 | 测试 NMAE (%) | 耗时 (s) | 加速 | 相对误差增加 | 峰值增量显存 (MiB) |',
        '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['name']} | {row['test_nmae_percent']:.6f} | {row['seconds']:.4f} | {row['speedup']:.3f}× | {row['relative_error_increase_percent']:.2f}% | {row['max_peak_increment_mib']:.1f} |")
    text += ['', '名称 nearR_blendW_strideS 表示近场权重在 R−W 至 R Å 内平滑降为零，远场使用每方向 S 倍稀疏网格。near_only2 与 near2_blend1_stride2 使用完全相同的近场项，但不补远场；它不是重新训练的短截断模型。', '',
        f"测试误差为 {len(result['records'])} 个结构各自测试 NMAE 的算术平均；耗时为每结构 {settings['repeat']} 次同步计时的中位数，再跨结构平均。显存取各次测量的最大增量 allocated，非进程总显存。", '',
        '## 验证集筛选（先锁定，再计算测试指标）', '',
        f"原模型验证 NMAE：{100*selection['original_validation_nmae']:.6f}%。事先约定可接受门槛为其 1.05 倍，即 {105*selection['original_validation_nmae']:.6f}%。", '',
        '| 候选 | 验证 NMAE (%) | 单次平均预估耗时 (s) | 进入 5% 门槛 |',
        '|---|---:|---:|---|']
    for name, row in selection['candidates'].items():
        text.append(f"| {name} | {100*row['validation_nmae']:.6f} | {row['provisional_seconds']:.4f} | {'是' if name in selection['eligible_5percent'] else '否'} |")
    text += ['', f"最低验证误差：{selection['best_accuracy']}。门槛内预估最快：{selection['fastest_within_5percent'] or '无候选达标'}。", '',
        '候选单次时间仅用于验证阶段的选择，不能替代最终重复计时。九个候选及筛选规则在本次 smoke 前写定，正式测试报告预先指定的近场对照、R=2 共享方案、验证最准确方案及门槛内最快方案。', '',
        '披露：正式运行前，sample 0 的固定 R=2,w=1,stride=2 smoke 曾输出该固定方案和近场对照的测试指标。之后未修改候选、实现或选择规则；正式方案选择仅使用验证指标。这不是从未查看过的全新独立测试集，历史实验也沿用相同结构及空间划分。', '',
        '## 如何解释', '',
        '共享远场与近场对照的差异衡量补回空间消息的作用；与完整模型的差异衡量插值误差。更大的 R 缩小需要插值的远场部分，但也增加逐点边消息计算。R=4 的近场图已经包含原图全部边，因此即使其精度接近原模型，也不能据此推断有计算优势。', '',
        '远场仅指原始 4 Å 截断内较远的原子消息，仍使用原模型的逐层原子特征。它不是整个晶胞的 Fourier 全局场，也不是 GPWNO 复现。远场插值成功或失败均不能直接推断 GPWNO 的性能。', '',
        '每个结构的粗网格特征都重新计算，不在不同结构间直接复用学习特征。测试证明的是同一结构内不同查询点之间的共享。', '',
        '本批网格为 56×56×140，共 439,040 个查询点。stride=2 的远场使用 54,880 个锚点；3 层各缓存 128 个标量及 3×128 个向量分量，float32 特征缓存本身约 321.6 MiB，此外还需要图和工作缓冲。共享会减少边计算，但不会消除细网格上的门控、非线性更新和读出。', '',
        '## 正确性与可复现性', '',
        '- 5 个测试通过：新近／远场测试 3 个，原快速推理回归测试 2 个。覆盖互补消息重构、stride=1 重构原模型、stride=2 锚点一致、周期边界、倾斜晶胞、坐标和元素变动刷新。',
        '- 正式验证阶段逐结构、逐候选检查粗网格锚点与完整模型输出一致（含 stride=4）。',
        '- 最终源文件及原 checkpoint 哈希与运行开始时一致。',
        f"- Checkpoint SHA256: `{settings['checkpoint_sha256']}`。",
        f"- GPU: {settings['device']}；PyTorch {settings['torch']}；TF32 关闭；chunk={settings['chunk']}。", '',
        '每次计时都包含原子图、原子网络、近／远场查询图、远场特征计算、插值、原门控／PaiNN 更新与读出，不包含读取数据文件和 checkpoint。原始权重、基准实现和历史实验没有改动。', '',
        '沿用 8 个 NMC 结构中的 80/10/10 空间点划分；同一结构的其他空间点曾用于原模型训练。本结果不是按未见结构划分的泛化测试。没有新训练，因此没有新增训练种子重复。', '',
        '## 文件与命令', '',
        f'- 原始记录：`{args.run.as_posix()}/result.json`。',
        f'- 全部验证结果：`{args.run.as_posix()}/validation.json`。',
        f'- 锁定选择：`{args.run.as_posix()}/selection.json`。',
        f'- 配置与来源：`{args.run.as_posix()}/settings.json`。',
        '- 实现：`scripts/deepdft_near_far.py`。',
        '- 运行：`python -m scripts.verify_deepdft_near_far`。',
        '- 汇总：`python -m scripts.summarize_deepdft_near_far`。', '']
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'report.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
