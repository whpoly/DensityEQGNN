"""Select the measured faster path and document graph reuse limitations."""
import json
from pathlib import Path
import numpy as np
from scripts.nmc_edengnn import write_json
from scripts.nmc_probe_holdout import sha256


def main():
    root=Path('runs/nmc_inference_graph_2026-10-29')
    report=Path('reports/nmc_inference_graph_2026-10-29')
    result=json.loads((root/'result.json').read_text())
    assert result['complete']
    assert all(sha256(Path(p))==digest for p,digest in result['source_sha256'].items())
    summary=json.loads((report/'summary.json').read_text())
    selected='graph'
    assert summary[selected]['fresh_inputs_prepared_graph_seconds']<summary['combined']['fresh_inputs_prepared_graph_seconds']
    before=summary['baseline']['fresh_inputs_prepared_graph_seconds']
    after=summary[selected]['fresh_inputs_prepared_graph_seconds']
    summary['selected']=selected
    summary['selected_speedup']=before/after
    summary['selected_latency_reduction_percent']=100*(1-after/before)
    summary['selected_approx_calls_to_amortize']=int(np.ceil(summary['graph_setup_seconds']/(before-after)))
    summary['selection_source_sha256']=sha256(Path(__file__))
    write_json(report/'summary.json',summary)
    lines=['# GNN 推理优化：采用执行图重放，保留原快速解码','',
           f'本轮同卡交错测量，上一轮快速实现 {before:.6f} 秒，已有匹配执行图时 {after:.6f} 秒，'
           f'耗时减少 {summary["selected_latency_reduction_percent"]:.2f}%。','',
           '| 实现 | 测试 NMAE (%) | 完整推理 (s) |','|---|---:|---:|']
    labels={'baseline':'上一轮快速实现','angular':'仅改变收缩顺序（不采用）',
            'graph':'仅 GNN CUDA Graph（采用）','combined':'GNN Graph + 改变收缩顺序（不采用）'}
    for name,label in labels.items():
        v=summary[name]
        lines.append(f'| {label} | {100*v["test_nmae"]:.6f} | {v["fresh_inputs_prepared_graph_seconds"]:.6f} |')
    lines += ['', '**表中 Graph 时间要求已有匹配形状的执行图。** 每次仍重新生成原子图、网格边、SAD、网络输出和完整密度，包含输入值复制。',
              f'首次录制/预热 GNN 平均额外需要 **{summary["graph_setup_seconds"]:.6f} 秒**。'
              f'按本轮均值估计，约 **{summary["selected_approx_calls_to_amortize"]} 次**同形状调用才能摊平成本。',
              '这里的形状包括原子数和邻接边数。坐标、元素、邻接关系和位移可以更新；张量形状改变时需重新录制。',
              '使用有界 LRU 缓存；原子数量不变也不保证命中，因为邻接边数可能变化。没有测量真实 MD/SCF 轨迹命中率。',
              '新接口的冷启动、不同结构切换、缓存淘汰及坐标变化另见 `runs/nmc_inference_graph_2026-10-29/api_audit.json`。',
              '独立新结构的一次性推理仍使用上一轮 FastAdaptiveR16；不把重复调用速度当作任意首次输入速度。','',
              '## 分项耗时','', '| 部分 | 秒 |','|---|---:|']
    lines += [f'| {k} | {v:.6f} |' for k,v in summary['components'].items()]
    lines += ['', 'GNN 内部大量小算子的主机调度开销是此次有效优化点。解码的三个候选（直接 stencil、直接 stencil 的重放、改变收缩顺序）均未超过现有分组实现，未集成到推荐接口。',
              '没有实现自定义融合 CUDA 内核；未降低精度或改变模型权重、R16 基组、角阶、截断。','',
              '## 验证与边界','',
              '- 相同冻结检查点、原 8 个结构及空间划分。每结构四种实现交错重复五次，汇总结构中位数的平均。',
              '- 模型/POTCAR/CHGCAR 文件加载不计入；录制单列。分项与完整计时独立，不将分项之和充作完整时间。',
              f'- 所有候选的完整网格与上一轮输出最大绝对差 {summary["max_abs_density_difference"]:.8g} e/Å³，包含独立 GNN 前向浮点归约差异。',
              '- 重放改变坐标、元素、边索引、周期位移后仍与 eager 一致；输出克隆防止后续重放覆盖。',
              '- 没有重新训练；精度不变，因此与 EdenGNN 的精度差距没有进一步缩小。',
              '- 此次未重测 EdenGNN，且它也可能采用 CUDA Graph；不跨轮次推算新的相对 Eden 加速比。',
              '- 8 个已见结构的实现验证，不能据此声称未知结构泛化或 DFT/SCF 总耗时改善。','',
              '接口用法见同目录 `usage.md`。原始记录为 `runs/nmc_inference_graph_2026-10-29/result.json`。','']
    (report/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ('selected','selected_speedup','selected_latency_reduction_percent',
                                           'selected_approx_calls_to_amortize','graph_setup_seconds')},indent=2))


if __name__=='__main__':
    main()
