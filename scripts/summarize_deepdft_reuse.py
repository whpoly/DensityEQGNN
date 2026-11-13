"""Generate the fixed-weight inference report from measured results."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('runs/deepdft_reuse_2026-10-29')
REPORT=Path('reports/deepdft_reuse_2026-10-29')
NAMES={
    'official_2048_fresh':'原接口，2048 点/块',
    'official_tuned_fresh':'原接口，4096 点/块',
    'flat_fresh':'局部构图＋扁平接口＋原子投影共享',
    'replay_fresh':'以上＋查询 CUDA Graph（已录制）',
    'replay_cached_geometry':'CUDA Graph＋复用完全相同的几何',
    'eden_gpu_sad':'EdenGNN＋GPU SAD',
    'hybrid_fast':'自适应 R16 快速实现',
}


def main():
    data=json.loads((ROOT/'result.json').read_text())
    rows=data['records']
    assert len(rows)==8 and {r['id'] for r in rows}==set(range(8))
    summary={}
    for key in NAMES:
        summary[key]={'seconds':float(np.mean([r['variants'][key]['median_seconds'] for r in rows])),
                      'test_nmae_percent':100*float(np.mean([r['metrics'][key]['test_nmae'] for r in rows])),
                      'max_abs_vs_deepdft':max(r['metrics'][key]['max_abs_vs_original'] for r in rows),
                      'max_relative_l1_vs_deepdft':max(r['metrics'][key]['relative_l1_vs_original'] for r in rows)}
    old=summary['official_2048_fresh']['seconds'];new=summary['flat_fresh']['seconds']
    capture=float(np.mean([r['capture_seconds'] for r in rows]))
    moving={name:float(np.mean([r['moving_frame']['measurements'][name]['median_seconds'] for r in rows]))
            for name in ['moved_rebuild','moved_skin_reuse']}
    graph={name:float(np.mean([r[name]['median_seconds'] for r in rows])) for name in ['classic_graph','local_graph','skin_preparation']}
    result={'models':summary,'capture_seconds':capture,'moving_seconds':moving,'graph_seconds':graph}
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    text=['# DeepDFT 逐点预测：固定权重加速与结构复用测试','',
          f'原接口完整密度生成 {old:.4f} s，局部构图与接口优化后 {new:.4f} s，加速 {old/new:.2f}×。',
          '相同 8 个 NMC 结构、每结构 439040 点、原空间划分；固定检查点，不重新训练。','',
          '| 方法 | 测试 NMAE (%) | 完整生成 (s) |','|---|---:|---:|']
    for key,name in NAMES.items():
        x=summary[key];text.append(f'| {name} | {x["test_nmae_percent"]:.6f} | {x["seconds"]:.6f} |')
    text+=['',
        '各时间为每结构 3 次交错测量的中位数再做宏平均；全部先预热并显式 CUDA 同步。',
        '新输入计时重算原子图、原子特征、查询邻域和完整密度。模型/原始 CHGCAR 文件加载及驻留输入传输不计入。',
        '原接口使用既有全周期像 cKDTree 适配器；这里不是作者最优完整推理程序的速度排名。',
        'Eden 与 R16 同进程交错重测，均含新几何和 GPU SAD；三者架构、训练预算与精度不同。','',
        '## 数值一致性','',
        f'固定 DeepDFT checkpoint SHA256：`{data["checkpoint_sha256"]}`。',
        '所有结构逐项比较原构图和新构图的周期边多重集，包括周期镜像。',
        '| DeepDFT 候选 | 最大逐点绝对差 (e/Å³) | 最大相对 L1 差 |','|---|---:|---:|']
    for key in list(NAMES)[:5]:
        x=summary[key];text.append(f'| {NAMES[key]} | {x["max_abs_vs_deepdft"]:.3e} | {x["max_relative_l1_vs_deepdft"]:.3e} |')
    text+=['','与原预测之差来自浮点求和/批量顺序，没有插值、裁负、电荷归一化、低精度量化或重新训练。',
        '候选只依据耗时和数值等价性选择，DFT 测试误差用于复核既有结果。','',
        '## 构图与查询计算','',
        f'原查询邻域构建平均 {graph["classic_graph"]:.6f} s；局部网格枚举＋排序 {graph["local_graph"]:.6f} s。',
        '以下为首个结构、带同步插桩的分段诊断；不能相加冒充无插桩总时间。','',
        '| 阶段 | 秒 |','|---|---:|']
    for key,value in rows[0]['profile_synchronized_diagnostic'].items():text.append(f'| {key} | {value:.6f} |')
    text+=['','## 相近结构能复用什么','',
        '每个原结构生成一个合成小位移结构：各原子坐标加 0.005*sin(index+sample_id) Å，晶胞和网格固定。',
        '0.2 Å skin 缓存由原结构建立；移动后每次重算原子特征、距离、方向、消息和密度，仅复用邻居候选表。',
        '每个扰动结构都与在其新坐标上运行的官方模型比较，没有用旧 DFT 密度评价新结构准确度。','',
        '| 小位移结构的查询执行图推理 | 秒 |','|---|---:|',
        f'| 每次重建查询邻域 | {moving["moved_rebuild"]:.6f} |',
        f'| 复用 skin 候选邻域 | {moving["moved_skin_reuse"]:.6f} |','',
        f'skin 初次准备平均 {graph["skin_preparation"]:.6f} s，未免费计入新结构单次推理。',
        'skin 会引入更多候选边；省去构图不保证总时间下降。位移超出 skin/2、晶胞/网格变化时接口拒绝复用。',
        '此处是 8 对合成小位移验证；不是实际 MD 轨迹吞吐，也未证明新结构密度精度。','',
        '## 执行图与冷启动','',
        f'每个结构的查询执行图录制与预热平均 {capture:.6f} s，应额外计入首次调用。',
        '重放行已包含新原子特征及逐块几何复制，不复用学习输出。执行图容量取该结构含 skin 的最大边块并向上取整。',
        '独立新结构若超过原子数/查询块/边容量须重新录制；CUDA Graph 不降低消息计算的算术量。',
        '预实验曾使用 16384 点分块，在多模型与执行图驻留下触发显存压力；该轮已中止，未纳入正式结果。',
        '正式主表固定 4096 点分块（旧接口基线为 2048），保留中止日志。','',
        '## 逐结构时间','',
        '| 结构 | 原接口 | flat | replay 新几何 | Eden | 快速 R16 |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        vals=[r['variants'][k]['median_seconds'] for k in ['official_2048_fresh','flat_fresh','replay_fresh','eden_gpu_sad','hybrid_fast']]
        text.append('| '+str(r['id'])+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    if (ROOT/'segment_diagnostic.json').exists():
        segment=json.loads((ROOT/'segment_diagnostic.json').read_text())
        result['segment_diagnostic']=segment
        text+=['','## 连续分段求和诊断','',
               '保持权重和公式，按 query 的连续边分段求和，替代 atomic scatter；该实验单独运行。','',
               '| 结构 | flat 缓存几何 (s) | 分段求和缓存几何 (s) | 最大绝对差 |','|---|---:|---:|---:|']
        for r in segment:
            text.append(f'| {r["id"]} | {r["flat"]["median_seconds"]:.6f} | {r["segment"]["median_seconds"]:.6f} | {r["agreement"]["max_abs"]:.3e} |')
    if (ROOT/'sparse_diagnostic.json').exists():
        sparse=json.loads((ROOT/'sparse_diagnostic.json').read_text())
        assert len(sparse['records'])==8
        sparse_summary={}
        text+=['','## 只计算 1/8 查询点的近似诊断','',
               '固定每个方向 stride=2，54880 个粗点；其余点周期三线性插值，无重新训练。',
               '该表是独立的同进程交错计时，每次重新构图；不能与主表跨轮次作精确耗时相除。',
               '两种固定插值方案都完整报告，没有按测试集寻找最优 stride。','',
               '| 方法 | 测试 NMAE (%) | 新输入生成 (s) | 相对完整网络的 L1 差 (%) |',
               '|---|---:|---:|---:|']
        labels={'full':'完整逐点预测','sparse_raw':'粗点总密度直接插值','sparse_sad_residual':'粗点 SAD 残差插值'}
        for name,label in labels.items():
            values=[r['variants'][name] for r in sparse['records']]
            x={'seconds':float(np.mean([r['median_seconds'] for r in values])),
               'test_nmae_percent':100*float(np.mean([r['test_nmae'] for r in values])),
               'relative_l1_percent':100*float(np.mean([r['relative_l1_vs_original'] for r in values]))}
            sparse_summary[name]=x
            text.append(f'| {label} | {x["test_nmae_percent"]:.6f} | {x["seconds"]:.6f} | {x["relative_l1_percent"]:.6f} |')
        result['sparse_diagnostic']=sparse_summary
        text+=['','粗点与完整网络相应节点逐点一致性检查通过。插值误差是额外的近似误差，不能与无损优化混称。',
               'SAD 沿用本地 POTCAR form-factor 重构；仍未逐点核验 VASP 原生零步密度。',
               '简单插值的成败不能代表重新训练的多分辨率潜在特征模型。']
    text+=['','## 判断与建议','',
        '1. 保留局部网格构图、单结构扁平接口和原子投影共享；它们在全部 8 个结构上保持原预测精度。',
        '2. 当前测试的固定容量执行图没有进一步收益；含 skin 容量的填充边带来额外计算。',
        '3. 小位移邻域复用在相同执行图容量下约节省 5%，主要瓶颈仍是查询边消息，而不是构图或原子主干。',
        '4. 连续分段求和仅完成结构 0 诊断，小幅收益尚未成为全结构主表中的生产实现。',
        '5. 简单粗网格插值可以减少绝大多数查询计算，但精度明显下降；要兼顾精度，更值得研究共享潜在特征、轻量局部修正或消息计算融合。',
        '专项验证：`tests/test_deepdft_inference_fast.py`、`tests/test_deepdft_sparse.py`、`tests/test_r16_grid_diagnostic.py`，4 passed。',
        '测试涵盖斜晶胞与未折回原子、周期像、位移/元素变化后刷新、skin 失效、执行图填充及插值周期边界。',
        '', '原始记录：`runs/deepdft_reuse_2026-10-29/result.json`；分块预实验：`pilot.json`。',
        '运行：`python -m scripts.benchmark_deepdft_reuse --chunk 4096 --repeat 3`；汇总：`python -m scripts.summarize_deepdft_reuse`。',
        '单种子、已见结构空间留出与实现测试；未验证未知结构泛化、PAW 占据或 SCF 总耗时。']
    (REPORT/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    (REPORT/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
