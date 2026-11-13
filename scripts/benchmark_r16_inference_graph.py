"""Paired fixed-model benchmark, explicitly separating cold capture and reuse."""
from datetime import datetime,timezone
from pathlib import Path
import json
import time
import numpy as np
import torch

from scripts.predict_adaptive_r16_fast import FastAdaptiveR16
from scripts.compare_local_nmc_oracles import load_sample
from scripts.nmc_edengnn import atom_data,write_json
from scripts.nmc_probe_holdout import DEFAULT_ROOT,save_split,sha256
from scripts.r16_grid_diagnostic import uniform_grid_edges
from scripts.r16_inference_fast import decode_grouped
from scripts.r16_inference_graph import GraphCoefficients,decode_angular_first
from scripts.fit_nmc_holdout_oracle import BASIS
from scripts.benchmark_nmc_edengnn import timed,metrics

ROOT=Path('runs/nmc_inference_graph_2026-10-29')
REPORT=Path('reports/nmc_inference_graph_2026-10-29')
CHECKPOINT=Path('runs/nmc_adaptive_r16_2026-10-29/adaptive/best.pt')
NAMES=('baseline','angular','graph','combined')


def render(result):
    summary={name:{'test_nmae':float(np.mean([r['metrics'][name]['test_nmae'] for r in result['records']])),
                   'fresh_inputs_prepared_graph_seconds':float(np.mean([np.median(r['timing'][name]) for r in result['records']]))}
             for name in NAMES}
    summary['cold_combined_seconds']=float(np.mean([r['cold_combined']['median_seconds'] for r in result['records']]))
    summary['graph_setup_seconds']=float(np.mean([r['captures'][0]['setup_seconds'] for r in result['records']]))
    summary['components']={k:float(np.mean([r['components'][k]['median_seconds'] for r in result['records']]))
                           for k in result['records'][0]['components']}
    summary['max_abs_density_difference']=max(max(r['max_abs_vs_baseline'].values()) for r in result['records'])
    summary['baseline_vs_combined_speedup']=(summary['baseline']['fresh_inputs_prepared_graph_seconds']/
                                          summary['combined']['fresh_inputs_prepared_graph_seconds'])
    saving=summary['baseline']['fresh_inputs_prepared_graph_seconds']-summary['combined']['fresh_inputs_prepared_graph_seconds']
    summary['approx_calls_to_amortize']=int(np.ceil((summary['cold_combined_seconds']-
          summary['combined']['fresh_inputs_prepared_graph_seconds'])/saving)) if saving>0 else None
    write_json(REPORT/'summary.json',summary)
    labels={'baseline':'上一轮快速实现','angular':'仅改变收缩顺序','graph':'仅 GNN CUDA Graph',
            'combined':'GNN CUDA Graph + 改变收缩顺序'}
    lines=['# 下一步推理优化：GNN 重放与解码收缩顺序','',
           '相同冻结检查点、8 个结构、原空间划分；没有训练或改变基组/角阶/截断/数值精度。','',
           '| 实现 | 测试 NMAE (%) | 完整推理，已有匹配执行图 (s) |','|---|---:|---:|']
    for name in NAMES:
        v=summary[name]
        lines.append(f'| {labels[name]} | {100*v["test_nmae"]:.6f} | {v["fresh_inputs_prepared_graph_seconds"]:.6f} |')
    lines += ['',f'组合版本首次遇到新形状、包含录制的完整推理平均 **{summary["cold_combined_seconds"]:.6f} 秒**。',
              f'GNN 录制/预热平均 {summary["graph_setup_seconds"]:.6f} 秒；估计约 {summary["approx_calls_to_amortize"]} 次同形状调用摊平成本。',
              '此估计来自同形状重复计时，不保证任意 MD 轨迹的收益。邻接边数改变会触发新执行图；有限缓存也可能淘汰旧图。','',
              '## 计时边界','',
              '- 每个结构均重新建立空执行图缓存，先实际计量一次包含录制的完整调用；再对四种实现交错重复五次。',
              '- 重复调用仅复用执行图。每次都重新构建原子图、网格边、SAD，重新执行 GNN 和密度重构，包含输入值复制。',
              '- 不缓存预测系数、密度或输入几何。模型/POTCAR/CHGCAR 文件加载仍不计入。',
              '- GNN 缓存键包括输入张量的形状、类型和设备；位置、元素、边索引、周期位移每次全部更新。',
              '- 改变坐标、元素、邻接关系后的重放已与 eager 对照；返回值独立克隆，不会被下一次重放覆盖。',
              '- 此次没有重测 EdenGNN，不能把这里的加速与前轮绝对时间相除，或假设 Eden 不能应用 CUDA Graph。','',
              '## 分项耗时','', '| 部分 | 秒 |','|---|---:|']
    lines += [f'| {k} | {v:.6f} |' for k,v in summary['components'].items()]
    lines += ['',f'完整网格与上一轮实现的最大绝对差：{summary["max_abs_density_difference"]:.8g} e/Å³。',
              '密度比较包含独立 GNN 前向的 CUDA 浮点归约差异。测试 NMAE 用同一原始划分重新计算。','',
              '## 未采用的候选','',
              '规则 stencil 直接构造并解码可省去边列表，但保守球内多余候选点增加了浮点运算。',
              '在结构 0 上，直接路径和其 CUDA Graph 版均慢于上一轮分组解码，未用于上表的组合版本。',
              '原始诊断包含在 `runs/nmc_inference_graph_2026-10-29/preflight.json`，无自定义融合 CUDA 内核。','',
              '单种子、8 个已见结构的工程验证，不代表未知结构泛化、MD/SCF 轨迹速度或 DFT 总体加速。','']
    (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    frozen=json.loads((CHECKPOINT.parent/'selection_frozen.json').read_text())
    assert sha256(CHECKPOINT)==frozen['checkpoint_sha256']
    predictor=FastAdaptiveR16()
    model=predictor.model
    result={'started_utc':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':sha256(CHECKPOINT),
            'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'records':[],
            'source_sha256':{str(p):sha256(p) for p in [Path(__file__),Path('scripts/r16_inference_graph.py'),
              Path('scripts/r16_inference_fast.py'),Path('scripts/predict_adaptive_r16_fast.py')]}}
    for sid in range(8):
        s=load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        split,meta=save_split(DEFAULT_ROOT/'spatial_splits',len(s['target']),sid,s['metadata']['sha256'])
        s['split']={k:torch.tensor(v,device='cuda') for k,v in split.items()}
        # Explicitly empty cache for each cold-start measurement.
        graph=GraphCoefficients(model,max_graphs=1)
        def fresh(name):
            data=atom_data(s)
            c=graph(data) if name in ('graph','combined') else model.coefficients(data)
            e=uniform_grid_edges(s['positions'],s['cell'],s['metadata']['grid_shape'],BASIS.cutoff)
            decoder=decode_angular_first if name in ('angular','combined') else decode_grouped
            residual=decoder(c,-e.displacement,e.batch,e.neighbor,e.center,len(s['probes']),BASIS)
            return residual.double()+predictor.sad(s).double()
        row={'sample_id':sid,'source':s['metadata'],'split':meta,'timing':{},'metrics':{},
             'max_abs_vs_baseline':{},'components':{}}
        # Cold means graph setup included, not file-loading/library initialization.
        row['cold_combined'],cold=timed(lambda:fresh('combined'),1,warmup=0)
        reference=fresh('baseline')
        torch.testing.assert_close(cold,reference,rtol=3e-5,atol=5e-5)
        del cold
        for name in NAMES:
            output=fresh(name)
            torch.testing.assert_close(output,reference,rtol=3e-5,atol=5e-5)
            row['max_abs_vs_baseline'][name]=float((output-reference).abs().max())
            row['metrics'][name]=metrics(output,s,s['split'])
            row['timing'][name]=[]
            del output
        del reference
        for repeat in range(5):
            shift=(sid+repeat)%len(NAMES)
            for name in NAMES[shift:]+NAMES[:shift]:
                torch.cuda.synchronize()
                tick=time.perf_counter()
                output=fresh(name)
                torch.cuda.synchronize()
                row['timing'][name].append(time.perf_counter()-tick)
                del output
        data=atom_data(s)
        row['components']['network_eager'],c=timed(lambda:model.coefficients(data),3)
        row['components']['network_graph'],_=timed(lambda:graph(data),3)
        row['components']['grid'],e=timed(lambda:uniform_grid_edges(
            s['positions'],s['cell'],s['metadata']['grid_shape'],BASIS.cutoff),3)
        row['components']['decoder_radial_first'],_=timed(lambda:decode_grouped(
            c,-e.displacement,e.batch,e.neighbor,e.center,len(s['probes']),BASIS),3)
        row['components']['decoder_angular_first'],_=timed(lambda:decode_angular_first(
            c,-e.displacement,e.batch,e.neighbor,e.center,len(s['probes']),BASIS),3)
        row['components']['sad'],_=timed(lambda:predictor.sad(s),3)
        row['captures']=graph.captures
        assert len(graph.captures)==1
        result['records'].append(row)
        write_json(ROOT/'result.json',result)
        print('RESULT '+json.dumps({'sample_id':sid,'cold':row['cold_combined']['median_seconds'],
                     'seconds':{k:float(np.median(v)) for k,v in row['timing'].items()}}),flush=True)
        del graph,c,e,data
    result['complete']=True
    result['completed_utc']=datetime.now(timezone.utc).isoformat()
    write_json(ROOT/'result.json',result)
    render(result)


if __name__=='__main__':
    main()
