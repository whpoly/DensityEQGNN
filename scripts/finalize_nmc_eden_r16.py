"""Final sealed accuracy and fresh-input timings for Eden-inspired R16 arms."""
from datetime import datetime,timezone
import gc
import json
from pathlib import Path
import types

import numpy as np
import torch
from scripts.nmc_eden_r16 import ROOT,ARMS,build,prepare,coefficients,prediction,evaluate
from scripts.nmc_edengnn import write_json,atom_data,make_sad
from scripts.nmc_probe_holdout import sha256
from scripts.fit_nmc_holdout_oracle import BASIS
from scripts.evaluate_vasp_sad_nmc import read_form_factors
from scripts.benchmark_nmc_edengnn import timed,r16_edges
from scripts.r16_decoder_diagnostic import decode
from scripts.r16_grid_diagnostic import uniform_grid_edges

REPORT=Path('reports/nmc_eden_r16_2026-10-29')


def make_report(results):
    REPORT.mkdir(parents=True,exist_ok=True)
    prior=json.loads(Path('reports/nmc_edengnn_2026-10-28/summary.json').read_text())
    summary={}
    labels={'r16_sad':'原 R16 + SAD','eden_r16_sad':'Eden 官方主干 + R16 + SAD'}
    for arm,value in results['models'].items():
        timing=value['timing']
        summary[arm]={'test_macro_nmae':value['metrics']['test_macro_nmae'],
                      'train_macro_nmae':value['metrics']['train_macro_nmae'],
                      'validation_macro_nmae':value['metrics']['validation_macro_nmae'],
                      'fresh_seconds':float(np.mean([r['fresh']['median_seconds'] for r in timing])),
                      'warm_seconds':float(np.mean([r['warm']['median_seconds'] for r in timing])),
                      'decoder_seconds':float(np.mean([r['decoder']['median_seconds'] for r in timing])),
                      'local_grid_fresh_seconds':float(np.mean([r['local_grid_fresh']['median_seconds'] for r in timing])),
                      'local_grid_geometry_seconds':float(np.mean([r['local_grid_geometry']['median_seconds'] for r in timing])),
                      'parameter_count':value['settings']['parameter_count']}
    write_json(REPORT/'summary.json',{'models':summary,'sad_test_nmae':results['sad']['test_macro_nmae'],'references':prior})
    lines=['# 直接借鉴 EdenGNN：SAD 与官方主干的 R16 实验','',
           '相同 8 个 NMC 结构、原空间划分、seed 42、各 16,000 步及每步 1,024 训练点。'
           '两臂从头训练；全程未使用教师系数或既有已训练权重。','',
           '| 模型 | 参数量 | 训练 NMAE (%) | 测试 NMAE (%) | 完整推理含准备 (s) |',
           '|---|---:|---:|---:|---:|',
           f'| 原 R16，原实现（既有参照） | 521,472 | 2.07143 | {100*prior["r16"]["test_macro_nmae"]:.5f} | '
           f'{prior["r16"]["fresh_mean_of_structure_medians_seconds"]:.4f} |',
           f'| 原 R16，代数重排解码（既有参照） | 521,472 | 2.07143 | {100*prior["r16_algebraic_diagnostic"]["test_macro_nmae"]:.5f} | '
           f'{prior["r16_algebraic_diagnostic"]["fresh_seconds"]:.4f} |']
    for arm,s in summary.items():
        lines.append(f'| {labels[arm]} | {s["parameter_count"]:,} | {100*s["train_macro_nmae"]:.5f} | '
                     f'{100*s["test_macro_nmae"]:.5f} | {s["fresh_seconds"]:.4f} |')
    lines+=[f'| 完整 EdenGNN（既有参照） | 6,194,410 | 0.28477 | {100*prior["eden"]["test_macro_nmae"]:.5f} | '
            f'{prior["eden"]["fresh_mean_of_structure_medians_seconds"]:.4f} |','',
            f'SAD 单独的测试 NMAE：**{100*results["sad"]["test_macro_nmae"]:.5f}%**。'
            'NMAE 为逐结构 Σ|预测−DFT|/Σ|DFT| 的宏平均，不做电荷重归一化或负值裁剪。','',
            '## 结果解释','']
    for arm,s in summary.items():
        lines.append(f'- {labels[arm]}：相对原 R16 的误差变化 '
                     f'{100*(s["test_macro_nmae"]/prior["r16"]["test_macro_nmae"]-1):+.2f}%；'
                     f'误差为完整 EdenGNN 的 {s["test_macro_nmae"]/prior["eden"]["test_macro_nmae"]:.2f} 倍。')
    lines+=['','r16_sad 在原 R16 上只增加 SAD，是此次最直接的残差学习对照。'
            '混合模型同时改变主干、原子图和读出头，不能把结果单独归因于主干。'
            '混合模型保留固定 R16 基组，没有使用 Eden 的可学习径向函数，因此不能将它称为完整 EdenGNN。','',
            '## 分段计时','',
            '| 模型 | 新输入完整推理 (s) | 缓存几何/SAD 后 (s) | 固定系数重构 (s) |','|---|---:|---:|---:|']
    for arm,s in summary.items():
        lines.append(f'| {labels[arm]} | {s["fresh_seconds"]:.5f} | {s["warm_seconds"]:.5f} | {s["decoder_seconds"]:.5f} |')
    lines+=['','本轮每结构预热 1 次、重复 3 次；上述为各结构重复中位数的算术平均。'
            '新输入包括原子图、原生产网格边构造、POTCAR SAD、网络、代数重排解码；模型及 POTCAR 文件加载不计入。'
            '缓存推理明确复用几何/SAD，不复用学习输出。计时与训练串行、CUDA 显式同步。'
            '既有参照是在同一 GPU/环境上的上一轮 5 次重复测量，不是本轮交错重复。','',
            '### 借鉴局部网格枚举的额外诊断','',
            '参照 Eden 的局部 stencil 思路，用保守候选球包住真实原子 3 Å 邻域，再按真实距离筛选。'
            '保持 R16 严格截断和全部周期镜像；仅改变邻域枚举，不修改密度模型或训练。'
            '该实现只适用于这里的规则周期网格。已检查倾斜晶胞、未折回原子和周期重数；'
            '每个 NMC 结构进一步比较边数及完整密度场。','',
            '| 模型 | 原构图完整推理 (s) | 局部构图完整推理 (s) | 局部构图本身 (s) |','|---|---:|---:|---:|']
    for arm,s in summary.items():
        lines.append(f'| {labels[arm]} | {s["fresh_seconds"]:.5f} | {s["local_grid_fresh_seconds"]:.5f} | {s["local_grid_geometry_seconds"]:.5f} |')
    lines+=['','两种构图使用同一检查点，不依据测试标签调节截断；额外优化版本单列，'
            '未将其速度冒充原生产实现速度。每次完整推理仍包括 SAD 和新几何生成。','',
            '## 逐结构测试','',
            '| 结构 | SAD (%) | 原 R16 + SAD (%) | Eden 主干 + R16 + SAD (%) |','|---|---:|---:|---:|']
    for sid in range(8):
        values=[results['sad']['records'][sid]['test_nmae']]+[results['models'][a]['metrics']['records'][sid]['test_nmae'] for a in ARMS]
        lines.append(f'| {sid} | '+' | '.join(f'{100*x:.5f}' for x in values)+' |')
    lines+=['','## 实现与审计','',
            '- R16 主干保持原初始化哈希、521,472 参数，元素标量基线仍可学习，但现在属于残差。',
            '- 混合模型采用固定官方 commit 的 Encoder，新等变线性头输出 16×25 个系数。'
            '官方主干 y,z,x 坐标通过 Wigner D 转回 x,y,z；旋转等变、读出和主干梯度测试通过。',
            '- 两臂保持 R16 l≤4、16 径向函数、3 Å 支持、原点正则、白化和 float64 重构。'
            '训练使用已审核的代数重排，最终精度以原生产解码器计算；完整网格另作两解码器一致性检查。',
            '- 混合模型官方主干使用经跨参数更新全梯度审核的 CUDA Graph 训练重放；最终评价/计时保持 eager。',
            '- Adam lr 1e-3→1e-5 余弦，16k 更新，梯度裁剪 5，无 weight decay。'
            '每 500 步用原固定 8,192 验证点选模；两臂都冻结权重后才评价测试。',
            '- 每结构实际训练点覆盖与原 R16 一致，验证/测试点参与反向传播数为 0。',
            '- SAD 与此前 Eden 对照使用同一本地 POTCAR form-factor 重构，不拟合目标密度。'
            '尚未与 VASP 原生零步密度核验；本轮没有 PAW/SCF 计算。',
            '- 单种子、8 个已见结构的空间插值；既有测试集已公开，不能据此声称未知结构泛化或统计显著性。','',
            '原始结果、逐次计时、显存、哈希、冻结记录与训练历史位于 '
            '`runs/nmc_eden_r16_2026-10-29/`。协议为 `reports/nmc_eden_r16_protocol_2026-10-29.md`。','']
    for arm in ARMS:
        f=results['frozen'][arm]
        lines.append(f'- {labels[arm]} 最佳步骤 {f["best_step"]}，固定验证子集 NMAE {100*f["best_validation_nmae"]:.5f}%。')
    (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


@torch.no_grad()
def main():
    frozen={a:json.loads((ROOT/a/'selection_frozen.json').read_text()) for a in ARMS}
    for arm,f in frozen.items():
        assert f['training_complete'] and sha256(ROOT/arm/'best.pt')==f['checkpoint_sha256']
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    results={'test_started_utc':datetime.now(timezone.utc).isoformat(),'frozen':frozen,'models':{}}
    if (ROOT/'result.json').exists():
        results=json.loads((ROOT/'result.json').read_text())
        assert results['frozen']==frozen
    factors=read_form_factors(Path('potpaw54.zip'))
    for arm in ARMS:
        if results['models'].get(arm,{}).get('complete'):
            continue
        model=build(arm,'cuda').eval()
        state=torch.load(ROOT/arm/'best.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(state['model'])
        settings=state['settings']
        del state
        samples=prepare(model,arm)
        if 'sad' not in results:
            results['sad']=evaluate(model,samples,arm,final=True,sad_only=True)
        value={'settings':settings,'metrics':evaluate(model,samples,arm,final=True),'timing':[]}
        # Remove training geometry cache so fresh measurements rebuild atom graphs.
        if arm=='r16_sad':
            model._geometry=types.MethodType(type(model)._geometry,model)
        for sample in samples:
            edges=r16_edges(sample,BASIS)
            def decode_edges(c,e):
                return decode(c,-e.displacement,e.batch,e.neighbor,e.center,len(sample['probes']),BASIS)
            def fresh():
                local=dict(sample)
                if arm=='eden_r16_sad':
                    local['atom_data']=atom_data(local)
                c=coefficients(model,local,arm)
                new_edges=r16_edges(local,BASIS)
                sad=make_sad(local,factors)
                return decode_edges(c,new_edges).double()+sad.double()
            row={'sample_id':sample['id']}
            row['fresh'],full=timed(fresh,3)
            row['local_grid_geometry'],local_edges=timed(lambda:uniform_grid_edges(
                sample['positions'],sample['cell'],sample['metadata']['grid_shape'],BASIS.cutoff),3)
            assert len(local_edges.center)==len(edges.center)
            def local_fresh():
                local=dict(sample)
                if arm=='eden_r16_sad':
                    local['atom_data']=atom_data(local)
                c=coefficients(model,local,arm)
                e=uniform_grid_edges(local['positions'],local['cell'],local['metadata']['grid_shape'],BASIS.cutoff)
                return decode_edges(c,e).double()+make_sad(local,factors).double()
            row['local_grid_fresh'],local_full=timed(local_fresh,3)
            torch.testing.assert_close(local_full,full,rtol=3e-5,atol=5e-5)
            row['local_vs_original_grid_max_abs']=float((local_full-full).abs().max())
            row['grid_edges']=len(edges.center)
            del local_edges,local_full
            c=coefficients(model,sample,arm)
            row['decoder'],decoded=timed(lambda:decode_edges(c,edges).double()+sample['sad'].double(),3)
            if arm=='r16_sad':
                original=model._geometry
                model._geometry=types.MethodType(lambda self,*unused:sample['geometry'],model)
            row['warm'],_=timed(lambda:decode_edges(coefficients(model,sample,arm),edges).double()+sample['sad'].double(),3)
            if arm=='r16_sad':
                model._geometry=original
            torch.testing.assert_close(decoded,full,rtol=3e-5,atol=5e-5)
            # Independent production decoder at every grid point, not sparse only.
            ids=torch.arange(len(sample['probes']),device='cuda')
            direct=torch.cat([prediction(sample,c,part) for part in ids.split(2048)])
            torch.testing.assert_close(decoded,direct,rtol=3e-5,atol=5e-5)
            row['fast_vs_production_max_abs']=float((decoded-direct).abs().max())
            value['timing'].append(row)
            results['models'][arm]=value
            write_json(ROOT/'result.json',results)
            print(f'TIMED arm={arm} sample={sample["id"]} seconds={row["fresh"]["median_seconds"]:.4f}',flush=True)
            del c,edges,decoded,direct,full,ids
        value['complete']=True
        results['models'][arm]=value
        write_json(ROOT/'result.json',results)
        del model,samples,value
        gc.collect()
        torch.cuda.empty_cache()
    results['complete']=True
    results['completed_utc']=datetime.now(timezone.utc).isoformat()
    write_json(ROOT/'result.json',results)
    make_report(results)


if __name__=='__main__':
    main()
