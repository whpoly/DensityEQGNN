"""Frozen accuracy and interleaved fresh-input timings on the same device."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.nmc_adaptive_r16 import ROOT,REPORT,ARMS,BASE,BASE_SHA,build
from scripts.nmc_eden_r16 import prepare,prediction,evaluate
from scripts.nmc_edengnn import EdenDensity,atom_data,make_sad,write_json
from scripts.nmc_probe_holdout import sha256
from scripts.fit_nmc_holdout_oracle import BASIS
from scripts.benchmark_nmc_edengnn import timed,prepare_native_eden,metrics
from scripts.r16_grid_diagnostic import uniform_grid_edges
from scripts.r16_decoder_diagnostic import decode
from scripts.evaluate_vasp_sad_nmc import read_form_factors


def render(result):
    REPORT.mkdir(parents=True,exist_ok=True)
    summary={}
    for name,item in result['models'].items():
        summary[name]={'test_macro_nmae':item['metrics']['test_macro_nmae'],
                       'validation_macro_nmae':item['metrics']['validation_macro_nmae'],
                       'train_macro_nmae':item['metrics']['train_macro_nmae'],
                       'fresh_seconds':float(np.mean([np.median(r['fresh_seconds']) for r in item['timing']])),
                       'stages':{k:float(np.mean([r['stages'][k]['median_seconds'] for r in item['timing']]))
                                 for k in item['timing'][0]['stages']}}
    reference=summary['eden']
    for s in summary.values():
        s['speedup_vs_eden']=reference['fresh_seconds']/s['fresh_seconds']
        s['relative_error_vs_eden']=s['test_macro_nmae']/reference['test_macro_nmae']
    write_json(REPORT/'summary.json',summary)
    labels={'eden':'EdenGNN（原 16k）','baseline':'旧混合模型（16k）',
            'continued':'原头继续训练（累计 24k）','adaptive':'原子自适应头（累计 24k）'}
    lines=['# 混合模型继续优化：精度与速度','',
           '相同 8 个已见 NMC 结构及原空间划分。两臂在测试前固定预算、验证选模、冻结权重。',
           '这是已公开测试集上的迭代诊断，不能据此声称未知结构泛化或统计显著性。','',
           '| 模型 | 训练 NMAE (%) | 测试 NMAE (%) | 完整密度生成 (s) | 相对 Eden 加速 |',
           '|---|---:|---:|---:|---:|']
    for name,s in summary.items():
        lines.append(f'| {labels[name]} | {s["train_macro_nmae"]*100:.5f} | {s["test_macro_nmae"]*100:.5f} | '
                     f'{s["fresh_seconds"]:.5f} | {s["speedup_vs_eden"]:.2f}× |')
    lines+=['','## 结论','']
    for name in ARMS:
        s=summary[name]
        hit=s['test_macro_nmae']<=reference['test_macro_nmae'] and s['fresh_seconds']<reference['fresh_seconds']
        lines.append(f'- {labels[name]}：'+('达到本轮精度与速度双目标。' if hit else '未同时达到本轮精度与速度双目标。')+
                     f'误差为 Eden 的 {s["relative_error_vs_eden"]:.3f} 倍；速度为 {s["speedup_vs_eden"]:.2f}×。')
    ratio=summary['adaptive']['test_macro_nmae']/summary['continued']['test_macro_nmae']
    lines += [f'- 同样增加 8000 步后，自适应头相对原头的测试误差变化为 {(ratio-1)*100:+.2f}%。',
              '- Eden 参照没有增加训练；不能把混合模型与 Eden 的差异解释为等训练预算的优劣。',
              '- 保持 R16 基组、角阶、截断和重构不变，新增计算全部发生在原子上。',
              '- 这证明的是实现与小样本诊断效果，尚不能证明方法新颖性或 DFT/SCF 总耗时改善。','',
              '## 分项耗时','',
              '| 模型 | 原子图 (s) | 网络及系数头 (s) | 网格邻域 (s) | SAD (s) | 解码 (s) |',
              '|---|---:|---:|---:|---:|---:|']
    for name,s in summary.items():
        lines.append('| '+labels[name]+' | '+' | '.join(f'{s["stages"][k]:.5f}' for k in
                     ('atom_graph','network','grid_geometry','sad','decoder'))+' |')
    lines+=['','完整计时在同一进程中逐结构轮换模型顺序，预热一次、交错重复五次；CUDA 显式同步。',
            '表中完整耗时为各结构重复中位数的平均。包括新原子图、网格几何、SAD、网络与密度解码；',
            '不含模型/POTCAR/CHGCAR 文件加载。分项独立测量，不将分项和当作完整耗时。',
            'Eden 保持官方原生 4 Å rounded-grid stencil，混合模型为真实 3 Å 球截断。',
            '各模型共享只读输入以公平比较；缓存模式只用于分项计时，完整推理重新生成几何和 SAD。','',
            '## 精度与审计','',
            'NMAE = 逐结构 Σ|预测−DFT|/Σ|DFT| 后做宏平均，不裁负、不强制电荷归一化。',
            '两臂最终精度由原生产解码器计算；优化版本在所有完整网格上逐点比较。',
            '新头通过零残差初始化一致性、不当旋转等变性及梯度测试；真实数据上审核 CUDA Graph 全参数梯度。',
            'SAD 沿用同一本地 POTCAR form-factor 重构，尚未与 VASP 原生零步密度核验。','',
            '| 结构 | Eden (%) | 旧混合 (%) | 继续训练 (%) | 自适应头 (%) |',
            '|---|---:|---:|---:|---:|']
    for sid in range(8):
        lines.append(f'| {sid} | '+' | '.join(f'{result["models"][n]["metrics"]["records"][sid]["test_nmae"]*100:.5f}'
                     for n in ('eden','baseline',*ARMS))+' |')
    lines+=['','原始记录：`runs/nmc_adaptive_r16_2026-10-29/result.json`。',
            '预先固定的方案：`reports/nmc_adaptive_r16_protocol_2026-10-29.md`。','']
    (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


@torch.no_grad()
def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    frozen={a:json.loads((ROOT/a/'selection_frozen.json').read_text()) for a in ARMS}
    for arm,f in frozen.items():
        assert f['training_complete'] and sha256(ROOT/arm/'best.pt')==f['checkpoint_sha256']
        settings=json.loads((ROOT/arm/'settings.json').read_text())
        assert all(sha256(Path(p))==digest for p,digest in settings['source_sha256'].items())
    eden_path=Path('runs/edengnn_nmc_2026-10-28/eden_eight/best.pt')
    eden_frozen=json.loads((eden_path.parent/'selection_frozen.json').read_text())
    assert eden_frozen['training_complete'] and sha256(eden_path)==eden_frozen['checkpoint_sha256']
    models={'eden':EdenDensity().cuda().eval(),'baseline':build('continued').eval(),
            **{arm:build(arm).eval() for arm in ARMS}}
    eden_state=torch.load(eden_path,map_location='cuda',weights_only=False)
    models['eden'].load_state_dict(eden_state['model'])
    del eden_state
    for arm in ARMS:
        state=torch.load(ROOT/arm/'best.pt',map_location='cuda',weights_only=False)
        models[arm].load_state_dict(state['model'])
        del state
    samples=prepare(models['baseline'],'eden_r16_sad')
    factors=read_form_factors(Path('potpaw54.zip'))
    result={'test_started_utc':datetime.now(timezone.utc).isoformat(),'frozen':frozen,
            'base_sha256':BASE_SHA,'eden_sha256':sha256(eden_path),'models':{}}
    for name,model in models.items():
        value={'parameter_count':sum(p.numel() for p in model.parameters()),'timing':[]}
        if name!='eden':
            value['metrics']=evaluate(model,samples,'eden_r16_sad',final=True)
        else:
            value['metrics']={'records':[]}
        result['models'][name]=value
        print('EVALUATED '+name,flush=True)
    for s in samples:
        native=prepare_native_eden(s,factors)
        edges=uniform_grid_edges(s['positions'],s['cell'],s['metadata']['grid_shape'],BASIS.cutoff)
        def reconstruct(c,e):
            return decode(c,-e.displacement,e.batch,e.neighbor,e.center,len(s['probes']),BASIS)
        def fresh(name):
            if name=='eden':
                return models[name].native(prepare_native_eden(s,factors))
            c=models[name].coefficients(atom_data(s))
            e=uniform_grid_edges(s['positions'],s['cell'],s['metadata']['grid_shape'],BASIS.cutoff)
            return reconstruct(c,e).double()+make_sad(s,factors).double()
        # Validate every output before measuring. Drop full-grid temporaries between calls.
        rows={n:{'sample_id':s['id'],'fresh_seconds':[],'stages':{}} for n in models}
        for name,model in models.items():
            full=fresh(name)
            if name=='eden':
                result['models'][name]['metrics']['records'].append({'sample_id':s['id'],**metrics(full,s,s['split'])})
            else:
                c=model.coefficients(s['atom_data'])
                ids=torch.arange(len(s['probes']),device='cuda')
                direct=torch.cat([prediction(s,c,part) for part in ids.split(2048)])
                torch.testing.assert_close(full,direct,rtol=3e-5,atol=5e-5)
                rows[name]['fast_vs_production_max_abs']=float((full-direct).abs().max())
                del direct,c,ids
            del full
        # Cyclic order changes with structure and repeat. No training runs concurrently.
        names=list(models)
        for repeat in range(5):
            shift=(s['id']+repeat)%len(names)
            for name in names[shift:]+names[:shift]:
                torch.cuda.synchronize()
                tick=time.perf_counter()
                output=fresh(name)
                torch.cuda.synchronize()
                rows[name]['fresh_seconds'].append(time.perf_counter()-tick)
                del output
        for name,model in models.items():
            stages=rows[name]['stages']
            stages['atom_graph'],_=timed(lambda:atom_data(s),3)
            stages['sad'],_=timed(lambda:make_sad(s,factors),3)
            if name=='eden':
                from scripts.nmc_edengnn import EdenStencil
                stages['grid_geometry'],_=timed(lambda:EdenStencil(s,csr=False),3)
                stages['network'],rep=timed(lambda:model.encode(s['atom_data']),3)
                stages['decoder'],output=timed(lambda:model.native(native,representation=rep),3)
                del rep
            else:
                stages['grid_geometry'],_=timed(lambda:uniform_grid_edges(
                    s['positions'],s['cell'],s['metadata']['grid_shape'],BASIS.cutoff),3)
                stages['network'],c=timed(lambda:model.coefficients(s['atom_data']),3)
                stages['decoder'],output=timed(lambda:reconstruct(c,edges).double()+s['sad'].double(),3)
                del c
            del output
            result['models'][name]['timing'].append(rows[name])
        write_json(ROOT/'result.json',result)
        print(f'TIMED sample={s["id"]}',flush=True)
        del native,edges
    for group in ('train','validation','test'):
        result['models']['eden']['metrics'][group+'_macro_nmae']=float(np.mean(
            [r[group+'_nmae'] for r in result['models']['eden']['metrics']['records']]))
    result['complete']=True
    result['completed_utc']=datetime.now(timezone.utc).isoformat()
    write_json(ROOT/'result.json',result)
    render(result)


if __name__=='__main__':
    main()
