"""Frozen-model speed experiments; no training or accuracy-based selection."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import torch

from scripts.nmc_adaptive_r16 import build
from scripts.nmc_edengnn import atom_data,make_sad,write_json,EdenDensity,EdenStencil
from scripts.compare_local_nmc_oracles import load_sample
from scripts.nmc_probe_holdout import DEFAULT_ROOT,save_split,sha256
from scripts.evaluate_vasp_sad_nmc import read_form_factors
from scripts.fit_nmc_holdout_oracle import BASIS
from scripts.r16_grid_diagnostic import uniform_grid_edges
from scripts.r16_decoder_diagnostic import decode
from scripts.r16_inference_fast import decode_atomwise,decode_grouped,TorchSAD
from scripts.benchmark_nmc_edengnn import timed,metrics,prepare_native_eden

ROOT=Path('runs/nmc_inference_fast_2026-10-29')
REPORT=Path('reports/nmc_inference_fast_2026-10-29')
CHECKPOINT=Path('runs/nmc_adaptive_r16_2026-10-29/adaptive/best.pt')


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    frozen=json.loads((CHECKPOINT.parent/'selection_frozen.json').read_text())
    assert sha256(CHECKPOINT)==frozen['checkpoint_sha256']
    model=build('adaptive').eval()
    state=torch.load(CHECKPOINT,map_location='cuda',weights_only=False)
    model.load_state_dict(state['model'])
    del state
    factors=read_form_factors(Path('potpaw54.zip'))
    gpu_sad=TorchSAD(factors)
    eden=None
    if not args.smoke:
        eden_path=Path('runs/edengnn_nmc_2026-10-28/eden_eight/best.pt')
        eden_frozen=json.loads((eden_path.parent/'selection_frozen.json').read_text())
        assert sha256(eden_path)==eden_frozen['checkpoint_sha256']
        eden=EdenDensity().cuda().eval()
        state=torch.load(eden_path,map_location='cuda',weights_only=False)
        eden.load_state_dict(state['model'])
        del state
    result={'started_utc':datetime.now(timezone.utc).isoformat(),'checkpoint_sha256':sha256(CHECKPOINT),
            'smoke':args.smoke,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'records':[],
            'source_sha256':{str(p):sha256(p) for p in [Path(__file__),Path('scripts/r16_inference_fast.py')]}}
    for sid in range(1 if args.smoke else 8):
        s=load_sample(Path(f'data/nmc_cache/{sid}.CHGCAR.lz4'),'cuda')
        s['id']=sid
        splits,metadata=save_split(DEFAULT_ROOT/'spatial_splits',len(s['target']),sid,s['metadata']['sha256'])
        s['split']={k:torch.tensor(v,device='cuda') for k,v in splits.items()}
        data=atom_data(s)
        c=model.coefficients(data)
        edges=uniform_grid_edges(s['positions'],s['cell'],s['metadata']['grid_shape'],BASIS.cutoff)
        def reconstruction(decoder,coef=c,e=edges):
            return decoder(coef,-e.displacement,e.batch,e.neighbor,e.center,len(s['probes']),BASIS)
        row={'sample_id':sid,'source':s['metadata'],'split':metadata,'components':{},'fresh':{},'metrics':{}}
        row['components']['old_decoder'],old=timed(lambda:reconstruction(decode),3)
        row['components']['atom_decoder'],atom=timed(lambda:reconstruction(decode_atomwise),3)
        row['components']['grouped_decoder'],new=timed(lambda:reconstruction(decode_grouped),3)
        torch.testing.assert_close(atom,old,rtol=3e-5,atol=5e-5)
        del atom
        torch.testing.assert_close(new,old,rtol=3e-5,atol=5e-5)
        row['decoder_max_abs']=float((old-new).abs().max())
        row['components']['cpu_sad'],cpu=timed(lambda:make_sad(s,factors),3)
        row['components']['gpu_sad'],gpu=timed(lambda:gpu_sad(s),3)
        torch.testing.assert_close(gpu,cpu,rtol=2e-6,atol=2e-7)
        row['sad_max_abs']=float((gpu-cpu).abs().max())
        reference=old.double()+cpu.double()
        candidate=new.double()+gpu.double()
        torch.testing.assert_close(candidate,reference,rtol=3e-5,atol=5e-5)
        row['full_max_abs']=float((candidate-reference).abs().max())
        row['full_relative_l1']=float((candidate-reference).abs().sum()/reference.abs().sum())
        if not args.smoke:
            row['metrics']['baseline']=metrics(reference,s,s['split'])
            row['metrics']['optimized']=metrics(candidate,s,s['split'])
        del old,new,cpu,gpu,reference,candidate
        if args.smoke:
            result['records'].append(row)
            print(json.dumps(row['components'],indent=2),flush=True)
            continue
        def fresh(name):
            if name=='eden':
                return eden.native(prepare_native_eden(s,factors))
            if name=='eden_gpu_sad':
                local=dict(s)
                local['atom_data']=atom_data(s)
                stencil=EdenStencil(s,csr=False)
                local['native_data']=local['atom_data'].clone()
                for key,value in stencil.native_fields().items():
                    local['native_data'][key]=value
                local['sad']=gpu_sad(s)
                return eden.native(local)
            coef=model.coefficients(atom_data(s))
            e=uniform_grid_edges(s['positions'],s['cell'],s['metadata']['grid_shape'],BASIS.cutoff)
            decoder=decode if name=='baseline' else decode_grouped
            sad=make_sad(s,factors) if name=='baseline' else gpu_sad(s)
            return reconstruction(decoder,coef,e).double()+sad.double()
        names=['eden','eden_gpu_sad','baseline','optimized']
        for name in names:
            output=fresh(name)
            if name.startswith('eden'):
                row['metrics'][name]=metrics(output,s,s['split'])
            else:
                row['fresh_consistency_'+name]=metrics(output,s,s['split'])['test_nmae']
            del output
            row['fresh'][name]=[]
        for repeat in range(5):
            shift=(sid+repeat)%len(names)
            for name in names[shift:]+names[:shift]:
                torch.cuda.synchronize()
                tick=time.perf_counter()
                output=fresh(name)
                torch.cuda.synchronize()
                row['fresh'][name].append(time.perf_counter()-tick)
                del output
        result['records'].append(row)
        write_json(ROOT/'result.json',result)
        print('RESULT '+json.dumps({'sample_id':sid,'fresh':{k:float(np.median(v)) for k,v in row['fresh'].items()},
                                   'full_max_abs':row['full_max_abs']}),flush=True)
    result['complete']=True
    result['completed_utc']=datetime.now(timezone.utc).isoformat()
    write_json(ROOT/('smoke.json' if args.smoke else 'result.json'),result)
    if not args.smoke:
        summary={n:{'test_nmae':float(np.mean([r['metrics'][n]['test_nmae'] for r in result['records']])),
                    'fresh_seconds':float(np.mean([np.median(r['fresh'][n]) for r in result['records']]))}
                 for n in ('eden','eden_gpu_sad','baseline','optimized')}
        summary['components']={k:float(np.mean([r['components'][k]['median_seconds'] for r in result['records']]))
                               for k in result['records'][0]['components']}
        summary['max_abs_density_difference']=max(r['full_max_abs'] for r in result['records'])
        summary['max_relative_l1_difference']=max(r['full_relative_l1'] for r in result['records'])
        write_json(REPORT/'summary.json',summary)
        lines=['# 固定混合模型的进一步推理加速','',
               '不训练、不改变检查点、R16 基组、截断、角阶或 SAD 定义；内部重构保持 FP64。',
               '将每条边重复系数收缩改为逐原子的矩阵乘法；SAD 的同一自然三次样条和傅里叶求和迁移到 GPU。','',
               '| 模型/实现 | 测试 NMAE (%) | 新输入完整生成 (s) |','|---|---:|---:|']
        for n in ('eden','eden_gpu_sad','baseline','optimized'):
            lines.append(f'| {n} | {summary[n]["test_nmae"]*100:.6f} | {summary[n]["fresh_seconds"]:.6f} |')
        lines+=['','每结构预热一次、同进程轮换顺序交错重复五次，8 个结构中位数的平均。',
                '完整时间含原子图、网络、网格邻域、SAD 与解码；不含模型、POTCAR、CHGCAR 文件加载。',
                '仅 POTCAR 常量表作为模型常量驻留 GPU；不复用输入几何、密度或学习输出。','',
                '| 分项 | 秒 |','|---|---:|']
        lines += [f'| {k} | {v:.6f} |' for k,v in summary['components'].items()]
        lines += ['',f'所有完整网格的新旧输出最大绝对差：{summary["max_abs_density_difference"]:.8g} e/Å³。',
                  f'逐结构新旧输出相对 L1 差的最大值：{summary["max_relative_l1_difference"]:.8g}。',
                  '测试划分沿用原始固定划分；本轮只检查实现等价性，不用测试标签调整模型。',
                  'GPU SAD 属于双方都可应用的工程优化，因此同时测量 Eden 原实现与 Eden + GPU SAD。',
                  'Windows 桌面其他 CPU 工作负载可能影响墙钟耗时；以上为本轮交错测量，不与旧轮次绝对耗时直接相除。',
                  '只适用于此处 8 个已见结构和规则周期网格，尚未验证 DFT/SCF 总体加速。','']
        (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
        print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
