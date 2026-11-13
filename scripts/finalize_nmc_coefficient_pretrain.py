"""Unseal tests only after BOTH predeclared two-stage arms are frozen."""
from __future__ import annotations

import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from scripts.prepare_nmc_coefficient_teachers import ROOT
from scripts.fit_nmc_holdout_oracle import BASIS
from scripts.nmc_edengnn import write_json
from scripts.nmc_probe_holdout import DEFAULT_ROOT,compact_prediction,sha256
from scripts.train_nmc_coefficient_pretrain import initialize,prepare_data,coefficient_loss
from scripts.train_nmc_probe_holdout import representation

ARMS=['coefficient_mse','density_metric']
REPORT=Path('reports/nmc_coefficient_pretrain_2026-10-29')


@torch.no_grad()
def score(sample,coeff):
    size=len(sample['target'])
    labels=torch.zeros(size,dtype=torch.long,device=coeff.device)
    labels[sample['split']['validation']]=1
    labels[sample['split']['test']]=2
    sums=torch.zeros(3,4,dtype=torch.float64,device=coeff.device)
    for ids in torch.arange(size,device=coeff.device).split(2048):
        prediction=compact_prediction(sample,coeff,ids,BASIS).double()
        truth=sample['target'][ids]
        sums.index_add_(0,labels[ids],torch.stack(((prediction-truth).abs(),truth.abs(),prediction-truth,(-prediction).clamp_min(0)),1))
    row={'sample_id':sample['id'],'coefficient_relative_mse':float(coefficient_loss(coeff,sample['teacher'])),
         'coefficient_rms':float(coeff.double().square().mean().sqrt())}
    for partition,s in zip(['train','validation','test'],sums):
        row[partition+'_nmae']=float(s[0]/s[1])
        row[partition+'_charge_error']=float(s[2]/s[1])
        row[partition+'_negative_fraction']=float(s[3]/s[1])
    total=sums.sum(0)
    row['full_grid_nmae']=float(total[0]/total[1])
    return row


def make_report(results):
    REPORT.mkdir(parents=True,exist_ok=True)
    baseline=json.loads((DEFAULT_ROOT/'r16_eight/result.json').read_text())
    eden=json.loads(Path('reports/nmc_edengnn_2026-10-28/summary.json').read_text())['eden']['test_macro_nmae']
    summary={name:{group+'_macro_nmae':float(np.mean([r[group+'_nmae'] for r in value['records']]))
                   for group in ['train','validation','test','full_grid']} for name,value in results['models'].items()}
    original=float(np.mean([r['test_nmae'] for r in baseline['records']]))
    write_json(REPORT/'summary.json',{'models':summary,'original_r16_test_nmae':original,'eden_test_nmae':eden})
    mse_fallback=results['models']['coefficient_mse_pretrain'].get('selected_step')==0
    labels={'teacher':'训练点最小二乘教师（非预测模型）',
            'coefficient_mse_pretrain':'系数 MSE：验证选模回退初始权重' if mse_fallback else '系数 MSE：预训练后',
            'coefficient_mse_finetune':'初始权重回退后微调（非有效预训练对照）' if mse_fallback else '系数 MSE：密度微调后',
            'density_metric_pretrain':'密度加权系数监督：预训练后',
            'density_metric_finetune':'密度加权系数监督：密度微调后'}
    lines=['# R16 系数预训练与密度微调：8 个 NMC 结构','',
           '使用原结构 0–7、原空间 80/10/10 划分及同一 R16 架构。每臂 8,000 步系数预训练 + '
           '8,000 步真实密度 L1 微调；没有根据测试结果追加训练。','',
           '## 精度结果','',
           '| 方案 | 训练 NMAE (%) | 验证 NMAE (%) | 测试 NMAE (%) |','|---|---:|---:|---:|',
           f'| 原 R16，直接密度训练 16k | {100*np.mean([r["train_nmae"] for r in baseline["records"]]):.5f} | '
           f'{100*np.mean([r["validation_nmae"] for r in baseline["records"]]):.5f} | {100*original:.5f} |']
    for name,label in labels.items():
        s=summary[name]
        lines.append(f'| {label} | {100*s["train_macro_nmae"]:.5f} | {100*s["validation_macro_nmae"]:.5f} | {100*s["test_macro_nmae"]:.5f} |')
    lines+= [f'| EdenGNN，直接密度训练 16k（既有参照） | — | — | {100*eden:.5f} |','',
             'NMAE = Σ|预测−DFT| / Σ|DFT|，先逐结构计算，再取宏平均。预测没有电荷重归一化或负值裁剪。','']
    if mse_fallback:
        history=json.loads((ROOT/'coefficient_mse/history.json').read_text())
        first=next(r for r in history if r['phase']=='pretrain' and r['step']==0)
        last=next(r for r in history if r['phase']=='pretrain' and r['step']==8000)
        lines+=['**重要：普通系数 MSE 臂预训练失败，验证选出的步骤为 0。** '
                '按预先固定规则，微调实际上从随机初始权重开始，只进行了 8,000 步低学习率密度训练。'
                '其 6.167% 不能解释成“有效系数预训练使微调变差”，也不能与原 16k 密度训练作等有效训练预算比较。',
                f'预训练 8,000 步的相对系数 MSE 从 {first["coefficient_relative_mse"]:.4f} 降到 '
                f'{last["coefficient_relative_mse"]:.4f}，但固定验证子集密度 NMAE 从 '
                f'{100*first["selection_macro_nmae"]:.3f}% 升至 {100*last["selection_macro_nmae"]:.3f}%。'
                '没有额外评价该未选中终点的测试误差，也没有根据测试结果重跑。','']
    lines+= [
             '## 相对改善与剩余差距','']
    for arm in ARMS:
        before=summary[arm+'_pretrain']['test_macro_nmae']
        after=summary[arm+'_finetune']['test_macro_nmae']
        if arm=='coefficient_mse' and mse_fallback:
            lines.append('- 普通系数 MSE 未产生验证密度优于初始化的预训练检查点；后续回退结果单独列出，不作预训练收益推断。')
            continue
        lines.append(f'- {labels[arm+"_finetune"]}：相对原 R16 误差变化 {(after/original-1)*100:+.2f}%；'
                     f'相对本臂预训练后变化 {(after/before-1)*100:+.2f}%；误差为 EdenGNN 的 {after/eden:.2f} 倍。')
    lines+=['','## 逐结构测试','',
            '| 结构 | 教师 (%) | 原 R16 (%) | 系数 MSE＋微调 (%) | 密度加权＋微调 (%) |','|---|---:|---:|---:|---:|']
    per_structure=[]
    for sid in range(8):
        row={'sample_id':sid,'original_r16_test_nmae':baseline['records'][sid]['test_nmae']}
        for name,value in results['models'].items():
            row[name+'_test_nmae']=value['records'][sid]['test_nmae']
        per_structure.append(row)
        lines.append(f'| {sid} | {100*row["teacher_test_nmae"]:.5f} | {100*row["original_r16_test_nmae"]:.5f} | '
                     f'{100*row["coefficient_mse_finetune_test_nmae"]:.5f} | {100*row["density_metric_finetune_test_nmae"]:.5f} |')
    with (REPORT/'per_structure.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(per_structure[0]))
        writer.writeheader()
        writer.writerows(per_structure)
    lines+=['','## 系数诊断与选模','',
            '| 方案 | 平均相对系数 MSE | 平均系数 RMS | 本阶段最优步骤 |','|---|---:|---:|---:|']
    for name,label in labels.items():
        value=results['models'][name]
        relative=np.mean([r['coefficient_relative_mse'] for r in value['records']])
        rms=np.mean([r['coefficient_rms'] for r in value['records']])
        lines.append(f'| {label} | {relative:.6f} | {rms:.6f} | {value.get("selected_step", "—")} |')
    teacher_seconds=sum(json.loads((ROOT/'teachers'/f'sample_{i:04d}'/'complete.json').read_text())['elapsed_seconds'] for i in range(8))
    lines+=['','系数 MSE 是每结构 mean((c−c_teacher)²)/mean(c_teacher²) 后宏平均。'
            '原子基函数有重叠，不同系数分配可能重构出相近密度，因此该量不能替代密度精度。','',
            '## 方法与局限','',
            f'- 教师离线拟合耗时合计 {teacher_seconds/60:.2f} 分钟（含生产解码核验）；系数来自全部训练点，'
            '验证/测试点未参与求解。固定归一化 Gram ridge=1e-9，未按验证或测试调节。',
            '- 教师使用既有 R16 基组的较小范数最小二乘初始解；不使用旧全网格 oracle 或后期大系数 L1 解。',
            '- 两臂与原 R16 随机初始化哈希完全相同，架构、基组、精度及元素标量基线不变。',
            '- 预训练 lr 1e-3→1e-5，微调重置 Adam，lr 1e-4→1e-6；余弦调度、无衰减、梯度裁剪 5。',
            '- 系数 MSE 监督全部原子系数；密度加权臂在每步 1,024 训练点上计算 A(c−c_teacher) 的归一化 MSE。'
            '微调均使用真实密度归一化 L1。教师已利用完整训练分区，不能宣称与原基线等标签暴露或等计算量。',
            '- 每 500 步以原 8,192 验证点的密度 NMAE 选模，预训练最优权重进入微调；'
            '微调保留起点为候选。两臂都冻结 checkpoint 后才评价测试。',
            '- 训练使用已通过数值/梯度核验的 float64 代数重排解码，验证及最终结果使用原生产解码器。',
            '- 本轮只回答已见 8 个结构内的空间插值拟合问题，单种子、固定预算。既有测试集已公开，'
            '不能据此声称新结构泛化或统计显著性，也不能用一次负结果否定所有系数预训练方案。',
            '- EdenGNN 的参数量、SAD 先验、径向表示和角动量配置不同；它是实用精度参照，不是仅改变预训练的消融。','',
            '## 文件与复现','',
            '- 协议：`reports/nmc_coefficient_pretrain_protocol_2026-10-29.md`。',
            '- 原始结果、哈希、冻结记录、采样审计和历史：`runs/nmc_coefficient_pretrain_2026-10-29/`。',
            '```powershell','python -m scripts.prepare_nmc_coefficient_teachers',
            'python -m scripts.train_nmc_coefficient_pretrain --arm coefficient_mse --resume',
            'python -m scripts.train_nmc_coefficient_pretrain --arm density_metric --resume',
            'python -m scripts.finalize_nmc_coefficient_pretrain','```','']
    (REPORT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    for arm,label in zip(ARMS,['Coefficient MSE','Density-weighted coefficients']):
        history=json.loads((ROOT/arm/'history.json').read_text())
        ax.plot([r['step']+(8000 if r['phase']=='finetune' else 0) for r in history],
                [100*r['selection_macro_nmae'] for r in history],label=label)
    ax.axvline(8000,color='gray',linestyle=':',label='Start density fine-tuning')
    ax.set_yscale('log')
    ax.set_xlabel('Optimizer updates')
    ax.set_ylabel('Validation selection NMAE (%)')
    ax.legend(fontsize=8)
    fig.savefig(REPORT/'validation.png',dpi=180)
    plt.close(fig)
    print(json.dumps(summary,indent=2),flush=True)


@torch.no_grad()
def main():
    frozen={}
    for arm in ARMS:
        frozen[arm]=json.loads((ROOT/arm/'selection_frozen.json').read_text())
        assert frozen[arm]['training_complete']
        for phase in ['pretrain','finetune']:
            assert sha256(ROOT/arm/phase/'best.pt')==frozen[arm]['checkpoints'][phase]['checkpoint_sha256']
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    model,_=initialize(torch.device('cuda'))
    samples=prepare_data(model,torch.device('cuda'))
    output=ROOT/'result.json'
    results={'test_started_utc':datetime.now(timezone.utc).isoformat(),'frozen':frozen,'models':{}}
    if output.exists():
        results=json.loads(output.read_text())
        assert results['frozen']==frozen
    for name in ['teacher']+[f'{a}_{p}' for a in ARMS for p in ['pretrain','finetune']]:
        if name in results['models'] and len(results['models'][name]['records'])==8:
            continue
        info={'records':[]}
        if name!='teacher':
            arm,phase=name.rsplit('_',1)
            path=ROOT/arm/phase/'best.pt'
            state=torch.load(path,map_location='cuda',weights_only=False)
            model.load_state_dict(state['model'])
            model.eval()
            info.update(selected_step=state['step'],checkpoint_sha256=sha256(path))
            del state
        for sample in samples:
            coeff=sample['teacher'] if name=='teacher' else representation(model,sample,'r16')
            row=score(sample,coeff)
            info['records'].append(row)
            results['models'][name]=info
            write_json(output,results)
            print('FINAL '+json.dumps({'model':name,**row}),flush=True)
        results['models'][name]=info
    results['complete']=True
    results['completed_utc']=datetime.now(timezone.utc).isoformat()
    write_json(output,results)
    make_report(results)


if __name__=='__main__':
    main()
