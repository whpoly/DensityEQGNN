"""Paired scientific summary of the completed, locked Eden fine-tune."""
import json
from pathlib import Path
import numpy as np
from scripts.finetune_public_eden import ROOT,REPORT,BASE,read
from scripts.benchmark_public_density import save

def main():
    cfg=read(BASE/'settings.json');ids=cfg['paired_test'];ft=read(ROOT/'accuracy.json')
    assert len(ids)==497
    by={}
    for arm in ('deep_frozen','eden_frozen','eden_finetuned'):
        old=None if arm=='eden_finetuned' else read(BASE/f'{arm}_final.json')
        both=[]
        for seed in ('42','43'):
            records=ft[seed]['records'] if old is None else old['seeds'][seed]['paired_test']['records']
            lookup={r['id']:r for r in records}; rr=[lookup[i] for i in ids]
            both.append([[r['mae'],r['relative_l1'],*[r['by_l'][str(l)]['mae'] for l in range(3)]] for r in rr])
        by[arm]=np.array(both)
    timing=read(ROOT/'timing.json');assert len(timing)==50
    deep,eden=([r[k]['median'] for r in timing] for k in ('DeepDFT','Eden'))
    delta=(by['eden_finetuned'][:,:,0]-by['deep_frozen'][:,:,0]).mean(0)
    rng=np.random.default_rng(20261102)
    boot=np.mean(delta[rng.integers(0,len(delta),(5000,len(delta)))],1)
    result=dict(common_test=497,means={k:v.mean((0,1)).tolist() for k,v in by.items()},
                columns=['MAE','relative_L1_fraction','L0_MAE','L1_MAE','L2_MAE'],
                paired_delta_eden_minus_deep=float(delta.mean()),paired_structure_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                n_eden_lower=int(sum(delta<0)),seconds={'DeepDFT':float(np.mean(deep)),'Eden':float(np.mean(eden))},
                ratio_total_eden_deep=sum(eden)/sum(deep))
    save(REPORT/'paired_summary.json',result)
    old=read(REPORT/'summary.json')
    lines=['# EdenGNN 全参数微调与 DeepDFT：NMC occupation 对照','',
        '主表使用同一批 497 个结构；官方 500 个测试结构中有 3 个曾用于 Eden 的密度预训练，所以仅作为补充给出 500 结构结果。模型选择没有使用本轮测试指标。','',
        '| 模型 | seed | occupation MAE | 相对 L1 (%) |','|---|---:|---:|---:|']
    for arm,seed,m,r in old['primary497']:lines.append(f'| {arm} | {seed} | {m:.6f} | {r:.3f} |')
    lines+=['','| 两种子平均 | MAE | 相对 L1 (%) | L=0 MAE | L=1 MAE | L=2 MAE |','|---|---:|---:|---:|---:|---:|']
    for arm,(m,r,*rank) in result['means'].items():lines.append(f'| {arm} | {m:.6f} | {100*r:.3f} | '+' | '.join(f'{v:.6f}' for v in rank)+' |')
    lo,hi=result['paired_structure_bootstrap95']
    lines+=['',f'按结构配对，微调 Eden 减 DeepDFT 的 MAE 差为 {delta.mean():+.6f}；结构自助抽样描述性 95% 区间 [{lo:+.6f}, {hi:+.6f}]。两种子取平均后，Eden 在 {sum(delta<0)}/497 个结构上的 MAE 更低。区间不包含训练种子或数据分布变化的不确定性。','',
    '| seed | 初始验证 MAE | 选中轮次 | 最佳微调验证 MAE | 含重叠的 500 结构 MAE |','|---|---:|---:|---:|---:|']
    for seed in ('42','43'):
        s=old['selections'][seed];lines.append(f"| {seed} | {s['initial_validation_mae']:.6f} | {s['epoch']} | {s['validation_mae']:.6f} | {ft[seed]['all500']['macro_mae']:.6f} |")
    lines+=['',f'50 结构预热计时：DeepDFT {np.mean(deep)*1000:.2f} ms/结构，Eden 全参数微调模型 {np.mean(eden)*1000:.2f} ms/结构。Eden/DeepDFT 耗时比 {sum(eden)/sum(deep):.2f}。每个结构交替顺序、各 2 次预热和 5 次同步重复，按每结构中位数再平均。只测 occupation，含新构图与编码，不含文件读写、模型初始化和密度网格。','',
    '两个 Eden 种子分别从对应的冻结头检查点继续训练（1450/50/500 划分），解冻完整编码器和 augmentation 头，每种子新增 4 个完整 epoch。验证选择限于微调轮次；初始模型作为独立对照。DeepDFT 报告的是之前锁定的冻结官方主干和新训练的几何头。预算、预训练数据和头容量不同，不作为架构上限排名。非磁性 NMC、L≤2；没有评估高阶 PAW、联合密度模型或 VASP 收敛。','',
    '协议：protocol.md。全参数权重、梯度更新量、每结构/角阶/元素误差与各次计时保存在 runs/eden_full_finetune_2026-11-02；训练入口 scripts/finetune_public_eden.py。']
    lines+=['','运行记录：种子 43 的首次第 1 轮因 Windows 共享显存分页减速，在生成检查点前中断。从原始检查点和相同种子重跑，仅增加了释放闲置 CUDA 缓存的条件。首次日志、原始源码和修复说明分别保存在 initial_run.log、original_training_source.py 和 runtime_note.txt。']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for s in ('42','43'):
        h=read(ROOT/f'seed_{s}/history.json')
        ax[0].plot(range(5),[old['selections'][s]['initial_validation_mae']]+[r['validation_mae'] for r in h],marker='o',label='seed '+s)
    ax[0].set(title='Eden fine-tuning',xlabel='Additional full epochs',ylabel='Validation MAE',xticks=range(5));ax[0].legend()
    x=by['deep_frozen'].mean(0)[:,0];y=by['eden_finetuned'].mean(0)[:,0]
    lo0=min(x.min(),y.min())*.9;hi0=max(x.max(),y.max())*1.1
    ax[1].scatter(x,y,s=8,alpha=.45,color='#147d80');ax[1].plot([lo0,hi0],[lo0,hi0],'k--',lw=1)
    ax[1].set(xscale='log',yscale='log',xlim=(lo0,hi0),ylim=(lo0,hi0),title='497 common structures',xlabel='DeepDFT MAE',ylabel='Tuned Eden MAE')
    vals=[np.mean(deep)*1000,np.mean(eden)*1000]
    bars=ax[2].bar(['DeepDFT','Tuned Eden'],vals,color=['#147d80','#df8238']);ax[2].bar_label(bars,fmt='%.1f')
    ax[2].set(title='Occupation-only inference',ylabel='Mean milliseconds / structure',ylim=(0,max(vals)*1.20))
    fig.savefig(REPORT/'comparison.png',dpi=170);plt.close(fig)
    lines+=['','![验证曲线、配对测试误差和推理时间](comparison.png)']
    (REPORT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if __name__=='__main__':main()
