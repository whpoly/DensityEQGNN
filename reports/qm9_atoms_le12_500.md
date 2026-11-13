# 小原子数 QM9 密度 benchmark

日期：2026-07-24  
状态：首轮本地验证完成

## 结论

在 500 个总原子数不超过 12 的 QM9/VASP 分子上，使用 360/40/100
的训练、验证、测试划分：

| 方法 | 测试 density NMAE |
|---|---:|
| 固定球形原子密度叠加 | 11.384% |
| 本项目：原子图 GNN + 可微密度重构 | **3.911%** |
| 离线 LS-fit 参考系数 | 8.679% |

本项目相对固定原子叠加把 NMAE 降低了 65.6%。这证明环境相关的原子
密度修正在小数据上可以学习。但是 3.911% 仍显著高于 ChargE3Net 在完整
QM9 benchmark 上报告的 0.196%，而且本实验评估的是每个分子 4,096 个固定
随机 probe，不是完整 FFT 网格，因此不能把两个数字视作同条件排名。

本机没有 VASP 和有许可证的 POTCAR，所以本报告没有测实际 SCF 电子步数。
当前结论仅限于密度预测误差。

## 数据

原始数据来自公开的
[QM9 Charge Densities and Energies Calculated with VASP](https://data.dtu.dk/articles/dataset/QM9_Charge_Densities_and_Energies_Calculated_with_VASP/16794500)。

- 候选按照包含氢在内的总原子数升序筛选；
- 取前 500 个总原子数不超过 12 的分子；
- 外层随机划分为 400 个 development 和 100 个 test；
- development 再分为 360 train 和 40 validation；
- test 只在训练和选模结束后评估一次；
- 元素集合为 H、C、N、O、F；
- 原始压缩包、筛选列表和处理结果保存在 `D:\density-eqgnn-data`。

公开压缩包只含最终 CHGCAR，不含 VASP 第 0 步 SAD 网格或真实 POTCAR。
所以本文的“固定球形原子密度叠加”是由 360 个训练分子的总密度联合拟合
得到的 5 种固定原子密度。它与 SAD 具有相同的数学形式，但不是 VASP 原生
SAD 的逐点复现。

## 模型

模型只在原子图上运行一次：

\[
\{\mathbf R_i,Z_i\}
\xrightarrow{\mathrm{GNN}}
\{c_i\}
\xrightarrow{\mathrm{basis}}
\rho_\theta(\mathbf r).
\]

设置如下：

- 4 层距离消息传递；
- hidden dimension 128；
- cutoff 4 Å；
- 8 个径向 Gaussian；
- \(l_\max=2\)；
- 每原子输出 \(8(1+3+9)=104\) 个数；
- 输出包含 scalar、vector 和 symmetric-traceless tensor 通道；
- 初始输出严格等于固定球形原子叠加；
- 网络只学习环境相关修正。

对任意 probe，密度通过固定的可微原子中心基函数重构：

\[
\rho_\theta(\mathbf r_g)
=
\sum_i\left[
\sum_n c^{(0)}_{in}R_n(r_{ig})
+
\sum_n \mathbf c^{(1)}_{in}\cdot\hat{\mathbf r}_{ig}R_n(r_{ig})
+
\sum_n\hat{\mathbf r}_{ig}^{T}C^{(2)}_{in}
\hat{\mathbf r}_{ig}R_n(r_{ig})
\right].
\]

这与 ChargE3Net 都使用 probe 上的最终 DFT 密度监督，但结构不同：
ChargE3Net 让 probe 参与图计算并直接输出该点密度；本模型先预测紧凑的
每原子系数，再解析地生成任意位置的密度。

## 训练

- 每个训练分子每步随机使用 1,024 个 probe；
- validation/test 每个分子使用固定的 4,096 个 probe；
- loss 是与 ChargE3Net 指标同形的 density NMAE；
- oracle/LS 系数监督权重为 0；
- AdamW，初始学习率 \(10^{-3}\)；
- ReduceLROnPlateau；
- 最多 200 epochs；
- validation 最佳 epoch 为 198。

第一版直接监督离线全局拟合系数，validation NMAE 最好只有约 118%，说明
不同原子重叠基函数导致的系数规范不适合作为独立原子监督标签。第二版因此
改为从固定原子密度起步、只对总密度进行端到端监督。

## 测试结果

最佳 checkpoint 在 100 个测试分子上的结果：

| 指标 | 数值 |
|---|---:|
| 全测试集加权 NMAE | **3.9106%** |
| 逐分子 NMAE 平均 | 3.9653% |
| 逐分子 NMAE 中位数 | 3.9620% |
| 逐分子 NMAE 90 分位 | 4.5692% |
| 固定原子叠加加权 NMAE | 11.3844% |
| 离线 LS-fit 参考重构 NMAE | 8.6789% |

“离线 LS-fit 参考”不是严格 oracle 下限。它在另外 20,000 个点上用
L2+ridge 拟合，再在 4,096 个独立点上用 NMAE 评估。端到端模型直接优化
跨分子的 probe NMAE，因此可以优于这组参考系数。

## 10 个 Materials Project 周期晶体

通过 Materials Project API 下载了 10 个非磁性、四原子、具有最终 CHGCAR
的周期材料，用相同 \(l_\max=2\) 表示做径向数目消融。Materials Project
说明其电荷密度直接来自静态 DFT 的最终 CHGCAR：
[MP charge density 文档](https://docs.materialsproject.org/methodology/materials-methodology/charge-density)。

材料为：AgHgPd2、ScAlIr2、AcH3、LiSc2Ir、LiZnAg2、Ti2AlTc、Ti2AlRe、
Ca2MgTl、ThCdRh2 和 HoInPd2。

| radial 数 | 每原子系数数 | 平均 LS-fit NMAE | 中位数 |
|---:|---:|---:|---:|
| 4 | 52 | 30.096% | 26.016% |
| 8 | 104 | 4.303% | 4.255% |
| 12 | 156 | **1.956%** | 1.956% |
| 16 | 208 | 1.744% | 1.569% |

增加 radial 明显提高表达能力，8 到 12 的收益很大，12 到 16 开始呈现边际
收益递减。因此下一版优先使用 12 radial，而不是直接扩到 16。

## 与论文结果的关系

[ChargE3Net](https://www.nature.com/articles/s41524-024-01343-1) 在完整
QM9 数据和完整论文设置下报告 0.196% NMAE。该论文还在 Materials Project
未见非磁性材料上把 SAD 初始化的 SCF 中位步数从 15 降至 11，即减少
26.7%。这两个结果来自不同数据集：0.196% 是 QM9 密度误差，26.7% 是 MP
上的 SCF 加速。

本报告不能复现 SCF 加速，因为本机没有 VASP/POTCAR。Materials Project
API 提供 POTCAR 标识和计算元数据，但不会分发受许可证约束的 POTCAR 内容；
正式 SCF 验证需在有 VASP 和对应 PAW 库的服务器上完成。

## 本地环境

- Windows 11；
- Intel Core i5-14400F；
- NVIDIA GeForce RTX 5060 Ti 16 GB；
- Python 3.11.15；
- PyTorch 2.11.0+cu128；
- 4 个单元/等变性测试全部通过。

## 复现

数据根目录：

```powershell
$env:DENSITY_EQGNN_DATA_ROOT = "D:\density-eqgnn-data"
```

固定原子叠加基线：

```powershell
python scripts/evaluate_atomic_superposition.py `
  --config configs/qm9_atoms_le12_500.yaml `
  --data-root D:\density-eqgnn-data\qm9_atoms_le12_500 `
  --output runs/qm9_atoms_le12_500/atomic_superposition_metrics.json
```

训练：

```powershell
python -m density_eqgnn.training.train `
  --config configs/qm9_atoms_le12_500.yaml
```

测试：

```powershell
python -m density_eqgnn.training.evaluate `
  --config configs/qm9_atoms_le12_500.yaml `
  --checkpoint runs/qm9_atoms_le12_500/best.pt `
  --split test `
  --output runs/qm9_atoms_le12_500/test_metrics.json
```

## 下一步

1. 把 radial 从 8 增至 12，并重新生成 QM9 系数/密度数据；
2. 将小原子数样本从 500 扩至最初计划的 3,500；
3. 评估完整网格 NMAE，而不只使用 4,096 probes；
4. 在有 VASP 和相同 POTCAR 的服务器上比较原生 SAD、ML 初猜和最终 SCF
   的电子步数与墙钟时间。
