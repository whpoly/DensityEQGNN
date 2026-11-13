# QM9 小分子：逐原子系数监督 GNN 实验

## 目的

验证以下路线是否可行：

\[
\rho_{\mathrm{SCF}}
\xrightarrow{\text{global least squares}}
\{c_{i n l m}\}
\xrightarrow{\text{GNN supervision}}
\{\hat c_{i n l m}\}
\xrightarrow{\text{basis reconstruction}}
\hat\rho .
\]

本实验只用拟合得到的逐原子系数监督 GNN，不使用 probe-density loss。

## 数据和划分

- 数据：500 个 QM9/VASP 小分子，原子数不超过 12。
- 划分：360 train / 40 validation / 100 test。
- 密度基组：12 个径向通道，\(l_{\max}=3\)。
- 每个原子的输出维度：

\[
12(1+3+5+7)=192.
\]

- 系数标签：用 8,192 个拟合点对所有原子的系数进行全局最小二乘求解。
- 密度评价：使用与拟合点独立的 4,096 个固定 probe points。
- GNN：严格等变输出 \(l=0,1,2,3\) 系数。
- 训练损失：归一化 coefficient MSE；没有直接使用密度误差。

## 基组表达能力

下表是直接使用最小二乘系数重构密度的独立 probe NMAE。它衡量的是表示和拟合能力，不是机器学习预测误差。

| \(l_{\max}\) | radial | 系数/原子 | 平均 NMAE | 中位数 | P90 |
|---:|---:|---:|---:|---:|---:|
| 2 | 8 | 72 | 11.004% | 10.937% | 13.652% |
| 2 | 12 | 108 | 8.234% | 8.568% | 11.259% |
| 3 | 8 | 128 | 5.981% | 5.792% | 8.224% |
| 3 | 12 | 192 | **3.002%** | **2.923%** | **4.435%** |

因此，增加径向通道和 \(l\) 都能显著提高重构能力。

## 纯系数监督结果

最佳 validation checkpoint 出现在 epoch 53，训练在 epoch 83 early stop。

| 指标 | Validation | Test |
|---|---:|---:|
| GNN 重构密度的 global NMAE | 76.654% | **80.313%** |
| LS 标签自身的密度 NMAE | 2.975% | **2.954%** |
| normalized coefficient RMSE | 0.685 | 0.667 |
| sample NMAE 中位数 | 80.993% | 80.190% |
| sample NMAE P90 | 119.260% | 133.990% |

作为对照，此前直接用 probe-density loss 训练的 \(l_{\max}=2\)、radial=8 模型，测试 global NMAE 为 **3.911%**；固定原子密度叠加基线为 **11.384%**。

## 结论

这次纯系数监督实验失败了。失败并不是因为 \(l=3\)、radial=12 的基组不能表示密度：最小二乘标签自身能把测试密度重构到约 2.95% NMAE。真正的问题是，多中心重叠基组的逐原子系数不是唯一且稳定的局部物理量。

全局最小二乘可以在不同原子和不同基函数之间重新分配相互抵消的贡献，而总密度几乎不变。因此：

\[
\text{相似的局部原子环境}
\not\Rightarrow
\text{相似的最小二乘逐原子系数}.
\]

局部 GNN 被要求预测一个受全体系条件数和 gauge 影响的标签，导致 coefficient RMSE 看起来有限，但系数误差经过大量重叠基函数叠加后产生很大的密度误差。

对同一个 QM9 分子使用三组不同的 LS 拟合采样点时，系数向量相对变化达到 24.6%–27.3%，但对应重构密度之间的 NMAE 只有 3.2%–3.6%。该样本的设计矩阵有效条件数约为 \(9.9\times10^{13}\)，1920 个系数方向中约 1732 个在数值阈值下有效。这直接说明系数存在严重病态性和近零空间。

## 推荐后续路线

保留“GNN 输出系数、系数重构连续密度”的模型形式，但不要把全局最小二乘系数当作唯一监督标签。应直接优化重构后的总密度：

\[
\{R_i,Z_i\}
\xrightarrow{\mathrm{equivariant\ GNN}}
\{\hat c_{i n l m}\}
\xrightarrow{\mathrm{reconstruction}}
\hat\rho(\mathbf r_g)
\xrightarrow{\mathrm{density\ loss}}
\rho_{\mathrm{SCF}}(\mathbf r_g).
\]

也就是让系数作为 latent variables，用 probe-density NMAE/MSE 端到端训练。若必须直接监督系数，则需要先定义唯一且稳定的 gauge，例如固定的原子密度分区、正交化辅助基或带正则化的规范。

## MACE-style 端到端验证

随后实现了两层 tensor-product interaction 的 MACE-style 网络。隐藏特征和输出均包含 \(l=0,1,2,3\)，各非标量阶至少包含 12 个 multiplicity，以匹配 12 个径向输出通道。

同一个模型比较两种训练方式：

| 方法 | Validation NMAE | Test NMAE |
|---|---:|---:|
| MACE-style + 直接 coefficient MSE | 86.358% | 92.303% |
| MACE-style + 重构 density NMAE | **3.032%** | **3.107%** |

测试集的逐样本统计：

- 平均 NMAE：3.108%
- 中位数：2.971%
- P90：3.875%
- LS 表示参考：2.954%

因此最终可行的训练图是：

\[
\{Z_i,\mathbf R_i\}
\xrightarrow{\mathrm{MACE\text{-}style\ GNN}}
\{c_{inlm}\}
\xrightarrow{\sum c_{inlm}R_nY_{lm}}
\hat\rho(\mathbf r_g)
\xrightarrow{\mathrm{NMAE}}
\rho_{\mathrm{SCF}}(\mathbf r_g).
\]

训练不再需要预先回推出 LS 系数。LS 只作为离线 basis-capacity/oracle 诊断工具。这里的 \(c_{inlm}\) 是网络自行选择的 latent variables，不应解释为唯一的原子物理系数。

## 结果文件

- `runs/qm9_l3_r12_coeff_500/test_metrics.json`
- `runs/qm9_l3_r12_mace_like_coeff_500/test_metrics.json`
- `runs/qm9_l3_r12_mace_like_density_500/test_metrics.json`
- `runs/qm9_l3_r12_coeff_500/history.json`
- `runs/qm9_atoms_le12_500/oracle_l3_r12_test100.json`
- `configs/qm9_l3_r12_coeff_500.yaml`
