# 周期边界修复与精度路线复核

日期：2026-10-24。结论：当前应把正确性、表示能力与预测精度放在前面；修复前的 NMC 误差距离同基准公开结果仍约 19 倍，现阶段不足以把效率作为主要贡献。本文区分已完成的修复、历史实验和待验证的建议。

**已完成的修复**

原子图和密度展开共用 [periodic.py](<C:/Users/User/Desktop/density EQGNN/src/density_eqgnn/periodic.py>) 的周期邻居枚举。晶胞使用行向量约定，原子图保留满足

\[
\lVert R_j+nA-R_i\rVert<r_c,\quad n\in\mathbb Z^3
\]

的所有边，仅排除 `i=j, n=0`，同一原子的非零周期平移仍参与消息传递。邻居表允许同一源原子出现多次，不再用一个原子对槽位代替所有周期像。MACE-like、compact 和旧 Cartesian 三种模型均已接入。

枚举范围由倒晶格确定：若位移小于 cutoff，则其第 k 个分数坐标的绝对值不超过 `cutoff * norm(inv(cell)[:, k])`。分数坐标取整只用于把搜索起点移回有限范围，不再把取整后的位移视为斜晶胞的最近像。由此覆盖一般可逆斜晶胞及跨多个晶胞的 cutoff。

密度重构累加支持半径内每个原子像的贡献；球谐方向统一为“原子像指向 probe”，避免奇数 l 的符号错误。修复涵盖 compact Torch、旧 Cartesian Torch、NumPy 设计矩阵及使用该矩阵的拟合路径，球形基线拟合也同步使用全部像。仅用于距离选择的 NumPy 最近像辅助函数改为真正的最近像搜索。旧 Cartesian Torch 的二阶基同时对齐 NumPy 的去迹约定；模型输出本来就是无迹张量。

离散邻居选择不求导，选中边的位移仍从原位置和晶胞张量计算，保留位置、晶胞和系数梯度。枚举按 probe 分块，重构按有效边分块；本次没有测量或宣称修复后运行速度。

**验证结果**

[test_periodic.py](<C:/Users/User/Desktop/density EQGNN/tests/test_periodic.py>) 新增 15 个测试实例。独立参考使用固定的宽周期像盒枚举，不调用生产代码的搜索边界，以避免两个实现共享同一缺陷。

| 范围 | 已验证内容 |
|---|---|
| 斜晶胞 | 完整边集合与独立像枚举相等；最近像不能由分数坐标取整代替；l≤4 密度与独立求和一致 |
| 单原子晶胞 | 2 Å 立方胞、2.1 Å cutoff 保留 6 个自身周期像；5 Å 胞、3 Å 密度 cutoff 正确累加两侧像 |
| 原胞与超胞 | 三种模型在斜晶胞和 2×2×1 超胞中给出一致的每原子系数及端到端密度 |
| 边界与批处理 | 跨分数坐标半胞分界连续；周期等价点相等；padding 不贡献；独立平移原子/probe、旋转晶胞和批处理一致 |
| 可微性 | 系数、原子位置、probe 位置和晶胞通过 float64 gradcheck；密度损失反向传播；空邻居返回可微零值 |
| 兼容性 | 没有额外像进入 cutoff 时，PBC 与非 PBC 重构一致；旧周期 checkpoint 不允许静默续训 |

全套测试：**26 passed in 6.13 s**。命令：

```powershell
& 'C:\Users\User\.conda\envs\kan-cgcnn-cuda\python.exe' -m pytest -q
```

真实 NMC sample 0001（41 原子，原子图 cutoff 6 Å）的独立验证：

| 检查 | 结果 |
|---|---:|
| 修复前原子图边数，来自此前诊断 | 1,362 |
| 修复后边数 / 独立枚举边数 | 4,016 / 4,016 |
| 原胞与 2×2×1 超胞系数最大绝对差，float64 | 4.97×10⁻¹⁴ |
| 端到端密度相对 L1 差，128 个随机 probe | 4.72×10⁻¹⁶ |
| 固定原胞系数复制后的密度相对 L1 差 | 3.01×10⁻¹⁶ |
| CUDA，batch=2，每结构 2,048 probe | 前向、反向通过；梯度有限且非零 |

这组一致性检查使用旧 checkpoint 的参数，但没有把它当作新算子的精度评测。参考 [诊断脚本](<C:/Users/User/Desktop/density EQGNN/runs/pbc_fix_20261024/check_real_nmc.py>) 和 [JSON 结果](<C:/Users/User/Desktop/density EQGNN/runs/pbc_fix_20261024/diagnostics.json>)。此前 0.07606% 的胞表示差异与本次使用不同 probe，不能当成严格同点的误差下降；独立参考和新回归测试才是本次正确性证据。

**训练和旧结果的处理**

新 checkpoint 保存 `periodic_geometry_version=all_images_v1`。周期训练会拒绝直接恢复旧优化器和训练历史，主要评估入口会对旧 checkpoint 给出提示。参数形状仍兼容，所以旧参数可以做诊断，但旧验证最优值不能沿用为新训练的 best。

新增 [nmc_full_l2_to_l4_mace_pbc.yaml](<C:/Users/User/Desktop/density EQGNN/configs/nmc_full_l2_to_l4_mace_pbc.yaml>)，只更换输出目录为 `runs/nmc_full_l2_to_l4_mace_pbc`，保留同一架构、数据划分和训练参数，便于先测量 PBC 修复本身的影响。完整 NMC 流水线已指向这个配置及独立报告。历史结果目录保留。

原始密度标签和 probe 坐标可复用；旧周期系数标签、基线拟合系数和 oracle 数值需要重新拟合。当前 NMC 主配置是 density-only，因此不依赖旧 LS 系数标签。

本次尚未进行完整 NMC 重训，也没有新的测试集 NMAE。当前机器未挂载 README/流水线默认的 `D:\density-eqgnn-data`，工作区也没有完整 `nmc_full_probes`；已有小样本和保存的 POSCAR 足够完成本次正确性验证，但不能替代官方完整划分。数据恢复后可运行：

```powershell
# DataRoot 指向包含 nmc_raw 和 nmc_full_probes 的真实数据根目录。
powershell -ExecutionPolicy Bypass -File scripts/run_nmc_full_benchmark.ps1 `
  -DataRoot '<实际数据根目录>'
```

**精度差距：哪些可以直接比较**

统一用百分数 NMAE：每个结构先计算 `sum(abs(pred-target))/sum(abs(target))`，再对结构取平均。仓库 JSON 中的比值需乘 100；例如 `0.00060` 才是 `0.060%`。跨结构总分子/总分母的 global NMAE 不能混作 macro NMAE。

NMC 的对照最直接：本项目已有官方 1,450/50/500 划分和 500 个测试结构的完整网格结果。

| NMC 方法 | 完整网格平均 NMAE（%） | 解释 |
|---|---:|---|
| 本项目，修复前 epoch 176 | 1.1474 | 历史基线；修复后必须重训 |
| invDeepDFT | 0.089 ± 0.001 | 公开对照 |
| equiDeepDFT | 0.061 ± 0.001 | 公开对照 |
| ChargE3Net | 0.060 ± 0.001 | 公开对照 |

公开结果来自 [ChargE3Net 原文 Table 1](https://arxiv.org/html/2312.05388v2#S2.T1)。1.1474/0.060≈19.1，因此“优于原子叠加”不足以说明达到有竞争力的精度。equiDeepDFT 使用 PaiNN 的标量/向量特征，已接近 ChargE3Net；这反驳了“主要因为 trunk 只有 l≤2”这一未经验证的解释。高阶特征可能有益，但不能代替表示和训练诊断。

QM9 则必须分开看：本项目约 **3.107%** 来自最多 12 原子的 500 分子子集，训练仅 360 个结构，且使用采样点评估。下面是完整 QM9/VASP 协议结果，不能据此把差值全部归因于模型结构。SCDP 使用 123,835/50/10,000 划分和完整网格，训练规模约为本地小实验的 344 倍。[SCDP，实验设置与 Table 1](https://papers.nips.cc/paper_files/paper/2024/file/12c118ef87fde56a10bd858842781b34-Paper-Conference.pdf)

| QM9/VASP 方法 | NMAE（%） | 原文 |
|---|---:|---|
| ChargE3Net | 0.196 | [Table 1](https://arxiv.org/html/2312.05388v2#S2.T1) |
| SCDP 最佳设置 | 0.178 | [Table 1](https://papers.nips.cc/paper_files/paper/2024/file/12c118ef87fde56a10bd858842781b34-Paper-Conference.pdf) |
| ELECTRA | 0.176 | [2026-01 v5，Table 1](https://arxiv.org/html/2503.08305v5) |
| BOA small | 0.1381 ± 0.0003 | [ICLR 2026，Table 1](https://proceedings.iclr.cc/paper_files/paper/2026/file/718573eff1cb169316783d3e08514b5b-Paper-Conference.pdf) |
| BOA large | 0.1339 ± 0.0005 | [ICLR 2026，Table 1](https://proceedings.iclr.cc/paper_files/paper/2026/file/718573eff1cb169316783d3e08514b5b-Paper-Conference.pdf) |

BOA 中引用 ELECTRA 为 0.177，ELECTRA v5 自己报告 0.176，这里保留版本区别。BOA 的 QM9/PySCF 是另一套密度标签，不与 QM9/VASP 或 NMC 混排。这里所列分子论文结果没有证明其方法已在我们的 NMC 数据上达到同样误差。

**其他文章真正支持什么改进**

| 原文消融 | NMAE 变化（%，QM9） | 对本项目的启示（待实测） |
|---|---|---|
| SCDP：加深并提高阶数，K=4/L=3 → K=8/L=6，β=2 | 0.504 → 0.434 | 扩大骨干有收益，但该消融同时改深度和阶数，不能单独归因于 l |
| SCDP：相同 K=8/L=6/β=2，加入虚拟轨道 | 0.434 → 0.237 | 基中心位置值得优先检查，特别是原子间区域 |
| SCDP：最佳固定基再微调尺度 | 0.196 → 0.178 | 可先训练系数，再解冻受限尺度参数 |
| BOA small：线性 → 二次密度展开，其余超参数固定 | 0.2716 → 0.1381 | 仅加原子中心基的数量，可能不如改变表达形式有效 |

前 3 行依据 [SCDP Table 1](https://papers.nips.cc/paper_files/paper/2024/file/12c118ef87fde56a10bd858842781b34-Paper-Conference.pdf)，最后一行依据 [BOA Table 5](https://proceedings.iclr.cc/paper_files/paper/2026/file/718573eff1cb169316783d3e08514b5b-Paper-Conference.pdf)。BOA 的 overlap 消息传递及包含双中心贡献的二次展开，和当前固定原子中心线性密度展开有本质区别；不能把它简化为“另一种系数回归器”。

ELECTRA 则学习 Gaussian 的位置、形状和权重；其消融中，去掉 floating orbitals 为 6.069%，禁止负 Gaussian 权重为 2.027%，完整模型为 0.176%。这些是该特定 Gaussian 架构内部的证据，不能推出所有原子中心球谐展开都不可行。负展开权重也不等于允许总电子密度任意为负。[ELECTRA Table 4](https://arxiv.org/html/2503.08305v5)

我的判断是：当前项目应先检验“是否把足够的表示能力放在正确的空间位置和径向尺度上”，再讨论压缩网络。上述消融支持这个实验优先级，并不保证照搬到过渡金属氧化物就能获得相同收益。

**建议的精度实验顺序**

1. **先建立修复后的同设置基线。** 使用新配置从头训练，保持旧划分与超参数，完整网格评估。训练中加入原胞/超胞一致性抽检，但不要把一致性误差当作预测误差。本次的回归测试负责阻止 PBC 缺陷再次出现。

2. **把表示误差与预测误差分开测。** 在训练/验证结构中固定一小组诊断样本，例如 20 个训练结构、5 个验证结构；所有候选在相同完整网格上比较直接优化系数与 GNN 输出。监测优化收敛，区分采样点拟合和未参与优化的网格点误差，不用官方测试集反复选基组。旧 3 结构结果从同组初始 1.2248% 降到 0.6705%，只能说明存在更好系数；有限步优化得到的是最优误差的上界，既不是基组下限，也不能跨到 500 结构直接做误差分解。修复后需要重测。

3. **优先做基函数消融。** 当前 l>0 通道径向函数在 r→0 时不消失，允许方向相关极限。先比较规范化 `r^l exp(-αr²)Y_lm` 或 solid harmonics；将最小径向宽度与 cutoff 解耦，再比较元素/l 相关尺度。接着加入确定的周期键中心基：中心应从完整周期边生成并去重，复制超胞后不能凭空增加物理自由度。固定基预训练稳定后，再尝试有界尺度微调。每个改动单独测 oracle 和 GNN；暂不同时改变骨干、基组和采样。

4. **若直接拟合已低误差而 GNN 仍明显落后，再改训练和容量。** 从完整网格动态采样，替换目前从固定 16,384 点中反复抽 2,048 点的方式；均匀采样作为基准，混合近核/成键采样时保留对应权重。比较学习率衰减、训练预算和通道宽度。继续以密度监督为主，避免直接恢复病态基下的普通系数 MSE。高阶 readout 的扫描须满足实际角动量耦合上限；当前 2⊗2 最多到 l=4。

5. **用精度门槛决定是否进入效率研究。** 我建议将完整 NMC 平均 NMAE <0.1% 作为阶段目标，随后争取接近公开约 0.06% 的水平；这是项目决策目标，不是普适的 SCF 精度阈值。报告至少 3 个随机种子、macro NMAE、P90、积分电子数偏差、负密度积分，以及按最近原子距离/元素/成键区分解的误差。等这些稳定后，再做等精度条件下的速度和 SCF 收敛对照。

判断分支应很明确：若多种充分优化后的系数拟合仍难以达到目标，继续改基函数、中心或二次展开；若拟合已很好而网络差距大，资源转向特征、监督和采样。不要用单次 oracle 失败证明表示不可行，也不要用更快但误差大一个数量级的模型证明方法已经具有优势。
