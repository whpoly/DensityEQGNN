# Density EQGNN

这个仓库的当前模型采用以下研究路线：

```text
最终 SCF 电子密度
  -> 训练集联合拟合固定球形原子基线
  -> 等变 GNN 预测环境相关原子系数
  -> 重构连续电子密度
  -> 在 probe 上直接以最终密度监督
  -> 用作后续 SCF 初猜
```

## 密度表示

当前基准使用 \(l_{\max}=2\)：

\[
\rho(\mathbf r)=\sum_{i,n}R_n(r_i)
\left[
c^{(0)}_{in}
+
\mathbf c^{(1)}_{in}\cdot\hat{\mathbf r}_i
+
\hat{\mathbf r}_i^\mathsf T C^{(2)}_{in}\hat{\mathbf r}_i
\right].
\]

- \(c^{(0)}\)：旋转不变标量；
- \(\mathbf c^{(1)}\)：随旋转变换的向量；
- \(C^{(2)}\)：按 \(C\mapsto RCR^\mathsf T\) 变换的对称无迹张量。

第一版离线 LS-fit 系数只作为参考。正式模型不监督这些非唯一的原子系数，
而是从固定原子密度叠加开始，直接最小化重构总密度在 probe 上的 NMAE。

## 本地 QM9 小分子基准

首轮基准选取 500 个公开 QM9/VASP 密度，要求总原子数（含 H）不超过
12。外层按固定随机种子分成 80% 开发集和 20% 测试集；开发集内部再划出
10% 用于验证，因此实际为：

```text
360 optimization / 40 validation / 100 test
```

已完成结果：

| 方法 | 100 个测试分子的 density NMAE |
|---|---:|
| 固定球形原子密度叠加 | 11.384% |
| 等变 GNN + 可微密度重构 | **3.911%** |
| 离线 LS-fit 参考系数 | 8.679% |

完整实验设置、MP 10 材料 radial 消融和限制见
[benchmark 报告](reports/qm9_atoms_le12_500.md)。

原始大文件、压缩成员和处理后数据都放在 `D:\density-eqgnn-data`。

```powershell
$env:DENSITY_EQGNN_DATA_ROOT = "D:\density-eqgnn-data"
$python = "C:\Users\User\.conda\envs\kan-cgcnn-cuda\python.exe"

# 下载第一个 QM9/VASP tar；支持断点续传和 MD5 校验
& $python scripts/download_qm9_benchmark.py --archive 000xxx.tar

# 从 QM9 元数据选择总原子数 <= 12 的 500 个分子
& $python scripts/select_qm9_subset.py `
  --metadata D:\density-eqgnn-data\qm9\qm9_structures.parquet `
  --output D:\density-eqgnn-data\qm9\qm9_atoms_le12_500_ids.txt `
  --samples 500 `
  --max-atoms 12

# 从 CHGCAR 拟合每原子系数并生成固定划分
& $python scripts/prepare_qm9_benchmark.py `
  --samples 500 `
  --selection-file D:\density-eqgnn-data\qm9\qm9_atoms_le12_500_ids.txt `
  --output D:\density-eqgnn-data\qm9_atoms_le12_500 `
  --max-atoms 12 `
  --workers 8

# 训练集联合拟合固定球形原子密度基线
& $python scripts/evaluate_atomic_superposition.py `
  --config configs/qm9_atoms_le12_500.yaml `
  --data-root D:\density-eqgnn-data\qm9_atoms_le12_500 `
  --output runs/qm9_atoms_le12_500/atomic_superposition_metrics.json

# GPU 训练和独立测试
& $python -m density_eqgnn.training.train `
  --config configs/qm9_atoms_le12_500.yaml

& $python -m density_eqgnn.training.evaluate `
  --config configs/qm9_atoms_le12_500.yaml `
  --checkpoint runs/qm9_atoms_le12_500/best.pt `
  --split test `
  --output runs/qm9_atoms_le12_500/test_metrics.json
```

主要指标是 ChargE3Net 使用的归一化绝对误差：

\[
\varepsilon_\mathrm{mae}
=\frac{\sum_g|\rho_g-\hat\rho_g|}{\sum_g|\rho_g|}.
\]

本项目 JSON 中保存的是比值，例如 `0.08` 表示 `8%`。零密度预测的误差为
`1.0 = 100%`。当前小基准用每个分子的 4,096 个固定随机网格点估计该指标；
ChargE3Net 论文的最终结果使用完整网格，因此只能比较定义，不能把数值当成严格复现。

## NMC smoke test

周期体系的最小冒烟测试仍保留在 `configs/nmc_small.yaml`：

```powershell
& $python scripts/prepare_nmc_subset.py --samples 8
& $python -m density_eqgnn.training.train --config configs/nmc_small.yaml
```

## 完整 NMC：低阶 MACE 到高阶密度

正式 NMC 对比使用官方 2,000 个结构和 1,450/50/500 划分。模型的消息传递
只保留到 \(l_\text{hidden}=2\)，最后一次局域 tensor-product readout 使用
\(2\otimes2\) 的完整 Clebsch–Gordan 通道输出到
\(l_\text{density}=4\)。因此高阶只出现在 readout 和解析密度重构中，不会让
每一层 MACE 都承担 \(l=4\) 的成本。

完整流水线支持数据断点续传与训练断点续训：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_nmc_full_benchmark.ps1
```

它依次完成官方 NMC 下载、probe 数据准备、L1 density 训练、500 个测试结构的
完整网格 NMAE，以及与官方 ChargE3Net NMC 结果的汇总。主配置为
`configs/nmc_full_l2_to_l4_mace_pbc.yaml`，训练输出保存到独立的
`runs/nmc_full_l2_to_l4_mace_pbc`，避免覆盖修复前的实验。

2026-10-24 周期边界修复：原子图与密度重构现在都累加 cutoff 内的全部周期像，
仅原子图的零平移自身边被排除；斜晶胞用倒晶格确定枚举范围。
旧 `nmc_full_l2_to_l4_mace` checkpoint 的误差属于旧算子，不能代表修正后的结果。
新 checkpoint 带 `periodic_geometry_version`，周期训练会拒绝直接续训旧历史，
评估旧 checkpoint 会提示算子变化。密度标签和 probe 坐标可以复用；若使用离线拟合的
系数标签，则需要用新重构器重新拟合。详细验证及精度路线见
[周期边界修复与精度分析](reports/pbc_fix_accuracy_review_2026-10-24.md)。

2026-10-25 本地 8 个 NMC 结构完成修复后的完整网格 oracle 比较：当前原子基、
原点正则径向基、周期键中心基，以及近似等系数预算的增宽原子基。
结果、求解器精度检查、旧 GNN 算子对照及下一轮训练方案见
[本地 NMC oracle 诊断](reports/nmc_oracle_review_2026-10-25.md)。
这些是逐结构直接拟合的可行误差，不是新 GNN 的泛化成绩或认证的表示误差下限。

紧凑解码器支持 `regular_radial`、`radial_whitening` 和 `radial_minimum`。
例如正则 R16 的基配置为：

```yaml
basis:
  num_radial: 16
  lmax: 4
  cutoff: 3.0
  pbc: true
  regular_radial: true
  radial_whitening: true
  radial_minimum: 0.09375
```

正则基采用 solid harmonics，原点处的值和导数不依赖方向归一化的 epsilon。
径向白化是保留全部方向的可逆 SVD 变换，按 `r² dr` 度量构造，
在 float64 中计算特征与密度求和。输出仍采用原有的 radial-major 系数布局，
但白化后的系数不能与原始 Gaussian 系数直接互用。checkpoint 保存基配置和
实际白化矩阵，续训与评估会检查坐标约定；旧配置默认保持原函数与系数含义。

8 个本地 NMC 结构的过拟合实验使用以下独立入口；它从原始缓存读取完整网格，
每次更新动态均匀采样，每次评估遍历全部网格，不依赖完整 NMC 数据目录：

```powershell
& $python -m scripts.train_local_nmc_overfit --variant r12 --output-dir runs/nmc_overfit_r12_seed42 --steps 4000 --eval-every 500 --probes 8192 --independent-final
& $python -m scripts.train_local_nmc_overfit --variant r16 --output-dir runs/nmc_overfit_r16_seed42 --steps 4000 --eval-every 500 --probes 8192 --independent-final
```

两组都使用白化坐标；R12 保留当前函数空间，R16 使用正则函数。训练不使用 oracle
系数监督，也不读取旧模型权重。已有实验必须显式 `--resume`，设置不同则需要新目录。
固定网格训练缓存只保存周期几何、只对系数求导；需要原子/网格/晶胞梯度时使用普通解码器。

如需复现本轮相同的后续精修，可在两组 Adam 完成后分别运行：

```powershell
& $python -m scripts.polish_local_nmc_overfit --parent runs/nmc_overfit_r12_seed42 --output-dir runs/nmc_overfit_r12_seed42_polish --rounds 3 --iterations 80 --probes 16384
& $python -m scripts.polish_local_nmc_overfit --parent runs/nmc_overfit_r16_seed42 --output-dir runs/nmc_overfit_r16_seed42_polish --rounds 3 --iterations 80 --probes 16384
```

精修每轮固定一组新的均匀网格点，全部 8 个结构共同产生 L-BFGS 梯度；
模型选择和最终误差仍使用完整网格。精修保留父实验的最佳模型作为候选，输出保存到新目录。

2026-10-26 完成上述两组实验：完整网格宏平均 NMAE 分别为 R12 的 2.53577%
和 R16 的 2.43859%，仍为对应 oracle 的 9.51 倍和 16.79 倍。
[完整报告、逐样本比较与下一步建议](reports/nmc_overfit_2026-10-26/report.md)
记录了训练控制、独立解码器复核及数值测试。这里只进行了 8 个本地结构的过拟合诊断，
没有启动完整训练。

```powershell
& $python -m scripts.summarize_local_nmc_overfit --r12 runs/nmc_overfit_r12_seed42 --r16 runs/nmc_overfit_r16_seed42 --polish-r12 runs/nmc_overfit_r12_seed42_polish --polish-r16 runs/nmc_overfit_r16_seed42_polish --output-dir reports/nmc_overfit_2026-10-26 --report-date 2026-10-26
```

## NMC 空间留出：官方等变 DeepDFT 对照

2026-10-27 汇总了已完成的单结构与 8 结构配对实验。每个结构的原始网格固定分为
80% 训练、10% 验证和 10% 测试；两种模型均从头训练，并只用验证点选择权重。
最终测试点 NMAE：单结构为 DeepDFT **0.25609%**、R16 **2.13180%**；
8 结构宏平均为 DeepDFT **0.21409%**、R16 **2.06069%**。
这是已知结构内未参与训练的空间网格点预测，尚未检验新结构泛化。

[完整报告、逐结构结果与验证曲线](reports/nmc_probe_holdout_2026-10-27/report.md)
记录了官方代码版本、数据划分、训练预算、采样审计与实验限制。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_nmc_probe_holdout.ps1
& $python -m scripts.summarize_nmc_probe_holdout
```

## NMC 表示与读出瓶颈诊断

2026-10-28 完成了沿用相同空间划分的 8 结构诊断。仅用训练点拟合的 R16 自由系数，
留出测试点宏平均 NMAE 为 **0.16365%**。同骨干架构、相同初始化和 16,000 步预算下，
系数读出为 **2.06069%**，查询点读出为 **0.74323%**；另加 4,000 步、每步 4,096 点
精修后为 **0.71268%**，尚未达到约 0.3% 的目标。官方 DeepDFT 的同划分参照为 **0.21409%**。

自由系数结果证明 R16 的表示能力有余量，但它是逐结构拟合，不是新结构预测成绩。
读出对照支持优先解决系数学习与密度重构的瓶颈；精修模型的训练误差 **0.71353%**
与测试误差接近，当前主要表现为拟合不足。详见
[完整报告、逐结构结果与验证曲线](reports/nmc_bottleneck_2026-10-27/report.md)。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_nmc_bottleneck.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_nmc_bottleneck_refine.ps1
& $python -m scripts.finalize_nmc_bottleneck
```

最终评价前固定验证选择和检查点哈希；额外精修与原始等预算对照分别报告。

## 验证

空间划分不变的后续瓶颈诊断使用
`powershell -ExecutionPolicy Bypass -File scripts/run_nmc_bottleneck.ps1`。
它先仅用训练点拟合 R16 自由系数，再训练与原系数模型共享骨干架构、初始化和
近似参数预算的查询点读出；测试评价单独执行。控制条件见
[预定实验协议](reports/nmc_bottleneck_protocol_2026-10-27.md)。

```powershell
& $python -m pytest -q
```

测试覆盖全局拟合、数据旋转增强、旋转等变性，以及斜晶胞完整周期像、单原子自身像、
三种模型的原胞/超胞密度一致性、跨边界连续性、padding、非折叠坐标和可微重构。

数据来源：

- [QM9 Charge Densities and Energies Calculated with VASP](https://data.dtu.dk/articles/dataset/QM9_Charge_Densities_and_Energies_Calculated_with_VASP/16794500)
- [ChargE3Net paper](https://www.nature.com/articles/s41524-024-01343-1)

## 官方预训练 DeepDFT：完整 NMC occupation 实验

2026-11-02：冻结官方 NMC 密度模型，在 1,450 / 50 / 500 个结构的正式划分上
训练和评估等变 PAW occupation 头。两个种子平均测试系数 MAE 为 **0.002202**，
相对 L1 为 **0.8163%**，相对训练元素均值基线的 MAE 降低 **94.19%**。
50 个固定测试结构上，密度单独推理为 **0.5388 s/结构**，联合预测为
**0.5463 s/结构**；完整网格相对原网络输出的最大归一化 L1 差为 **1.315e-7**。
这些是预热后推理时间，排除文件处理及模型/kernel 初始化。

有效标签仅覆盖非磁性 Li/O/Mn/Co/Ni 的 **L≤2**；L=3、4 均为缺失信息的
零槽位，不能当作完整 PAW 初始化。尚未验证 VASP SCF、总 DFT 时间或能量/力。
Eden 冻结头的迁移对照单独排除了 3 个预训练重叠测试结构，且预训练预算不同。

[完整报告](reports/public_paw_2026-11-02/report.md) 与
[固定协议](reports/public_paw_2026-11-02/protocol.md) 包含逐角阶/元素结果、
原始结果位置、计时边界和复现说明。

```python
from scripts.public_paw_predictor import PublicPAWPredictor

predictor = PublicPAWPredictor(seed=42)
result = predictor.predict(geometry)  # CUDA positions/numbers/cell/probes + metadata
# result: density, augmentation, augmentation_mask, supported_lmax=2
```

也可仅从 POSCAR 输出预测 NPZ：

```powershell
python -m scripts.predict_public_paw --structure POSCAR --grid 56 56 140 --output prediction.npz
```

NPZ 保存 canonical/原生投影子顺序及有效掩码，不是完整 VASP 重启 CHGCAR。
原始模型与数据保存在 `runs/public_paw_2026-11-02`；顺序复现入口为
`python -m scripts.run_public_paw_experiment`，保留已完成结果并跳过完成的阶段。
