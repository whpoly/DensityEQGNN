# NMC 探针 EdenGNN 与 DeepDFT+PAW 基准

## 当前比较

两种模型使用种子 42，分别从随机初值开始，在同一批 1,450 个训练结构上训练。EdenGNN 至少训练 1,000,000 步，此后满足下述早停规则时结束，上限为 4,965,000 步；DeepDFT+PAW 已完成 4,965,000 步。EdenGNN 使用作者的 NMC 编码器、两个输出头和 POTCAR 重构 SACD，所有更新均使用探针。[配置](../../runs/nmc_eden_fullgrid/probe_adamw_l1_seed42/settings.json) · [首次更新的标签与损失](../../runs/nmc_eden_fullgrid/probe_adamw_l1_seed42/first_update.json)

每步复用 DeepDFT 的训练结构顺序和随机探针序列，每个结构均匀有放回抽取 1,000 个原始格点，累积两个结构的平均梯度。两边的 PAW 监督均覆盖全部有效系数。每 5,000 步在同一批 50 个验证结构上比较，密度取每结构相同的 5,000 个固定探针，PAW 取全部有效系数。[训练与早停实现](../../scripts/nmc_eden_early_stopping.py) · [等步验证记录](../../runs/nmc_eden_fullgrid/probe_adamw_l1_seed42/paired_validation.json)

EdenGNN 的每结构损失为 0.9 倍探针密度 MAE 加 0.1 倍有效 PAW 占据 MAE，直接以原始物理数值计算。使用 AdamW，初始学习率 0.005，betas 为 (0.99, 0.999)，eps 为 1e-8，启用 AMSGrad，权重衰减为 0。每次验证用固定探针估计按电子数归一化的密度误差，乘以 0.9 后加 0.1 倍有效 PAW MAE，按结构等权平均。此分数驱动 ReduceLROnPlateau，patience 为 5，衰减因子为 0.5，最低学习率为 0.00002。DeepDFT+PAW 沿用其 Adam、初始学习率 0.0001 及密度和标准化 PAW 的加权 MSE。先前采用 Adam 和 MSE 的纯探针 EdenGNN 已归档，保留第 18,000 步断点及原训练代码。[上一轮归档记录](../../runs/nmc_eden_fullgrid/probe_matched_deep_seed42/superseded.json)

EdenGNN 在第 1,000,000 步之前不因验证指标改善小而停止。从第 1,000,000 步验证起，在每个验证步 t 比较相邻两段各 50,000 步的验证最佳值：前段为 (t−100,000, t−50,000]，后段为 (t−50,000, t]，每段各 10 个验证点，分别取密度 NMAE 和 PAW 相对 L1 的最小值。每项相对改善为 1−后段最佳值/前段最佳值；两项均严格小于 0.5%，且最近一次降学习率后已完成至少 50,000 步，才触发早停。任何一项仍有至少 0.5% 的改善就继续；再次降学习率会重置观察期。这里的 0.5% 是相对改善，不是 0.5 个百分点。该规则只读取验证集。[早停规则](../../scripts/nmc_eden_stop_policy.py) · [当前判断](../../runs/nmc_eden_fullgrid/probe_adamw_l1_seed42/early_stopping_status.json)

训练停止后，按各自最低验证密度 MAE 冻结一个检查点，用同一套权重评价两个输出，在同一批官方 500 个测试结构上计算全格点密度 NMAE、有效 PAW 占据相对 L1 和 MAE。两项相对误差按结构等权平均，以百分点报告 EdenGNN 减 DeepDFT+PAW 的配对差和 10,000 次逐结构 bootstrap 的 95% 区间。主结果写入 `reports/nmc_eden_fullgrid/paired_early_stop_seed42.json`，分别记录两边实际停止步、停止原因及选中步。最终测试不强制相同步数；相同步数的比较保留在验证记录中。DeepDFT+PAW 已完成的 500 结构测试采用第 4,900,000 步的验证最佳权重。[停止后测试实现](../../scripts/nmc_eden_early_stop_test.py)

当前训练命令为 `python -X utf8 -u -m scripts.nmc_eden_early_stopping --adopt-early-stopping`，使用原 CUDA Python。2026-11-18 在第 277,000 步的已保存断点加入停止规则及至少百万步的限制；迁移逐项保留模型、优化器、学习率调度器、验证历史、最佳权重和 CPU/CUDA 随机状态。后续运行自动续训，早停或达到上限后自动作配对测试。[启动记录](../../runs/nmc_eden_fullgrid/probe_adamw_l1_seed42/launch.json) · [规则迁移记录](../../runs/nmc_eden_fullgrid/probe_adamw_l1_seed42/protocol_changes/early_stopping_v2/migration.json) · [状态一致性检查](../../runs/nmc_eden_fullgrid/probe_adamw_l1_seed42/protocol_changes/early_stopping_v2/state_preservation_audit.json)

2026-11-18 用户另行要求立即评估当前模型在同一批 500 个测试结构上的表现。此次中期测试复制第 287,000 步断点，依据其中截至第 285,000 步的 57 次验证，固定最低验证密度 MAE 对应的第 265,000 步权重；两个输出均使用该检查点。选模与冻结先于测试，测试结果不用于更新训练参数、模型选择或停止规则，训练继续。独立测试目录为 `runs/nmc_eden_fullgrid/probe_adamw_l1_seed42/interim_test_20261118`，结果写入 `reports/nmc_eden_fullgrid/interim_test_20261118.json`，不替代训练停止后的主报告。使用同标签 DeepDFT+PAW 的第 4,900,000 步验证最佳权重作配对对照，明确两边训练步数不同。另报告按作者电子数归一化的全网格密度 L1 和有效 PAW MAE，与论文的 0.096% 和 0.00085 作数值参考；论文重算了 DFT 标签，不能将此比较称为同标签复现。[中期测试实现](../../scripts/nmc_eden_interim_test.py) · [冻结选择](../../runs/nmc_eden_fullgrid/probe_adamw_l1_seed42/interim_test_20261118/selection.json)

此次中期测试已完成全部 500 个结构，覆盖 219,520,000 个密度格点和 937,485 个有效 PAW 系数。EdenGNN 的密度 NMAE 为 0.09468733%，PAW 占据相对 L1 为 0.39508977%，有效 PAW MAE 为 0.001066518；同标签 DeepDFT+PAW 分别为 0.06064723%、0.23298204% 和 0.000627276。按作者电子数归一化的 EdenGNN 密度相对 L1 为 0.09468733%，与本次 NMAE 在上述显示精度下一致。EdenGNN 减 DeepDFT 的密度差为 0.03404010 个百分点，95% 配对 bootstrap 区间为 [0.03315180, 0.03498576]；PAW 相对 L1 差为 0.16210772 个百分点，区间为 [0.15567186, 0.16841069]。本次快照的两项误差均未达到已完成训练的 DeepDFT 对照；主训练继续执行至少百万步的约定。[中期测试结果及 500 个配对记录](interim_test_20261118.json) · [完整性核对](../../runs/nmc_eden_fullgrid/probe_adamw_l1_seed42/interim_test_20261118/final_audit.json)

## 原论文的 NMC 设置

补充讨论第 1 节给出了 NMC 专用参数。原子邻域为 4 Å，密度展开半径为 3 Å，每个角动量阶有 3 个可学习的径向通道，最高角动量为 6。模型共享编码器，同时输出赝电荷密度和 PAW augmentation occupancies。每步用一个结构的原始 FFT 全网格，先在各原子的局域范围展开密度修正，再周期累加并加上该结构的 SACD。训练的密度项取全网格绝对误差的均值。计算分块只影响一次处理的原子与格点对数。[补充讨论 1、4、5、6 及补充表 2](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-026-01326-5/MediaObjects/42256_2026_1326_MOESM1_ESM.pdf) · [官方密度层](https://github.com/rubenlee11/EdenGNN/blob/f37c9a1864803bf04d75fa91c932474040220021/src/edengnn/model/model.py)

作者用指定的赝势重新计算了原始 NMC 几何对应的自洽密度和 PAW 标签，并用 VASP 的 `ICHARG=12`、`NELM=0`、`ISPIN=1` 生成原子叠加密度。在这组标签上，作者报告 500 个测试结构的相对密度误差为 **0.096%**，occupation MAE 为 **0.00085**。补充材料同段的 0.061% 指 ChargE3Net。公开数据归档的 NMC 目录提供五种赝势的版本和哈希。[论文及补充材料](https://www.nature.com/articles/s42256-026-01326-5) · [作者数据归档](https://zenodo.org/records/21736087)

## 已归档的全网格配对实验

本地使用 DeepDFT 原始 NMC 的收敛 CHGCAR，按官方 1,450 / 50 / 500 划分。每个结构含 439,040 个格点。EdenGNN 的训练、验证和测试覆盖各自结构的全部格点。PAW 监督只使用原文件有信息的 L≤2 系数，并按各元素 POTCAR 的投影子顺序映射到作者的输出表示。五种 POTCAR 与作者 NMC 哈希逐一匹配。[数据与基线核对](../../runs/nmc_eden_fullgrid/data_check.json) · [原始 NMC 数据](https://doi.org/10.11583/DTU.16837721)

从随机种子 42 开始，密度与有效 PAW 系数的平均绝对误差分别乘 0.9 和 0.1。模型沿用作者发布的 MD 配置并修改上述两项 NMC 参数。按补充表 2 的 MD 设置采用 batch 1、AdamW、初始学习率 0.005、连续 5 轮验证无改善后按 `ReduceLROnPlateau` 规则减半。优化器沿用作者代码的 betas (0.99, 0.999)、AMSGrad、零权重衰减。每轮验证 50 个结构，选模分数为 0.9 倍按电子数归一化的密度误差加 0.1 倍有效 occupation MAE。采用代码默认的早停等待 30 个验证轮次和最多 1,000,000 步。实际更新预算和停止原因写入 `selection.json`。补充表没有单列 NMC 的优化器设置，这里明确沿用 MD 行。[论文方法 §4.2](https://arxiv.org/html/2510.00788v2#S4.SS2) · [作者训练器](https://github.com/rubenlee11/EdenGNN/blob/f37c9a1864803bf04d75fa91c932474040220021/src/edengnn/cli.py)

双任务对照使用独立训练的 DeepDFT+PAW（种子 42）。它每次更新累积两个结构的梯度，每个结构均匀采样 1,000 个密度格点，并使用其全部有效 PAW 系数。训练预算为 4,965,000 步，每 5,000 步验证一次，以验证密度 MAE 最低的一个检查点评估两个输出。EdenGNN 沿用上面的联合验证分数选取一个检查点。两边分别记录实际停止步、停止原因和选中步。[DeepDFT+PAW 单模型训练协议](../joint_paper_steps/nmc_single_run.md)

两套冻结权重在同一批 500 个测试结构上分别预测完整密度和有效 PAW 系数。对每个结构，两项指标分别用预测和参考的绝对误差之和除以参考绝对值之和，再以结构为单位等权平均，乘 100 报百分数。汇总程序核对官方结构编号、检查点、网格大小、两边参考值的绝对值之和及有效 PAW 系数数目，同时复核各自已有的测试汇总。两项差值均为 EdenGNN 减 DeepDFT+PAW，以百分点报告，使用相同的 10,000 次逐结构配对 bootstrap 抽样给出各自的 95% 区间。每项以 EdenGNN 的平均误差小于或等于对照作为达到标准，两项分别判断。官方单任务 eqDeepDFT 权重在这些测试结构上的密度 NMAE 为 **0.0609834%**，另列作密度参考。[官方密度参考](../public_benchmark_2026-10-30/nmc_report.md) · [逐结构配对比较程序](../../scripts/compare_nmc_joint_test.py)

## 已归档的全网格热启动探针对照

从原全网格第 10,350 步的同一断点继续，保留模型、AdamW 和学习率调度器状态、POTCAR 重构先验与种子 42。两条路径按原顺序各再看 1,450 个训练结构。全网格路径每步用一个结构，取第 11,800 步权重。探针路径每两个结构更新一次，每个结构均匀有放回抽取 1,000 个 FFT 格点，每次更新两个结构的损失各占一半，PAW 项使用全部有效系数；取额外 725 次更新后的第 11,075 步权重。探针路径按累计结构数对齐原训练轮次的学习率调度。此次比较同时检验探针数与 batch size 的改动。[共同起点](../../runs/nmc_eden_fullgrid/probe_pair_trial_seed42/origin.json) · [训练及验证代码](../../scripts/nmc_eden_probe_trial.py)

在同一批 50 个验证结构上，两套权重都预测每结构全部 439,040 个格点和全部有效 PAW 系数。分别计算逐结构等权平均的密度 NMAE 和 PAW 占据相对 L1。用户指定的界线是两项均满足“探针组均值 ≤ 1.01 × 全网格组均值”。配对 bootstrap 95% 区间作为不确定度另报。判定只使用验证集：两项均通过则从试验断点继续探针训练，否则继续原全网格训练。最终报告注明采用的方法和各阶段步数，再与 DeepDFT+PAW 在官方 500 个测试结构上配对比较。单轮通过支持在当前训练阶段采用探针，最终精度以收敛后的测试结果为准。

## 实现与已完成的验证

- 训练直接调用作者的 `DensityDataset`、`EfficientDensity`、Fortran 网格枚举及周期重构。已核对 Zenodo 存档和所用 GitHub 版本中 Encoder、DensityLayer、AugmentationLayer 三个类的语法树。完整模型包装器的差异为无损失权重的总电荷日志项。NMC 训练样本 0 和 2 的全网格密度、原子次序、有效 PAW 数值和掩码与作者 Fortran 读取结果逐项一致，覆盖全部五种元素。[原始标签核对](../../runs/nmc_eden_fullgrid/nmc_label_audit.json)
- 用作者发布的 Al₂O₃ 零步及自洽 CHGCAR 做了两次真实参数更新，每次使用全部 501,760 个 FFT 格点。编码器、密度头和 PAW 头都有有限非零梯度。改变计算分块后的两次全网格输出相对 L1 差为 2.62×10⁻⁷。[全网格反传记录](../../runs/nmc_eden_fullgrid/native_fullgrid_check.json)
- 在缺少 NMC 原生零步输出时，可另外采用从 POTCAR 原子电荷表重构的 SACD。重构使用 POTCAR 的 local-part 倒空间网格和原生 FFT 的 Nyquist 平面约定。对作者发布且赝势哈希匹配的 13 个外部样例，重构与原生 SACD 的全网格相对 L1 差为 **0.0000151% 至 0.001005%**。另一个样例的 Zr 赝势哈希不匹配，未计入。没有 NMC 原生 SACD 时，这些数字只衡量外部样例上的输入近似。[逐样例差异](../../runs/nmc_eden_fullgrid/sacd_native_examples.json)

## 已归档全网格实验的运行方式

从仓库根目录运行。原生模式从给定目录下的 `<官方结构编号>/CHGCAR` 读取 VASP 零步输出。两种 SACD 来源的检查点和测试结果使用独立目录，重复运行同一命令会续训。

```powershell
python -m scripts.verify_nmc_eden_fullgrid
python -m scripts.nmc_eden_fullgrid --native-sacd C:\path\to\nmc_sacd
```

使用 POTCAR 重构先验的独立对照命令如下。

```powershell
python -m scripts.nmc_eden_sacd
python -m scripts.nmc_eden_fullgrid --potcar-sacd
```

本机使用 `kan-cgcnn-cuda` Python 环境和 `runs/nmc_eden_fullgrid/EdenGNN` 下固定的官方代码。两个 Fortran 扩展已经在本机编译。训练保存断点与验证历史，完成后在运行目录写入 `benchmark.json` 和逐结构测试记录。

当前探针实验的主结果和验证选模结果见本页“当前比较”。
