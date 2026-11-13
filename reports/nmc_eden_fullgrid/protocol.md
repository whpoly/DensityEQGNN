# NMC 全网格 EdenGNN 与 DeepDFT 密度基准

## 原论文的 NMC 设置

补充讨论第 1 节给出了 NMC 专用参数。原子邻域为 4 Å，密度展开半径为 3 Å，每个角动量阶有 3 个可学习的径向通道，最高角动量为 6。模型共享编码器，同时输出赝电荷密度和 PAW augmentation occupancies。每步用一个结构的原始 FFT 全网格，先在各原子的局域范围展开密度修正，再周期累加并加上该结构的 SACD。训练的密度项取全网格绝对误差的均值。计算分块只影响一次处理的原子与格点对数。[补充讨论 1、4、5、6 及补充表 2](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs42256-026-01326-5/MediaObjects/42256_2026_1326_MOESM1_ESM.pdf) · [官方密度层](https://github.com/rubenlee11/EdenGNN/blob/f37c9a1864803bf04d75fa91c932474040220021/src/edengnn/model/model.py)

作者用指定的赝势重新计算了原始 NMC 几何对应的自洽密度和 PAW 标签，并用 VASP 的 `ICHARG=12`、`NELM=0`、`ISPIN=1` 生成原子叠加密度。在这组标签上，作者报告 500 个测试结构的相对密度误差为 **0.096%**，occupation MAE 为 **0.00085**。补充材料同段的 0.061% 指 ChargE3Net。公开数据归档的 NMC 目录提供五种赝势的版本和哈希。[论文及补充材料](https://www.nature.com/articles/s42256-026-01326-5) · [作者数据归档](https://zenodo.org/records/21736087)

## 本地配对实验

本地使用 DeepDFT 原始 NMC 的收敛 CHGCAR，按官方 1,450 / 50 / 500 划分。每个结构含 439,040 个格点。训练、验证和测试覆盖各自结构的全部格点。PAW 监督只使用原文件有信息的 L≤2 系数，并按各元素 POTCAR 的投影子顺序映射到作者的输出表示。五种 POTCAR 与作者 NMC 哈希逐一匹配。[数据与基线核对](../../runs/nmc_eden_fullgrid/data_check.json) · [原始 NMC 数据](https://doi.org/10.11583/DTU.16837721)

从随机种子 42 开始，密度与有效 PAW 系数的平均绝对误差分别乘 0.9 和 0.1。模型沿用作者发布的 MD 配置并修改上述两项 NMC 参数。按补充表 2 的 MD 设置采用 batch 1、AdamW、初始学习率 0.005、连续 5 轮验证无改善后按 `ReduceLROnPlateau` 规则减半。优化器沿用作者代码的 betas (0.99, 0.999)、AMSGrad、零权重衰减。每轮验证 50 个结构，选模分数为 0.9 倍按电子数归一化的密度误差加 0.1 倍有效 occupation MAE。采用代码默认的早停等待 30 个验证轮次和最多 1,000,000 步。实际更新预算和停止原因写入 `selection.json`。补充表没有单列 NMC 的优化器设置，这里明确沿用 MD 行。[论文方法 §4.2](https://arxiv.org/html/2510.00788v2#S4.SS2) · [作者训练器](https://github.com/rubenlee11/EdenGNN/blob/f37c9a1864803bf04d75fa91c932474040220021/src/edengnn/cli.py)

最终比较使用同一组 500 个原始测试结构，对每个晶胞计算 `sum(abs(pred-rho))/sum(abs(rho))` 后等权平均。官方 eqDeepDFT 权重在这批数据上实测 **0.0609834%**。EdenGNN 从验证集选出单个检查点后才评分，记录两模型逐结构误差、宏平均差及配对 bootstrap 区间。比较是否达到 DeepDFT，以本地同标签的测试均值为准。[本地 DeepDFT 对照](../public_benchmark_2026-10-30/nmc_report.md)

## 实现与已完成的验证

- 训练直接调用作者的 `DensityDataset`、`EfficientDensity`、Fortran 网格枚举及周期重构。已核对 Zenodo 存档和所用 GitHub 版本中 Encoder、DensityLayer、AugmentationLayer 三个类的语法树。完整模型包装器的差异为无损失权重的总电荷日志项。NMC 训练样本 0 和 2 的全网格密度、原子次序、有效 PAW 数值和掩码与作者 Fortran 读取结果逐项一致，覆盖全部五种元素。[原始标签核对](../../runs/nmc_eden_fullgrid/nmc_label_audit.json)
- 用作者发布的 Al₂O₃ 零步及自洽 CHGCAR 做了两次真实参数更新，每次使用全部 501,760 个 FFT 格点。编码器、密度头和 PAW 头都有有限非零梯度。改变计算分块后的两次全网格输出相对 L1 差为 2.62×10⁻⁷。[全网格反传记录](../../runs/nmc_eden_fullgrid/native_fullgrid_check.json)
- 在缺少 NMC 原生零步输出时，可另外采用从 POTCAR 原子电荷表重构的 SACD。重构使用 POTCAR 的 local-part 倒空间网格和原生 FFT 的 Nyquist 平面约定。对作者发布且赝势哈希匹配的 13 个外部样例，重构与原生 SACD 的全网格相对 L1 差为 **0.0000151% 至 0.001005%**。另一个样例的 Zr 赝势哈希不匹配，未计入。没有 NMC 原生 SACD 时，这些数字只衡量外部样例上的输入近似。[逐样例差异](../../runs/nmc_eden_fullgrid/sacd_native_examples.json)

## 运行

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
