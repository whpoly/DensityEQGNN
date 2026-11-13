# 冻结非线性特征的本地 NMC 线性读出校准

2026-10-26。两组主校准完成，仍只使用本地 8 个结构。没有启动完整 NMC 训练，也没有训练本项目自己的查询点（probe）模型。

## 结果

下表是 8 个结构各自完整网格 NMAE 的宏平均。每结构 439,040 点，共 3,512,320 点；全部数字为百分数。

| 表示 | 校准前 | 校准后 | 相对改善 | 同表示 oracle | 校准后/oracle |
|---|---:|---:|---:|---:|---:|
| R12＋白化 | 2.53577% | **1.73619%** | 31.53% | 0.26675% | 6.51× |
| 正则 R16＋白化 | 2.43859% | **1.69015%** | 30.69% | 0.14526% | 11.64× |

R16 在全部 8 个结构上更低，但相对 R12 的宏平均改善仅约 2.65%。仅调整已有线性输出参数就能降低约三成误差，说明之前的联合训练没有充分优化这些参数。剩余差距仍大；本实验不能把它全部归因于骨干容量。

![完整网格比较](<C:/Users/User/Desktop/density EQGNN/reports/nmc_linear_calibration_2026-10-26/comparison.png>)

| 样本 | R12 校准后 | R16 校准后 |
|---|---:|---:|
| 0 | 1.97583% | 1.90478% |
| 1 | 1.77736% | 1.74464% |
| 2 | 1.67929% | 1.64871% |
| 3 | 1.73952% | 1.66261% |
| 4 | 1.51074% | 1.49597% |
| 5 | 1.74465% | 1.70606% |
| 6 | 1.72992% | 1.69864% |
| 7 | 1.73217% | 1.65977% |

## 实验实际改了什么

起点为上一轮两组从头训练、再经 L-BFGS 精修的最佳模型。固定原子嵌入、所有消息传递层，以及读出径向 MLP 前两层。只校准 `readout.direct.weight`、`readout.radial_weights.4.weight/bias`、出现元素的 `learned_scalar_baseline.weight`。没有添加逐原子或逐结构独立参数，没有使用 oracle 系数监督。

把这些已有参数到全部原子系数的线性映射显式构造出来，在每个角动量阶内做 SVD。当前主结果保留相对奇异值大于 `1e-8` 的方向，优化这些方向上的增量，并完整保留父模型输出。因此这不是对所有线性参数方向的无截断最优求解，也不同于此前保留所有方向的径向白化。

冻结的非线性运算仍为原来的 float32。线性输出参数及其计算采用 float64，保持原非线性特征；checkpoint 记录 `linear_readout_dtype`，主要评估入口按该字段恢复精度，防止静默转回 float32。

每轮使用全部最近原子距离小于 0.35 Å 的网格点（各结构约 8,289–9,887 点），再从其余区域均匀无放回抽取 32,768 点，按入样概率加权，使 L1 目标对应各结构完整网格平均。两组数据 SHA256、求解设置及每轮采样索引哈希已逐一匹配。

先进行 400 步 CGLS 加权最小二乘初始化，再进行三轮、每轮最多 300 步平滑 L1 L-BFGS。每轮更换外层采样点，每 100 步完整网格评价并选模。两组最佳点均为第 3 轮第 100 步。继续降低采样损失没有继续改善完整网格，因此选取最佳点，不使用最后一步。

早期 R12 纯均匀采样预实验保存在 `runs/nmc_linear_r12_seed42`，在发现采样误差与完整网格误差脱节后终止；不与本表的配对主结果混合。

## 验证和结论边界

- 所有冻结参数的数值哈希保持不变。
- 导出实际网络与稳定线性映射的最大系数差：R12 为 `2.74e-10`，R16 为 `9.86e-10`。
- 最终重新计算原子图，并用无缓存的生产解码器遍历完整网格；缓存与直接密度重构最大差均为 `5.33e-15`。没有电荷重归一化或负密度截断。
- 全套 **43 项测试通过**，新增测试覆盖线性映射、参数回写/恢复、伴随与已知解求解、分层采样权重。
- R12 最佳点的加权采样 NMAE 为 1.54335%，完整网格为 1.73619%；R16 分别为 1.43203% 和 1.69015%。网格覆盖差距仍未消失。
- 求解尚无全局最优性证书，最佳点平滑损失梯度无穷范数分别约 `6.39e-5`、`7.36e-5`；不能把结果叫作冻结特征的精度极限。
- 高阶读出权重很大，l=4 参数 RMS 约为 R12 的 `2.03e6`、R16 的 `1.94e6`，反映了特征病态。权重数值不能与此前原子密度系数 RMS 混淆。
- 更宽松 SVD 阈值的配对拟合、float32 导出的密度稳定性审计尚未完成；已准备的宽阈值特征谱不等于已经完成了训练。当前结果是双精度读出的训练集诊断，不是稳定部署或泛化成绩。

## 对下一步的建议

当前证据支持同时保留两个待验证方向：改善系数输出的优化与参数化；对比更丰富的等变系数读出和直接预测密度的查询点读出。应先用同一骨干、同一 8 个结构和相同网格评价做受控对照。查询点读出改变了密度函数空间，R16 oracle 可作参照，不能当成查询点模型的精度上限。

先完成数值稳定性诊断，再据此判断哪些变化适合进入下一轮训练；本报告没有宣布骨干容量不足，也没有启动新架构或完整训练。

## 文件与复现

- [R12 结果](<C:/Users/User/Desktop/density EQGNN/runs/nmc_linear_stratified_r12_seed42/result.json>)；[checkpoint](<C:/Users/User/Desktop/density EQGNN/runs/nmc_linear_stratified_r12_seed42/best.pt>)。
- [R16 结果](<C:/Users/User/Desktop/density EQGNN/runs/nmc_linear_stratified_r16_seed42/result.json>)；[checkpoint](<C:/Users/User/Desktop/density EQGNN/runs/nmc_linear_stratified_r16_seed42/best.pt>)。
- [机器可读汇总](<C:/Users/User/Desktop/density EQGNN/reports/nmc_linear_calibration_2026-10-26/summary.json>)；[源码与实验哈希](<C:/Users/User/Desktop/density EQGNN/reports/nmc_linear_calibration_2026-10-26/manifest.json>)。

```powershell
& 'C:/Users/User/.conda/envs/kan-cgcnn-cuda/python.exe' -m scripts.calibrate_local_nmc_readout --parent runs/nmc_overfit_r12_seed42_polish --output-dir runs/nmc_linear_stratified_r12_seed42
& 'C:/Users/User/.conda/envs/kan-cgcnn-cuda/python.exe' -m scripts.calibrate_local_nmc_readout --parent runs/nmc_overfit_r16_seed42_polish --output-dir runs/nmc_linear_stratified_r16_seed42
```

独立复现使用新输出目录；已有实验可使用相同设置加 `--resume`，按已保存的完整轮恢复。数据与所有配置位于各实验目录，默认参数与主实验相同。
