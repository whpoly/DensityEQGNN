# NMC：compact coefficient model 与 ChargE3Net benchmark

## 口径

- NMC 官方划分：1,450 train / 50 validation / 500 test。
- 完整 CHGCAR 网格：`56 x 56 x 140 = 439,040` probes/结构。
- 指标：

  \[
  \mathrm{NMAE}=
  \frac{\sum_g|\hat\rho_g-\rho_g|}
       {\sum_g|\rho_g|}.
  \]

- compact coefficient model 使用停止训练时的最佳 checkpoint（epoch 176）。
- ChargE3Net 使用官方仓库 commit
  `18290c63c0307792ab6da5ff98ee535cdb7376d6` 和官方
  `charge3net_nmc.pt` checkpoint。

## 结果

| 模型 | 评估 | NMAE | 参数量 |
|---|---|---:|---:|
| compact coefficient model | 500 test，完整网格 | **1.1474%** | 477,148 |
| ChargE3Net | 100 test，每结构固定随机 1,000 probes | **0.06048%** | 1,909,389 |
| ChargE3Net | sample 1，完整网格 | **0.04708%** | 1,909,389 |
| ChargE3Net（论文） | 500 test，完整网格 | **0.060 ± 0.001%** | 1,909,389 |

100 结构采样结果和论文完整测试结果一致，sample 1 的完整网格结果
进一步确认了 checkpoint、密度单位、probe 坐标和 NMAE 实现均对齐。

### 同一结构的直接比较

sample 1（`Co10Li5Ni2O24`）：

| 项目 | compact coefficient model | ChargE3Net |
|---|---:|---:|
| full-grid NMAE | 0.9845% | 0.04708% |
| 参数量 | 477,148 | 1,909,389 |
| 模型与重建时间 | 0.922 s（首结构含 warm-up） | 577.7 s |

这个结构上 ChargE3Net 的误差低约 20.9 倍，但推理慢约 626 倍。
ChargE3Net 为该结构构造了 13,162,164 条 atom-to-probe 周期边，并将
439,040 probes 分成约 176 个批次逐批执行 equivariant probe network。

按实测速度串行完成 ChargE3Net 的 500 个完整网格约需 80 小时。

## 当前 coefficient basis 的表示上限

没有使用最小二乘 coefficient 标签。直接在完整网格上最小化 L1，并用
full-grid smooth-L1 L-BFGS 精修，前三个测试结构得到：

- 0.6819%
- 0.6082%
- 0.7215%

当前找到的平均可行值为 **0.6705%**。这是精确 L1 projection error 的
数值上界，不是经过证明的数学下界，但已经表明当前 12 radial、
`l_density <= 4`、3 Å support basis 是主要误差瓶颈。

## ChargE3Net 为什么能达到约 0.06%

### 1. 没有共享 coefficient bottleneck

ChargE3Net 对每个 probe 直接预测标量密度。probe 位置参与三层
equivariant tensor-product network，因此径向和角向响应可以随 probe
和局部环境非线性变化。

当前方法先把一个结构压缩成每原子 300 个 coefficient，再用固定的
12 个 Gaussian radial functions 和线性 spherical-harmonic expansion
重建所有 probes。推理快很多，但完整密度必须通过同一个低维函数空间。

### 2. 更大的 probe 网络

ChargE3Net 在 atom trunk 和 probe readout 中都使用 `l <= 4`、两种 parity、
约 500 total multiplicities，并在三层中持续执行 tensor products。
它有 1.91M 参数，是当前 477k 参数模型的约 4 倍。

更重要的是，高阶表示存在于每个 probe 的非线性网络中；它不等价于一次
线性的 `l <= 4` density expansion。

### 3. 可学习的径向映射

ChargE3Net 使用 20 个 Gaussian radial inputs，并通过隐藏宽度 100 的
radial network 生成 tensor-product weights。当前 density decoder 只有
12 个固定 Gaussian radial functions，且没有 probe-specific radial MLP。

### 4. 完整周期图

ChargE3Net 的 KD-tree graph constructor 显式复制周期像，并保留 4 Å 内
所有 atom-to-probe edges。当前 decoder 只使用 minimum-image displacement；
NMC 最短晶格边约 5.44--5.55 Å，而 3 Å 已超过约 2.72--2.78 Å 的
minimum-image 安全半径。

### 5. 更充分的 probe 覆盖

当前训练数据每结构只保存 16,384 个固定 probes，约为完整网格的 3.7%，
之后每轮只能在这个固定子集中重采样 2,048 点。

ChargE3Net 每次直接从完整 density grid 重采样。官方 checkpoint 在
363,375 optimizer steps 保存，训练使用 4 GPUs、batch size 8/GPU、
200 probes/结构，累计监督量达到十亿量级。

### 6. NMC benchmark 是强同分布插值

所有结构共享相同的 NMC 超胞家族、元素集合和位点拓扑。500 个 test
结构中有 358 个（71.6%）的化学式在训练集中完全出现过。模型主要学习
固定材料家族内的取代、Li vacancy 和局部弛豫，而不是跨材料体系外推。

## 学习率诊断

当前训练使用：

```text
ReduceLROnPlateau(factor=0.5, patience=12)
```

validation NMAE 有明显噪声，但每隔少量 epoch 仍会刷新 best；每次刷新
都会重置 patience。因此训练到 epoch 176 时 LR 始终是 0.001。

ChargE3Net 使用与 validation 无关的逐 step power decay：

\[
\mathrm{lr}(s)=0.01\times0.96^{s/10000}.
\]

官方 checkpoint 在 step 363,375 时的 LR 为 0.0008439。

下一次训练应使用确定性的 step-based cosine/power decay，而不应继续依赖
当前高噪声 validation 的 plateau scheduler。

## 下一步建议

1. 先修复 density decoder 的多周期像求和。
2. 将最小 radial width 与 support cutoff 解耦。
3. 用 full-grid direct-L1 oracle 扫描 radial basis、cutoff 和
   `l_density`，先把表示误差从约 0.67% 降下来。
4. 训练时从完整 CHGCAR 动态抽 probes，避免固定 3.7% 网格子集。
5. 改用 step-based LR decay，再训练新的 coefficient model。

