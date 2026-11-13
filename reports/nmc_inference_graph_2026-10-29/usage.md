# 使用 GNN 执行图重放

这条路径适合连续调用，并且能反复命中相同的原子数/邻接边数。每次重新计算输入几何、系数、SAD 和密度。
首次遇到新形状需要录制；邻接边数变化或缓存淘汰后可能再次录制。没有保证任意 MD 轨迹都能提速。

```python
import torch
from scripts.predict_adaptive_r16_replay import ReplayAdaptiveR16

torch.set_num_threads(2)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False

predictor = ReplayAdaptiveR16(max_graphs=2)

# 参数与上一轮一致：原子序数 [N]，Å 位置 [N,3]，Å 行向量晶胞 [3,3]。
# 不需要 DFT 密度标签。每次调用可以传入新的坐标、晶胞和元素。
rho = predictor(numbers, positions, cell, (56, 56, 140))

# 查看已经发生的录制；setup_seconds 是录制与预热时间。
print(predictor.graphs.captures)

# 需要时释放缓存的执行图。
predictor.clear_graphs()
```

输出为 GPU 上的 e/Å³ 密度张量，形状等于给定网格。接口用于固定权重模型的单调用者推理，不用于训练或多线程同时调用。
基础解码仍采用上一轮已验证的分组矩阵乘法，FP64 白化与重构保持不变。没有缓存之前的密度或 GNN 输出。

一次性预测独立新结构仍用 `scripts.predict_adaptive_r16_fast.FastAdaptiveR16`，避免执行图录制成本。

复测四种候选：`python -m scripts.benchmark_r16_inference_graph`。
依据结果生成采用方案报告：`python -m scripts.summarize_r16_graph_selection`。
接口与缓存淘汰核验：`python -m scripts.check_replay_predictor`。

当前检查点仅在 8 个已见 NMC 结构上完成诊断；加速不代表已经解决未知结构精度或完成 DFT/SCF 验证。
