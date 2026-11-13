# 使用优化后的混合模型

在仓库根目录、原有 `kan-cgcnn-cuda` 环境中运行。推理只需要原子序数、位置、晶胞和网格尺寸。
位置和晶胞单位为 Å，晶胞采用每行为一个晶格向量的约定。

```python
import torch
from scripts.predict_adaptive_r16_fast import FastAdaptiveR16

torch.set_num_threads(2)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32 = False

# 加载一次模型与 POTCAR 常量；默认采用本轮已冻结的自适应头检查点。
predictor = FastAdaptiveR16()

# numbers: [N]；positions: [N, 3]；cell: [3, 3]。
# 参数可以是 NumPy 数组或 torch Tensor，不需要目标电子密度。
rho = predictor(numbers, positions, cell, (56, 56, 140))
# rho 是 CUDA Tensor，形状为网格尺寸，单位 e/Å³。
```

每次调用都重新生成原子图、网格邻域、SAD 和模型输出，不缓存特定结构的预测结果。
基组与白化重构保持 FP64；主干和输出系数保持原模型的 FP32。GPU SAD 的样条、相位和 FFT 使用双精度。
本轮计时不包含构造 predictor 时的检查点/POTCAR 文件加载，也不包含最终输出到 CPU 或写盘。

复测：`python -m scripts.benchmark_r16_inference_fast`。
接口核验：`python -m scripts.check_fast_predictor`。

结果及局限见同目录的 `report.md`。当前检查点来自 8 个已见 NMC 结构，未验证未知结构泛化。
