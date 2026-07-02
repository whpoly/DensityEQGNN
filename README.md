# Density EQGNN

这个项目把你分享对话里的想法落成一个可训练的 research scaffold：

```text
DFT grid density -> atom latent teacher
atomic structure -> ChargeE3Net-like atom EQGNN -> predicted atom latent
predicted atom latent -> property head
predicted atom latent -> atom-to-grid decoder -> predicted density
```

核心目标是：训练时利用 DFT electron density 做 teacher，推理时只需要原子种类、坐标和晶胞，就能预测 property 和 grid/probe 上的 charge density。

## 模型思路

本仓库先参考 ChargE3Net 的三点设计：

1. atom graph 先做 message passing，获得原子表示。
2. probe/grid point 通过 atom-to-probe message decoding 预测 density。
3. benchmark 跟 ChargE3Net 对齐：QM9、NMC、Materials Project。

在此基础上加了你提出的 atom bottleneck：

1. `DensityToAtomEncoder` 从 DFT grid/probe density 聚合出 `z_rho`。
2. `ChargeE3LikeAtomNetwork` 只看结构，预测 `z_hat`。
3. `AtomToGridDecoder` 用 `z_hat` 解码 probe/grid density。
4. `PropertyHead` 从 `z_hat` 预测性质。

默认实现是纯 PyTorch 的 lightweight 版本，保留 scalar/vector message passing、radial basis、atom-to-probe decoder。完整 `e3nn` irreps 版本可以在 `src/density_eqgnn/models/` 中继续替换，不影响数据和训练接口。

## 数据格式

训练统一使用 `.npz`，每个样本包含：

```text
atomic_numbers   int64, shape [N]
positions        float32, shape [N, 3], Cartesian Angstrom
cell             float32, shape [3, 3], optional
probe_positions  float32, shape [K, 3], Cartesian Angstrom
density          float32, shape [K]
property         float32, shape [P], optional
probe_volume     float32 scalar or shape [K], optional
electron_count   float32 scalar, optional
```

VASP `CHGCAR` 可以先用脚本转成 `.npz`：

```bash
python scripts/prepare_chgcar_dataset.py \
  --input data/raw_chgcars \
  --output data/processed/mp \
  --pattern CHGCAR \
  --probes 5000
```

## Benchmark 配置

配置文件已经放在：

```text
configs/qm9.yaml
configs/nmc.yaml
configs/materials_project.yaml
```

默认路径约定：

```text
data/processed/qm9
data/processed/nmc
data/processed/materials_project
```

这三套数据集对应 ChargE3Net 文章/官方仓库里的主要 benchmark。你可以先只放少量 `.npz` 跑 smoke training，再换成完整数据。

## 训练

安装依赖：

```bash
pip install -e .
```

跑 QM9：

```bash
python -m density_eqgnn.training.train --config configs/qm9.yaml
```

跑 NMC：

```bash
python -m density_eqgnn.training.train --config configs/nmc.yaml
```

跑 Materials Project：

```bash
python -m density_eqgnn.training.train --config configs/materials_project.yaml
```

## Loss

总 loss：

```text
L = w_density * L_density
  + w_teacher_recon * L_teacher_recon
  + w_latent * L_latent
  + w_property * L_property
  + w_charge * L_charge
```

其中：

- `L_density`: 用结构预测 latent 解码出的 density 与 DFT density 的误差。
- `L_teacher_recon`: grid density teacher latent 自编码重建误差。
- `L_latent`: 结构模型 `z_hat` 逼近 density teacher `z_rho`。
- `L_property`: property prediction loss。
- `L_charge`: 积分电子数约束。

## 参考

- ChargE3Net paper: https://www.nature.com/articles/s41524-024-01343-1
- ChargE3Net code: https://github.com/AIforGreatGood/charge3net

