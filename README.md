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

或者一条命令自动准备 benchmark 数据：

```bash
python scripts/prepare_all_benchmarks.py --datasets qm9 nmc mp --mp-api-key "$MP_API_KEY"
```

正式准备完整 Materials Project benchmark：

```bash
export MP_API_KEY="your_materials_project_key"
bash scripts/prepare_mp_full.sh
```

`scripts/prepare_mp_full.sh` 默认不限制样本数，也不指定单个 `mp-id`，会扫描 Materials Project 中所有带 charge density 的材料并转换成训练用 `.npz`。如果中途断掉，重新运行会跳过已经转换好的样本继续准备。

先只测试 Materials Project 下载和转换流程：

```bash
export MP_API_KEY="your_materials_project_key"
bash scripts/prepare_mp_test.sh
```

可以在提交脚本时覆盖参数：

```bash
MP_LIMIT_SAMPLES=5 PROBES=2000 bash scripts/prepare_mp_test.sh
```

`scripts/prepare_mp_test.sh` 默认使用一个已验证可下载的 MP id 做快速测试：

```bash
MP_IDS="mp-1524357" bash scripts/prepare_mp_test.sh
```

如果只想临时调试完整 MP 脚本，可以给它加样本数限制：

```bash
MP_LIMIT_SAMPLES=100 bash scripts/prepare_mp_full.sh
```

说明：

- QM9/NMC 会从 DTU/Figshare 自动下载原始 tar 文件并转换。
- Materials Project 需要 Materials Project API key；建议通过环境变量 `MP_API_KEY` 传入，不要写进脚本。
- 完整 QM9/NMC 数据很大，可以先用 `--limit-files 1 --limit-samples 10` 做测试。

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

## 环境

创建并激活 conda 环境：

```bash
conda env create -f environment.yml
conda activate density-eqgnn
```

这个环境会设置 `PYTHONPATH=src`，所以不需要 `pip install -e .`。

如果你想完全复现当前 Windows 机器上解析出来的包版本，可以用：

```bash
conda env create -f environment-win64.lock.yml
```

## 训练

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

集群上一键准备完整 MP、训练并评估：

```bash
cd /home/wuhao/DensityEQGNN
sbatch --export=ALL,MP_API_KEY="your_materials_project_key" scripts/run_mp_full.sbatch
```

这个脚本会先检查 CUDA，然后运行 `scripts/prepare_mp_full.sh`。只有 `data/processed/materials_project/train.txt` 存在后才会训练，只有 `runs/materials_project/best.pt` 存在后才会评估。

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
