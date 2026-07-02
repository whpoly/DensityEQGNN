# Benchmark Notes

The initial benchmark set follows ChargE3Net:

1. QM9 charge densities.
2. NMC Li-ion battery cathode charge densities.
3. Materials Project charge densities.

Expected processed layout:

```text
data/processed/qm9/
  train.txt
  val.txt
  test.txt
  sample_000001.npz

data/processed/nmc/
  train.txt
  val.txt
  test.txt
  sample_000001.npz

data/processed/materials_project/
  train.txt
  val.txt
  test.txt
  mp-*.npz
```

Each split file lists `.npz` filenames relative to its dataset root.

## Metrics

The training loop reports:

- density MAE
- density normalized MAE
- total multi-task loss

For full-paper reproduction, add full-grid evaluation by setting `test_probes`
to null/omitting probe subsampling after enough memory profiling.

## One-Command Preparation

Prepare all supported benchmarks:

```bash
python scripts/prepare_all_benchmarks.py --datasets qm9 nmc mp --mp-api-key "$MP_API_KEY"
```

Prepare a small Materials Project subset first:

```bash
export MP_API_KEY="your_materials_project_key"
bash scripts/prepare_mp_test.sh
```

Smoke test with small downloads/conversions:

```bash
python scripts/prepare_all_benchmarks.py --datasets qm9 nmc --limit-files 1 --limit-samples 10
python scripts/prepare_all_benchmarks.py --datasets mp --limit-samples 5 --mp-api-key "$MP_API_KEY"
```

The full QM9 and NMC tar files are multi-GB downloads. Materials Project access
requires an API key and is therefore not stored in this repository.
