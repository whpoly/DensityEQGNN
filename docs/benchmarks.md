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

