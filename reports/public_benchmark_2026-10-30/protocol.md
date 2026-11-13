# Official pretrained density benchmark: fixed protocol

Frozen before this run's full official test-set evaluation, 2026-10-30. Local RTX 5060 Ti 16 GB. QM9 is excluded at the user's request. No retraining or test-driven choice of architecture, cutoff, precision, radial table size, or chunk size is allowed in this run.

## Data and checkpoints

| Dataset | Train | Validation | Test | Checkpoint |
|---|---:|---:|---:|---|
| NMC | 1450 | 50 | 500 | Official `nmc_painn/best_model.pth` |
| Ethylene carbonate (EC) | 7330 | 50 | 4000 | Official `ethylenecarbonate_painn/best_model.pth` |

Use the exact author-provided split files and original full density grids. NMC archive order is `cellrelax_000.tar`, `cellrelax_001.tar`. EC uses the author's `ethylenecarbonate.txt` order, archives 001xxx through 012xxx. EC split index 0 corresponds to trajectory frame 01000; the first 1000 frames are excluded. Verify contiguous member numbering and exact split membership before evaluation. Downloaded archives must match the publisher's byte count and MD5. Store SHA-256 receipts, tar offsets, model and source hashes in `settings.json`.

Sources: [NMC dataset](https://doi.org/10.11583/DTU.16837721), [EC dataset](https://doi.org/10.11583/DTU.16691825), [DeepDFT repository](https://github.com/peterbjorgensen/DeepDFT), commit `a6bab4deb5cf05d9b46ae397b72253d04ea3c694`. Both models have 3 interactions, 128 channels and a 4 Angstrom cutoff. Pymatgen parses the total CHGCAR density; divide stored VASP values by cell volume to obtain electrons/Angstrom^3. Preserve the full original periodic grid.

## Compared implementations

- `reference`: official pretrained atom and probe algebra through the local FlatProbe adapter, chunk 4096. This adapter already hoists atom-only projections and uses the local periodic graph builder. Its runtime is **not the unmodified upstream CLI runtime**.
- `fused_same_chunk`: identical weights and 4096 chunk, radial tables with 16385 samples and fused CUDA aggregation.
- `fused`: identical weights and 8192 chunk, the same tables and fused aggregation.

All 4 Angstrom neighbors and 128 channels are retained. The only persistent reuse between materials is weight-dependent constants; graph geometry, atom features and predictions are recalculated. No low-dimensional fitting, cached answer, PAW occupation head or test-set fine-tuning is used. FP32 inference, TF32 disabled, PyTorch CPU threads 2. Density targets are used only to calculate metrics after prediction.

## Accuracy and numerical agreement

Evaluate `reference` and `fused` on every official test structure and every grid point. Alternate execution order by index parity. Per structure report NMAE = sum(abs(prediction-target))/sum(abs(target)), MAE, RMSE, integrated electron-count bias and negative-density integral. JSON stores NMAE as a fraction; human reports multiply it by 100 and label percent.

Report unweighted mean, median, p95 and maximum per-structure NMAE; also report separately the ratio of pooled absolute errors to pooled absolute targets. Report paired NMAE difference and descriptive structure-bootstrap 95% intervals (seed 20261030, 2000 replicates); EC trajectory correlations mean these intervals do not measure independent-material generalization. Include the worst structure IDs.

Numerical equivalence is tested against the reference prediction, not against DFT: every point must meet `rtol=5e-5, atol=5e-5` electrons/Angstrom^3, and the per-structure relative L1 output discrepancy must be below 1e-6. Report every violation. Training-frame preflights additionally test both chunk sizes. A small direct upstream-forward check validates the adapter; this check is separate from measured performance.

## Timing and memory

Before evaluation, choose 50 official test indices with NumPy `default_rng(20261030).choice(test_indices, 50, replace=False)` and freeze them in the manifest. Run each variant in its own process, serially on the GPU. For each structure, use 2 warmups and 5 synchronized wall-clock repetitions. Every repetition builds a fresh periodic graph, encodes atoms and predicts the complete density grid. Record all repetitions, each structure's median, peak allocated CUDA memory and increase above the input/model baseline. Report distributions and paired speedups on the common successful structure set, always including failure counts.

These are warm inference seconds **per material structure**, not seconds per grid point or a complete DFT calculation. They exclude network transfer, CHGCAR parsing, target loading, imports, checkpoint loading and one-time table/kernel preparation. Record setup and input read/decode/transfer durations separately; do not claim these warm times as cold-start end-to-end times. Other GPU activity can affect timing; log utilization and processes and disclose interference.

## Failures, resuming and interpretation

Save results after each structure. A CUDA OOM is an explicit failure; never silently reduce chunk size, skip grid points or substitute a model. Other errors stop the stage while retaining partial results. Hash checks reject changed models, sources, split files or this protocol when resuming. Report partial results as partial; a completed loop with failures is not an all-pass result.

The implementation and small local NMC benchmark existed before this protocol; this is not a wholly blind method-development experiment. Official pretrained checkpoints, rather than the previous locally trained eight-structure model, are used here. The experiment measures preservation of the official model's accuracy under faster inference, not improved learning or out-of-distribution generalization. The [DeepDFT paper](https://www.nature.com/articles/s41524-022-00863-y) reports equivariant-model NMAE of 0.06% on NMC and 0.18% on EC; those are literature reference values, not this run's results.

EdenGNN public [release](https://zenodo.org/records/21736087) is audited separately. Its public universal/MD checkpoints and example density files do not supply a matched NMC/EC-trained baseline or the complete MP density test targets. Do not label a cross-domain universal-model comparison as a matched benchmark, reuse its paper's different-hardware timings as local timings, or present the old local EdenGNN fit as a public pretrained model. Complete PAW occupations, VASP SCF iterations and total DFT wall time are outside this density-only benchmark until the required data and DFT installation are available.
