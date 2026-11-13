# Full NMC occupation transfer experiment — 2026-11-02

Goal: attach a trainable equivariant PAW augmentation readout to the official
pretrained DeepDFT NMC density model while preserving its density predictions
and fused inference. This is a new full-split experiment, not the old 8-structure fit.

Use all 2000 official NMC structures and the fixed 1450/50/500 split. Audit
native augmentation sizes, projector mappings, angular availability, finite
values and exact geometry duplicates before training. Read local POTCAR
projector ordering explicitly. Historical exact POTCAR identity cannot be
recovered from CHGCAR alone. All-zero L=3/4 channels are unavailable labels,
never a successful prediction. Train/evaluate only supported L<=2. Nonmagnetic
NMC only; extending to EC or magnetic materials requires separate mappings and experiments.

Primary model: freeze official NMC DeepDFT encoder and density decoder; train
existing 49,899-parameter geometric equivariant PAWReadout. Comparison: training
element means and the existing official Eden augmentation architecture with a
frozen local 8-structure density encoder. Eden pretraining differs from official
DeepDFT pretraining; this is a transfer baseline, not a matched-pretraining
architecture ranking. Identify all 8 local pretraining structures in the full
archive. Exclude any overlapped official test structures from Eden results and
report both models on the common subset; DeepDFT additionally reports all 500.
Do not use the previous PAW-trained Eden checkpoint or its selected head.

Both learned arms: seeds 42 and 43, 100 full epochs, batch 16 structures,
Adam 1e-3 cosine to 1e-5, gradient clipping 5. Same epoch permutation for a given
seed. Standardized coefficient MSE, averaged per structure then per batch.
Statistics use only the 1450 training structures, and all m components of an
irrep share scaling. Eden feature RMS uses training atoms only. Frozen feature
caching is allowed in training, never in fresh-inference timing. Check validation
macro raw coefficient MAE every 5 epochs. Freeze all four selections before any
new test metrics. Report both seeds; no test-selected model, cutoff or precision.

Validation includes batched-versus-separate predictions and gradients, masks,
native mapping roundtrip, rotation/inversion/translation equivariance, and
finite gradients. Final metrics: macro and pooled coefficient MAE/relative L1,
per rank/element/structure errors, train/validation/test gap. Save predictions.
Density and occupation metric denominators are different; do not conflate them.

Joint performance: official full-grid NMC density, frozen DeepDFT plus seed-42
head, all 500 structures for occupation, a fixed 50 official test structure
subset (same as the previous density benchmark) for density verification and
timing. Compare fresh density-only and joint calls, 2 warmups and 5 CUDA-synchronized
repeats, independent processes per path, per-structure median then mean.
Retain original 4 A neighbors, 128 channels, FP32, TF32 off, fused chunk 8192.
No learned features or geometry reused across timed structures/calls. Record
all times, peak allocated and incremental memory. Exclude imports, file parsing,
checkpoint/kernel/table setup and file export from warm inference, disclose them.
Verify density numerical agreement and model hashes; measure fresh occupation-only
paths for DeepDFT and Eden on the common timing subset. Source/checkpoint/data
hashes and selections accompany results. Existing historical experiments remain intact.

SCF steps, end-to-end DFT time and final energy/forces require an executable VASP,
matching full PAW labels and settings. If unavailable, finish all supported
prediction experiments and explicitly mark this part unperformed. No oracle
augmentation or true grid values may enter geometry-only inference inputs.
