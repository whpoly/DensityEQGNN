# Matched NMC EdenGNN occupation benchmark

Reuse the exact DeepDFT PAW experiment's dataset, masks, training-only per-element means and per-block scales. Train cache 0–7, validate SCF 170/314, test SCF 361/796/825/1056/1749. The previously used five-structure test is already public from the DeepDFT experiment; no change in architecture, split, budget or checkpoint selection may be based on the new test results. This is a matched small-data diagnostic, not a fresh benchmark.

Only native L<=2 augmentation coefficients are supervised/evaluated. L=3,4 are unavailable in existing files and excluded. Metrics and baseline are imported unchanged from the DeepDFT experiment: per-structure macro MAE and relative L1, pooled metrics, by-L and by-element errors, and anisotropic-only errors. Relative coefficient L1 is not density NMAE.

Two arms, seeds 42 and 43 each:

1. frozen: the local eight-structure EdenGNN density encoder is frozen; train only its official atomwise equivariant linear augmentation head.
2. finetuned: start from the identical density encoder checkpoint; train a separate occupation network (encoder plus augmentation head). The original density model/checkpoint remains unchanged; the separately tuned encoder must not be silently substituted into density inference.

Keep the exact pinned Eden Encoder and AugmentationLayer source classes. Adapt the output irreps to the same canonical 138-slot NMC projector schema, mask unavailable higher ranks, and apply the same training-only residual baseline/scales as the DeepDFT experiment. This is a controlled local adaptation, not a reproduction of the publication's datasets or universal model. Eden's larger high-order encoder and original DeepDFT's smaller scalar/vector encoder are not parameter-matched; the output-head architectures also differ.

Both arms: 6000 optimizer updates, two distinct training structures per step, identical per-seed sample RNG sequence to DeepDFT, standardized MSE, Adam 1e-3→1e-5 cosine, gradient clipping 5, validation every 200 steps. Select by validation macro raw MAE. Both arms' seeds must have selected/frozen checkpoints before final test metrics. Report all seeds, not the best test seed. This matches update/sample budget, not training FLOPs or wall time.

Pre-test conditioning amendment: the preliminary unscaled frozen seed 42 run showed almost no training improvement (0.03770→0.03707 MAE); higher-order input RMS was about 0.0055 (L=1) and 0.0025 (L=2). The new zero-initialized head requires an appropriately scaled input to learn within the matched update budget. Divide every irrep channel by its training-atom RMS shared across all m, floor 1e-4. This fixed rotation-invariant scaling is used in BOTH arms and is fit before training using only training features. No test metrics were opened. Preserve the completed unscaled seed 42 and interrupted seed 43, logs and source snapshot under runs/edengnn_paw_2026-10-30; formal results use runs/edengnn_paw_scaled_2026-10-30. This changes optimization conditioning, not the representable linear function space.

Frozen arm caches immutable training/validation encoder outputs. Finetuned arm may use CUDA execution graphs only after forward and all-parameter gradient equivalence checks against eager, including two-structure gradient accumulation and updated parameters on all eight training structures. No learned outputs cached in that arm. Formal inference timing uses eager execution, fresh atom graph/encoder features each call; head-only timing is separately labeled.

Tests: official-head linear equivalence and nontrivial rotation/inversion covariance, native coefficient mask, checkpoint reload, cached/fresh frozen features, training gradient audit, real-structure rotation/inversion check, source/checkpoint hashes. Preserve DeepDFT results. Compare density preservation only for the frozen encoder; the fine-tuned arm is explicitly a separate occupation model. Do not claim SCF acceleration without VASP evaluation.
