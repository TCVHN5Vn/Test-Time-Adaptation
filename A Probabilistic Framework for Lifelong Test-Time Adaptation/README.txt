PETAL — Probabilistic lifElong Test-time Adaptation with seLf-training prior
Brahma & Rai, CVPR 2023
Adapted for tabular sensor data (HAR + Occupancy)

Paper-faithful version matching Kbaier et al., Section 4.5.

════════════════════════════════════════════════════
PROJECT STRUCTURE
════════════════════════════════════════════════════

petal_fixed/
├── src/
│   ├── swag.py            SWAG-D posterior (Maddox et al. 2019)
│   ├── train_swag.py      SGD training + snapshot collection
│   └── petal_adapter.py   PETAL Algorithm 2 (adaptation)
└── README.txt             This file

════════════════════════════════════════════════════
ALGORITHM SUMMARY (Algorithm 2, paper-faithful)
════════════════════════════════════════════════════

Phase 1 — Source Training (SGD + SWAG-D):
  - Burn-in: cosine LR decay, early stopping
  - SWAG collection: constant LR=0.01, 4 snapshots/epoch × 20 epochs
  - Posterior: q(θ) = N(μ, diag(σ²))  where μ=SWA mean, σ²=iterate variance
  - MAP model: θ_0 = μ  (used to initialize student and teacher)

Phase 2 — PETAL Adaptation (per batch):
  1. Pseudo-label from teacher:
       y' = plain teacher       if conf(θ_0, x) >= τ
       y' = aug-avg teacher     otherwise  (K=32 augmentations)
  2. Store prediction from y' (online evaluation, before adaptation)
  3. Loss: L = log q(θ) - λ̄ * Hxe(y', y_student)
  4. Adapt student: Adam, lr and weight_decay per retained config
  5. Update teacher: EMA with smoothing π
  6. FIM restoration: restore δ-quantile of least important params to θ_0
  7. Return the pre-update prediction from step 2

  Note: there is no teacher re-anchoring step after FIM restoration.
  Algorithm 2 in the paper does not include this step, so it is not
  part of this implementation.

════════════════════════════════════════════════════
RETAINED HYPERPARAMETERS (Kbaier et al., Sec. 4.5 grid search winner)
════════════════════════════════════════════════════

  π    = 0.95   (EMA smoothing)
  K    = 32     (augmentations per pseudo-label)
  λ̄   = 0.0    (cross-entropy weight; FIM restoration is the main
                anti-forgetting mechanism, not this regularizer)
  δ    = 0.1    (FIM quantile for restoration)
  τ    = 0.5    (source-confidence threshold — dataset-dependent)
  lr   = 1e-4   (Adam learning rate)
  weight_decay = 1e-4

  Selected via a 288-configuration grid search over:
    π          in {0.9, 0.95, 0.99, 0.999}
    τ          in {0.5, 0.6, 0.7, 0.9}
    lr         in {1e-3, 1e-4, 5e-5}
    δ          in {0.03, 0.1, 0.2}
    λ̄          in {0.0, 1.0}

  K=32 is kept aligned with the paper. π, τ, lr, δ, and λ̄ are the
  retained grid-search winner for this HAR setting, not universal
  paper defaults — the paper notes some hyperparameters are
  dataset-dependent and defers exact values to its appendix.

════════════════════════════════════════════════════
AUGMENTATION NOTE
════════════════════════════════════════════════════

  The paper uses image augmentations (random crops, flips, colour jitter).
  augment_tabular() uses Gaussian noise + random feature dropout instead.
  This is a modality-specific adaptation for HAR/sensor data, not a
  faithful reimplementation of the paper's augmentation step. The algorithmic
  role is the same: produce diverse views for the teacher to average over
  when source-model confidence is low.

════════════════════════════════════════════════════
EVALUATION METRICS
════════════════════════════════════════════════════

  Accuracy, Macro F1, Kappa, Balanced Accuracy, MCC  — standard classification
  NLL (Negative Log-Likelihood)                       — calibration
  Brier Score                                         — calibration
  Per-visit trajectory                                — lifelong TTA behaviour

════════════════════════════════════════════════════
EVALUATION SETUP NOTE
════════════════════════════════════════════════════

  The recurring-lifelong HAR evaluation replays the same chronologically
  ordered S2 stream for N visits, with adapter state (student, teacher)
  carrying over across all batches and visits without reset.

  This differs from the paper's original corruption-stream benchmarks
  (CIFAR-C / ImageNet-C), which present a sequence of distinct corruption
  domains that the model encounters once. The recurring-visit design here
  is a diagnostic convenience for HAR — it lets you observe convergence
  and drift over repeated exposure to the same distribution shift.
  It should not be described as a direct reproduction of the paper's
  evaluation protocol.

════════════════════════════════════════════════════
LOSS SIGN NOTE
════════════════════════════════════════════════════

  The paper maximises a log-posterior objective:
      max  log q(θ) - λ̄ * H(y', f_θ(x))

  The code minimises the negation of that objective:
      loss = -(log_q - lambda_bar * H)

  These are exactly equivalent. The README and docstrings use
  "minimise the negative log-posterior" throughout to avoid sign
  ambiguity.
