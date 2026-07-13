DIGA — Dynamically Instance-Guided Adaptation
Wang et al., CVPR 2023
Adapted for tabular sensor data (HAR + Occupancy)

Paper-faithful version matching Kbaier et al., Section 4.4 (Eqs. 2-6).

════════════════════════════════════════════════
PROJECT STRUCTURE
════════════════════════════════════════════════

diga_fixed_patched/
├── src/
│   ├── diga.py        Main DIGA adapter (DAM + SAM)
│   ├── dam.py          Distribution Adaptation Module
│   ├── sam.py          Semantic Adaptation Module
│   ├── har_model.py    HAR MLP architecture
│   ├── occ_model.py    Occupancy MLP architecture
│   └── constants.py    Dataset constants
└── README.txt          This file

════════════════════════════════════════════════
ALGORITHM — matches Kbaier et al. Sec. 4.4
════════════════════════════════════════════════

DIGA is BACKWARD-FREE — no gradient computation needed.

DAM — Distribution Adaptation Module (dam.py):
  Replaces each BatchNorm1d with mixed normalization (Eq. 2-3):
    BN_DAM(x) = w * [bn_lambda*src_norm + (1-bn_lambda)*ins_norm] + b
  bn_lambda=0 → instance statistics only
  bn_lambda=1 → source statistics only
  bn_lambda=0.95 → retained config: 95% source, 5% instance
  λ_BN weights the SOURCE statistics, matching the paper's Eq. 3.

SAM — Semantic Adaptation Module (sam.py):
  Builds a dynamic non-parametric classifier (Eqs. 4-6):
    1. Instance prototypes q^c_t: mean embedding of the top-K=32 most
       confident predictions per class, among those with confidence
       >= a single FIXED global threshold P0 (Eq. 4). No adaptive or
       per-class threshold schedule — P0 is constant throughout.
    2. Historical EMA prototypes (Eq. 5/7):
         q̄^c_t = proto_rho * q̄^c_{t-1} + (1 - proto_rho) * q^c_t
       proto_rho weights HISTORY; (1 - proto_rho) weights the new
       instance prototype.
    3. Non-parametric prediction: cosine similarity between the feature
       embedding and q̄^c_t directly — no additional re-mixing step.
    4. Fusion (Eq. 6): fusion_lambda*SAM + (1-fusion_lambda)*parametric
       fusion_lambda weights the NON-PARAMETRIC (SAM) output.

════════════════════════════════════════════════
RETAINED HYPERPARAMETERS (Kbaier et al., Sec. 4.4)
════════════════════════════════════════════════

  bn_lambda            = 0.95  BN mixing weight (source vs instance)
  fusion_lambda        = 0.80  classifier fusion weight
  confidence_threshold = 0.50  fixed candidate-selection threshold (P0)
  proto_rho            = 0.10  EMA weight on HISTORY
  top_k                = 32    max confident samples per class per batch

These were selected via grid search over:
  bn_lambda   in {0.80, 0.90, 0.95, 1.00}
  P0          in {0.50, 0.60, 0.65, 0.70}
  proto_rho   in {0.02, 0.10}
  top_k       in {16, 32}
(4 x 4 x 2 x 2 = 64 configurations.)

For HAR, the retained config is the one with the best target-domain
macro-F1 on held-out labels; for Occupancy Estimation, the one with
the best MCC.
