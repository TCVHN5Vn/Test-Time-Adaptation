# PeTTA-HAR: Persistent Test-Time Adaptation for Smart Building HAR

## Overview

A HAR-adapted implementation of PeTTA (Hoang et al., NeurIPS 2024) for
cross-resident activity recognition in smart buildings (S1 → S2).

- **Backbone:** MLP with BatchNorm (256→128→64), replaced by RobustBN1d during TTA
- **Recurring evaluation:** Dirichlet-mixed batches over `(year_day, time_block)`
  target domains, repeated across 20 visits
- **Alignment:** follows the paper's recurring-TTA evaluation spirit — not a
  direct reproduction of the original CIFAR-C / ImageNet-C / DomainNet benchmarks
- **Algorithm:** all three core PeTTA components faithfully adapted (divergence
  sensing, adaptive λ_t / α_t, anchor loss)

---

## Project Structure

```
upgraded_petta_har/
├── run_petta.py              ← standalone script entry point
├── train_source.py           ← source MLP training on S1
└── src/
    ├── model.py              MLP architecture (256→128→64 + BatchNorm)
    ├── petta_adapter.py      PeTTA Algorithm 1 — all fixes applied
    ├── data_loader.py        stratified split, StandardScaler, loaders
    ├── dirichlet_stream.py   domain builder + Dirichlet stream generator
    ├── persistence_metrics.py visit-level tracking + retention gap
    ├── constants.py          activity map, class names
    └── utils/
        ├── petta_memory.py   DivergenceScore (per-class), PrototypeMemory, PeTTAMemory
        ├── bn_layers.py      RobustBN1d (EMA-mixed BatchNorm)
        └── loss_func.py      self_training, softmax_entropy
```

---

## Dataset

| | S1 (source) | S2 (target) |
|---|---|---|
| Rows | 5031 | 3412 |
| Activities | 5 | 5 |
| Features | 92 (union of S1 + S2 sensors) | 92 |
| Window size | 1 timestep | 1 timestep |

**Activity mapping:**

| Code | Index | Label |
|------|-------|-------|
| 15 | 0 | Other |
| 60 | 1 | Cook |
| 65 | 2 | Sleep |
| 70 | 3 | Bathe |
| 85 | 4 | Toilet |

---

## Model Architecture

```
Input (B, 1, 92) → flatten → (B, 92)
  Linear(92, 256) + BatchNorm1d(256) + ReLU + Dropout(0.3)
  Linear(256, 128) + BatchNorm1d(128) + ReLU + Dropout(0.3)
  Linear(128, 64)  + BatchNorm1d(64)  + ReLU
  ↑ feat(x): 64-dim embedding used for divergence sensing (Eq. 6)
  Linear(64, 5)
```

- **Params:** ~200K
- **val_acc on S1:** 0.9364
- **source-only on S2:** 0.2541

---

## PeTTA Algorithm (src/petta_adapter.py)

| Paper element | Implementation | Eq. |
|---|---|---|
| Pseudo-labels Ŷ_t = f_{t-1}(X_t) | teacher forward pass | 1 |
| Student update L_CLS + L_AL + λ_t R(θ) | Adam on RobustBN1d params | 2 |
| EMA teacher update θ_t | update_ema() | 3 |
| Per-class Mahalanobis distance d^y_t | DivergenceScore.per_class_distance() | 6 |
| γ^y_t = 1 - exp(-d^y_t) per class then averaged | applied before averaging (nonlinear) | 6–7 |
| λ_t = γ̄_t · λ_0 | adaptive_lambda=True | 7 |
| α_t = (1 - γ̄_t) · α_0 | adaptive_alpha=True | 7 |
| Anchor loss L_AL | softmax_entropy(p_aug, anchor_out) = KL divergence | 8 |
| Category-balanced memory bank | PeTTAMemory | — |
| Final prediction from f_t(X_t) | updated teacher after EMA | Alg. 1 |
| Source statistics μ^y_0, Σ^y_0 | computed from full S1 (all rows) | App. E.4 |

---

## Recurring TTA Protocol (src/dirichlet_stream.py)

**Domain definition:**
Each domain = `(year_day, time_block)` where time_block ∈
{morning, afternoon, evening, night}. This yields finer-grained domains
than whole days alone (~4× more domains).

**Dirichlet batch mixing:**
Each batch is a mixture of samples from 3 nearby domains, with mixing
weights drawn from Dirichlet(β=0.1). Small β creates batches dominated
by one domain — simulating the temporally correlated test streams in the
paper.

**Recurring stream:**
Domains are traversed chronologically D1 → D2 → ... → DD for 20 visits.
Each visit generates fresh Dirichlet-sampled batches from the same ordered
domain sequence, so recurrence is preserved while intra-batch composition
varies across visits.

**Evaluation:**
After each visit's adaptation pass, the model is evaluated on the full
sorted S2 stream (no Dirichlet) — clean separation of adaptation and evaluation.

**Comparison with paper:**

| Paper (original benchmarks) | This project (HAR) |
|---|---|
| CIFAR-10/100-C, ImageNet-C, DomainNet | Smart-building sensor CSVs |
| Dirichlet-sampled temporally correlated batches | Dirichlet-mixed from (day, time_block) domains |
| Recurring TTA: D image-corruption domains × K=20 | Recurring TTA: D HAR domains × K=20 |
| ResNet / WideResNet backbone | MLP with BatchNorm |
| RobustBN from RoTTA | RobustBN1d adapted for 1D |

---

## Hyperparameters

This is the retained configuration reported in Kbaier et al., Sec. 4.6,
selected via grid search over:
  alpha_0 in {0.001, 0.005, 0.01, 0.05}, lambda_0 in {0.25, 1.0, 10.0},
  BN momentum in {0.01, 0.05, 0.1}, lr in {1e-3, 1e-4, 5e-5}.

| Parameter | Value | Notes |
|---|---|---|
| alpha_0 | 0.05 | initial EMA rate |
| lambda_0 | 10.0 | initial regularization weight |
| al_wgt | 1.0 | anchor loss weight |
| bn_momentum | 0.01 | RobustBN1d momentum |
| memory_size | 64 | category-balanced bank capacity |
| noise_std | 0.05 | Gaussian noise added to inputs during student updates |
| lr | 5e-5 | Adam learning rate |
| regularizer | cosine | R(θ) distance metric |
| loss_func | sce | L_CLS: self-training |
| beta | 0.1 | Dirichlet concentration (strong class correlation) |
| num_visits | 20 | recurring visits K |
| window_size_d | 3 | nearby domains mixed per batch |

---

## Results (HAR S1→S2, persistence reporting)

| Metric | Value |
|---|---|
| Source-only accuracy | 25.41% |
| PeTTA — visit 1 | rerun to get |
| PeTTA — best visit | 37.30% (visit 8, simple stream) |
| PeTTA — final visit | rerun to get |
| PeTTA — avg over 20 visits | rerun to get |
| Retention gap (final − best) | rerun to get |

> **Note:** "best visit" result (37.30%) was obtained with the earlier
> sequential replay setting. Rerun `run_petta.py` with the Dirichlet
> stream to get full persistence metrics.

---

## How to Run

From terminal:
```bash
python train_source.py   # train source model on S1
python run_petta.py      # run recurring TTA with Dirichlet stream
```
