"""
petta_adapter.py — Persistent Test-Time Adaptation
----------------------------------------------------
Faithful implementation of PeTTA (Hoang et al., NeurIPS 2024)
adapted for HAR sensor data (S1 → S2 domain shift).

Three core components (Sec. 4 of paper):

1. DIVERGENCE SENSING (Eq. 6):
   γ^y_t = 1 - exp(-(μ̂^y_t - μ^y_0)^T (Σ^y_0)^{-1} (μ̂^y_t - μ^y_0))
   Mahalanobis distance of the first moment of feature embedding vectors,
   mapped to [0,1] via 1-exp(-x).
   The covariance matrix Σ^y_0 is diagonal (paper Eq. 6 note).

2. ADAPTIVE PARAMETER ADJUSTMENT (Eq. 7):
   γ̄_t = mean over unique pseudo-labels of γ^y_t
   λ_t = γ̄_t · λ_0        (higher divergence → more regularization)
   α_t = (1 - γ̄_t) · α_0  (higher divergence → slower EMA update)

3. ANCHOR LOSS (Eq. 8):
   L_AL(X_t; θ) = -Σ_y Pr(y|X_t; θ_0) log Pr(y|X_t; θ)
   which is equivalent to minimizing KL divergence:
   D_KL(Pr(y|X_t; θ_0) || Pr(y|X_t; θ))
   (paper explicitly states this equivalence after Eq. 8)

Full PeTTA update (paper Eq. after Eq. 8):
   θ'_t = Optim_{θ'} E_Pt[L_CLS(Ŷ_t, X_t; θ') + L_AL(X_t; θ')] + λ_t R(θ')
   θ_t  = (1 - α_t) θ_{t-1} + α_t θ'_t

Image TTA transforms are replaced by Gaussian noise augmentation,
a standard approach for sensor time-series data.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from src.utils.loss_func import self_training, softmax_entropy
from src.utils.petta_memory import PeTTAMemory, PrototypeMemory, DivergenceScore
from src.utils.bn_layers import RobustBN1d


# ─────────────────────────────────────────────────────────────────────────────
# Sensor augmentation (replaces image TTA transforms for HAR sensor data)
# ─────────────────────────────────────────────────────────────────────────────

def sensor_augment(x: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    """Gaussian noise augmentation for HAR sensor readings."""
    return x + torch.randn_like(x) * noise_std


# ─────────────────────────────────────────────────────────────────────────────
# Configure model: freeze all params, replace BN with RobustBN1d
# ─────────────────────────────────────────────────────────────────────────────

def configure_model_for_tta(model: nn.Module, bn_momentum: float = 0.05) -> nn.Module:
    """
    Freeze all parameters and replace every BatchNorm1d with RobustBN1d.
    Only the RobustBN1d parameters remain trainable during TTA.
    """
    model.requires_grad_(False)

    bn_names = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.BatchNorm1d)
    ]

    for name in bn_names:
        parts      = name.split(".")
        parent     = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        child_name = parts[-1]
        old_bn     = getattr(parent, child_name)
        new_bn     = RobustBN1d(old_bn, bn_momentum)
        new_bn.requires_grad_(True)
        setattr(parent, child_name, new_bn)

    return model


# ─────────────────────────────────────────────────────────────────────────────
# PeTTA Adapter — Algorithm 1 (Appendix E.1 of paper)
# ─────────────────────────────────────────────────────────────────────────────

class PeTTAAdapter(nn.Module):
    """
    PeTTA: Persistent Test-Time Adaptation (Hoang et al., NeurIPS 2024).
    Adapted for HAR sensor data (SensorMLP backbone).

    Three model copies:
      student (θ')  : optimized each step via gradient descent
      teacher (θ_t) : EMA of student — provides pseudo-labels and predictions
      anchor  (θ_0) : frozen source model — used for anchor loss L_AL (Eq. 8)

    Divergence sensing (Eq. 6):
      Mahalanobis distance in the feature embedding space between
      the running class-wise mean μ̂^y_t and source statistics (μ^y_0, Σ^y_0).
      Diagonal covariance matrix Σ^y_0 is used (paper Eq. 6 note).

    Anchor loss (Eq. 8):
      L_AL = -Σ_y Pr(y|X; θ_0) log Pr(y|X; θ)
           = D_KL(Pr(y|X_t; θ_0) || Pr(y|X_t; θ))   [paper equivalence]
      Stabilizes the model under severe domain shifts by keeping
      predictions close to the well-behaved source model.

    Usage:
      adapter = PeTTAAdapter(model, cfg).cuda()
      adapter.compute_source_prototypes(source_loader)   # once before TTA
      for batch in target_loader:
          output = adapter(batch['image'].cuda(), label=batch['label'].cuda())
    """

    def __init__(self, model: nn.Module, cfg: dict):
        super().__init__()
        self.cfg         = cfg
        self.num_classes = cfg["num_classes"]

        # Three model copies (student, teacher, anchor)
        self.student = configure_model_for_tta(
            deepcopy(model), bn_momentum=cfg.get("bn_momentum", 0.01))
        self.teacher = self._build_ema(model)
        self.anchor  = deepcopy(model).cuda()
        self.anchor.requires_grad_(False)
        self.anchor.eval()

        # Initial state snapshot for regularizer R(θ)
        self.init_state = deepcopy(self.student.state_dict())

        # Optimizer — only RobustBN1d params are trainable
        trainable = [p for p in self.student.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(
            trainable,
            lr=cfg.get("lr", 5e-5),
            weight_decay=cfg.get("weight_decay", 0.0),
        )

        # ── Paper hyperparameters (Sec. 4 and Sec. 5.2) ──────────────────
        # α_0: initial EMA update rate (Eq. 3 / Eq. 7)
        self.alpha_0     = cfg.get("alpha_0",    0.05)
        self.alpha       = self.alpha_0           # current α_t (adaptive)

        # λ_0: initial regularization weight (Eq. 2 / Eq. 7)
        self.lambda_0    = cfg.get("lambda_0",   10.0)

        # Anchor loss weight (Eq. 8)
        self.al_wgt      = cfg.get("al_wgt",     1.0)

        # R(θ): cosine similarity or L2 (paper default: cosine)
        self.regularizer = cfg.get("regularizer", "cosine")

        # L_CLS: self-training or cross-entropy (paper default: self-training)
        self.loss_func   = cfg.get("loss_func",  "sce")

        # Adaptive adjustment flags (Eq. 7)
        self.adaptive_lambda = cfg.get("adaptive_lambda", True)
        self.adaptive_alpha  = cfg.get("adaptive_alpha",  True)

        # Augmentation noise std
        self.noise_std = cfg.get("noise_std", 0.05)

        # Category-balanced memory bank (from RoTTA [61], used in PeTTA)
        self.sample_mem = PeTTAMemory(
            capacity=cfg.get("memory_size", 64),
            num_class=self.num_classes,
            lambda_t=cfg.get("lambda_t", 1.0),
            lambda_u=cfg.get("lambda_u", 1.0),
        )

        # Initialized after compute_source_prototypes()
        self.proto_mem  = None   # PrototypeMemory: tracks μ̂^y_t
        self.divg_score = None   # DivergenceScore: computes Eq. 6
        self.step       = 0

    def _build_ema(self, model: nn.Module) -> nn.Module:
        ema = configure_model_for_tta(
            deepcopy(model), bn_momentum=self.cfg.get("bn_momentum", 0.01))
        for p in ema.parameters():
            p.requires_grad_(False)
        return ema.cuda()

    # ─────────────────────────────────────────────────────────────────────
    # Source prototype computation (called once before TTA)
    # Computes μ^y_0 and diagonal Σ^y_0 for Mahalanobis distance (Eq. 6)
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def compute_source_prototypes(self, source_loader,
                                   cache_path: str = None,
                                   recompute: bool = False):
        """
        Pre-compute per-class empirical mean μ^y_0 and diagonal covariance
        Σ^y_0 of feature embeddings from the source dataset (Eq. 6).

        Uses ALL source samples (drop_last=False) to avoid ignoring the
        last partial batch. A clean shuffle=False loader is built internally
        from the source_loader's dataset if drop_last=True is detected.
        """
        mean_path = f"{cache_path}_mean.pth" if cache_path else None
        cov_path  = f"{cache_path}_cov.pth"  if cache_path else None

        if (not recompute and mean_path
                and os.path.exists(mean_path) and os.path.exists(cov_path)):
            print("[PeTTA] Loading cached source prototypes ...")
            src_mean = torch.load(mean_path)
            src_cov  = torch.load(cov_path)
        else:
            print("[PeTTA] Computing source prototypes ...")

            # Build a clean loader that uses ALL source data (no drop_last)
            import torch.utils.data as D
            if hasattr(source_loader, 'dataset'):
                clean_loader = D.DataLoader(
                    source_loader.dataset,
                    batch_size=source_loader.batch_size or 64,
                    shuffle=False,
                    drop_last=False,   # use ALL samples
                    num_workers=0,
                )
            else:
                clean_loader = source_loader

            self.anchor.eval()
            all_feats, all_labels = [], []
            for x, y in clean_loader:
                x = x.cuda()
                f = self.anchor.feat(x)           # (B, D) feature embeddings
                all_feats.append(f.cpu())
                all_labels.append(y)
            all_feats  = torch.cat(all_feats,  dim=0)   # (N, D)
            all_labels = torch.cat(all_labels, dim=0)   # (N,)

            D_feat   = all_feats.size(1)
            src_mean = torch.zeros(self.num_classes, D_feat)   # μ^y_0
            src_cov  = torch.zeros(self.num_classes, D_feat)   # diag(Σ^y_0)

            for c in range(self.num_classes):
                mask = (all_labels == c)
                if mask.sum() == 0:
                    continue
                cf           = all_feats[mask]
                src_mean[c]  = cf.mean(dim=0)
                # Diagonal of covariance matrix Σ^y_0 (paper Eq. 6)
                if cf.size(0) > 1:
                    src_cov[c] = torch.diagonal(cf.T.cov())
                else:
                    src_cov[c] = torch.ones(D_feat)  # fallback for single sample

            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(src_mean, mean_path)
                torch.save(src_cov,  cov_path)
            print(f"[PeTTA] Source prototypes computed from {len(all_feats)} samples.")

        src_mean = src_mean.cuda()
        src_cov  = src_cov.cuda()

        # PrototypeMemory: maintains running mean μ̂^y_t (EMA updated)
        self.proto_mem  = PrototypeMemory(src_mean, self.num_classes)

        # DivergenceScore: computes per-class Mahalanobis distance (Eq. 6)
        self.divg_score = DivergenceScore(src_mean, src_cov)

    # ─────────────────────────────────────────────────────────────────────
    # Regularization R(θ): distance between student and initial weights
    # Paper allows cosine similarity or L2 (Sec. 4)
    # ─────────────────────────────────────────────────────────────────────

    def regularization_loss(self) -> torch.Tensor:
        reg = 0.0; count = 0
        for name, param in self.student.named_parameters():
            if not param.requires_grad:
                continue
            ref = self.init_state[name].cuda()
            if self.regularizer == "l2":
                reg += ((param - ref) ** 2).sum()
            elif self.regularizer == "cosine":
                reg += -F.cosine_similarity(
                    param[None, ...], ref[None, ...].cuda()).mean()
            count += 1
        return reg / max(count, 1)

    # ─────────────────────────────────────────────────────────────────────
    # EMA teacher update — Eq. 3: θ_t = (1 - α_t) θ_{t-1} + α_t θ'_t
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def update_ema(ema_model: nn.Module, model: nn.Module, alpha: float):
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data[:] = (1 - alpha) * ema_p.data[:] + alpha * p.data[:]

    # ─────────────────────────────────────────────────────────────────────
    # Core adaptation step — Algorithm 1 (Appendix E.1 of paper)
    # ─────────────────────────────────────────────────────────────────────

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data: torch.Tensor, label: torch.Tensor):
        assert self.proto_mem is not None, \
            "Call compute_source_prototypes() before running TTA."
        self.step += 1

        # ── Step 1: Pseudo-labels from teacher (Eq. 1) ───────────────────
        with torch.no_grad():
            self.teacher.eval()
            ema_feat   = self.teacher.feat(batch_data)       # (B, D)
            p_ema      = self.teacher.clsf(ema_feat)         # (B, C) logits
            predict    = torch.softmax(p_ema, dim=1)
            pseudo_lbl = torch.argmax(predict, dim=1)        # Ŷ_t
            entropy    = torch.sum(
                -predict * torch.log(predict + 1e-6), dim=1)

        # ── Step 2: Add samples to memory bank ───────────────────────────
        for i, data_i in enumerate(batch_data):
            self.sample_mem.add_instance(
                (data_i, pseudo_lbl[i].item(), entropy[i].item(), label[i]))

        sup_data, _ = self.sample_mem.get_memory()
        if len(sup_data) == 0:
            return p_ema
        sup_data = torch.stack(sup_data).cuda()              # (M, 1, F)

        # ── Step 3: Unique pseudo-labels Ŷ_t in batch ────────────────────
        lbl_uniq = torch.unique(pseudo_lbl)

        # ── Step 4: Divergence sensing — Mahalanobis distance (Eq. 6+7) ───
        # Per-class: d^y_t = (μ̂^y_t - μ^y_0)^T (Σ^y_0)^{-1} (μ̂^y_t - μ^y_0)
        # Per-class: γ^y_t = 1 - exp(-d^y_t)    ∈ [0, 1]
        # Batch avg: γ̄_t  = mean_{y ∈ Ŷ_t} γ^y_t
        # NOTE: exp() applied per-class BEFORE averaging (Eq. 7) — nonlinear!
        current_proto = self.proto_mem.mem_proto[lbl_uniq]         # (K, D)
        dist_y  = self.divg_score.per_class_distance(
            current_proto, lbl_uniq)                                # (K,)
        gamma_y = 1.0 - torch.exp(-dist_y)                         # (K,) ∈ [0,1]
        divg_scr = gamma_y.mean()                                   # γ̄_t scalar

        # ── Step 5: Update running prototype μ̂^y_t (Eq. 6) ──────────────
        self.proto_mem.update(feats=ema_feat.detach(), pseudo_lbls=pseudo_lbl)

        # ── Step 6: Adaptive λ_t and α_t (Eq. 7) ─────────────────────────
        # λ_t = γ̄_t · λ_0        → higher divergence = more regularization
        # α_t = (1 - γ̄_t) · α_0  → higher divergence = slower EMA update
        reg_wgt = self.lambda_0
        if self.adaptive_lambda:
            reg_wgt = divg_scr * self.lambda_0
        if self.adaptive_alpha:
            self.alpha = (1 - divg_scr) * self.alpha_0

        # ── Step 7: Student and teacher forward ──────────────────────────
        self.teacher.train()
        ema_sup_feat = self.teacher.feat(sup_data)
        x_ema        = self.teacher.clsf(ema_sup_feat)       # (M, C)

        self.student.train()
        self.anchor.eval()

        p_ori     = self.student(sup_data)                   # (M, C)
        init_feat = self.anchor.feat(sup_data)
        init_out  = self.anchor.clsf(init_feat)              # (M, C)

        sup_aug  = sensor_augment(sup_data, noise_std=self.noise_std)
        stu_feat = self.student.feat(sup_aug)
        p_aug    = self.student.clsf(stu_feat)               # (M, C)

        # ── Step 8: Classification loss L_CLS (Eq. 2) ────────────────────
        if self.loss_func == "sce":
            cls_lss = self_training(x=p_ori, x_aug=p_aug, x_ema=x_ema).mean()
        else:
            cls_lss = softmax_entropy(p_aug, x_ema).mean()

        # ── Step 9: Regularization R(θ) ──────────────────────────────────
        reg_lss = self.regularization_loss()

        # ── Step 10: Anchor loss L_AL (Eq. 8) ────────────────────────────
        # L_AL = -Σ_y Pr(y|X; θ_0) log Pr(y|X; θ)
        #      = D_KL(Pr(y|X_t; θ_0) || Pr(y|X_t; θ))   [paper equivalence]
        anchor_lss = softmax_entropy(p_aug, init_out).mean()

        # ── Step 11: Total loss (paper Eq. after Eq. 8) ──────────────────
        # θ'_t = Optim E_Pt[L_CLS + L_AL] + λ_t R(θ')
        total_lss = cls_lss + self.al_wgt * anchor_lss + reg_wgt * reg_lss

        self.optimizer.zero_grad()
        total_lss.backward()
        self.optimizer.step()

        # ── Step 12: EMA teacher update (Eq. 3) ──────────────────────────
        # θ_t = (1 - α_t) θ_{t-1} + α_t θ'_t
        self.update_ema(self.teacher, self.student, self.alpha)

        # Final prediction: UPDATED teacher f_t(X_t) — Algorithm 1 yields
        # the prediction AFTER the update, not before (f_{t-1}(X_t))
        self.teacher.eval()
        with torch.no_grad():
            out = self.teacher(batch_data)
        return out

    def forward(self, x: torch.Tensor, label: torch.Tensor = None):
        return self.forward_and_adapt(x, label)
