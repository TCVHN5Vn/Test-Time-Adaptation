"""
PETAL Adapter — Algorithm 2 from Brahma & Rai CVPR 2023 (paper-faithful).

Per batch:
  1. Pseudo-labels from teacher (Eq. 2):
       y' = ŷ'  if conf(θ_0, x) >= τ   → plain teacher
       y' = ỹ'  otherwise               → augmentation-averaged teacher (K=32)
  2. STORE prediction from y_prime BEFORE adaptation (online evaluation)
  3. Loss (Eq. 9): minimise -(log q(θ) - λ̄ * Hxe(y', y_student))
  4. Update student with Adam
  5. Update teacher via EMA: θ' ← π*θ' + (1-π)*θ
  6. FIM-based restoration of student (Eq. 12):
       restore parameters where F_p < quantile(F, δ) back to θ_0
  7. Return stored pre-update predictions

Note: Step 6 in the paper (Fix 1 in the old code) — re-anchoring the
teacher after FIM restore — is NOT in Algorithm 2 of the paper and has
been removed for paper faithfulness.

Augmentation note:
  The paper uses image augmentations (random crops, flips, colour jitter).
  augment_tabular() below uses Gaussian noise + random feature dropout —
  a modality-specific adaptation for tabular sensor data.  K=32 is kept
  aligned with the paper.  pi, tau, lr, delta, and lambda_bar below are
  the retained grid-search winner reported in Kbaier et al. Sec. 4.5,
  not universal paper defaults.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from src.swag import SWAGD


def augment_tabular(x: torch.Tensor, noise_std=0.1, drop_rate=0.1):
    """
    Modality-specific augmentation for tabular/sensor data.

    Randomly applies one of:
      - Gaussian noise:         x + N(0, noise_std²)
      - Random feature dropout: zero each feature with prob drop_rate

    NOTE: this is a tabular adaptation of PETAL's augmentation role,
    not the paper's image augmentation. Described as such in the paper.
    """
    shape  = x.shape
    x_flat = x.view(x.size(0), -1).clone()
    if torch.rand(1).item() < 0.5:
        x_flat = x_flat + torch.randn_like(x_flat) * noise_std
    else:
        mask   = (torch.rand_like(x_flat) > drop_rate).float()
        x_flat = x_flat * mask
    return x_flat.view(shape)


class PETALAdapter:

    # Retained configuration (Kbaier et al., Sec. 4.5 grid search winner).
    # Selected from a 288-configuration grid:
    #   pi         in {0.9, 0.95, 0.99, 0.999}
    #   tau        in {0.5, 0.6, 0.7, 0.9}
    #   lr         in {1e-3, 1e-4, 5e-5}
    #   delta      in {0.03, 0.1, 0.2}
    #   lambda_bar in {0.0, 1.0}
    DEFAULTS = {
        'pi':           0.95,
        'K':            32,
        'lambda_bar':   0.0,
        'delta':        0.1,
        'tau':          0.5,
        'lr':           1e-4,
        'weight_decay': 1e-4,
        'noise_std':    0.1,
        'drop_rate':    0.1,
    }

    def __init__(self, source_model: nn.Module, swag: SWAGD,
                 cfg: dict = None, device: str = 'cpu'):
        self.cfg    = {**self.DEFAULTS, **(cfg or {})}
        self.device = device
        self.swag   = swag

        # θ_0: frozen source MAP for confidence estimation and restoration
        self.theta_0 = deepcopy(source_model).to(device).eval()
        for p in self.theta_0.parameters(): p.requires_grad_(False)

        # Student θ
        self.student = deepcopy(source_model).to(device)
        self.student.train()

        # Teacher θ' (EMA of student)
        self.teacher = deepcopy(source_model).to(device).eval()
        for p in self.teacher.parameters(): p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(
            self.student.parameters(),
            lr=self.cfg['lr'],
            weight_decay=self.cfg.get('weight_decay', 0.0))

        # Flat source weights for restoration (Eq. 12)
        self._theta_0_flat = torch.cat([
            p.data.view(-1) for p in self.theta_0.parameters()
        ]).to(device)

        print(f"[PETAL] π={self.cfg['pi']}  K={self.cfg['K']}  "
              f"λ̄={self.cfg['lambda_bar']}  δ={self.cfg['delta']}  "
              f"τ={self.cfg['tau']}  lr={self.cfg['lr']}  "
              f"weight_decay={self.cfg.get('weight_decay', 0.0)}")

    def adapt_and_predict(self, x: torch.Tensor) -> dict:
        """
        Paper-faithful PETAL adaptation step.

        Prediction timing follows Algorithm 2: y_prime is computed BEFORE
        the gradient update and returned as the online prediction, matching
        the paper's 'By = By ∪ {y'}' line which occurs before adaptation.

        Returns:
          'preds'  : (B,)    hard class indices from y_prime (CPU)
          'probs'  : (B, C)  softmax probabilities from y_prime (CPU)
        """
        x   = x.to(self.device)
        cfg = self.cfg
        pi, K             = cfg['pi'], cfg['K']
        lambda_bar, delta = cfg['lambda_bar'], cfg['delta']
        tau               = cfg['tau']

        # ── Step 1: Pseudo-labels y' (Eq. 2) ──────────────────────────────────
        with torch.no_grad():
            src_conf = F.softmax(self.theta_0(x), dim=1).max(1).values

            y_hat = F.softmax(self.teacher(x), dim=1)

            aug_sum = torch.zeros_like(y_hat)
            for _ in range(K):
                x_aug    = augment_tabular(x, cfg['noise_std'], cfg['drop_rate'])
                aug_sum += F.softmax(self.teacher(x_aug), dim=1)
            y_tilde = aug_sum / K

            use_plain = (src_conf >= tau).unsqueeze(1).float()
            y_prime   = use_plain * y_hat + (1 - use_plain) * y_tilde

        # ── Step 2: Store online prediction (before adaptation) ───────────────
        # Paper Algorithm 2: By = By ∪ {y'} happens before the gradient step.
        preds_out = y_prime.argmax(dim=1).cpu()
        probs_out = y_prime.cpu()

        # ── Step 3: Loss (Eq. 9) ──────────────────────────────────────────────
        self.optimizer.zero_grad()
        self.student.train()
        y_student = F.softmax(self.student(x), dim=1)

        H     = -(y_prime * torch.log(y_student + 1e-8)).sum(1).mean()
        log_q = self.swag.log_q(self.student, device=self.device)
        loss  = -(log_q - lambda_bar * H)
        loss.backward()

        # Collect gradients for FIM before optimizer step
        grads = torch.cat([p.grad.view(-1) if p.grad is not None
                           else torch.zeros(p.numel(), device=self.device)
                           for p in self.student.parameters()])
        fim = grads ** 2

        # ── Step 4: Update student ────────────────────────────────────────────
        self.optimizer.step()

        # ── Step 5: EMA teacher update ────────────────────────────────────────
        with torch.no_grad():
            for p_t, p_s in zip(self.teacher.parameters(),
                                 self.student.parameters()):
                p_t.data = pi * p_t.data + (1 - pi) * p_s.data

        # ── Step 6: FIM-based restoration of student (Eq. 12) ─────────────────
        # Restore low-importance parameters back to θ_0.
        # No second teacher EMA after this — not in the paper.
        with torch.no_grad():
            gamma  = torch.quantile(fim, delta)
            mask   = (fim < gamma).float()
            offset = 0
            for p in self.student.parameters():
                n       = p.numel()
                m_slice = mask[offset:offset+n].view_as(p.data)
                s_slice = self._theta_0_flat[offset:offset+n].view_as(p.data)
                p.data  = m_slice * s_slice + (1 - m_slice) * p.data
                offset += n

        return {
            'preds': preds_out,
            'probs': probs_out,
        }
