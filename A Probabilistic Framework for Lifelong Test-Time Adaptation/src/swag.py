"""
SWAG-Diagonal posterior approximation.
Maddox et al. 2019 — used in PETAL (Brahma & Rai, CVPR 2023).
"""

import torch
import torch.nn as nn
from copy import deepcopy


class SWAGD:
    def __init__(self, model: nn.Module, collect_freq: int = 4, eps: float = 1e-6):
        self.model        = model
        self.collect_freq = collect_freq
        self.eps          = eps
        self._snapshots   = []
        self.mu           = None
        self.sigma        = None
        self._finalized   = False

    def collect(self):
        flat = torch.cat([p.data.view(-1) for p in self.model.parameters()])
        self._snapshots.append(flat.detach().clone().cpu())

    def finalize(self):
        if len(self._snapshots) == 0:
            raise RuntimeError("No snapshots collected.")
        stacked    = torch.stack(self._snapshots, dim=0)
        self.mu    = stacked.mean(dim=0)
        var        = stacked.var(dim=0, unbiased=True)
        self.sigma = torch.sqrt(var + self.eps)
        self._finalized = True
        print(f"[SWAG-D] Finalized: {len(self._snapshots)} snapshots  "
              f"mu_norm={self.mu.norm():.4f}  mean_sigma={self.sigma.mean():.6f}")

    def get_map_model(self) -> nn.Module:
        if not self._finalized:
            raise RuntimeError("Call finalize() first.")
        map_model = deepcopy(self.model)
        offset = 0
        for p in map_model.parameters():
            n = p.numel()
            p.data.copy_(self.mu[offset:offset+n].view_as(p.data))
            offset += n
        return map_model

    def log_q(self, model: nn.Module, device='cpu') -> torch.Tensor:
        """log q(θ) = -0.5 * Σ_p (θ_p - μ_p)² / σ_p²   (Eq. 9 regularizer)"""
        if not self._finalized:
            raise RuntimeError("Call finalize() first.")
        mu    = self.mu.to(device)
        sigma = self.sigma.to(device)
        theta = torch.cat([p.view(-1) for p in model.parameters()])
        return -0.5 * ((theta - mu) ** 2 / (sigma ** 2)).sum()

    def save(self, path: str):
        torch.save({'mu': self.mu, 'sigma': self.sigma,
                    'snapshots': self._snapshots,
                    'finalized': self._finalized,
                    'collect_freq': self.collect_freq}, path)
        print(f"[SWAG-D] Saved → {path}")

    def load(self, path: str):
        d = torch.load(path, map_location='cpu', weights_only=False)
        self.mu           = d['mu']
        self.sigma        = d['sigma']
        self._snapshots   = d['snapshots']
        self._finalized   = d['finalized']
        self.collect_freq = d['collect_freq']
        print(f"[SWAG-D] Loaded ← {path}  snapshots={len(self._snapshots)}")
