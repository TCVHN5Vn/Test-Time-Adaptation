"""
dam.py — Distribution Adaptation Module
-----------------------------------------
Wang et al. CVPR 2023 — Dynamically Instance-Guided Adaptation (DIGA)

DAM replaces each BatchNorm1d layer with a mixed normalization layer that
combines source domain statistics (stored running mean/var) with instance
statistics (computed from the current test batch).

BN_DAM(x) = weight * [bn_lambda * src_norm + (1 - bn_lambda) * ins_norm] + bias

where:
  src_norm = (x - mu_src) / sqrt(var_src + eps)    source statistics (frozen)
  ins_norm = (x - mu_ins) / sqrt(var_ins + eps)    instance statistics (current batch)
  bn_lambda in [0, 1]:
    0 → use instance statistics only (= AdaBN)
    1 → use source statistics only (= standard BN inference)
    0.8 → paper default: 80% source, 20% instance  (eq. 3 of Wang et al. CVPR 2023)

Note: the paper's λ_BN weights the SOURCE statistics, i.e.:
  μ̄_T = λ_BN · μ̄_S + (1 − λ_BN) · μ_T   (eq. 3)

Paper: Sec. 3.1 Distribution Adaptation Module
"""

import torch
import torch.nn as nn


class DAM_BN(nn.Module):
    """
    Distribution Adaptation Module BN layer.
    Replaces BatchNorm1d during test-time adaptation.

    Mixes source domain BN statistics with current instance statistics:
      out = bn_lambda * src_norm + (1 - bn_lambda) * ins_norm    ← λ_BN=0.8 → 80% source, 20% instance
      return weight * out + bias

    Args:
        bn       : original BatchNorm1d layer (source statistics extracted)
        bn_lambda: mixing weight ∈ [0,1], default=0.8 (paper default)
    """
    def __init__(self, bn: nn.BatchNorm1d, bn_lambda: float = 0.8):
        super().__init__()
        self.bn_lambda = bn_lambda
        self.eps       = bn.eps

        # Trainable affine parameters (kept from source model)
        self.weight = nn.Parameter(bn.weight.data.clone())
        self.bias   = nn.Parameter(bn.bias.data.clone())

        # Frozen source domain statistics
        self.register_buffer('src_mean', bn.running_mean.clone())
        self.register_buffer('src_var',  bn.running_var.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Source domain normalized output
        src_norm = (x - self.src_mean) / torch.sqrt(self.src_var + self.eps)

        # Instance (current batch) normalized output
        if x.size(0) > 1:
            ins_mean = x.mean(dim=0)
            ins_var  = x.var(dim=0, unbiased=False)
        else:
            # Single sample: fall back to source statistics
            ins_mean = self.src_mean
            ins_var  = self.src_var
        ins_norm = (x - ins_mean) / torch.sqrt(ins_var + self.eps)

        # Mix: DAM output — paper eq(3): μ̄_T = λ_BN · μ̄_S + (1 − λ_BN) · μ_T
        # bn_lambda weights the SOURCE statistics (λ_BN=0.8 → 80% source, 20% instance)
        out = self.bn_lambda * src_norm + (1.0 - self.bn_lambda) * ins_norm
        return self.weight * out + self.bias
