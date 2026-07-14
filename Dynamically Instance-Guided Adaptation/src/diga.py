"""
diga.py — Dynamically Instance-Guided Adaptation
--------------------------------------------------
Wang et al. CVPR 2023 — CVPR paper: "Dynamically Instance-Guided Adaptation:
A Backward-Free Approach for Test-Time Domain Adaptive Semantic Segmentation"

DIGA is a backward-free TTA method — NO gradient computation, NO optimizer.
Adaptation is done purely through:
  1. DAM: Distribution Adaptation Module (BN statistics mixing)
  2. SAM: Semantic Adaptation Module (prototype-based classifier)

Algorithm (per batch):
  1. Replace all BatchNorm1d with DAM_BN (done once at init)
  2. For each test batch x:
     a. Forward pass through DAM-equipped model → mixed BN statistics
     b. Extract feature embeddings z = feat(x)
     c. Compute parametric logits = clsf(z)
     d. SAM.predict(z, logits) → update prototypes + fuse predictions
     e. Return final predictions

Retained hyperparameters (Kbaier et al., Sec. 4.4 grid search winner):
  bn_lambda            = 0.95  BN mixing weight, source vs instance (Eq. 2-3)
  fusion_lambda         = 0.80  classifier fusion, parametric vs non-parametric (Eq. 6)
  confidence_threshold = 0.50  fixed candidate-selection threshold P0 (Eq. 4)
  proto_rho            = 0.10  EMA weight on HISTORY in the prototype update (Eq. 5/7)
  top_k                = 32    max confident samples per class used per batch (Eq. 4)

Adaptation for tabular sensor data:
  - BatchNorm1d used instead of BatchNorm2d (for MLP instead of CNN)
  - No image-specific augmentations needed (DAM and SAM are feature-level)
  - All other algorithmic details follow the paper exactly
"""

import torch
import torch.nn as nn
from copy import deepcopy

from src.dam import DAM_BN
from src.sam import SAM


class DIGA:
    """
    DIGA: Dynamically Instance-Guided Adaptation (Wang et al. CVPR 2023).

    Backward-free test-time adaptation using:
      - DAM: mixed BN statistics (source + instance)
      - SAM: dynamic non-parametric classifier with prototype mixing

    Usage:
        diga = DIGA(source_model, num_classes=5, feat_dim=64)
        for batch in target_loader:
            preds = diga.predict(batch['image'].to(device))
    """

    DEFAULTS = {
        'bn_lambda':            0.95,
        'fusion_lambda':        0.80,
        'confidence_threshold': 0.50,
        'proto_rho':            0.10,
        'top_k':                32,
    }

    def __init__(self,
                 source_model: nn.Module,
                 num_classes:  int,
                 feat_dim:     int = 64,
                 cfg:          dict = None):
        """
        Args:
            source_model : trained source model (must have .feat() and .clsf() methods)
            num_classes  : number of output classes
            feat_dim     : feature embedding dimension
            cfg          : hyperparameter overrides (see DEFAULTS)
        """
        self.cfg = {**self.DEFAULTS, **(cfg or {})}

        # Deep copy model and install DAM (no gradients needed)
        self.model = deepcopy(source_model).eval()
        self._install_dam()

        # SAM: semantic adaptation module
        self.sam = SAM(
            num_classes          = num_classes,
            feat_dim             = feat_dim,
            fusion_lambda        = self.cfg['fusion_lambda'],
            confidence_threshold = self.cfg['confidence_threshold'],
            proto_rho            = self.cfg['proto_rho'],
            top_k                = self.cfg['top_k'],
        )

        print(f'[DIGA] bn_lambda={self.cfg["bn_lambda"]}  '
              f'fusion_lambda={self.cfg["fusion_lambda"]}  '
              f'conf_thresh={self.cfg["confidence_threshold"]}  '
              f'proto_rho={self.cfg["proto_rho"]}  '
              f'top_k={self.cfg["top_k"]}')

    def _install_dam(self):
        """Replace all BatchNorm1d layers with DAM_BN."""
        bn_names = [
            name for name, module in self.model.named_modules()
            if isinstance(module, nn.BatchNorm1d)
        ]
        for name in bn_names:
            parts  = name.split('.')
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            old_bn = getattr(parent, parts[-1])
            setattr(parent, parts[-1], DAM_BN(old_bn, self.cfg['bn_lambda']))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels for a batch of test samples.
        Updates SAM prototypes as a side effect.

        Args:
            x: (B, 1, F) input tensor on device

        Returns:
            preds: (B,) predicted class indices (CPU)
        """
        self.model.eval()

        # Forward through DAM-equipped model
        feats  = self.model.feat(x)      # (B, D) — DAM applied inside feat
        logits = self.model.clsf(feats)  # (B, C) — parametric classifier

        # SAM: update prototypes + fuse predictions
        preds = self.sam.predict(feats, logits)

        return preds.cpu()
