"""
sam.py — Semantic Adaptation Module
--------------------------------------
Wang et al. CVPR 2023 — Dynamically Instance-Guided Adaptation (DIGA)

Paper-faithful implementation matching Kbaier et al. Section 4.4, Eqs. 4-6:

  1. Instance prototypes q^c_t (Eq. 4): mean feature embedding of the
     top-K=32 most confident predictions per class, restricted to
     predictions with confidence >= P0 (fixed threshold, no adaptive
     per-class schedule — the paper uses a single global P0).
  2. Historical EMA (Eq. 5/7):
       q̄^c_t = ρ_P · q̄^c_{t-1} + (1 − ρ_P) · q^c_t
     ρ_P weights the OLD/historical prototype; (1 − ρ_P) weights the
     new instance prototype. With ρ_P = 0.10 (paper default), the
     historical prototype is replaced quickly by new evidence.
  3. Non-parametric prediction: cosine similarity between the feature
     embedding and each class's EMA prototype q̄^c_t. There is no further
     mixing step — q̄^c_t computed in step 2 IS the prototype used here.
  4. Classifier association / final fusion (Eq. 6):
       p = λ_F · p̃(c|x) + (1 − λ_F) · p̂(c|x)
     where p̃ is the non-parametric (SAM) output and p̂ is the parametric
     classifier output. λ_F = 0.80 → 80% weight on the SAM output.

Args:
  num_classes          : number of classes C
  feat_dim             : feature embedding dimension D
  fusion_lambda        : λ_F in Eq. 6, weight on the non-parametric (SAM)
                          output. Paper default = 0.80.
  confidence_threshold : P0 in Eq. 4, fixed global confidence threshold
                          for candidate samples. Paper default = 0.50.
  proto_rho            : ρ_P in Eq. 5/7, weight on the HISTORICAL
                          prototype in the EMA update. Paper default = 0.10.
  top_k                : K in Eq. 4, max number of most-confident samples
                          per class used to form the instance prototype.
                          Paper default = 32.

Paper: Sec. 3.2 Semantic Adaptation Module + Sec. 3.3 Classifier Association
"""

import torch
import torch.nn.functional as F


class SAM:
    """
    Semantic Adaptation Module (Wang et al. CVPR 2023).

    Maintains one EMA historical prototype per class (Eq. 5/7) and uses it
    directly, together with the parametric classifier, to produce the
    final fused prediction (Eq. 6).
    """

    def __init__(self,
                 num_classes:          int,
                 feat_dim:             int,
                 fusion_lambda:        float = 0.80,
                 confidence_threshold: float = 0.50,
                 proto_rho:            float = 0.10,
                 top_k:                int   = 32):
        self.num_classes          = num_classes
        self.feat_dim             = feat_dim
        self.fusion_lambda        = fusion_lambda
        self.confidence_threshold = confidence_threshold
        self.proto_rho            = proto_rho
        self.top_k                = top_k

        # Historical prototypes q̄^c_t (C, D) — initialized to zeros
        self.hist_proto = torch.zeros(num_classes, feat_dim)
        self.hist_init  = torch.zeros(num_classes, dtype=torch.bool)

    def _compute_instance_prototypes(
        self,
        feats:  torch.Tensor,   # (B, D)
        probs:  torch.Tensor,   # (B, C) softmax probabilities
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-class instance prototypes q^c_t (Eq. 4).

        Candidates: predictions with confidence >= confidence_threshold (P0).
        Among those candidates, only the top-K most confident per class are
        averaged into the instance prototype (paper: "the top-K = 32 most
        confident samples in the batch are used for prototype estimation").

        Returns:
          inst_proto : (C, D) instance prototypes
          inst_mask  : (C,) bool — which classes have at least one candidate
        """
        conf, pred = probs.max(dim=1)
        conf_cpu   = conf.detach().cpu()
        pred_cpu   = pred.detach().cpu()
        feats_cpu  = feats.detach().cpu()

        candidate_mask = conf_cpu >= self.confidence_threshold

        inst_proto = torch.zeros(self.num_classes, self.feat_dim)
        inst_mask  = torch.zeros(self.num_classes, dtype=torch.bool)

        for c in range(self.num_classes):
            c_idx = (candidate_mask & (pred_cpu == c)).nonzero(as_tuple=True)[0]
            if c_idx.numel() == 0:
                continue
            # Rank candidates for this class by confidence, keep top-K
            c_conf  = conf_cpu[c_idx]
            k       = min(self.top_k, c_idx.numel())
            top_idx = c_idx[torch.topk(c_conf, k).indices]
            inst_proto[c] = feats_cpu[top_idx].mean(dim=0)
            inst_mask[c]  = True

        return inst_proto, inst_mask

    def _update_historical_prototypes(
        self,
        inst_proto: torch.Tensor,  # (C, D)
        inst_mask:  torch.Tensor,  # (C,) bool
    ):
        """
        EMA update of historical prototypes (Eq. 5/7):
          q̄^c_t = ρ_P · q̄^c_{t-1} + (1 − ρ_P) · q^c_t
        ρ_P (proto_rho) weights HISTORY; (1 − ρ_P) weights the new
        instance prototype. Only classes with a confident instance
        prototype this batch are updated.
        """
        for c in range(self.num_classes):
            if not inst_mask[c]:
                continue
            if not self.hist_init[c]:
                # First time seeing this class: initialize directly
                self.hist_proto[c] = inst_proto[c].clone()
                self.hist_init[c]  = True
            else:
                self.hist_proto[c] = (
                    self.proto_rho * self.hist_proto[c]
                    + (1 - self.proto_rho) * inst_proto[c]
                )

    def predict(
        self,
        feats:         torch.Tensor,  # (B, D) feature embeddings
        logits_param:  torch.Tensor,  # (B, C) parametric classifier output
    ) -> torch.Tensor:
        """
        SAM prediction with classifier association (Eqs. 4-6).

        1. Compute instance prototypes q^c_t (Eq. 4)
        2. Update historical EMA prototypes q̄^c_t (Eq. 5/7)
        3. Non-parametric prediction via cosine similarity to q̄^c_t
        4. Fuse with parametric classifier output (Eq. 6)

        Returns: (B,) predicted class indices
        """
        probs = F.softmax(logits_param, dim=1)

        inst_proto, inst_mask = self._compute_instance_prototypes(feats, probs)
        self._update_historical_prototypes(inst_proto, inst_mask)

        if not self.hist_init.any():
            # No historical prototypes yet anywhere — fall back to parametric
            return logits_param.argmax(dim=1).cpu()

        # Non-parametric cosine similarity prediction using q̄^c_t directly
        # (no further re-mixing — Eq. 5/7's output IS the prototype used here)
        feats_n = F.normalize(feats.detach().cpu(), dim=1)          # (B, D)
        proto_n = F.normalize(self.hist_proto, dim=1)               # (C, D)
        sim_np  = torch.mm(feats_n, proto_n.t())                    # (B, C)

        # Classifier association — Eq. 6: p = λ_F · p̃ + (1 − λ_F) · p̂
        logits_cpu   = logits_param.detach().cpu()
        logits_final = (self.fusion_lambda * sim_np
                        + (1 - self.fusion_lambda) * logits_cpu)

        return logits_final.argmax(dim=1)
