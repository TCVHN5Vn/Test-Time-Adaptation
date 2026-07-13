"""
petta_memory.py
---------------
Exact copy of src/utils/petta_memory.py from the PeTTA repository.
Three classes:
  - DivergenceScore   : Mahalanobis-like divergence between current and source prototypes
  - PrototypeMemory   : EMA-updated per-class feature mean (mu_hat_t)
  - PeTTAMemory       : Category-balanced sample memory bank (from RoTTA)
"""

import torch
import torch.nn.functional as F
import math
from copy import deepcopy
from torch import nn


# ──────────────────────────────────────────────────────────────────────────────
# Helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_feat_mean(feats: torch.Tensor, pseudo_lbls: torch.Tensor):
    lbl_uniq  = torch.unique(pseudo_lbls)
    lbl_group = [torch.where(pseudo_lbls == l)[0] for l in lbl_uniq]
    group_avgs = []
    for lbl_idcs in lbl_group:
        group_avgs.append(feats[lbl_idcs].mean(axis=0).unsqueeze(0))
    return lbl_uniq, group_avgs


# ──────────────────────────────────────────────────────────────────────────────
# Divergence score  (Lines #8-#9 of Alg. 1)
# ──────────────────────────────────────────────────────────────────────────────

class DivergenceScore(nn.Module):
    """
    Computes per-class Mahalanobis distance between current feature means μ̂^y_t
    and source prototype means μ^y_0, using diagonal covariance Σ^y_0.

    Implements Eq. 6 + Eq. 7 of PeTTA (Hoang et al., NeurIPS 2024):

      Per-class distance (Eq. 6):
        d^y_t = (μ̂^y_t - μ^y_0)^T (Σ^y_0)^{-1} (μ̂^y_t - μ^y_0)
              = Σ_d (μ̂^y_{t,d} - μ^y_{0,d})^2 / σ^y_{0,d}^2   [diagonal Σ]

      Per-class divergence score (Eq. 6):
        γ^y_t = 1 - exp(-d^y_t)   ∈ [0, 1]

      Batch divergence score (Eq. 7):
        γ̄_t = mean_{y ∈ Ŷ_t} γ^y_t

    IMPORTANT: the exp() is applied per class BEFORE averaging.
    Averaging the raw distances before exp() is NOT equivalent (exp is nonlinear).

    per_class_distance() returns raw distances d^y_t (K,) for K unique classes.
    The caller applies 1-exp and averages to get γ̄_t.
    """

    def __init__(self, src_prototype: torch.Tensor, src_prototype_cov: torch.Tensor):
        super().__init__()
        self.src_proto     = src_prototype       # (C, D)
        self.src_proto_cov = src_prototype_cov   # (C, D)

    def per_class_distance(self, current_proto: torch.Tensor,
                            class_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute per-class Mahalanobis distance (un-exponentiated).

        Args:
            current_proto : (K, D) current running means μ̂^y_t for K classes
            class_ids     : (K,)   class indices

        Returns:
            dist : (K,) per-class Mahalanobis distances d^y_t
        """
        diff = current_proto - self.src_proto[class_ids]          # (K, D)
        dist = (diff.pow(2) / (self.src_proto_cov[class_ids]
                               + 1e-6)).sum(dim=1)                 # (K,)
        return dist

    def forward(self, feats: torch.Tensor, pseudo_lbls: torch.Tensor) -> torch.Tensor:
        """Legacy forward — kept for compatibility. Returns γ̄_t directly."""
        lbl_uniq = torch.unique(pseudo_lbls)
        current_proto = torch.stack(
            [feats[pseudo_lbls == y].mean(0) for y in lbl_uniq])  # (K, D)
        dist   = self.per_class_distance(current_proto, lbl_uniq)  # (K,)
        gamma  = 1.0 - torch.exp(-dist)                            # (K,)
        return gamma.mean()                                         # γ̄_t


# ──────────────────────────────────────────────────────────────────────────────
# Prototype memory  (Line #10 of Alg. 1)
# ──────────────────────────────────────────────────────────────────────────────

class PrototypeMemory:
    """
    Stores and EMA-updates per-class feature means (mu_hat_t).
    Initialised from source prototypes (mu_0).
    """

    def __init__(self, src_prototype: torch.Tensor, num_classes: int) -> None:
        self.src_proto   = src_prototype.squeeze(1)        # (C, D)
        self.mem_proto   = deepcopy(self.src_proto)        # (C, D)  – will be updated
        self.num_classes = num_classes
        self.src_proto_l2 = torch.cdist(self.src_proto, self.src_proto, p=2)

    def update(self, feats: torch.Tensor, pseudo_lbls: torch.Tensor, nu: float = 0.05):
        """EMA update of mu_hat_t  (Line #10 of Alg. 1)."""
        lbl_uniq  = torch.unique(pseudo_lbls)
        lbl_group = [torch.where(pseudo_lbls == l)[0] for l in lbl_uniq]
        for i, lbl_idcs in enumerate(lbl_group):
            psd_lbl   = lbl_uniq[i]
            batch_avg = feats[lbl_idcs].mean(axis=0)
            self.mem_proto[psd_lbl] = (
                (1 - nu) * self.mem_proto[psd_lbl] + nu * batch_avg
            )

    def get_mem_prototype(self) -> torch.Tensor:
        return self.mem_proto


# ──────────────────────────────────────────────────────────────────────────────
# Sample memory bank  (adapted from RoTTA)
# ──────────────────────────────────────────────────────────────────────────────

class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0, true_label=None):
        self.data        = data
        self.uncertainty = uncertainty
        self.age         = age
        self.true_label  = true_label

    def increase_age(self):
        if not self.empty():
            self.age += 1

    def get_data(self):
        return self.data, self.uncertainty, self.age

    def empty(self):
        return self.data == "empty"


class PeTTAMemory:
    """
    Category-balanced sample memory bank (adapted from RoTTA / Alg. 1 of PeTTA).
    Keeps at most `capacity` samples, balanced across `num_class` classes,
    using a heuristic score based on age and prediction uncertainty.
    """

    def __init__(
        self,
        capacity:   int,
        num_class:  int,
        lambda_t:   float = 1.0,
        lambda_u:   float = 1.0,
    ):
        self.capacity  = capacity
        self.num_class = num_class
        self.per_class = self.capacity / self.num_class
        self.lambda_t  = lambda_t
        self.lambda_u  = lambda_u
        self.data: list[list[MemoryItem]] = [[] for _ in range(self.num_class)]

    # ── Occupancy helpers ────────────────────────────────────────────────

    def get_occupancy(self) -> int:
        return sum(len(d) for d in self.data)

    def get_accuracy(self) -> float:
        acc = 0.0
        for clss, dat in enumerate(self.data):
            for item in dat:
                acc += (item.true_label == clss).item()
        return acc / (self.get_occupancy() + 1e-6)

    def per_class_dist(self) -> list[int]:
        return [len(cls_list) for cls_list in self.data]

    # ── Insertion / removal ──────────────────────────────────────────────

    def add_instance(self, instance):
        assert len(instance) == 4
        x, prediction, uncertainty, true_label = instance
        new_item  = MemoryItem(data=x, uncertainty=uncertainty, age=0,
                               true_label=true_label)
        new_score = self.heuristic_score(0, uncertainty)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self.add_age()

    def remove_instance(self, cls: int, score: float) -> bool:
        class_list     = self.data[cls]
        class_occupied = len(class_list)
        all_occupancy  = self.get_occupancy()
        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            else:
                majority_classes = self.get_majority_classes()
                return self.remove_from_classes(majority_classes, score)
        else:
            return self.remove_from_classes([cls], score)

    def remove_from_classes(self, classes: list[int], score_base: float) -> bool:
        max_class = max_index = max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                score = self.heuristic_score(item.age, item.uncertainty)
                if max_score is None or score >= max_score:
                    max_score = score
                    max_index = idx
                    max_class = cls
        if max_class is not None:
            if max_score > score_base:
                d = self.data[max_class].pop(max_index)
                del d
                return True
            else:
                return False
        else:
            return True

    def get_majority_classes(self) -> list[int]:
        dist        = self.per_class_dist()
        max_occupied = max(dist)
        return [i for i, occ in enumerate(dist) if occ == max_occupied]

    # ── Scoring ──────────────────────────────────────────────────────────

    def heuristic_score(self, age: int, uncertainty: float) -> float:
        return (
            self.lambda_t * 1 / (1 + math.exp(-age / self.capacity))
            + self.lambda_u * uncertainty / math.log(self.num_class)
        )

    def add_age(self):
        for cls_list in self.data:
            for item in cls_list:
                item.increase_age()

    # ── Retrieval ────────────────────────────────────────────────────────

    def get_memory(self):
        tmp_data = []
        tmp_age  = []
        for cls_list in self.data:
            for item in cls_list:
                tmp_data.append(item.data)
                tmp_age.append(item.age)
        tmp_age = [a / self.capacity for a in tmp_age]
        return tmp_data, tmp_age
