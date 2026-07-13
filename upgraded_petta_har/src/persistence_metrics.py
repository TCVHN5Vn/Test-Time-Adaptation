"""
persistence_metrics.py
-----------------------
Persistence metrics for recurring TTA evaluation.

Aligned with the PeTTA paper's (Hoang et al., NeurIPS 2024) recurring-TTA evaluation spirit:
  - Per-visit: accuracy, balanced accuracy, macro F1
  - Summary: best, final, average (Avg = paper's primary metric)
  - Persistence gap: best_visit_metric - final_visit_metric
  - Collapse detection: when acc drops below source-only

Paper Table 1 format:
  Method | v1 | v2 | ... | v20 | Avg
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from sklearn.metrics import (
    f1_score, cohen_kappa_score, balanced_accuracy_score,
    matthews_corrcoef, classification_report
)


@dataclass
class VisitResult:
    visit:        int
    accuracy:     float
    balanced_acc: float
    macro_f1:     float
    kappa:        float
    mcc:          float
    delta:        float   # vs source-only


@dataclass
class PersistenceReport:
    """
    Tracks per-visit metrics and computes persistence summary.
    Matches PeTTA paper Table 1 / Table 2 reporting format.
    """
    method_name:     str
    source_only_acc: float
    visit_results:   List[VisitResult] = field(default_factory=list)

    # Filled by compute()
    best_acc:          float         = 0.0
    best_visit:        int           = 0
    best_preds:        Optional[np.ndarray] = None
    final_acc:         float         = 0.0
    avg_acc:           float         = 0.0   # paper "Avg" metric
    stability:         float         = 0.0   # std of per-visit accuracy
    retention_gap:     float         = 0.0   # final_acc - best_acc; 0=stable, negative=degradation
    collapse_visit:    Optional[int] = None  # first visit where acc < source-only
    _all_preds:        list          = field(default_factory=list)
    _ref_labels:       Optional[np.ndarray] = None

    def add_visit(self, visit: int,
                  preds: np.ndarray, labels: np.ndarray):
        """Record results for one visit."""
        acc  = float((preds == labels).mean())
        bacc = float(balanced_accuracy_score(labels, preds))
        mf1  = float(f1_score(labels, preds, average='macro', zero_division=0))
        kap  = float(cohen_kappa_score(labels, preds))
        mcc  = float(matthews_corrcoef(labels, preds))

        self.visit_results.append(VisitResult(
            visit=visit, accuracy=acc, balanced_acc=bacc,
            macro_f1=mf1, kappa=kap, mcc=mcc,
            delta=acc - self.source_only_acc,
        ))
        self._all_preds.append(preds)
        if self._ref_labels is None:
            self._ref_labels = labels

        # Track collapse
        if acc < self.source_only_acc and self.collapse_visit is None:
            self.collapse_visit = visit

    def compute(self):
        """Compute aggregate persistence metrics after all visits."""
        if not self.visit_results:
            return self

        accs = [r.accuracy for r in self.visit_results]

        self.best_acc   = float(max(accs))
        self.best_visit = int(np.argmax(accs)) + 1
        self.best_preds = self._all_preds[self.best_visit - 1]
        self.final_acc  = accs[-1]
        self.avg_acc    = float(np.mean(accs))
        self.stability  = float(np.std(accs))

        # Persistence gap: how much worse is final vs best
        # Negative = performance degraded (bad), zero/positive = stable (good)
        self.retention_gap = self.final_acc - self.best_acc  # 0=stable, negative=degraded from peak

        return self

    def summary_table(self) -> str:
        """
        Print per-visit table in PeTTA paper Table 1 style.
        Columns: Visit | Acc | BalAcc | MF1 | Kappa | Delta
        """
        K     = len(self.visit_results)
        lines = []
        lines.append(f'\n{"="*72}')
        lines.append(f'Method          : {self.method_name}')
        lines.append(f'Source-only acc : {self.source_only_acc:.4f}')
        lines.append(f'{"="*72}')

        # Per-visit table
        lines.append(
            f'{"Visit":>6}  {"Acc":>7}  {"BalAcc":>7}  '
            f'{"MF1":>7}  {"Kappa":>7}  {"Delta":>8}')
        lines.append('-' * 62)
        for r in self.visit_results:
            lines.append(
                f'  {r.visit:>4}  {r.accuracy:>7.4f}  {r.balanced_acc:>7.4f}  '
                f'{r.macro_f1:>7.4f}  {r.kappa:>7.4f}  {r.delta:>+8.4f}')
        lines.append('-' * 62)

        # Summary row (paper "Avg")
        avg_bacc = np.mean([r.balanced_acc for r in self.visit_results])
        avg_mf1  = np.mean([r.macro_f1     for r in self.visit_results])
        avg_kap  = np.mean([r.kappa         for r in self.visit_results])
        lines.append(
            f'  {"Avg":>4}  {self.avg_acc:>7.4f}  {avg_bacc:>7.4f}  '
            f'{avg_mf1:>7.4f}  {avg_kap:>7.4f}')

        # Persistence summary
        lines.append(f'\n{"="*72}')
        lines.append('PERSISTENCE METRICS  (paper Table 1 format)')
        lines.append(f'{"="*72}')
        lines.append(f'  Visit 1 accuracy       : {self.visit_results[0].accuracy:.4f}')
        lines.append(f'  Best accuracy          : {self.best_acc:.4f}  (visit {self.best_visit})')
        lines.append(f'  Final accuracy         : {self.final_acc:.4f}  (visit {K})')
        lines.append(f'  Avg accuracy           : {self.avg_acc:.4f}  ← Avg metric used in PeTTA paper Table 1')
        lines.append(f'  Stability (std)        : {self.stability:.4f}  (lower = more stable)')
        lines.append(f'  Retention gap          : {self.retention_gap:+.4f}  '
                     f'(final - best; closer to 0 = more stable, negative = degraded from peak)')
        if self.collapse_visit:
            lines.append(f'  Collapse detected      : visit {self.collapse_visit}  ⚠')
        else:
            lines.append(f'  Collapse detected      : None  ✓  (never dropped below source-only)')
        lines.append(f'\n  Delta best vs source   : {self.best_acc - self.source_only_acc:+.4f} pp  '
                     f'/ {(self.best_acc - self.source_only_acc)/self.source_only_acc*100:+.2f}%')
        lines.append(f'  Delta avg  vs source   : {self.avg_acc  - self.source_only_acc:+.4f} pp  '
                     f'/ {(self.avg_acc  - self.source_only_acc)/self.source_only_acc*100:+.2f}%')
        lines.append(f'  Delta final vs source  : {self.final_acc - self.source_only_acc:+.4f} pp  '
                     f'/ {(self.final_acc - self.source_only_acc)/self.source_only_acc*100:+.2f}%')

        return '\n'.join(lines)

    def best_visit_report(self, class_names: list) -> str:
        """Full sklearn classification report for the best visit."""
        preds  = self.best_preds
        labels = self._ref_labels
        lines  = [
            f'\n{"="*60}',
            f'{self.method_name} — visit {self.best_visit} (best)',
            '='*60,
            classification_report(labels, preds,
                                   target_names=class_names, digits=4),
            f'Accuracy          : {self.best_acc:.4f}',
            f'Macro F1          : {f1_score(labels, preds, average="macro"):.4f}',
            f'Weighted F1       : {f1_score(labels, preds, average="weighted"):.4f}',
            f'Balanced Accuracy : {balanced_accuracy_score(labels, preds):.4f}',
            f'Cohen Kappa       : {cohen_kappa_score(labels, preds):.4f}',
            f'MCC               : {matthews_corrcoef(labels, preds):.4f}',
        ]
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON export."""
        return {
            'method_name':     self.method_name,
            'source_only_acc': self.source_only_acc,
            'best_acc':        self.best_acc,
            'best_visit':      self.best_visit,
            'final_acc':       self.final_acc,
            'avg_acc':         self.avg_acc,
            'stability':       self.stability,
            'retention_gap': self.retention_gap,  # closer to 0 = more stable; negative = degradation
            'collapse_visit':  self.collapse_visit,
            'delta_best_pp':   self.best_acc  - self.source_only_acc,
            'delta_avg_pp':    self.avg_acc   - self.source_only_acc,
            'delta_final_pp':  self.final_acc - self.source_only_acc,
            'visits': [{
                'visit':        r.visit,
                'accuracy':     r.accuracy,
                'balanced_acc': r.balanced_acc,
                'macro_f1':     r.macro_f1,
                'kappa':        r.kappa,
                'mcc':          r.mcc,
                'delta':        r.delta,
            } for r in self.visit_results],
        }
