"""
dirichlet_stream.py
--------------------
Recurring TTA stream with Dirichlet-based batch mixing.

Implements a HAR-adapted recurring TTA protocol inspired by PeTTA (Hoang et al., NeurIPS 2024) Sec. 5.2.
This is not a dataset-level reproduction of the CIFAR-C / ImageNet-C benchmarks —
it is an adaptation of the recurring-TTA evaluation spirit to smart-building sensor data.
smart-building HAR sensor data.

Key design decisions (aligned with the paper's recurring-TTA spirit):
  1. DOMAIN DEFINITION
     Each domain = one (year_day, time_block) pair, where time_block is:
       - 'morning'   : 06:00 – 11:59
       - 'afternoon' : 12:00 – 17:59
       - 'evening'   : 18:00 – 23:59
       - 'night'     : 00:00 – 05:59
     This gives finer-grained domains than whole days alone.

  2. DIRICHLET BATCH MIXING
     Each batch is a mixture of samples from 2-4 *nearby* domains,
     with mixing weights drawn from Dirichlet(beta).
     - Small beta (e.g. 0.1): batches dominated by one domain → strong correlation
     - Large beta (e.g. 1.0): uniform mix across domains
     This avoids trivial "same day over and over" replay.

  3. RECURRING STREAM (not repeated loader)
     - Domains are ordered chronologically: D1 → D2 → ... → DD
     - Each visit generates fresh Dirichlet-sampled batches from the same ordered
       domain sequence — so order is preserved but intra-batch composition varies
     - This is aligned with the paper's recurring TTA spirit

  4. EVALUATION
     Evaluation is done on ALL target samples (sorted, no Dirichlet) after
     each visit's adaptation pass — clean separation of adaptation and evaluation.
"""

import numpy as np
import torch
from typing import List, Dict, Any


# ─────────────────────────────────────────────────────────────────────────────
# Domain definition
# ─────────────────────────────────────────────────────────────────────────────

def get_time_block(time_val) -> str:
    """
    Map a time value (seconds since midnight, or HH:MM:SS string) to a
    coarse time-of-day block.
    """
    try:
        if isinstance(time_val, str):
            parts = time_val.split(':')
            hour  = int(parts[0])
        else:
            # Assume seconds since midnight
            hour = int(time_val) // 3600
        if 6  <= hour < 12: return 'morning'
        if 12 <= hour < 18: return 'afternoon'
        if 18 <= hour < 24: return 'evening'
        return 'night'
    except Exception:
        return 'unknown'


def build_domains(df_target, union_cols: list, activity_map: dict) -> List[Dict]:
    """
    Split target data into (year_day, time_block) domains, ordered
    chronologically. Each domain contains all samples from that
    day × time-block combination.

    Returns:
        List of domain dicts, each with:
          - domain_id  : int
          - day        : int (year_day)
          - time_block : str
          - label_str  : str e.g. "day42_morning"
          - X          : np.ndarray (N, F) float32
          - y          : np.ndarray (N,)   int64
          - n_samples  : int
    """
    import pandas as pd

    df = df_target.copy()
    df['label'] = df['activity'].map(activity_map).astype(int)
    df = df.sort_values(['year_day', 'time']).reset_index(drop=True)
    df['time_block'] = df['time'].apply(get_time_block)

    domains = []
    domain_id = 0
    for (day, tblock), grp in df.groupby(['year_day', 'time_block'], sort=False):
        grp = grp.sort_values('time').reset_index(drop=True)
        X   = grp[union_cols].values.astype(np.float32)
        y   = grp['label'].values.astype(np.int64)
        if len(X) < 4:
            continue  # skip tiny domains
        domains.append({
            'domain_id':  domain_id,
            'day':        day,
            'time_block': tblock,
            'label_str':  f'day{day}_{tblock}',
            'X':          X,
            'y':          y,
            'n_samples':  len(X),
        })
        domain_id += 1

    # Sort chronologically
    block_order = {'night': 0, 'morning': 1, 'afternoon': 2, 'evening': 3}
    domains.sort(key=lambda d: (d['day'], block_order.get(d['time_block'], 9)))

    print(f'[Dirichlet] {len(domains)} domains  '
          f'(days={len(df["year_day"].unique())}  blocks per day≈'
          f'{len(domains)//len(df["year_day"].unique())})')
    for d in domains[:4]:
        print(f'  {d["label_str"]:25s}  n={d["n_samples"]:4d}')
    if len(domains) > 4:
        print(f'  ... ({len(domains)-4} more)')

    return domains


# ─────────────────────────────────────────────────────────────────────────────
# Dirichlet batch builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_mixed_batch(
    domains:      List[Dict],
    active_ids:   List[int],
    batch_size:   int,
    beta:         float,
    rng:          np.random.Generator,
) -> Dict[str, torch.Tensor]:
    """
    Build one batch by Dirichlet-mixing samples from active_ids domains.

    Steps:
      1. Sample mixing proportions p ~ Dirichlet(beta * ones(K))
      2. n_k = round(p_k * batch_size) samples from domain k
      3. Shuffle the combined batch
    """
    K     = len(active_ids)
    props = rng.dirichlet(beta * np.ones(K))
    counts = np.round(props * batch_size).astype(int)
    # Fix rounding to exactly batch_size
    diff = batch_size - counts.sum()
    counts[counts.argmax()] += diff
    counts = np.maximum(counts, 0)

    X_parts, y_parts = [], []
    for k, did in enumerate(active_ids):
        n = int(counts[k])
        if n == 0:
            continue
        dom = domains[did]
        # Sample with replacement if needed
        idx = rng.choice(dom['n_samples'], size=n, replace=(n > dom['n_samples']))
        X_parts.append(dom['X'][idx])
        y_parts.append(dom['y'][idx])

    if not X_parts:
        return None

    X_all = np.concatenate(X_parts, axis=0)  # (B, F)
    y_all = np.concatenate(y_parts, axis=0)  # (B,)

    # Shuffle
    perm  = rng.permutation(len(X_all))
    X_all = X_all[perm]
    y_all = y_all[perm]

    return {
        'X': torch.from_numpy(X_all[:, np.newaxis, :]).float(),  # (B, 1, F)
        'y': torch.from_numpy(y_all).long(),                      # (B,)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main stream builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dirichlet_stream(
    domains:         List[Dict],
    batch_size:      int   = 64,
    beta:            float = 0.1,
    num_visits:      int   = 20,
    window_size:     int   = 3,
    batches_per_dom: int   = 3,
    seed:            int   = 42,
) -> List[Dict[str, Any]]:
    """
    Build the full recurring TTA stream with Dirichlet-mixed batches.

    For each visit:
      - Traverse domains in chronological order D1 → D2 → ... → DD
      - At each domain Di, create `batches_per_dom` batches mixed from
        Di and its `window_size-1` nearest neighbors (nearby domains)
      - Each batch's class composition is sampled from Dirichlet(beta)

    This gives:
      - Temporal correlation (nearby domains mixed together)
      - Class distribution shift (Dirichlet mixing)
      - True recurrence (same domain order across K visits, fresh mixing each time)

    Args:
        domains        : output of build_domains()
        batch_size     : samples per batch
        beta           : Dirichlet concentration (0.1 = paper-inspired default)
        num_visits     : number of recurring passes (K=20, paper-inspired default)
        window_size    : how many nearby domains to mix (2-4 recommended)
        batches_per_dom: batches generated per domain per visit
        seed           : base random seed

    Returns:
        List of stream items, each: {visit, domain_id, label_str, X, y}
    """
    rng    = np.random.default_rng(seed)
    D      = len(domains)
    stream = []

    total_batches = 0
    for visit in range(1, num_visits + 1):
        # Fresh seed per visit so batch composition varies across visits
        visit_rng = np.random.default_rng(seed + visit * 10000)

        for i, dom in enumerate(domains):
            # Select nearby domain window: [i-w//2, ..., i+w//2]
            half    = window_size // 2
            lo      = max(0, i - half)
            hi      = min(D, i + half + 1)
            active  = list(range(lo, hi))

            for _ in range(batches_per_dom):
                batch = _build_mixed_batch(
                    domains    = domains,
                    active_ids = active,
                    batch_size = batch_size,
                    beta       = beta,
                    rng        = visit_rng,
                )
                if batch is None:
                    continue
                stream.append({
                    'visit':      visit,
                    'domain_id':  dom['domain_id'],
                    'label_str':  dom['label_str'],
                    'X':          batch['X'],
                    'y':          batch['y'],
                })
                total_batches += 1

    print(f'\n[Dirichlet stream]')
    print(f'  Domains      : {D}  (day × time_block)')
    print(f'  Visits       : {num_visits}')
    print(f'  beta         : {beta}  ({"strong" if beta <= 0.2 else "moderate"} class correlation)')
    print(f'  Window size  : {window_size}  domains mixed per batch')
    print(f'  Total batches: {total_batches}  ({total_batches//num_visits} per visit)')

    return stream


def group_by_visit(stream: List[Dict]) -> Dict[int, List[Dict]]:
    """Group stream items by visit number."""
    grouped = {}
    for item in stream:
        v = item['visit']
        if v not in grouped:
            grouped[v] = []
        grouped[v].append(item)
    return grouped
