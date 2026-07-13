"""
run_petta.py
------------
Recurring Test-Time Adaptation on S2 using PeTTA.

Upgraded evaluation protocol:
  - Domains: (year_day, time_block) — finer than whole days
  - Batches: Dirichlet-mixed from 3 nearby domains (beta=0.1)
  - Recurrence: 20 visits over the same domain order, fresh mixing each visit
  - Metrics: visit-1, best, final, avg (paper-inspired Avg metric), retention gap

This follows a HAR-adapted recurring TTA protocol inspired by PeTTA (Hoang et al., NeurIPS 2024),
aligned with the paper's evaluation spirit but not a direct reproduction of its benchmarks
adapted for smart-building HAR sensor data.

Usage
-----
python run_petta.py
"""

import os
import pickle
import torch
import numpy as np
import pandas as pd

from src.data_loader         import get_loader, build_union_columns, align_features
from src.model               import build_model
from src.petta_adapter       import PeTTAAdapter
from src.constants           import NUM_CLASSES, CLASS_NAMES, ACTIVITY_MAP
from src.dirichlet_stream    import build_domains, build_dirichlet_stream, group_by_visit
from src.persistence_metrics import PersistenceReport


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CFG = {
    # Paths
    "source_path":   "data/S1.csv",
    "target_path":   "data/S2.csv",
    "scaler_path":   "checkpoints/scaler.pkl",
    "ckpt_path":     "checkpoints/best_model.pth",
    "output_dir":    "outputs",
    # Data
    "window_size":   1,
    "batch_size":    64,
    "val_split":     0.2,
    "num_workers":   2,
    "seed":          42,
    # PeTTA hyperparameters — retained config (Kbaier et al., Sec. 4.6)
    "num_classes":   NUM_CLASSES,
    "alpha_0":       0.05,
    "lambda_0":      10.0,
    "al_wgt":        1.0,
    "bn_momentum":   0.01,
    "memory_size":   64,
    "lambda_t":      0.25,
    "lambda_u":      0.50,
    "regularizer":   "cosine",
    "loss_func":     "sce",
    "adaptive_lambda": True,
    "adaptive_alpha":  True,
    "noise_std":     0.05,
    "lr":            5e-5,
    "weight_decay":  1e-4,
    # Recurring TTA stream (Dirichlet protocol)
    "num_visits":    20,     # K recurring visits (paper-inspired)
    "beta":          0.1,    # Dirichlet concentration (paper-inspired)
    "window_size_d": 3,      # nearby domains to mix per batch
    "batches_per_dom": 3,    # batches generated per domain per visit
}


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_petta(cfg: dict = CFG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[PeTTA] device = {device}")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # ── Step 1: Load data ─────────────────────────────────────────────────
    train_loader, val_loader, target_loader, meta = get_loader(
        source_path      = cfg["source_path"],
        target_path      = cfg["target_path"],
        scaler_save_path = cfg["scaler_path"],
        batch_size       = cfg["batch_size"],
        val_split        = cfg["val_split"],
        window_size      = cfg["window_size"],
        num_workers      = cfg["num_workers"],
        seed             = cfg["seed"],
    )

    # ── Step 2: Load source model ─────────────────────────────────────────
    ckpt  = torch.load(cfg["ckpt_path"], map_location=device, weights_only=False)
    model = build_model(
        len_features = ckpt["len_features"],
        num_classes  = ckpt["num_classes"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"[PeTTA] Source model loaded — val_acc={ckpt['val_acc']:.4f}")

    # ── Step 3: Source-only baseline ──────────────────────────────────────
    sp, sl = [], []
    with torch.no_grad():
        for batch in target_loader:
            x = batch["image"].to(device)
            sp.append(model(x).argmax(1).cpu())
            sl.append(batch["label"])
    sp     = torch.cat(sp).numpy()
    sl     = torch.cat(sl).numpy()
    src_acc = float((sp == sl).mean())
    print(f"[PeTTA] Source-only: acc={src_acc:.4f}  err={1-src_acc:.4f}\n")

    # ── Step 4: Build Dirichlet recurring stream ──────────────────────────
    # Load scaled target data for stream building
    df_s2 = pd.read_csv(cfg["target_path"])
    df_s1 = pd.read_csv(cfg["source_path"])
    union_cols = build_union_columns([df_s1, df_s2])
    df_s2      = align_features(df_s2, union_cols)

    with open(cfg["scaler_path"], "rb") as f:
        scaler = pickle.load(f)
    df_s2[union_cols] = scaler.transform(
        df_s2[union_cols].values.astype(np.float32))

    # Build (year_day, time_block) domains
    domains = build_domains(df_s2, union_cols, ACTIVITY_MAP)

    # Build Dirichlet stream: 20 visits, beta=0.1, window=3
    stream        = build_dirichlet_stream(
        domains         = domains,
        batch_size      = cfg["batch_size"],
        beta            = cfg["beta"],
        num_visits      = cfg["num_visits"],
        window_size     = cfg["window_size_d"],
        batches_per_dom = cfg["batches_per_dom"],
        seed            = cfg["seed"],
    )
    visit_batches = group_by_visit(stream)

    # ── Step 5: Build PeTTA adapter ───────────────────────────────────────
    adapter = PeTTAAdapter(model, cfg).to(device)
    # Build a full-source loader (train + val) for cleaner prototype statistics.
    # The paper (Appendix E.4) recommends computing source statistics from the
    # full source distribution you permit yourself to access.
    from src.data_loader import SensorDataset, align_features, build_union_columns
    import torch.utils.data as D_mod
    import pandas as pd_mod

    df_s1_full = pd.read_csv(cfg["source_path"])
    df_s1_full = align_features(df_s1_full, meta["union_cols"])
    df_s1_full[meta["union_cols"]] = meta["scaler"].transform(
        df_s1_full[meta["union_cols"]].values.astype(np.float32))
    src_full_ds     = SensorDataset(df_s1_full, meta["union_cols"], cfg["window_size"])
    src_full_loader = D_mod.DataLoader(
        src_full_ds, batch_size=cfg["batch_size"],
        shuffle=False, drop_last=False, num_workers=0)

    proto_cache = os.path.join(cfg["output_dir"], "protos")
    adapter.compute_source_prototypes(
        source_loader = src_full_loader,
        cache_path    = proto_cache,
        recompute     = False,
    )

    # ── Step 6: Recurring TTA with persistence reporting ─────────────────
    report = PersistenceReport(
        method_name     = "PeTTA (Dirichlet stream)",
        source_only_acc = src_acc,
    )

    print(f"{'Visit':>6}  {'Acc':>7}  {'BalAcc':>7}  "
          f"{'MF1':>7}  {'Kappa':>7}  {'Delta':>8}")
    print("-" * 62)

    for visit in range(1, cfg["num_visits"] + 1):

        # ── Adapt on Dirichlet-mixed batches ──────────────────────────────
        adapter.train()
        for item in visit_batches.get(visit, []):
            x = item["X"].to(device)
            y = item["y"].to(device)
            if x.size(0) < 2:
                continue
            adapter(x, label=y)

        # ── Evaluate on full S2 (clean sorted stream, no Dirichlet) ───────
        adapter.eval()
        pv, lv = [], []
        with torch.no_grad():
            for batch in target_loader:
                out = adapter.teacher(batch["image"].to(device))
                pv.append(out.argmax(1).cpu())
                lv.append(batch["label"])
        vp = torch.cat(pv).numpy()
        vl = torch.cat(lv).numpy()

        report.add_visit(visit, vp, vl)
        r = report.visit_results[-1]
        print(f"  {visit:>4}  {r.accuracy:>7.4f}  {r.balanced_acc:>7.4f}  "
              f"{r.macro_f1:>7.4f}  {r.kappa:>7.4f}  {r.delta:>+8.4f}")

    # ── Step 7: Compute and print persistence metrics ─────────────────────
    report.compute()
    print(report.summary_table())
    print(report.best_visit_report(CLASS_NAMES))

    # ── Step 8: Save results ──────────────────────────────────────────────
    import json, numpy as np

    # Text report
    report_path = os.path.join(cfg["output_dir"], "petta_report.txt")
    with open(report_path, "w") as f:
        f.write(report.summary_table() + "\n")
        f.write(report.best_visit_report(CLASS_NAMES))
    print(f"\n[PeTTA] Report saved → {report_path}")

    # JSON results
    json_path = os.path.join(cfg["output_dir"], "petta_results.json")
    results_d = report.to_dict()
    results_d["config"] = cfg
    with open(json_path, "w") as f:
        json.dump(results_d, f, indent=2)
    print(f"[PeTTA] JSON saved  → {json_path}")

    # Numpy predictions
    np.save(os.path.join(cfg["output_dir"], "best_preds.npy"),  report.best_preds)
    np.save(os.path.join(cfg["output_dir"], "ref_labels.npy"),  report._ref_labels)
    np.save(os.path.join(cfg["output_dir"], "all_preds.npy"),
            np.stack(report._all_preds))

    return report


if __name__ == "__main__":
    run_petta()
