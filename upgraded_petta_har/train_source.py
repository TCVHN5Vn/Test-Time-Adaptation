"""
train_source.py  (fixed)
------------------------
Fix: class weights computed from actual training split, not hardcoded values.
"""

import os
import torch
import torch.nn as nn
from src.data_loader import get_loader
from src.model import build_model
from src.constants import NUM_CLASSES

CFG = {
    "source_path":  "data/S1.csv",
    "target_path":  "data/S2.csv",
    "scaler_path":  "checkpoints/scaler.pkl",
    "ckpt_path":    "checkpoints/best_model.pth",
    "window_size":  1,
    "batch_size":   64,
    "val_split":    0.2,
    "num_workers":  2,
    "seed":         42,
    "epochs":       150,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "patience":     20,
}

def train(cfg=CFG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")
    os.makedirs("checkpoints", exist_ok=True)

    train_loader, val_loader, _, meta = get_loader(
        source_path=cfg["source_path"],  target_path=cfg["target_path"],
        scaler_save_path=cfg["scaler_path"], batch_size=cfg["batch_size"],
        val_split=cfg["val_split"], window_size=cfg["window_size"],
        num_workers=cfg["num_workers"], seed=cfg["seed"],
    )
    len_features = meta["len_features"]

    model = build_model(len_features=len_features, num_classes=NUM_CLASSES).to(device)
    print(f"[train] Model params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"[train] Input size   : {len_features}")

    # FIX 4: compute class weights from actual split, not hardcoded counts
    all_labels = torch.cat([y for _, y in train_loader])
    counts = torch.zeros(NUM_CLASSES)
    for c in range(NUM_CLASSES):
        counts[c] = (all_labels == c).sum().float()
    counts  = counts.clamp(min=1.0)
    weights = (counts.sum() / (NUM_CLASSES * counts)).to(device)
    print(f"[train] Class counts : {counts.int().tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"])

    best_val_acc   = 0.0
    patience_count = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        tl = tc = tt = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tl += loss.item() * y.size(0)
            tc += (out.argmax(1) == y).sum().item()
            tt += y.size(0)
        scheduler.step()

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vc += (model(x).argmax(1) == y).sum().item()
                vt += y.size(0)
        val_acc = vc / vt

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{cfg['epochs']}  "
                  f"loss={tl/tt:.4f}  train_acc={tc/tt:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            patience_count = 0
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "val_acc": val_acc, "len_features": len_features,
                "num_classes": NUM_CLASSES, "cfg": cfg,
            }, cfg["ckpt_path"])
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                print(f"[train] Early stopping at epoch {epoch}.")
                break

    print(f"\n[train] Best val acc : {best_val_acc:.4f}")
    print(f"[train] Checkpoint   : {cfg['ckpt_path']}")
    return model, meta

if __name__ == "__main__":
    train()
