"""
Source training with SGD + SWAG-D snapshot collection.
Phase 1: cosine LR decay (burn-in)
Phase 2: constant high LR + collect 4 snapshots/epoch
"""

import torch
import torch.nn as nn
from copy import deepcopy
from src.swag import SWAGD


def train_with_swag(train_loader, val_loader, model, criterion, cfg, device='cpu'):
    epochs       = cfg.get('epochs',       150)
    swag_epochs  = cfg.get('swag_epochs',  20)
    collect_freq = cfg.get('collect_freq', 4)
    lr           = cfg.get('lr',           0.01)
    swag_lr      = cfg.get('swag_lr',      0.01)
    momentum     = cfg.get('momentum',     0.9)
    weight_decay = cfg.get('weight_decay', 1e-4)
    patience_cfg = cfg.get('patience',     20)

    burn_in = epochs - swag_epochs
    swag    = SWAGD(model, collect_freq=collect_freq)

    # ── Phase 1: Burn-in with cosine LR ───────────────────────────────────────
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                 momentum=momentum, weight_decay=weight_decay,
                                 nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=burn_in)

    best_val = 0.0; patience = 0; best_state = None

    print(f"[SWAG] Phase 1 — burn-in  epochs={burn_in}  lr={lr}")
    for epoch in range(1, burn_in + 1):
        model.train()
        tl = tc = tt = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if y.size(0) < 2: continue
            optimizer.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            tl += loss.item()*y.size(0)
            tc += (out.argmax(1)==y).sum().item()
            tt += y.size(0)
        scheduler.step()

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vc += (model(x).argmax(1)==y).sum().item()
                vt += y.size(0)
        val_acc = vc/vt

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{burn_in}  loss={tl/tt:.4f}  "
                  f"train={tc/tt:.4f}  val={val_acc:.4f}")
        if val_acc > best_val:
            best_val = val_acc; patience = 0
            best_state = deepcopy(model.state_dict())
            print(f"  Saved val_acc={val_acc:.4f}")
        else:
            patience += 1
            if patience >= patience_cfg:
                print(f"  Early stop at epoch {epoch}"); break

    model.load_state_dict(best_state)
    print(f"\n[SWAG] Best burn-in val_acc={best_val:.4f}")

    # ── Phase 2: SWAG collection with constant high LR ────────────────────────
    print(f"\n[SWAG] Phase 2 — collection  swag_epochs={swag_epochs}  "
          f"collect_freq={collect_freq}  swag_lr={swag_lr}")

    opt_swag = torch.optim.SGD(model.parameters(), lr=swag_lr,
                                momentum=momentum, weight_decay=weight_decay,
                                nesterov=True)
    n_batches        = len(train_loader)
    collect_interval = max(1, n_batches // collect_freq)

    for epoch in range(1, swag_epochs + 1):
        model.train()
        tl = tc = tt = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            if y.size(0) < 2: continue
            opt_swag.zero_grad()
            out  = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt_swag.step()
            tl += loss.item()*y.size(0)
            tc += (out.argmax(1)==y).sum().item()
            tt += y.size(0)
            if (i+1) % collect_interval == 0:
                swag.collect()

        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                vc += (model(x).argmax(1)==y).sum().item()
                vt += y.size(0)
        val_acc = vc/vt
        print(f"  SWAG epoch {epoch:3d}/{swag_epochs}  loss={tl/tt:.4f}  "
              f"val={val_acc:.4f}  snapshots={len(swag._snapshots)}")

    swag.finalize()
    map_model = swag.get_map_model().to(device)
    map_model.eval()

    vc = vt = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            vc += (map_model(x).argmax(1)==y).sum().item()
            vt += y.size(0)
    print(f"\n[SWAG] MAP model val_acc={vc/vt:.4f}")
    return map_model, swag
