"""
data_loader.py  (fixed)
-----------------------
Fixes vs original:
  1. Stratified ROW-level split instead of day-level split
     → guarantees all activity classes appear in both train and val
  2. .reshape() instead of .view() in __getitem__
     → prevents RuntimeError with non-contiguous numpy slices
  3. Last-label rule for windows instead of majority-vote
     → preserves rare/short activity classes
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.constants import ACTIVITY_MAP, NUM_CLASSES


def get_sensor_columns(df):
    return sorted([c for c in df.columns if c.startswith("sensor_id_")])


def build_union_columns(dfs):
    union = set()
    for df in dfs:
        union |= set(get_sensor_columns(df))
    return sorted(union)


def align_features(df, union_cols):
    df = df.copy()
    for col in union_cols:
        if col not in df.columns:
            df[col] = 0
    return df


def fit_scaler(X, save_path=None):
    scaler = StandardScaler()
    scaler.fit(X)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(scaler, f)
    return scaler


class SensorDataset(data.Dataset):
    def __init__(self, df, union_cols, window_size=1):
        self.union_cols  = union_cols
        self.window_size = window_size
        df = df.copy()
        df["label"] = df["activity"].map(ACTIVITY_MAP)
        df = df.dropna(subset=["label"]).reset_index(drop=True)
        df["label"] = df["label"].astype(int)
        self.X       = df[union_cols].values.astype(np.float32)
        self.y       = df["label"].values.astype(np.int64)
        self.indices = list(range(len(self.X) - self.window_size + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end   = start + self.window_size
        # FIX 3: last-label (not majority-vote) — preserves rare classes
        y = int(self.y[end - 1])
        # FIX 2: reshape not view — works on non-contiguous numpy slices
        x = torch.from_numpy(self.X[start:end]).reshape(1, -1)
        return x, torch.tensor(y, dtype=torch.long)


class StreamDataset(data.Dataset):
    def __init__(self, df, union_cols, window_size=1):
        self.union_cols  = union_cols
        self.window_size = window_size
        df = df.copy().sort_values(["year_day", "time"]).reset_index(drop=True)
        df["label"]  = df["activity"].map(ACTIVITY_MAP).astype(int)
        self.X       = df[union_cols].values.astype(np.float32)
        self.y       = df["label"].values.astype(np.int64)
        day_to_idx   = {d: i for i, d in enumerate(sorted(df["year_day"].unique()))}
        self.domains = np.array([day_to_idx[d] for d in df["year_day"].values])
        self.indices = list(range(len(self.X) - self.window_size + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start  = self.indices[idx]
        end    = start + self.window_size
        y      = int(self.y[end - 1])
        domain = int(self.domains[end - 1])
        return {
            "image":  torch.from_numpy(self.X[start:end]).reshape(1, -1),
            "label":  torch.tensor(y,      dtype=torch.long),
            "domain": torch.tensor(domain, dtype=torch.long),
        }


def get_loader(source_path, target_path,
               scaler_save_path="checkpoints/scaler.pkl",
               batch_size=64, val_split=0.2, window_size=1,
               num_workers=2, seed=42):

    df_s1 = pd.read_csv(source_path)
    df_s2 = pd.read_csv(target_path)

    union_cols   = build_union_columns([df_s1, df_s2])
    len_features = len(union_cols) * window_size

    df_s1 = align_features(df_s1, union_cols)
    df_s2 = align_features(df_s2, union_cols)

    # FIX 1: stratified row-level split → all classes in train AND val
    df_s1["_label"] = df_s1["activity"].map(ACTIVITY_MAP)
    train_idx, val_idx = train_test_split(
        df_s1.index, test_size=val_split,
        stratify=df_s1["_label"], random_state=seed)
    df_s1_train = df_s1.loc[train_idx].drop(columns=["_label"]).reset_index(drop=True)
    df_s1_val   = df_s1.loc[val_idx].drop(columns=["_label"]).reset_index(drop=True)

    X_train_raw = df_s1_train[union_cols].values.astype(np.float32)
    scaler = fit_scaler(X_train_raw, save_path=scaler_save_path)
    for df in [df_s1_train, df_s1_val, df_s2]:
        df[union_cols] = scaler.transform(df[union_cols].values.astype(np.float32))

    train_ds  = SensorDataset(df_s1_train, union_cols, window_size)
    val_ds    = SensorDataset(df_s1_val,   union_cols, window_size)
    target_ds = StreamDataset(df_s2,       union_cols, window_size)

    train_loader  = data.DataLoader(train_ds,  batch_size=batch_size,
                                    shuffle=True,  num_workers=num_workers, drop_last=True)
    val_loader    = data.DataLoader(val_ds,    batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    target_loader = data.DataLoader(target_ds, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)

    print(f"[data] Union sensors  : {len(union_cols)}")
    print(f"[data] len_features   : {len_features}  (window_size={window_size})")
    print(f"[data] S1 train rows  : {len(df_s1_train)}  ({len(train_ds)} samples)")
    print(f"[data] S1 val   rows  : {len(df_s1_val)}   ({len(val_ds)} samples)")
    print(f"[data] S2 target rows : {len(df_s2)}  ({len(target_ds)} samples)")

    return train_loader, val_loader, target_loader, {
        "union_cols":   union_cols,
        "len_features": len_features,
        "num_classes":  NUM_CLASSES,
        "scaler":       scaler,
        "window_size":  window_size,
    }
