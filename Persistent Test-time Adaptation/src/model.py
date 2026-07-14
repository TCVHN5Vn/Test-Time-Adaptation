"""
model.py  (fixed)
-----------------
Replaces Conv1d(kernel_size=1) with a proper FC-MLP.

WHY: Conv1d(k=1) applies an independent transform to each sensor position.
It never sees that "sensor_23 AND sensor_44 fired together". For binary
sparse HAR data, the ONLY discriminative signal is co-activation patterns.
A Linear(F→256) first layer sees all sensors jointly and learns those patterns.

Architecture:
    Input (B, 1, F)
    → flatten → (B, F)
    → Linear(F, 256) + BN + ReLU + Dropout(0.3)
    → Linear(256, 128) + BN + ReLU + Dropout(0.3)
    → Linear(128, 64)  + BN + ReLU          ← 64-d embedding
    → Linear(64, C)                          ← logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureExtractor(nn.Module):
    """
    Input  : (B, 1, F)
    Output : (B, 64)  embedding
    """
    def __init__(self, len_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),          nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),           nn.BatchNorm1d(64),  nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))   # (B,1,F) → (B,F) → (B,64)


class Classifier(nn.Module):
    def __init__(self, num_classes, embed_dim=64):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class SensorMLP(nn.Module):
    def __init__(self, len_features, num_classes):
        super().__init__()
        self.feat = FeatureExtractor(len_features)
        self.clsf = Classifier(num_classes)

    def forward(self, x):
        return self.clsf(self.feat(x))

    def encode(self, x):
        return self.feat(x)


def build_model(len_features, num_classes):
    return SensorMLP(len_features=len_features, num_classes=num_classes)
