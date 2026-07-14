"""
occ_model.py — Occupancy MLP
------------------------------
Architecture matches PeTTA Occupancy experiments.
Uses separate FeatExtractor and Classifier submodules matching
checkpoint keys: feat.net.* and clsf.fc.*
"""

import torch.nn as nn


class FeatExtractor(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 128), nn.BatchNorm1d(128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class Classifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(64, num_classes)
    def forward(self, z):
        return self.fc(z)


class OCC_MLP(nn.Module):
    def __init__(self, num_features: int = 9, num_classes: int = 2):
        super().__init__()
        self.feat = FeatExtractor(num_features)
        self.clsf = Classifier(num_classes)

    def forward(self, x):
        return self.clsf(self.feat(x))
