"""
bn_layers.py
------------
Exact copy of src/utils/bn_layers.py from the PeTTA repository.
RobustBN1d / RobustBN2d replace standard BN during TTA:
  - In train mode : uses an EMA blend of source statistics and batch statistics
  - In eval  mode : uses the (updated) source statistics only
"""

import torch
import torch.nn as nn
from copy import deepcopy


class MomentumBN(nn.Module):
    def __init__(self, bn_layer: nn.BatchNorm1d, momentum):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        if (
            bn_layer.track_running_stats
            and bn_layer.running_var is not None
            and bn_layer.running_mean is not None
        ):
            self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
            self.register_buffer("source_var",  deepcopy(bn_layer.running_var))
            self.source_num = bn_layer.num_batches_tracked
        self.weight = deepcopy(bn_layer.weight)
        self.bias   = deepcopy(bn_layer.bias)
        self.register_buffer("target_mean", torch.zeros_like(self.source_mean))
        self.register_buffer("target_var",  torch.ones_like(self.source_var))
        self.eps           = bn_layer.eps
        self.current_mu    = None
        self.current_sigma = None

    def forward(self, x):
        raise NotImplementedError


class RobustBN1d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(
                x, dim=0, unbiased=False, keepdim=False
            )  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var  = (1 - self.momentum) * self.source_var  + self.momentum * b_var
            self.source_mean = deepcopy(mean.detach())
            self.source_var  = deepcopy(var.detach())
            mean, var = mean.view(1, -1), var.view(1, -1)
        else:
            mean = self.source_mean.view(1, -1)
            var  = self.source_var.view(1, -1)
        x      = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1)
        bias   = self.bias.view(1, -1)
        return x * weight + bias


class RobustBN2d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(
                x, dim=[0, 2, 3], unbiased=False, keepdim=False
            )  # (C,)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var  = (1 - self.momentum) * self.source_var  + self.momentum * b_var
            self.source_mean = deepcopy(mean.detach())
            self.source_var  = deepcopy(var.detach())
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            mean = self.source_mean.view(1, -1, 1, 1)
            var  = self.source_var.view(1, -1, 1, 1)
        x      = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias   = self.bias.view(1, -1, 1, 1)
        return x * weight + bias
