"""
loss_func.py
------------
Exact copy of src/utils/loss_func.py from the PeTTA repository.
"""

import torch


@torch.jit.script
def softmax_entropy(x: torch.Tensor, x_ema: torch.Tensor) -> torch.Tensor:
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


# Self-training loss from https://arxiv.org/abs/2211.13081
@torch.jit.script
def self_training(
    x: torch.Tensor,
    x_aug: torch.Tensor,
    x_ema: torch.Tensor,
) -> torch.Tensor:
    return (
        -0.25 * (x_ema.softmax(1) * x.log_softmax(1)).sum(1)
        - 0.25 * (x.softmax(1) * x_ema.log_softmax(1)).sum(1)
        - 0.25 * (x_ema.softmax(1) * x_aug.log_softmax(1)).sum(1)
        - 0.25 * (x_aug.softmax(1) * x_ema.log_softmax(1)).sum(1)
    )
