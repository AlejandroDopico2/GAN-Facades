from typing import Tuple
import numpy as np
from torch import nn
import torch


def count_parameters(model: nn.Module, only_trainable: bool = False):
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def gaussian2d(
    height: int,
    width: int,
    mean: Tuple[float, float] = (0.0, 0.0),
    std: Tuple[float, float] = (0.4, 0.4),
):
    gaus = lambda x, y: np.exp(
        -(
            ((x - mean[0]) ** 2) / (2 * std[0] ** 2)
            + ((y - mean[1]) ** 2) / (2 * std[1] ** 2)
        )
    )
    x = torch.zeros(height, width)
    for h in range(height):
        i = h / height - 0.5
        for w in range(width):
            j = w / width - 0.5
            x[h, w] = gaus(i, j)
    return x
