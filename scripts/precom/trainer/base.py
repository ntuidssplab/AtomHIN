from __future__ import annotations

from typing import Callable, NamedTuple, TypedDict

import torch
from torch import nn
from torch.utils.data import Dataset


class HGNNReturnT(TypedDict):
    model: nn.Module
    dataset: Dataset
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler | None
    forward_fn: Callable
    eval_forward_fn: Callable | None
    loss_fn: Callable


class BatchData(NamedTuple):

    batch_indices: torch.Tensor
    features: tuple
    labels: torch.Tensor

    @classmethod
    def handle(cls, data: tuple):
        return cls(*data)
