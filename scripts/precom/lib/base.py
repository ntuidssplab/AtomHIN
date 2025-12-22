from __future__ import annotations

from typing import Callable, TypedDict

import torch
from torch.utils.data import Dataset


class HGNNReturnT(TypedDict):
    model: int
    dataset: Dataset
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler | None
    forward_fn: Callable
    eval_forward_fn: Callable | None
    loss_fn: Callable
