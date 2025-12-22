from __future__ import annotations

from typing import Callable, TypedDict

import torch
from torch import nn

from dhgl.data.link_prediction import LinkPredDatasetLike


class HGNNReturnT(TypedDict):
    model: nn.Module
    dataset: LinkPredDatasetLike
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler | None
    forward_fn: Callable
    eval_forward_fn: Callable | None
    loss_fn: Callable
