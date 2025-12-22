from __future__ import annotations

import abc
import sys
from typing import Callable, Mapping, TypedDict, TypeVar

import torch

if sys.version_info.major >= 3 and sys.version_info.minor >= 11:
    # 3.11+
    from typing import Self  # pylint: disable=ungrouped-imports
else:
    from typing_extensions import Self

from ...data.base.base_schema import BaseGraphSchema


class BasePredData(abc.ABC):
    """This is an abc of the output of forward_fn for interface alignment.
    In the simplest case, the predicted logits would be subscripted by indices
        and be fed to loss function.
    For example,
    >>> logits = forward_fn(...)
    >>> loss = loss_fn(logits[train_indices], labels[train_indices])

    However, there are some cases more complicated. Models may output more data than just logits.

    Take the case of TreeXGNN as an example. Model outputs (logits, pred_tree),
        both in shape (#samples, ...), and both of them are used for loss_fn.
    Therefore, it requires to do something like:
    >>> logits, pred_tree = TreeXGNN.forward(...)
    >>> loss = loss_fn((logits[train_indices], pred_tree[train_indices]), labels[train_indices])

    Furthermore, there are some more different scenario which the outputs are not all in shape
    (#samples, ...). For instance, NodeFormer outputs logits (in shape [#samples, ...])
        and adjencies losses (in shape [#adj, ...]):
    >>> logits, link_losses = NodeFormer.forward(...)
    >>> loss = loss_fn((logits[train_indices], link_losses), labels[train_indices])
    """

    @abc.abstractmethod
    def __init__(self, *args):
        ...

    @abc.abstractmethod
    def __getitem__(self, item) -> Self:
        ...

    @property
    @abc.abstractmethod
    def logits(self) -> torch.Tensor:
        ...


PredDataT = TypeVar('PredDataT', bound=BasePredData)


class HGNNReturnT(TypedDict):
    hg: BaseGraphSchema
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    """Deprecated."""
    forward_fn: Callable[[BaseGraphSchema, Mapping[str, torch.Tensor]],
                         torch.Tensor | PredDataT]
    scheduler: torch.optim.lr_scheduler._LRScheduler | None
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None
    eval_forward_fn: Callable[[BaseGraphSchema, Mapping[str, torch.Tensor]],
                              torch.Tensor | PredDataT] | None
    """Default to forward_fn"""
