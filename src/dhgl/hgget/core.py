# pylint: disable=unused-argument
from __future__ import annotations

import sys
import typing
from typing import TYPE_CHECKING, overload

import torch
from dgl import DGLGraph

from ..data.base import NdataDictView, TargetNodeData
from ..type import NType, Split
from . import dgl

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from . import pyg

if sys.version_info.major >= 3 and sys.version_info.minor >= 11:
    from typing import assert_never  # python 3.11+
else:

    def assert_never(arg):
        raise ValueError(arg)


__all__ = [
    'tgt_ntype', 'tgt_data', 'tgt_feat', 'mask', 'label', 'n_classes',
    'is_multi_label', 'index', 'device', 'ndata', 'feat_dim'
]


def _designate(fn):

    name = fn.__name__

    def getter_fn(hdata, *args, **kwargs):
        if isinstance(hdata, DGLGraph):
            return getattr(dgl, name)(hdata, *args, **kwargs)
        assert pyg is not None  # pylint: disable=used-before-assignment
        return getattr(pyg, name)(hdata, *args, **kwargs)

    return getter_fn


@_designate
def tgt_ntype(hg) -> str:
    """Get target node type"""


class SplitDict(typing.TypedDict):
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor


@_designate
def tgt_data(hg) -> TargetNodeData:
    """Get target node data"""
    assert_never(hg)


@_designate
def tgt_feat(hg, split: Split = None) -> torch.Tensor:
    """Get target node features.
    If split is not specified, a dict for all splits would be returned.
    """
    assert_never(hg)


@overload
def mask(hg) -> torch.Tensor:
    ...


@overload
def mask(hg, split: Split) -> torch.Tensor:
    ...


@_designate
def mask(hg, split: Split = None):
    """Get mask for splits
    If split is not specified, a dict for all splits would be returned.
    """
    assert_never(hg)


@_designate
def label(hg, split: Split = None) -> torch.Tensor:
    """Get labels for splits
    If split is not specified, labels for all target nodes would be returned
    """
    assert_never(hg)


@_designate
def is_multi_label(hg) -> bool:
    """Get labels for splits
    If split is not specified, labels for all target nodes would be returned
    """
    assert_never(hg)


@_designate
def n_classes(hg) -> int:
    """Get labels for splits
    If split is not specified, labels for all target nodes would be returned
    """
    assert_never(hg)


@_designate
def index(hg, split: Split = None) -> torch.Tensor:
    """index of different split (train, val, or test)"""
    assert_never(hg)


def device(hg) -> torch.device:
    if isinstance(hg, DGLGraph):
        return hg.device
    return tgt_feat(hg).device


def ndata(hg, key: str) -> NdataDictView:
    if isinstance(hg, DGLGraph):
        return hg.ndata[key]
    assert isinstance(hg, HeteroData)
    return hg.collect(key)


@_designate
def feat_dim(hg) -> dict[NType, int]:
    """Feature dimensions. This handle the case features represented by ids"""
    assert_never(hg)
