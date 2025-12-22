from __future__ import annotations

import typing

import torch

from ..data.base import BaseHeteroGraphLike
from ..type import Split
from .context import my_cache


class SplitDict(typing.TypedDict):
    train: torch.Tensor
    val: torch.Tensor
    test: torch.Tensor


@my_cache(max_size=1)
def tgt_ntype(hg: BaseHeteroGraphLike):
    return next(iter(hg.ndata['label']))


def tgt_data(hg: BaseHeteroGraphLike):
    return hg.nodes[tgt_ntype(hg)].data


@my_cache(max_size=4)
def tgt_feat(hg: BaseHeteroGraphLike, split: Split = None) -> torch.Tensor:
    if split is None:
        return hg.nodes[tgt_ntype(hg)].data['feat']
    return hg.nodes[tgt_ntype(hg)].data['feat'][mask(hg, split)]


@typing.overload
def mask(hg: BaseHeteroGraphLike) -> torch.Tensor:
    ...


@typing.overload
def mask(hg: BaseHeteroGraphLike, split: Split) -> torch.Tensor:
    ...


@my_cache(max_size=4)
def mask(hg: BaseHeteroGraphLike, split: Split = None):
    if split is None:
        d = tgt_data(hg)
        return (d['train_mask'] | d['val_mask'] | d['test_mask'])
    assert split in typing.get_args(Split)
    return tgt_data(hg)[f'{split}_mask']


@my_cache(max_size=4)
def label(hg: BaseHeteroGraphLike, split: Split = None) -> torch.Tensor:
    labels = tgt_data(hg)['label']
    if split is None:
        return labels

    assert split in typing.get_args(Split)
    return labels[mask(hg, split)]


def is_multi_label(hg: BaseHeteroGraphLike) -> bool:
    return len(label(hg).shape) == 2


@my_cache(max_size=1)
def n_classes(hg: BaseHeteroGraphLike) -> int:
    labels = label(hg)

    if len(labels.shape) == 2:  # multi-label
        return labels.shape[-1]

    return labels.max().item() + 1


@my_cache(max_size=4)
def index(hg: BaseHeteroGraphLike, split: Split = None):
    """index of different split (train, val, or test)"""
    return torch.nonzero(mask(hg, split)).squeeze()


@my_cache(max_size=4)
def feat_dim(hg: BaseHeteroGraphLike):
    """Feature dimensions. This handle the case features represented by ids"""
    dims = {}
    for ntype, nfeat in hg.ndata['feat'].items():
        if len(nfeat.shape) == 2:
            dims[ntype] = nfeat.shape[-1]
        else:
            assert len(nfeat.shape) == 1
            dims[ntype] = nfeat.max().item() + 1
    return dims
