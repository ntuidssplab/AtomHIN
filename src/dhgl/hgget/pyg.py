from __future__ import annotations

import typing

import torch
from torch_geometric.data import HeteroData

from ..type import Split
from . import core


def tgt_ntype(hdata: HeteroData):
    return next(iter(hdata.collect('label')))


def tgt_data(hdata: HeteroData):
    return hdata[tgt_ntype(hdata)]


def tgt_feat(hdata: HeteroData, split: Split = None) -> torch.Tensor:
    if split is None:
        return hdata[tgt_ntype(hdata)]['feat']
    return hdata[tgt_ntype(hdata)]['feat'][core.mask(hdata, split)]


def mask(hdata: HeteroData, split: Split = None):
    if split is None:
        d = tgt_data(hg)
        return (d['train_mask'] | d['val_mask'] | d['test_mask'])
    assert split in typing.get_args(Split)
    return tgt_data(hdata)[f'{split}_mask']


def label(hdata: HeteroData, split: Split = None) -> torch.Tensor:
    labels = tgt_data(hdata)['label']
    if split is None:
        return labels

    assert split in typing.get_args(Split)
    return labels[mask(hdata, split)]


def is_multi_label(hdata: HeteroData) -> bool:
    return len(label(hdata).shape) == 2


def n_classes(hdata: HeteroData) -> int:
    labels = label(hdata)

    if len(labels.shape) == 2:  # multi-label
        return labels.shape[-1]

    return labels.max().item() + 1


def index(hdata: HeteroData, split: Split = None):
    """index of different split (train, val, or test)"""
    return torch.nonzero(mask(hdata, split)).squeeze()


def feat_dim(hdata: HeteroData):
    """Feature dimensions. This handle the case features represented by ids"""
    raise NotImplementedError
