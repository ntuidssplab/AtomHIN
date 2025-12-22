from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
from dgl.data import load_graphs as dgl_load_graphs
from dgl.data import save_graphs as dgl_save_graphs
from scipy import sparse

from ..base.base_schema import BaseHeteroGraphLike


def mat2tensor(mat: np.ndarray):

    if isinstance(mat, np.ndarray):
        return torch.from_numpy(mat).type(torch.FloatTensor)

    assert sparse.isspmatrix(mat)

    coo: sparse.coo_matrix = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse_coo_tensor(i, v, torch.Size(shape))


def load_graphs(filename, idx_list=None):
    """Deal with the issue that dgl.data.save_graphs cannot save graph with sparse tensor data
    """

    g_list, labels = dgl_load_graphs(filename, idx_list)

    sparse_data = defaultdict(dict)

    def yield_key_to_del():
        for key, data in labels.items():
            key: str
            key_ = key.split('|')
            if len(key_) != 4:
                continue
            idx, ntype, data_key, idx_or_val = key_
            idx = int(idx)
            sparse_data[(idx, ntype, data_key)][idx_or_val] = data
            yield key

    for key in list(yield_key_to_del()):
        labels.pop(key)

    for (idx, ntype, data_key), data in sparse_data.items():
        g_list[idx].nodes[ntype].data[data_key] = torch.sparse_coo_tensor(
            **data
        )

    return g_list, labels


def save_graphs(
    filename, g_list: list[BaseHeteroGraphLike], labels=None, formats=None
):
    """Deal with the issue that dgl.data.save_graphs cannot save graph with sparse tensor data
    """

    if not isinstance(g_list, Iterable):
        g_list = [g_list]

    if labels is None:
        labels = {}

    sparse_data = {}

    for i, hg in enumerate(g_list):
        for data_key, ndata in hg.ndata.items():

            if isinstance(ndata, torch.Tensor):
                assert len(hg.ntypes) == 1
                ndata = {hg.ntypes[0]: ndata}

            for ntype, data in ndata.items():
                data: torch.Tensor
                if data.layout == torch.strided:
                    continue

                data = data.coalesce()
                sparse_data[(i, ntype, data_key)] = {
                    'indices': data.indices(),
                    'values': data.values(),
                }
                hg.nodes[ntype].data[data_key] = torch.zeros(
                    (data.shape[0], 0)
                )
                # discard sparse tensor data

    for key, data in sparse_data.items():
        labels['|'.join(map(str, (*key, 'indices')))] = data['indices']
        labels['|'.join(map(str, (*key, 'values')))] = data['values']

    dgl_save_graphs(filename, g_list, labels, formats)

    for (i, ntype, data_key), data in sparse_data.items():
        g_list[i].nodes[ntype].data[data_key] = torch.sparse_coo_tensor(**data)

    return
