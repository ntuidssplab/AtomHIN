"""
Purpose:
    DGL stores edges/adjacency sparsely. When a dense adjacency is needed,
    store it under `hg.gdata`.
"""
from __future__ import annotations

from typing import Mapping

import dgl
import torch

from .. import transforms
from ..type import CEType

GDataValueT = torch.Tensor | dict[str | tuple, torch.Tensor]
GDataT = dict[str | GDataValueT]
_GDATA = 'gdata'
_NOT_IMPLEMENT = (
    "Unsupported gdata: expected a Mapping[str, Tensor | Mapping[str|tuple, Tensor]]."
)


def gdata(hg: dgl.DGLHeteroGraph) -> GDataT:
    gdata = getattr(hg, _GDATA, {})
    if not isinstance(gdata, Mapping):
        raise NotImplementedError(_NOT_IMPLEMENT)
    for data in gdata.values():
        if not isinstance(gdata, (Mapping, torch.Tensor)):
            raise NotImplementedError(_NOT_IMPLEMENT)
        if any(not isinstance(v, torch.Tensor) for v in data.values()):
            raise NotImplementedError(_NOT_IMPLEMENT)
    return gdata


def has_gdata(hg: dgl.DGLHeteroGraph):
    return bool(gdata(hg))


def set_gdata(hg: dgl.DGLHeteroGraph, **kwargs):
    gd = gdata(hg)
    for k, v in kwargs.items():
        if not isinstance(v, (Mapping, torch.Tensor)):
            raise NotImplementedError(_NOT_IMPLEMENT)
        gd[k] = v
    hg.gdata = gd
    return


def to(hg: dgl.DGLHeteroGraph, device):
    """Move the graph and all tensors in `gdata(hg)` to `device`.

    Args:
        hg (dgl.DGLHeteroGraph): Input graph.
        device: Device.

    Returns:
        dgl.DGLHeteroGraph: Graph on `device` with `hg.gdata` tensors moved.
    """
    d = gdata(hg)
    hg = hg.to(device)

    for key, val in d.items():
        if isinstance(val, Mapping):
            for k, v in val.items():
                assert isinstance(v, torch.Tensor)
                d[key][k] = v.to(device)
        else:
            assert isinstance(v, torch.Tensor)
            d[key] = val.to(device)

    set_gdata(hg, **d)
    return hg


def dense_adjs_to_gdata(hg: dgl.DGLHeteroGraph, dense_etypes: list[CEType]):
    """Store dense, row-normalized adjacencies in `hg.gdata['adj']` and prune edges.

    For each specified etype:
      1) build row-normalized (sparse) adj using `row_normalized_adjs`,
      2) convert to dense, and
      3) remove those edges from the sparse graph structure.

    Args:
        hg (dgl.DGLHeteroGraph): Input graph with `edata['weight']`.
        dense_etypes (list[CEType]): Edge types to materialize densely.

    Returns:
        dgl.DGLHeteroGraph: New graph with dense adjs under `gdata['adj']`.
    """

    from ..utils.precomputation.adj import row_normalized_adjs
    adjs = row_normalized_adjs(hg, hg.edata['weight'], dense_etypes)
    adjs = {etype: adj.to_dense() for etype, adj in adjs.items()}

    dense_etypes = list(map(hg.to_canonical_etype, dense_etypes))
    data_dict = {
        etype: hg.edges(etype=etype)
        for etype in hg.canonical_etypes if etype not in dense_etypes
    }
    for etype in dense_etypes:
        data_dict[etype] = ([], [])  # keep the etype without edges

    new_hg = transforms.update_graph_structure(hg, data_dict, copy_edata=False)
    for key, edata in hg.edata.items():
        if key == 'weight':
            for etype, v in edata.items():
                if etype not in dense_etypes:
                    hg.edges[etype].data[key] = v
        else:
            hg.edata[key] = edata
    if adjs:
        new_hg.gdata = {'adj': adjs}
    return new_hg
