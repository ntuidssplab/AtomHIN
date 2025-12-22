from __future__ import annotations

import warnings
from itertools import accumulate
from typing import TypeVar

import dgl
import torch

from ...type import CEType, EType
from .metagraph import SelfLoop


def row_normalized_adjs(
    hg: dgl.DGLHeteroGraph,
    edge_weights: dict[EType, torch.Tensor] | None = None,
    etypes: list[EType] | None = None,
) -> dict[CEType, torch.Tensor]:
    """Get row-wise L1-normalized adjacency matrices for each edge type
    in a heterogeneous graph.

    The resulting adjacency matrices are in the format: (dst, src)
        (i.e., rows correspond to destination nodes, columns to source nodes).
    Note that the format is different from the one used in DGL's, which is (src, dst).

    Args:
        hg (dgl.DGLHeteroGraph): The input heterogeneous graph.
        edge_weights (dict[EType, torch.Tensor] | None, optional): Edge weights.
            If not provided, unweighted adjacency is used.
            >>> edge_weights=hg.edata['weight']  # Example of using edge weights
        etypes (list[EType]): List of etypes. Default to all etypes.

    Returns:
        dict[EType, torch.Tensor]: Edge type -> row-normalized adjacency matrices.
    """

    if edge_weights is not None:
        # Just in case the edge_weights are not in canonical form
        for etype, val in edge_weights.items():
            edge_weights[hg.to_canonical_etype(etype)] = val
    else:
        edge_weights = {}
        if set(hg.edata) & {'w', 'weights', 'weight'}:
            keys = set(hg.edata) & {'w', 'weights', 'weight'}
            warnings.warn(
                f'edge_weights not set while edata[{keys}] detected.'
            )
    adjs = {}

    for cetype in (hg.canonical_etypes if etypes is None else etypes):

        if isinstance(cetype, SelfLoop):
            adjs[cetype] = torch.sparse_coo_tensor(
                torch.arange(hg.num_nodes(cetype[0])).repeat((2, 1)),
                values=torch.ones(hg.num_nodes(cetype[0]))
            )
            continue
        cetype = hg.to_canonical_etype(cetype)
        if cetype in edge_weights:
            val = edge_weights[cetype]
            src, dst = hg.edges(etype=cetype)
            adj = torch.sparse_coo_tensor(
                torch.stack([dst, src]),
                values=val,
                size=(hg.num_nodes(cetype[-1]), hg.num_nodes(cetype[0])),
            )
            adj_abs = torch.sparse_coo_tensor(
                torch.stack([dst, src]),
                values=val.abs(),
                size=(hg.num_nodes(cetype[-1]), hg.num_nodes(cetype[0])),
            )
        else:
            adj: torch.Tensor = hg.adj_external(etype=cetype).T
            adj_abs = adj

        deg_row_inv = adj_abs.sum(dim=1).to_dense().pow(-1)
        deg_row_inv = deg_row_inv.masked_fill(deg_row_inv == torch.inf, 0)
        adj = adj * deg_row_inv.view(-1, 1)
        adjs[cetype] = adj

    return adjs


ETypeT = TypeVar('ETypeT', EType, CEType)


def adjs_to_homogeneous(
    hg: dgl.DGLHeteroGraph, hetero_adjs: dict[ETypeT, torch.Tensor]
) -> dict[ETypeT, torch.Tensor]:
    """Converts heterogeneous adjacencies to homogeneous format.

    Args:
        hg (dgl.DGLHeteroGraph): reference heterogeneous graph
        hetero_adjs (dict[EType, torch.Tensor]): adjacency matrices for each edge type

    Returns:
        dict[EType, torch.Tensor]: homogeneous adjacency matrices
    """
    num_nodes = sum(hg.num_nodes(ntype) for ntype in hg.ntypes)
    nid_offsets = accumulate(
        [0] + [hg.num_nodes(ntype) for ntype in hg.ntypes]
    )
    nid_offsets = dict(zip(hg.ntypes, nid_offsets))

    homo_adjs = {}
    for etype, adj in hetero_adjs.items():
        (s, _, d) = hg.to_canonical_etype(etype)
        adj = adj.coalesce()
        dst, src = adj.indices()

        homo_adjs[etype] = torch.sparse_coo_tensor(
            torch.stack([dst + nid_offsets[d], src + nid_offsets[s]]),
            values=adj.values(),
            size=(num_nodes, num_nodes),
        )
    return homo_adjs
