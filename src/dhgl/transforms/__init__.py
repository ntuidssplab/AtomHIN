"""Transformas for heterogeneous graphs
Please note that inplace transform should be avoided
"""
from __future__ import annotations

import warnings
from collections import defaultdict
from functools import reduce
from itertools import accumulate
from typing import Literal, Sequence

import dgl
import torch
from dgl import DGLGraph, DGLHeteroGraph, convert
from dgl import function as fn
from typing_extensions import deprecated

from .. import hgget as H
from ..data.base import BaseHeteroGraphLike
from ..type import CEType, EType, NType


def _get_data_dict(graph: DGLHeteroGraph):
    for src_ntype, etype, dst_ntype in graph.canonical_etypes:
        new_adj = graph.adj(etype)
        u, v = new_adj.indices()
        yield (src_ntype, etype, dst_ntype), (u, v)


def transfer_data(g_dst: DGLHeteroGraph, g_src: DGLHeteroGraph):
    g_dst = g_dst.clone()
    for data_key, data in g_src.ndata.items():
        data: dict
        for ntype, ndata in data.items():
            g_dst.nodes[ntype].data[data_key] = ndata

    for data_key, data in g_src.edata.items():
        data: dict
        for etype, edata in data.items():
            g_dst.edges[etype].data[data_key] = edata

    return g_dst


def add_self_loop(
    hg: BaseHeteroGraphLike, ntypes: NType | list[NType] = None,
    self_loop_etypes: dict[NType, str] = None
) -> BaseHeteroGraphLike:
    """Add self loop for uni-bipartite graph.

    Currently, dgl has not supported `dgl.add_self_loop` for unidirectional bipartite graphs.
    Therefore, to add self loop, a graph need to be recreate.

    Args:
        g (DGLHeteroGraph): unidirectional biparatite graph
        ntypes: (list[Ntype]): list of node types that would get self-loops added. If omit,
            add self-loop for nodes in all node types
        self_loop_etype (dict[Ntype]): New etype names for the created self loops. If omit,
            new self loop will use the name '{ntype}-self'

    Returns:
        DGLHeteroGraph: graph with self loop
    """
    # assert not bool(
    #     hg.edata
    # ), 'Not support add_self_loop for graph with edge data'

    self_loop_etypes = self_loop_etypes or {
        ntype: f'{ntype}-self'
        for ntype in hg.ntypes
    }
    data_dict = dict(_get_data_dict(hg))
    if isinstance(ntypes, NType):
        ntypes = [ntypes]
    ntypes = ntypes or hg.ntypes
    for ntype in ntypes:
        data_dict.update(
            {
                (ntype, self_loop_etypes[ntype], ntype): ([], []),
            }
        )
    new_g = update_graph_structure(
        hg, data_dict, copy_ndata=True, copy_edata=True
    )
    for ntype in ntypes:
        new_g = dgl.add_self_loop(new_g, etype=self_loop_etypes[ntype])

    return new_g


def to_homogeneous(hg: BaseHeteroGraphLike, order: list[NType]):
    """Transform a heterogeneous graph to homogeneous and the node ids of output homo-graph
        following the specific order of node type

    Args:
        hg (BaseGraphSchema): _description_
        order (list[Ntype]): _description_

    Returns:
        _type_: _description_

    Yields:
        _type_: _description_
    """

    num_nodes = [hg.num_nodes(ntype) for ntype in order]
    offsets_list = [0, *accumulate(num_nodes)]
    offsets = dict(zip(order, offsets_list))

    def gen():
        for src_ntype, etype, dst_ntype in hg.canonical_etypes:
            adj = hg.adj(etype)
            u, v = adj.indices()
            u += offsets[src_ntype]
            v += offsets[dst_ntype]
            yield u, v, torch.full_like(
                u, hg.get_etype_id(etype), dtype=torch.long
            )

    new_us, new_vs, etypes = list(zip(*gen()))
    new_u = torch.concat(new_us)
    new_v = torch.concat(new_vs)
    g = dgl.graph((new_u, new_v))
    g.ndata['_TYPE'] = torch.zeros(offsets_list[-1], dtype=torch.long)
    for i, (offset_l,
            offset_r) in enumerate(zip(offsets_list[:-1], offsets_list[1:])):
        g.ndata['_TYPE'][offset_l:offset_r] = i
    g.edata['_TYPE'] = torch.concat(etypes)
    return g


FeatMode = Literal['zero', 'ntype', 'nid', 'none']


def remove_non_target_feature(
    hg: BaseHeteroGraphLike,
    mode: FeatMode | dict[NType, FeatMode],
    fmt: Literal['id', 'coo'] | None = None,
    as_id: bool = False,
):
    """Remove features from non-target node types using the specified mode.

    Args:
        hg (BaseHeteroGraphLike): The input heterogeneous graph.
        mode (str | dict): Feature removal mode (globally or per node type):
            - 'none': Remove features without replacement.
            - 'zero': Replace with a 1D zero vector.
            - 'ntype': Replace with one-hot encoding of node type ID.
            - 'nid': Replace with node ID (identity matrix).
        fmt (str | None): Output format for replacement ('id' or 'coo'). Required for some modes.
        as_id (bool): Deprecated. Use `fmt='id'` instead when using embedding layers.

    Raises:
        ValueError: If `as_id=True` and `mode='none'`.
    """
    LEGAL_MODES = {'zero', 'ntype', 'nid', 'id', 'ntype_id', 'none'}
    non_tgt_ntypes = [ntype for ntype in hg.ntypes if ntype != H.tgt_ntype(hg)]

    if isinstance(mode, dict):
        assert set(mode) <= set(non_tgt_ntypes), (
            f'got ntypes: {set(mode)} but non-tgt node types are {non_tgt_ntypes}'
        )
        assert set(mode.values()) <= LEGAL_MODES
        for ntype, m in mode.items():
            hg = remove_node_features(hg, m, ntype, fmt, as_id)

        return hg

    assert mode in LEGAL_MODES
    hg = remove_node_features(hg, mode, non_tgt_ntypes, fmt, as_id)
    return hg


def remove_node_features(
    hg: BaseHeteroGraphLike,
    mode: Literal['zero', 'ntype', 'nid', 'none'],
    ntypes: str | Sequence[str],
    fmt: Literal['id', 'coo'] | None = None,
    as_id: bool = False,
    in_place: bool = False,
):
    """Remove node features for specified node types using a given mode.

    Args:
        hg (BaseHeteroGraphLike): The input heterogeneous graph.
        mode (str): Feature removal mode:
            - 'none': Remove features without replacement.
            - 'zero': Replace with a zero vector of size 1.
            - 'ntype': Replace with one-hot encoding of node type ID.
            - 'nid': Replace with node ID (i.e., identity matrix).
        ntypes (str | list[str]): Node types whose features are to be removed.
        fmt (str | None): Output format for replacement ('id' or 'coo'). Required for some modes.
        as_id (bool): Deprecated. Use `fmt='id'` instead when using embedding layers.

    Raises:
        ValueError: If `as_id=True` and mode is 'none'.
    """
    assert mode in ['zero', 'ntype_id', 'id', 'ntype', 'nid', 'none']
    if as_id:
        warnings.warn(
            'as_id has been deprecated. Use "fmt=\'id\'" instead.',
            DeprecationWarning,
        )
        assert fmt is None
        fmt = 'id'
    if as_id and mode == 'zero':
        mode = 'ntype'
        warnings.warn(
            'mode=zero as as_id is True is equivalent as using mode=ntype. Use "ntype" instead.',
        )
    if as_id and mode == 'none':
        raise ValueError(f'Canont set as_id while mode={mode}')

    if not in_place:
        hg = hg.clone()

    def ntype_feat(ntype: str, n_nodes: int, device):
        if fmt == 'coo':
            raise NotImplementedError(
                f'ntype feat not supported with format={fmt}'
            )
        t = torch.zeros((n_nodes, len(hg.ntypes)), device=device)
        t[:, hg.get_ntype_id(ntype)] = 1.
        return t

    def identity(n_nodes: int):
        if fmt == 'coo':
            return torch.sparse_coo_tensor(
                torch.stack([torch.arange(n_nodes),
                             torch.arange(n_nodes)]),
                values=torch.ones(n_nodes),
            )
        elif fmt == 'id':
            return torch.arange(n_nodes, dtype=torch.long)
        assert fmt is None
        return torch.eye(n_nodes)

    if isinstance(ntypes, str):
        ntypes = [ntypes]

    for ntype in ntypes:
        n_nodes = hg.num_nodes(ntype)
        if mode == 'none':
            assert fmt is None
            if 'feat' in hg.nodes[ntype].data:
                hg.nodes[ntype].data.pop('feat')
        elif mode == 'zero':
            assert fmt != 'id'
            hg.nodes[ntype].data['feat'] = torch.zeros(
                (n_nodes, 1), device=hg.device
            )
        elif mode == 'ntype':
            if fmt == 'id':
                hg.nodes[ntype].data['feat'] = torch.zeros(
                    (n_nodes, ), dtype=torch.long, device=hg.device
                )
            else:
                hg.nodes[ntype].data['feat'] = ntype_feat(
                    ntype, n_nodes, device=hg.device
                )
        else:
            assert mode == 'nid'
            hg.nodes[ntype].data['feat'] = identity(n_nodes).to(hg.device)
    return hg


def to_dense(hg: BaseHeteroGraphLike):
    hg = hg.clone()
    for ntype in hg.ntypes:
        for data_key, data in hg.nodes[ntype].data.items():
            data: torch.Tensor
            if data.is_sparse:
                hg.nodes[ntype].data[data_key] = data.to_dense()
    for etype in hg.canonical_etypes:
        for data_key, data in hg.edges[etype].data.items():
            data: torch.Tensor
            if data.is_sparse:
                hg.edges[etype].data[data_key] = data.to_dense()

    return hg


def to_homogeneous_wrt_metapaths(
    hg: DGLHeteroGraph,
    metapaths: list[list[EType]] | list[EType],
):
    """Transformed the given hetero-graph to
    homogeneous graph whose edges are subject to provided meta-paths.

    Args:
        hg (BaseHeteroGraphLike)
        metapaths_ (list[list[EType]] | list[Etype]): list of methpaths.
                Note that a metapath is a list of Etype.
    Returns:
        A homogeneous graph
    """

    assert len(metapaths) > 0

    if isinstance(metapaths[0], (str, tuple)):
        metapaths = [metapaths]

    def gen():
        for metapath in metapaths:
            yield reduce(lambda x, y: x @ y, map(hg.adj, metapath))

    homo_adj = reduce(lambda x, y: x + y, gen())

    return dgl.graph(
        (homo_adj.indices()[0], homo_adj.indices()[1]),
        num_nodes=homo_adj.shape[0]
    )


def add_nhop_edges(g: DGLGraph | DGLHeteroGraph, n: int):
    """Add nhop edges to the given graph. The n-hop edges contain no self-loop,
        and no added edges connect to the same node pair.

    Args:
        g (DGLGraph | DGLHeteroGraph): either heterogeneous and homogeneous. If it is heterogeneous,
            it would be first transformed to homogeneous via `dgl.to_homogeneous`.
        n (int): n hops. n >= 2

    Returns:
        A homogeneous graph with the added edges have a new edge type, which can be access via
        `g.edata[dgl.ETYPE]`.
        Similarly, the eids are available through `g.edata[dgl.EID]`
    """

    assert n > 1
    if g.is_homogeneous:
        g = g.clone()
    else:
        g = dgl.to_homogeneous(g)

    adj = g.adjacency_matrix().coalesce()

    nhop_adj = reduce(torch.sparse.mm, [adj] * n).coalesce()  # pylint: disable=not-callable

    indices = nhop_adj.indices()
    indices = indices[:, indices[0] != indices[1]]  # exclude self-loops

    edata = {}
    if dgl.ETYPE in g.edata:
        edata[dgl.ETYPE] = torch.full(
            (indices.shape[-1], ), fill_value=g.edata[dgl.ETYPE].max() + 1
        )
    if dgl.EID in g.edata:
        edata[dgl.EID] = torch.arange(indices.shape[-1])
    g.add_edges(indices[0], indices[1], edata)

    return g


def update_graph_structure(
    g, data_dict, num_nodes_dict: dict[NType, int] | None = None,
    copy_ndata=True, copy_edata=True
):
    r"""Update the structure of a graph.

    Parameters
    ----------
    g : DGLGraph
        The graph to update.
    data_dict : graph data
        The dictionary data for constructing a heterogeneous graph.
    num_nodes_dict: dict[NType, int]. Optional.
    copy_edata : bool
        If True, it will copy the edge features to the updated graph.

    Returns
    -------
    DGLGraph
        The updated graph.
    """
    device = g.device
    idtype = g.idtype
    if num_nodes_dict is None:
        num_nodes_dict = {}
        for ntype in g.ntypes:
            num_nodes_dict[ntype] = g.num_nodes(ntype)

    new_g = convert.heterograph(
        data_dict, num_nodes_dict=num_nodes_dict, idtype=idtype, device=device
    )

    # Copy features
    if copy_ndata:
        for ntype in num_nodes_dict:
            for key, feat in g.nodes[ntype].data.items():
                new_g.nodes[ntype].data[key] = feat

    if copy_edata:
        for c_etype in g.canonical_etypes:
            if c_etype in new_g.canonical_etypes:
                for key, feat in g.edges[c_etype].data.items():
                    new_g.edges[c_etype].data[key] = feat

    return new_g


def _get_unreachable_ntypes(
    ntypes: list[NType], etypes: list[CEType], target_ntype: NType
):

    get_ntype_id = {ntype: nid for nid, ntype in enumerate(ntypes)}

    def construct_metagraph(ntypes: list[NType], etypes: list[CEType]):
        edges = [
            (get_ntype_id[srctype], get_ntype_id[dsttype])
            for srctype, _, dsttype in etypes
        ]
        edges = torch.tensor(edges)
        return dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(ntypes))

    mg = construct_metagraph(ntypes, etypes)
    mg = dgl.add_self_loop(mg)
    mg.ndata['h'] = torch.eye(mg.num_nodes())
    current = mg.ndata['h']
    for i in range(99):
        mg.update_all(fn.copy_u('h', 'm'), fn.max('m', 'h'))
        if (mg.ndata['h'] == current).all():
            break
        current = mg.ndata['h']
        if i == 98:
            raise ValueError('This is likely a bug.')

    reachable = mg.ndata['h'][get_ntype_id[target_ntype]]
    return [
        ntype for ntype_id, ntype in enumerate(ntypes)
        if not reachable[ntype_id]
    ]


def remove_etypes(
    hg: DGLHeteroGraph,
    etypes_to_remove: list[EType],
    unreachable_mode: Literal['full', 'nfeat', 'none'] = 'nfeat',
):
    """Remove edges of the specified edge types and optionally prune unreachable parts.

    Removes all edges whose types are in ``etypes_to_remove`` and rebuilds the
    heterogeneous graph. Optionally prunes unreachable node types either by
    dropping only their features or by fully removing those node types.

    Args:
        hg (DGLHeteroGraph): Input heterogeneous graph.
        etypes_to_remove (list[EType]): Edge types to remove. Each entry may be a
            canonical etype or a shorthand accepted by ``hg.to_canonical_etype``.
        unreachable_mode (Literal['full', 'nfeat', 'none'], optional): Strategy
            for handling node types that become unreachable from target node types
            after edge removal.
            - ``'none'``: Do not prune; keep all node types and their data.
            - ``'nfeat'``: Keep unreachable node types but drop their ``'feat'``
              field to save memory.
            - ``'full'``: Remove unreachable node types entirely (nodes and all
              incident edges) and rebuild the graph on the remaining node types.
            Defaults to ``'nfeat'``.

    Returns:
        DGLHeteroGraph: A new graph with the specified edge types removed and
        optional pruning applied.

    Raises:
        ValueError: If the requested removal would delete all edge types.
    """
    assert all(
        hg.to_canonical_etype(etype) in hg.canonical_etypes
        for etype in etypes_to_remove
    )
    etypes_to_remove = list(map(hg.to_canonical_etype, etypes_to_remove))
    data_dict = {}
    for etype in hg.canonical_etypes:
        if etype in etypes_to_remove:
            continue
        data_dict[etype] = hg.edges(etype=etype)

    if not data_dict:
        raise ValueError('Removing all edge types is not allowed.')
    num_nodes_dict = {ntype: hg.num_nodes(ntype=ntype) for ntype in hg.ntypes}

    hg = update_graph_structure(
        hg, data_dict, num_nodes_dict=num_nodes_dict, copy_ndata=True,
        copy_edata=True
    )
    if unreachable_mode != 'none' and hg.ndata['label']:
        hg = remove_unreachable(hg, unreachable_mode)
    return hg


def remove_unreachable(
    hg: DGLHeteroGraph, mode: Literal['full', 'nfeat'] = 'full'
):
    """Remove or prune node types unreachable from target node type(s).

    Args:
        hg (DGLHeteroGraph): Input heterogeneous graph with ``ndata['label']`` set.
        mode (Literal['full', 'nfeat'], optional): Strategy for unreachable node types.
            - ``'full'``: Remove unreachable node types and all incident edges, and
            - ``'nfeat'``: Keep node types but drop their ``'feat'`` field if present.
              rebuild the graph.
            Defaults to ``'full'``.

    Raises:
        ValueError: If ``hg.ndata['label']`` is not set.
    """
    if not bool(hg.ndata['label']):
        raise ValueError('Require ndata "label" to be set on the hg.')

    data_dict = {etype: hg.edges(etype=etype) for etype in hg.canonical_etypes}

    if mode == 'nfeat' and hg.ndata['label']:
        for ntype in _get_unreachable_ntypes(
            hg.ntypes, list(data_dict), H.tgt_ntype(hg)
        ):
            if 'feat' in hg.nodes[ntype].data:
                hg.nodes[ntype].data.pop('feat')
    elif mode == 'full' and hg.ndata['label']:
        unreachable_ntypes = _get_unreachable_ntypes(
            hg.ntypes, list(data_dict), H.tgt_ntype(hg)
        )
        data_dict = {
            etype: edges
            for etype, edges in data_dict.items()
            if etype[0] not in unreachable_ntypes
            and etype[-1] not in unreachable_ntypes
        }
        num_node_dict = {
            ntype: hg.num_nodes(ntype)
            for ntype in hg.ntypes if ntype not in unreachable_ntypes
        }
        hg = update_graph_structure(
            hg,
            data_dict,
            num_nodes_dict=num_node_dict,
        )
    return hg


@deprecated('use dhgl.schema.prepropagate instead.')
def prepropagate_legacy(
    hg: BaseHeteroGraphLike,
    max_hops: int | None = None,
    reduce_fn: Literal['mean', 'max'] = 'mean',
    edge_weights: dict[EType, torch.Tensor] | None = None,
):
    """propagate node features to nodes without feature

    Args:
        hg (BaseHeteroGraphLike): The heterogeneous graph
        max_hops (int | None, optional): Number of max hops to propagate. The propagation
            terminates when either the max_hops reached or all nodes are filled with features.
            Defaults to 99.
        reduce_fn (Literal[&#39;mean&#39;, &#39;max&#39;], optional): either mean or max.
            Setting mean will generally allow different ntypes to have same accumulated weights.
            Setting max works effectively as union.
            Defaults to 'mean'.
        edge_weights (dict[EType, torch.Tensor] | None, optional): Required for weighted graphs.
        Defaults to None.

    Returns:
        The graph with propagated features
    """

    hg = hg.clone()
    # REDUCE_FN = getattr(fn, reduce_fn)
    max_hops = max_hops or 99

    def cross_reduce(results: list[torch.Tensor]):
        raise NotImplementedError
        return results[0]

    # tolerate ID (long) features
    feats = hg.ndata['feat']
    for ntype, feat in feats.items():
        if len(feat.shape) > 1:
            continue
        assert not torch.is_floating_point(feat)
        identity = torch.eye(feat.max() + 1)
        feats[ntype] = identity[feat]

    @hg.local_scope()
    def _propagate(srctype, feat):
        if edge_weights is not None:
            hg.edata['w'] = edge_weights
        elif set(hg.edata) & {'w', 'weights', 'weight'}:
            keys = set(hg.edata) & {'w', 'weights', 'weight'}
            warnings.warn(
                f'edge_weights not set while edata[{keys}] detected.'
            )

        hs = {srctype: feat}

        def weighted_union(u, v):

            def fn(nodes):
                indices = nodes.mailbox[u].abs().argmax(dim=1, keepdim=True)
                res = torch.take_along_dim(nodes.mailbox[u], indices, dim=1)
                return {v: res.squeeze(dim=1)}

            return fn

        def get_update_fns(etype):
            message_fn = (
                fn.u_mul_e('x', 'w', 'm')
                if 'w' in hg.edges[etype].data else fn.copy_u('x', 'm')
            )
            reduce_fn_ = (fn.mean if reduce_fn == 'mean' else weighted_union)
            return message_fn, reduce_fn_('m', 'h')

        for i in range(max_hops):
            valid_etypes = [
                (s, e, d) for s, e, d, in hg.canonical_etypes
                if s in hs and d not in hs
            ]
            if len(valid_etypes) == 0:
                break
            hg.ndata['x'] = hs

            hg.multi_update_all(
                {etype: get_update_fns(etype)
                 for etype in valid_etypes}, cross_reducer=cross_reduce
            )
            hs.update(hg.ndata['h'])
        return hs

    # ntype -> propgated-feats
    propagated_feat: dict[NType, list[torch.Tensor]] = defaultdict(list)
    for srctype, feat in feats.items():
        for ntype, h in _propagate(srctype, feat).items():
            propagated_feat[ntype].append(h)
    propagated_feat = {
        ntype: torch.concatenate(hs, dim=1)
        for ntype, hs in propagated_feat.items()
    }
    propagated_feat.update(hg.ndata['feat'])
    hg.ndata['feat'] = propagated_feat
    return hg


def merge_etypes(
    hg: BaseHeteroGraphLike,
    etype: str,
    etype_to_drop: str,
):
    """Merge edges from one edge type into another and drop the source edge type.

    The function takes the adjacency of ``etype`` and ``etype_to_drop``,
    computes their union (removing duplicates), and assigns the result
    to ``etype``. The ``etype_to_drop`` is then removed from the graph.
    Other edge types and all node types remain unchanged.

    Edge weights are not supported: if ``etype`` has an edge feature
    named ``"weight"``, a ``NotImplementedError`` is raised. Edge
    features of the merged edges are not preserved.

    Args:
        hg (BaseHeteroGraphLike):
            Input heterogeneous graph.
        etype (str):
            Target edge type to merge into. After merging, it will contain
            the union of its original edges and those from ``etype_to_drop``.
        etype_to_drop (str):
            Edge type to merge from and then remove.

    Raises:
        NotImplementedError:
            If ``etype`` contains an edge feature named ``"weight"``.
        AssertionError:
            If ``etype_to_drop`` does not exist in the graph.

    Returns:
        BaseHeteroGraphLike:
            A new heterograph with ``etype`` updated to contain merged
            edges and ``etype_to_drop`` removed.

    Examples:
        >>> # Merge 'cited' edges into 'cites', making 'cites' symmetric
        >>> hg = merge_etypes(hg, 'cites', etype_to_drop='cited')
    """

    if 'weight' in hg.edges[etype].data:
        raise NotImplementedError

    etype = hg.to_canonical_etype(etype)
    etype_to_drop = hg.to_canonical_etype(etype_to_drop)
    indices = (
        hg.adj_external(etype=etype) + hg.adj_external(etype=etype_to_drop)
    ).coalesce().indices()

    data_dict = {e: hg.edges(etype=e) for e in hg.canonical_etypes}
    assert data_dict.pop(
        hg.to_canonical_etype(etype_to_drop), None
    ) is not None
    data_dict[etype] = (indices[0], indices[1])

    hg = update_graph_structure(hg, data_dict)
    return hg
