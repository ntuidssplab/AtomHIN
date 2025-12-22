from __future__ import annotations

from collections import defaultdict
from typing import Literal, Mapping

import dgl
import numpy as np
import torch
from dgl import function as fn

from .. import hgget as H
from ..data.base import BaseHeteroGraphLike
from ..transforms import to_homogeneous_wrt_metapaths

__all__ = ['info', 'node_homophily', 'edge_homophily', 'metapaths']


def info(hg: BaseHeteroGraphLike):
    # pylint: disable=import-outside-toplevel
    from io import StringIO

    from tabulate import tabulate

    sio = StringIO()
    print(file=sio)
    print('Dimensions of node features:', file=sio)

    def gen_feat_dim_table():
        for ntype in hg.ntypes:
            ndata = hg.ndata['feat']
            if isinstance(ndata, torch.Tensor):
                assert len(hg.ntypes) == 1
                ndata = {hg.ntypes[0]: ndata}
            data = ndata.get(ntype, None)
            if data is not None:
                if len(data.shape) == 1 and not torch.is_floating_point(data):
                    yield ntype, data.shape[0], data.max().item() + 1
                else:
                    yield ntype, *data.shape
            else:
                yield ntype, hg.num_nodes(ntype), 0

    print(
        tabulate(
            gen_feat_dim_table(), headers=['ntype', '#samples', 'feat_dim']
        ), file=sio
    )

    # assert not bool(graph.edata), 'graph with edge data is not supported'

    def gen_table():
        for src_ntype, etype, dst_ntype in hg.canonical_etypes:
            num_edge = hg.num_edges(etype)
            avg_in_d = hg.in_degrees(etype=etype).float().mean().item()
            avg_out_d = hg.out_degrees(etype=etype).float().mean().item()
            yield (
                src_ntype,
                etype,
                dst_ntype,
                num_edge,
                round(avg_out_d, 2),
                round(avg_in_d, 2),
            )

    print('Edges:', file=sio)
    print(
        tabulate(
            gen_table(), headers=[
                'src_ntype',
                'etype',
                'dst_ntype',
                '#edges',
                'out_deg',
                'in_deg',
            ]
        ), file=sio
    )

    if 'label' in hg.ndata and hg.ndata['label']:
        target_type = list(hg.ndata['label'].keys())[0]

        target_node_data = hg.nodes[target_type].data
        label_ = target_node_data['label']
        task_type = 'multi' if len(label_.shape) != 1 else 'single'
        n_classes_ = label_.shape[-1] if task_type == 'multi' else (
            label_.max().item() + 1
        )

        print(
            f'Task: {n_classes_}-class {task_type}-label classification',
            file=sio
        )
        if task_type == 'multi':
            sample_per_class = [
                int(label_[:, c].sum().item()) for c in range(n_classes_)
            ]
        else:
            sample_per_class = [
                (label_ == c).sum().item() for c in range(n_classes_)
            ]

        if (label_ < 0).any():
            print(f'\t#Unlabeled: {(label_ < 0).sum().item()}', file=sio)
        print(f'\tSamples per class: {sample_per_class}', file=sio)
        print(f'\tTarget "{target_type}":', file=sio)
        splits = (
            target_node_data['train_mask'].sum().item(),
            target_node_data['val_mask'].sum().item(),
            target_node_data['test_mask'].sum().item(),
        )
        print(f'\tSplits (train, valid, test): {splits}', file=sio)

    gdata = getattr(hg, 'gdata', None)
    if gdata is not None:
        print(f'NOTE: graph data has not been supported by dgl.', file=sio)
        assert isinstance(gdata, Mapping)
        for key, data in gdata.items():
            if isinstance(data, Mapping):
                print(f'GData.{key}:', file=sio)
                for k, v in data.items():
                    print(f'\t{k}:', end=' ', file=sio)
                    if isinstance(v, torch.Tensor):
                        print(f'{tuple(v.shape)}', file=sio)
                    else:
                        print(str(v), file=sio)
            else:
                print(f'GData.{key}: {str(data)}', file=sio)

    info_ = sio.getvalue()
    sio.close()
    return info_


EType = str | tuple[str, str, str]


def _multi_class_node_homophily(
    graph: dgl.DGLGraph,
    y: torch.Tensor,
    mode: Literal['pos', 'neg', 'both'] = 'pos',
):
    if mode == 'both':
        return dgl.homophily.node_homophily(graph, y)

    with graph.local_scope():
        # Handle the case where graph is of dtype int32.
        src, dst = graph.edges()
        src, dst = src.long(), dst.long()
        y = y.long()

        if mode == 'pos':
            graph.edata['pos'] = (y[src] & y[dst]).float()
            graph.update_all(fn.copy_e('pos', 'p'), fn.mean('p', 'pos_degree'))
            return graph.ndata['pos_degree'][y == 1].mean(dim=0).item()

        assert mode == 'neg'
        graph.edata['neg'] = ((y[src] == 0) & (y[dst] == 0)).float()
        graph.update_all(fn.copy_e('neg', 'n'), fn.mean('n', 'neg_degree'))
        return graph.ndata['neg_degree'][y == 0].mean(dim=0).item()


def node_homophily(
    hg: BaseHeteroGraphLike,
    metapaths_: list[list[EType]] = None,
    allow_self_loop: bool = False,
    multi_label_mode: Literal['pos', 'neg', 'both'] = 'pos',
):
    """Calculate the node homophily of given hg. The hg will first be transformed to
    homogeneous graph whose edges are subject to provided meta-paths.

    Args:
        hg (BaseHeteroGraphLike)
        metapaths_ (list[list[EType]], optional): list of methpaths. If not provided,
            `hgget.metapaths(hg).values()` will be used.
            Note that a metapath is a list of Etype.
        allow_self_loop (bool): whether allow the self loops of the reduced homogeneous graph.
            Default to False.
        multi_label_mode (Literal['pos', 'neg', 'both']): mode to calculate the homophily.
            If 'pos', the homophily will be calculated only considering whether a pair of
                nodes that are connected are both positive.
            If 'neg', the homophily will be calculated only considering whether a pair of
                nodes that are connected are both negative.
            If 'both', the homophily will be calculated considering whether a pair of
                nodes that are connected are in the same class (either both
                positive or both negative)
            This argument would be ignored when the given `hg` is single-label

    Returns:
        return a float if the hg is single-label
        return a list[float] if the hg is multi-label
    """

    if metapaths_ is None:
        metapaths_ = list(metapaths(hg).values())

    labels = H.label(hg)
    g = to_homogeneous_wrt_metapaths(hg, metapaths_)
    if not allow_self_loop:
        g: dgl.DGLGraph = dgl.compact_graphs(dgl.remove_self_loop(g))
        labels = labels[g.ndata[dgl.NID]]

    homophilies = np.zeros(labels.shape[-1])
    if H.is_multi_label(hg):
        for i in range(labels.shape[-1]):
            homophilies[i] = _multi_class_node_homophily(
                g, labels[:, i], multi_label_mode
            )
        return homophilies

    homophily: float = dgl.homophily.node_homophily(g, labels)
    return homophily


def _multi_class_edge_homophily(
    graph: dgl.DGLGraph,
    y: torch.Tensor,
    mode: Literal['pos', 'neg', 'both'] = 'pos',
):
    if mode == 'both':
        return dgl.homophily.edge_homophily(graph, y)

    with graph.local_scope():
        # Handle the case where graph is of dtype int32.
        src, dst = graph.edges()
        src, dst = src.long(), dst.long()
        y = y.long()

        if mode == 'pos':
            mask = (y[src] | y[dst]).bool()
            pos_edges = (y[src] & y[dst])[mask].float()
            return pos_edges.mean(dim=0).item()

        assert mode == 'neg'
        y_src_neg = y[src] == 0
        y_dst_neg = y[dst] == 0
        mask = (y_src_neg | y_dst_neg).bool()
        neg_edges = (y_src_neg & y_dst_neg)[mask].float()
        return neg_edges.mean(dim=0).item()


def edge_homophily(
    hg: BaseHeteroGraphLike,
    metapaths_: list[list[EType]] = None,
    allow_self_loop: bool = False,
    multi_label_mode: Literal['pos', 'neg', 'both'] = 'pos',
):
    """Calculate the edge homophily of given hg. The hg will first be transformed to
    homogeneous graph whose edges are subject to provided meta-paths.

    Args:
        hg (BaseHeteroGraphLike)
        metapaths_ (list[list[EType]], optional): list of methpaths. If not provided,
            `hgget.metapaths(hg).values()` will be used.
            Note that a metapath is a list of Etype.
        allow_self_loop (bool): whether allow the self loops of the reduced homogeneous graph.
            Default to False.
        multi_label_mode (Literal['pos', 'neg', 'both']): mode to calculate the homophily.
            If 'pos', the homophily will be calculated only considering whether a pair of
                nodes that are connected are both positive.
            If 'neg', the homophily will be calculated only considering whether a pair of
                nodes that are connected are both negative.
            If 'both', the homophily will be calculated considering whether a pair of
                nodes that are connected are in the same class (either both
                positive or both negative)
            This argument would be ignored when the given `hg` is single-label

    Returns:
        return a float if the hg is single-label
        return a list[float] if the hg is multi-label
    """

    if metapaths_ is None:
        metapaths_ = list(metapaths(hg).values())

    labels = H.label(hg)
    g = to_homogeneous_wrt_metapaths(hg, metapaths_)
    if not allow_self_loop:
        g: dgl.DGLGraph = dgl.compact_graphs(dgl.remove_self_loop(g))
        labels = labels[g.ndata[dgl.NID]]

    homophilies = np.zeros(labels.shape[-1])
    if H.is_multi_label(hg):
        for i in range(labels.shape[-1]):
            homophilies[i] = _multi_class_edge_homophily(
                g, labels[:, i], multi_label_mode
            )
        return homophilies

    homophily: float = dgl.homophily.edge_homophily(g, labels)
    return homophily


def metapaths(hg: BaseHeteroGraphLike, max_hops: int = None):
    """get metapaths of a hgraph.

    Args:
        hg (BaseHeteroGraphLike)
        max_hops (int, optional): set the longest length of metapaths. If omitted,
            all of the metapaths regardless of length will be returned.

    Returns:
        dict[node_based_meta_path, edge_based_meta_path]

    Examples:
        >>> hgget.metapaths(imdb_hg)
        {
            ('movie', 'actor', 'movie'): ['stars', 'acts'],
            ('movie', 'director', 'movie'): ['directed-by', 'directed'],
            ('movie', 'keyword', 'movie'): ['contains', 'is-in']
        }
    """
    edict = {}
    etypes_from_src_ntype = defaultdict(list)
    dst_ntype_from_etype = {}
    for src_ntype, etype, dst_ntype in hg.canonical_etypes:
        edict[(src_ntype, dst_ntype)] = etype
        etypes_from_src_ntype[src_ntype].append(etype)
        assert etype not in dst_ntype_from_etype
        dst_ntype_from_etype[etype] = dst_ntype

    max_hops = max_hops or 100000

    def traverse_metapaths(
        ntype: str, path: list[str] = None, npath: tuple[str] = None
    ):

        if path is None:
            path = []
            npath = (ntype, )

        if len(path) >= max_hops:
            return

        for etype in etypes_from_src_ntype[ntype]:
            dst_ntype = dst_ntype_from_etype[etype]
            if etype in path:
                continue
            if dst_ntype == H.tgt_ntype(hg):
                yield (npath + (dst_ntype, ), path + [etype])
            else:
                yield from traverse_metapaths(
                    dst_ntype, path + [etype], npath + (dst_ntype, )
                )

    paths = dict(traverse_metapaths(H.tgt_ntype(hg)))
    return paths
