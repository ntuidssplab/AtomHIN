from __future__ import annotations

import contextlib
import hashlib
import os
import warnings
from typing import Literal, overload

import dgl
import dgl.core
import torch
from dgl import function as fn
from filelock import FileLock
from packaging import version
from tqdm import tqdm

from dhgl.type import EType, NType

from .mixed_tensor import MixedTensor

if version.parse(torch.__version__) >= version.parse('2.1.0'):
    from torch._tensor_str import printoptions
else:

    @contextlib.contextmanager
    def printoptions(*args, **kwargs):
        print_options = {
            'precision': torch._tensor_str.PRINT_OPTS.precision,
            'threshold': torch._tensor_str.PRINT_OPTS.threshold,
            'edgeitems': torch._tensor_str.PRINT_OPTS.edgeitems,
            'linewidth': torch._tensor_str.PRINT_OPTS.linewidth,
            'sci_mode': torch._tensor_str.PRINT_OPTS.sci_mode,
        }

        try:
            yield torch.set_printoptions(*args, **kwargs)
        finally:
            torch.set_printoptions(**print_options)


def _dense_prepropagate(
    hg: dgl.DGLHeteroGraph,
    srctype: NType,
    feat: torch.Tensor,
    max_hops: int | None = None,
    reduce_fn: Literal['mean', 'absmax'] = 'mean',
    edge_weights: dict[EType, torch.Tensor] | None = None,
    verbose: bool = False,
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
        coo features
    """

    def abs_max(u, v):

        def fn(nodes):
            indices = nodes.mailbox[u].abs().argmax(dim=1, keepdim=True)
            res = torch.take_along_dim(nodes.mailbox[u], indices, dim=1)
            return {v: res.squeeze(dim=1)}

        return fn

    def cross_reduce_fn(results: list[torch.Tensor]):
        if len(results) > 1:
            # TODO: Add doc for this
            raise NotImplementedError
        return results[0]

    with hg.local_scope():
        if edge_weights is not None:
            hg.edata['w'] = edge_weights
        elif set(hg.edata) & {'w', 'weights', 'weight'}:
            keys = set(hg.edata) & {'w', 'weights', 'weight'}
            warnings.warn(
                f'edge_weights not set while edata[{keys}] detected.'
            )

        hs = {srctype: feat}

        def get_update_fns(etype):
            message_fn = (
                fn.u_mul_e('x', 'w', 'm')
                if 'w' in hg.edges[etype].data else fn.copy_u('x', 'm')
            )
            reduce_fn_ = (fn.mean if reduce_fn == 'mean' else abs_max)
            return message_fn, reduce_fn_('m', 'h')

        for i in range(max_hops or 99):
            valid_etypes = [
                (s, e, d) for s, e, d, in hg.canonical_etypes
                if s in hs and d not in hs
            ]
            if len(valid_etypes) == 0:
                break
            hg.ndata['x'] = hs

            for dsttype in hg.ntypes:
                etype_dict = {
                    etype: get_update_fns(etype)
                    for etype in valid_etypes if etype[-1] == dsttype
                }
                if not etype_dict:
                    continue

                if len(etype_dict) == 1:
                    # NOTE: this aims to avoid dense adj using large amount of memory by batching
                    # There should be only one etype to update while the case
                    srctype, etype, dsttype = list(etype_dict)[0]
                    density = hg.num_edges(
                        etype=etype
                    ) / (hg.num_nodes(srctype) * hg.num_nodes(dsttype))
                    if density > 0.3:
                        batches = torch.chunk(
                            torch.arange(hg.num_nodes(dsttype)),
                            min(hg.num_nodes(dsttype), hg.num_nodes(srctype))
                        )
                        for v in tqdm(
                            batches,
                            delay=5,
                            desc=f'Propagating {(srctype, etype, dsttype)}...',
                            disable=not verbose,
                            leave=False,
                        ):
                            hg.pull(
                                v, *etype_dict[srctype, etype, dsttype],
                                etype=etype
                            )
                        # h_ = hg.nodes[dsttype].data.pop('h')
                        # hg.multi_update_all(
                        #     etype_dict,
                        #     cross_reducer=cross_reduce_fn,
                        # )
                        # if not torch.allclose(
                        #     h_, hg.nodes[dsttype].data['h'], atol=1e-4
                        # ):
                        #     breakpoint()
                        # else:
                        #     print(f'{etype} checked ok')
                        continue

                hg.multi_update_all(
                    etype_dict,
                    cross_reducer=cross_reduce_fn,
                )
            hs.update(hg.ndata['h'])
    return hs


def _sparse_csr_prepropagate(
    hg: dgl.DGLHeteroGraph,
    srctype: NType,
    feat: torch.Tensor,
    max_hops: int | None = None,
    reduce_fn: Literal['mean', 'absmax'] = 'absmax',
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
        coo features
    """

    # assert feat.is_sparse
    feat = feat.to_sparse_csr()
    out_feats = {ntype: {} for ntype in hg.ntypes}
    out_feats[srctype] = feat

    def copy_u(u, out, e=None):

        def fn(edges: dgl.core.EdgeBatch):
            s = edges.src[u]
            res = {
                out:
                s,
                f'{out}_type':
                torch.full_like(s, hg.get_ntype_id(edges.canonical_etype[0]))
            }
            if e is not None:
                res[e] = edges.data[e]
            return res

        return fn

    def handle_id_feat(u, v, e=None):

        def wrap_(reduce_fn):

            def fn(nodes: dgl.core.NodeBatch):
                nonlocal out_feats
                src: torch.Tensor = nodes.mailbox[u]
                is_id = not src.is_floating_point()
                if is_id:
                    src: torch.Tensor = nodes.mailbox[u]
                    srctype = hg.ntypes[nodes.mailbox[f'{u}_type'].view(-1)
                                        [0].item()]
                    shape = src.shape + out_feats[srctype].shape[1:]
                    src = torch.stack(
                        [
                            out_feats[srctype][s].to_dense()
                            for s in src.reshape(-1)
                        ]
                    )
                    if e is not None:
                        src *= nodes.mailbox[e].view(-1, 1)
                    src = src.reshape(shape)
                res = reduce_fn(src)
                if is_id:
                    res = res.to_sparse_csr()
                    for i, nid in enumerate(nodes.nodes().tolist()):
                        out_feats[nodes.ntype][nid + 1] = res[i]
                    return {v: nodes.nodes() + 1}  # ID:0 as zero vector
                return {v: res}

            return fn

        return wrap_

    def abs_max(src: torch.Tensor):
        indices = src.abs().argmax(dim=1, keepdim=True)
        res = torch.take_along_dim(src, indices, dim=1)
        res = res.squeeze(dim=1)
        return res

    def mean(src: torch.Tensor):
        return src.mean(dim=1)

    def cross_reduce_fn(results: list[torch.Tensor]):
        if len(results) > 1:
            raise NotImplementedError
        return results[0]

    with hg.local_scope():
        if edge_weights is not None:
            hg.edata['w'] = edge_weights
        elif set(hg.edata) & {'w', 'weights', 'weight'}:
            keys = set(hg.edata) & {'w', 'weights', 'weight'}
            warnings.warn(
                f'edge_weights not set while edata[{keys}] detected.'
            )

        hs = {srctype: torch.arange(len(feat))}

        def get_update_fns(etype):
            reduce_fn_ = (mean if reduce_fn == 'mean' else abs_max)
            if etype in edge_weights:
                return copy_u('x', 'm', 'w'), handle_id_feat('m', 'h',
                                                             'w')(reduce_fn_)
            return copy_u('x', 'm'), handle_id_feat('m', 'h')(reduce_fn_)

        for i in range(max_hops or 99):
            valid_etypes = [
                (s, e, d) for s, e, d, in hg.canonical_etypes
                if s in hs and d not in hs
            ]
            if len(valid_etypes) == 0:
                break
            hg.ndata['x'] = hs

            for dsttype in hg.ntypes:
                etype_dict = {
                    etype: get_update_fns(etype)
                    for etype in valid_etypes if etype[-1] == dsttype
                }
                if not etype_dict:
                    continue
                hg.multi_update_all(
                    etype_dict,
                    cross_reducer=cross_reduce_fn,
                )
                out_feats[dsttype] = torch.stack(
                    [
                        out_feats[dsttype].get(
                            i,
                            torch.zeros(
                                feat.shape[1:], layout=torch.sparse_coo
                            )
                        ) for i in hg.nodes[dsttype].data.pop('h').tolist()
                    ]
                )
                out_feats[dsttype] = out_feats[dsttype].to_sparse_csr()
                hs[dsttype] = torch.arange(hg.num_nodes(dsttype))
    return out_feats


def _slot_prop(
    hg: dgl.DGLHeteroGraph,
    ntype: NType,
    feat: torch.Tensor,
    max_hops: int | None = None,
    reduce_fn: Literal['mean', 'absmax'] = 'absmax',
    edge_weights: dict[EType, torch.Tensor] | None = None,
    cache_dir: str | None = None,
    check_feathash: bool = True,
    verbose: bool | None = None,
    pbar=None,
):
    # NOTE: feathash only checks parts of the feature and shape of the feature
    # This should be safe in most cases
    # COLLISION SUPPOSED TO BE AVOIDED BY SPECIFYING DIFFERENT CACHE_DIR
    with printoptions(profile='default'):
        feathash = hashlib.sha1(f'{feat}shape={feat.shape}'.encode()
                                ).hexdigest()

    if check_feathash:
        cache_file = os.path.join(cache_dir or '', f'{ntype}.pt')
    else:
        cache_file = os.path.join(
            cache_dir or '', f'{ntype}_{feathash[:8]}.pt'
        )
    if cache_dir is not None and os.path.isfile(cache_file):
        if verbose:
            print(f'cache found at {cache_file}', end='', file=pbar)
        cache = torch.load(cache_file)
        if cache.pop('feathash') != feathash:
            raise RuntimeError(
                f'Hash mismatch for {cache_file}! '
                'Specify a different cache_dir to avoid collisions, '
                'required after node features have changed.'
            )
        return cache

    if len(feat.shape) != 2 and not feat.is_floating_point():
        raise ValueError('Feature format not supported.')

    if feat.layout != torch.strided:
        feat_fmt = feat.layout
        xs = _sparse_csr_prepropagate(
            hg,
            ntype,
            feat.to_sparse_csr(),
            max_hops=max_hops,
            reduce_fn=reduce_fn,
            edge_weights=edge_weights,
        )
        for nt, x in xs.items():
            xs[nt] = x.to_sparse(layout=feat_fmt)
    else:
        xs = _dense_prepropagate(
            hg,
            ntype,
            feat,
            max_hops=max_hops,
            reduce_fn=reduce_fn,
            edge_weights=edge_weights,
            verbose=verbose,
        )
    if cache_dir is not None:
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        with FileLock(f'{cache_file}.lock'):
            torch.save({**xs, 'feathash': feathash}, cache_file)
        if verbose:
            print(f'cache saved at {cache_file}', end='', file=pbar)
    return xs


@overload
def prepropagate(
    hg: dgl.DGLHeteroGraph,
    feats: dict[NType, torch.Tensor],
    edge_weights: dict[EType, torch.Tensor] | None = None,
    max_hops: int | None = None,
    reduce_fn: Literal['mean', 'absmax'] = 'absmax',
    cache_dir: str | None = None,
    cache_check_feathash: bool | None = None,
    to_sparse_threshold: float | None = None,
    return_slots: False = False,
    verbose: bool | None = None,
) -> dict[NType, torch.Tensor]:
    ...


@overload
def prepropagate(
    hg: dgl.DGLHeteroGraph,
    feats: dict[NType, torch.Tensor],
    edge_weights: dict[EType, torch.Tensor] | None,
    max_hops: int | None = None,
    reduce_fn: Literal['mean', 'absmax'] = 'absmax',
    cache_dir: str | None = None,
    cache_check_feathash: bool | None = None,
    to_sparse_threshold: float | None = None,
    return_slots: True = True,
    verbose: bool | None = None,
) -> dict[NType, dict[NType, torch.Tensor]]:
    ...


def prepropagate(
    hg: dgl.DGLHeteroGraph,
    feats: dict[NType, torch.Tensor],
    edge_weights: dict[EType, torch.Tensor] | None = None,
    max_hops: int | None = None,
    reduce_fn: Literal['mean', 'absmax'] = 'absmax',
    cache_dir: str | None = None,
    cache_check_feathash: bool | None = None,
    to_sparse_threshold: float | None = None,
    return_slots: bool = False,
    verbose: bool | None = None,
) -> dict[NType, torch.Tensor | dict[NType, torch.Tensor]]:
    """Prepropagated node features on a heterogeneous graph.

    For each source node type, this function propagates its features to all
    node types up to `max_hops` using the specified reducer, then (by default)
    concatenates the per-source contributions for each destination node type.
    Dense and sparse input features are supported; outputs follow the input
    layout, and mixed dense/sparse outputs are wrapped in a `MixedTensor`.

    Args:
        hg (dgl.DGLHeteroGraph): Input heterogeneous graph.
        feats (dict[NType, torch.Tensor]): Node features per node type to be
            propagated. Each tensor can be dense or sparse (CSR/COO).
        edge_weights (dict[EType, torch.Tensor] | None, optional): Optional edge
            weights per edge type used during propagation. If ``None``, edges are
            treated as unweighted.
        max_hops (int | None, optional): Maximum propagation hops. If ``None``,
            an internal default is used.
        reduce_fn (Literal['mean', 'absmax'], optional): Aggregation used when
            combining messages at each hop. Use ``'mean'`` for averaging or
            ``'absmax'`` for elementwise maximum by absolute value. Defaults to
            ``'absmax'``.
        cache_dir (str | None, optional): Directory to cache/load precomputed
            propagated features. If specified, the function will attempt to load
            cached results (with a basic hash check) or write new results to this
            directory. A new directory must be specified if the input node
            features, graph structure, or dataset changes. Defaults to ``None``.
        to_sparse_threshold (float): 0-1 threshold to auto convert features that are too sparse
        into sparse format.
        return_slots (bool, optional): If ``False`` (default), returns, for each
            destination node type, the concatenation (along feature dim=1) of
            contributions from all source types. If ``True``, returns the
            un-concatenated per-source contributions as nested dicts.
        verbose (bool | None, optional): Enable verbose in dense backend.

    Returns:
        dict[NType, torch.Tensor | dict[NType, torch.Tensor]]: If
        ``return_slots`` is ``False``, a dict mapping each destination node type
        to a single tensor of concatenated propagated features. If
        ``return_slots`` is ``True``, a dict mapping each destination node type
        to a dict of per-source tensors (no concatenation).

    Examples:
        >>> feats = prepropagate(
        ...     hg, hg.ndata['feat'], edge_weights=hg.edata['weight']
        ... )
        >>> feats
        {'ntype1': tensor(...), 'ntype2': tensor(...)}

        >>> slot_feats = prepropagate(
        ...     hg, hg.ndata['feat'], edge_weights=hg.edata['weight'], return_slots=True
        ... )
        >>> slot_feats
        {'ntype1': {'ntype1': tensor(...), 'ntype2': tensor(...)},
         'ntype2': {'ntype1': tensor(...), 'ntype2': tensor(...)}}

        >>> torch.allclose(
        ...     feats['ntype1'],
        ...     torch.concat(list(slot_feats['ntype1'].values()), dim=1)
        ... )
        True
    """
    if reduce_fn == 'max':
        # backward compatibility
        warnings.warn(
            'reduce_fn=="max" has deprecated, use "absmax" to avoid confusion.'
        )
        reduce_fn = 'absmax'
    out_feats = {}
    with tqdm(list(feats), disable=(not verbose or len(feats) == 1)) as pbar:
        for srctype in pbar:
            pbar.set_description(f'Propagating feats for {srctype}')
            xs = _slot_prop(
                hg,
                srctype,
                feats[srctype],
                max_hops=max_hops,
                reduce_fn=reduce_fn,
                edge_weights=edge_weights,
                cache_dir=cache_dir,
                check_feathash=True
                if cache_check_feathash is None else cache_check_feathash,
                verbose=verbose,
                pbar=pbar,
            )
            for nt, x in xs.items():
                if nt not in out_feats:
                    out_feats[nt] = {}
                out_feats[nt][srctype] = x

    if return_slots:
        if to_sparse_threshold is not None:
            warnings.warn(
                f'{to_sparse_threshold=} is set but has not effect since return_slots=True.'
            )
        return out_feats

    to_sparse_threshold = to_sparse_threshold or 0

    def _concat(xs: dict[str, torch.Tensor]):
        density = [
            (x != 0).sum() / x.numel() if x.layout == torch.strided else -1
            for x in xs.values()
        ]
        dense_feats = [
            x for d, x in zip(density, xs.values()) if d >= to_sparse_threshold
        ]
        sparse_feats = [
            x for d, x in zip(density, xs.values()) if d < to_sparse_threshold
        ]
        if len(sparse_feats) == 0:
            return torch.concat(dense_feats, dim=1)
        sparse_feats = [x.to_sparse_coo() for x in sparse_feats]
        if verbose and any(d >= 0 for d in density):
            ntypes = [f'{ntype}({d:.2e})' for ntype, d in zip(xs, density)]
            print(
                f'Sparse feat detected(threshold={to_sparse_threshold}): {", ".join(ntypes)}'
            )
        if len(dense_feats) == 0:
            return torch.concat(sparse_feats, dim=1)
        warnings.warn(
            'Detected both sparse and dense features during prepropagation. '
            'This results in the creation of a custom torch.Tensor extension: MixedTensor. '
            'MixedTensor is experimental and currently only supports being passed to '
            'torch.nn.Linear. Use with caution.'
        )
        dense_feats = torch.concat(dense_feats, dim=1)
        sparse_feats = torch.concat(sparse_feats, dim=1)
        return MixedTensor(dense_feats, sparse_feats, dim=1)

    # def _post_concat(xs: dict[str, torch.Tensor]):
    #     dense_feats = [x for x in xs.values() if x.layout == torch.strided]
    #     sparse_feats = [x for x in xs.values() if x.layout != torch.strided]
    #     if len(sparse_feats) == 0:
    #         return torch.concat(dense_feats, dim=1)
    #     sparse_feats = [x.to_sparse_coo() for x in sparse_feats]
    #     if len(dense_feats) == 0:
    #         return torch.concat(sparse_feats, dim=1)
    #     dense_feats = torch.concat(dense_feats, dim=1)
    #     sparse_feats = torch.concat(sparse_feats, dim=1)
    #     return MixedTensor(dense_feats, sparse_feats, dim=1)

    return {ntype: _concat(xs) for ntype, xs in out_feats.items()}
