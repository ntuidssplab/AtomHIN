from __future__ import annotations

import warnings
from typing import Iterable, Literal, TypedDict, get_args

import torch
from dgl import DGLHeteroGraph
from dgl import function as fn
from dgl.heterograph import DGLBlock
from torch import nn
from torch.nn import functional as F

from ...type import CEType, EType, NType
from ...utils import gdata as gdata_utils
from ..common import MaxNormLinear, SharedSpaceProjection
from ..tools import scatter_edge_softmax

LinearByT = Literal['identity', 'ntype', 'shared', 'etype']


class RGCNConv(nn.Module):

    def __init__(
        self,
        in_size: int,
        rel_emb_size: int,
        head_size: int,
        num_heads: int,
        ntypes: list[str],
        cetypes: list[CEType],
        feat_drop=0.5,
        edge_drop=0.5,
        use_norm=False,
        residual: bool | float = True,
        linear_by: LinearByT = 'ntype',
        activation=None,
        aggregation: Literal['macro', 'micro'] = 'macro',
        softmax_tau=1.,
    ):
        super().__init__()
        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.use_norm = use_norm

        self.etypes = cetypes
        if len(cetypes):
            assert isinstance(cetypes[0], tuple)
        self.tau = softmax_tau
        self._rel_alpha = (rel_emb_size / num_heads)**0.5
        self._rel_w = nn.Parameter(
            torch.Tensor(size=(len(cetypes), num_heads))
        )
        assert linear_by in get_args(LinearByT)
        self.linear_by = linear_by
        if linear_by == 'ntype':
            self.linear = nn.ModuleDict(
                {
                    ntype: nn.Linear(in_size, head_size * num_heads)
                    for ntype in ntypes
                }
            )
        elif linear_by == 'shared':
            self._linear = nn.Linear(in_size, head_size * num_heads)
            self.linear = {ntype: self._linear for ntype in ntypes}
        elif linear_by == 'identity':
            self._linear = nn.Identity()
            self.linear = {ntype: self._linear for ntype in ntypes}
        else:
            self.linear = nn.ModuleDict(
                {
                    etype[1]: nn.Linear(in_size, head_size * num_heads)
                    for etype in cetypes
                }
            )
        self.feat_drop = nn.Dropout(feat_drop)
        self.edge_drop = nn.Dropout1d(edge_drop)
        self.norms = None
        if use_norm:
            self.norms = nn.ModuleDict(
                {
                    ntype: nn.GroupNorm(num_heads, head_size * num_heads)
                    for ntype in ntypes
                }
            )
        self._etype_to_id = {etype: eid for eid, etype in enumerate(cetypes)}
        self._srctype_to_id = {
            srctype: i
            for i, srctype in
            enumerate(dict.fromkeys(cetype[0] for cetype in self.etypes))
        }
        self._num_srctypes = len(self._srctype_to_id)
        self._srctype_indicator = nn.Parameter(
            torch.tensor(
                [self._srctype_to_id[cetype[0]] for cetype in self.etypes]
            ),
            requires_grad=False,
        )
        self._dsttype_to_id = {
            dsttype: i
            for i, dsttype in
            enumerate(dict.fromkeys(cetype[-1] for cetype in self.etypes))
        }
        self._num_dsttypes = len(self._dsttype_to_id)
        self._dsttype_indicator = nn.Parameter(
            torch.tensor(
                [self._dsttype_to_id[cetype[-1]] for cetype in self.etypes]
            ),
            requires_grad=False,
        )
        if residual and in_size != head_size * num_heads:
            if linear_by == 'ntype':
                self.residual_fc = nn.ModuleDict(
                    {
                        ntype: nn.Linear(in_size, head_size * num_heads)
                        for ntype in ntypes
                    }
                )
            elif linear_by == 'shared':
                self._residual_fc = nn.Linear(in_size, head_size * num_heads)
                self.residual_fc = {
                    ntype: self._residual_fc
                    for ntype in ntypes
                }
            else:
                raise NotImplementedError
        self.residual = residual
        self.reset_parameters()
        self.activation = activation
        self.aggregation = aggregation
        return

    @property
    def rel_w(self):
        return self._rel_w * self._rel_alpha

    def reset_parameters(self):
        nn.init.normal_(self._rel_w, std=(1 / self._rel_alpha))
        return

    def homo_softmax(
        self,
        logits: torch.Tensor,
        hg: DGLHeteroGraph,
        srctypes: list[str] | None = None,
        edge_weight: dict | None = None,
    ):
        if srctypes is not None and len(srctypes) < self._num_srctypes:
            srctype_ids = {
                self._srctype_to_id[srctype]
                for srctype in srctypes
            }
            logits = logits.clone()
            for srctype_id in range(self._num_srctypes):
                if srctype_id in srctype_ids:
                    continue
                mask = self._srctype_indicator == srctype_id
                logits[mask] = -1e10
        with hg.local_scope():
            for eid, etype in enumerate(self.etypes):
                n_edges = hg.num_edges(etype=etype)
                if n_edges == 0:
                    continue
                tem = logits[eid].view(1, -1).tile(n_edges, 1)
                if etype in edge_weight:
                    tem *= edge_weight[etype].view(-1, 1)
                hg.edges[etype].data['a'] = tem
            return scatter_edge_softmax(hg, hg.edata['a'])

    def dst_wise_softmax(
        self, logits: torch.Tensor, srctypes: list[NType] | None = None
    ):

        if srctypes is not None and len(srctypes) < self._num_srctypes:
            srctype_ids = {
                self._srctype_to_id[srctype]
                for srctype in srctypes
            }
            logits = logits.clone()
            for srctype_id in range(self._num_srctypes):
                if srctype_id in srctype_ids:
                    continue
                mask = self._srctype_indicator == srctype_id
                logits[mask] = -1e10
        res = torch.zeros_like(logits)
        for dsttype_id in range(self._num_dsttypes):
            mask = self._dsttype_indicator == dsttype_id
            res[mask] = torch.softmax(logits[mask], 0)
        return res

    def message(
        self,
        src_key: str,
        dst_key: str,
        etype: CEType,
        ref_hg: DGLHeteroGraph,
        rel_weight: torch.Tensor = None,
        att_weight: torch.Tensor = None,
        edge_weight_key: str | None = None,
    ):
        """Message function."""

        if att_weight is not None:
            ref_hg.edges[etype].data['a_'] = self.edge_drop(
                att_weight[etype],
            ).unsqueeze(-1)

        def edge_attention(edges):
            if self.linear_by == 'etype':
                m = self.linear[etype[1]](edges.src[src_key])
                m = m.view(-1, self.num_heads, self.head_size)
            else:
                m = edges.src[src_key]

            if rel_weight is not None:
                assert att_weight is None
                w = rel_weight.view(1, -1).tile(m.shape[0], 1)
                # w = self.drop(w)
                w = self.edge_drop(w)
                m *= w.view(m.shape[0], self.num_heads, 1)
            else:
                assert rel_weight is None
                m *= edges.data['a_']
            if edge_weight_key is not None:
                m *= edges.data[edge_weight_key].view(-1, 1, 1)
            return {dst_key: m}

        return edge_attention

    def dense_update_all(
        self, hg: DGLHeteroGraph, etype: CEType, src_key,
        rel_weight: torch.Tensor, dst_key
    ):
        """message passing for dense adj.
        The adj should have been row normalized so that this is equivalent to mean reduce.
        """
        assert hg.num_edges(etype) == 0
        x = hg.nodes[etype[0]].data[src_key]
        adj = F.dropout(
            gdata_utils.gdata(hg)['adj'][etype], p=self.edge_drop.p,
            training=self.training
        )
        x = torch.einsum('ab,bcd->acd', adj, x)
        w = rel_weight.view(1, -1).tile(x.shape[0], 1)
        # w = self.edge_drop(w)
        x *= w.view(x.shape[0], self.num_heads, 1)
        hg.nodes[etype[-1]].data[dst_key] += x
        return

    def forward(
        self,
        hg: DGLHeteroGraph,
        xs: dict[NType, torch.Tensor],
        edge_weight: dict[EType, torch.Tensor | str] | None,
    ):
        if edge_weight is not None:
            edge_weight = {
                hg.to_canonical_etype(etype): eweight
                for etype, eweight in edge_weight.items()
            }
        else:
            edge_weight = {}
        if 'adj' in gdata_utils.gdata(hg):  # hg with dense adjs
            assert not hg.is_block
        if hg.is_block:
            raise NotImplementedError
            xs_src = {}
            xs_dst = {}
            for ntype, x in xs.items():
                xs_src[ntype] = x
                xs_dst[ntype] = x[:hg.num_dst_nodes(ntype)]
        else:
            xs_src = xs
            xs_dst = xs

        with hg.local_scope():
            if edge_weight:
                hg.edata['w'] = edge_weight
            if self.linear_by == 'etype':
                for ntype in hg.srctypes:
                    if ntype in xs_src:
                        h = self.feat_drop(xs_src[ntype])
                        hg.srcnodes[ntype].data['v'] = h
            else:
                for ntype in hg.srctypes:
                    if ntype in xs_src:
                        h = self.linear[ntype](self.feat_drop(xs_src[ntype]))
                        h = h.view(-1, self.num_heads, self.head_size)
                        hg.srcnodes[ntype].data['v'] = h

            if self.aggregation == 'macro':
                # rel_w: #etypes x #heads
                w = self.rel_w
                rel_w = self.dst_wise_softmax(w / self.tau, srctypes=xs_src)

                hg.multi_update_all(
                    {
                        cetype: (
                            self.message(
                                'v',
                                'm',
                                cetype,
                                hg,
                                rel_weight=rel_w[i],
                                edge_weight_key='w'
                                if cetype in edge_weight else None,
                            ), fn.mean('m', 'h')
                        )
                        for i, cetype in enumerate(self.etypes)
                        if hg.num_edges(cetype) and cetype[0] in xs_src
                    }, cross_reducer='sum'
                )
                if 'adj' in gdata_utils.gdata(hg):
                    # propagate for dense etypes
                    for etype in gdata_utils.gdata(hg)['adj']:
                        self.dense_update_all(
                            hg, etype, 'v', rel_w[self._etype_to_id[etype]],
                            'h'
                        )

            else:
                assert self.aggregation == 'micro'
                # rel_w: Dict[etype, #edges x #heads]
                att = self.homo_softmax(
                    self.rel_w / self.tau, hg, srctypes=xs_src,
                    edge_weight=edge_weight
                )
                hg.multi_update_all(
                    {
                        etype: (
                            self.message('v', 'm', etype, hg,
                                         att_weight=att), fn.sum('m', 'h')
                        )
                        for i, etype in enumerate(self.etypes)
                        if hg.num_edges(etype)
                    }, cross_reducer='sum'
                )
                if 'adj' in gdata_utils.gdata(hg):
                    # propagate for dense etypes
                    rel_w = self.dst_wise_softmax(
                        self.rel_w / self.tau, srctypes=xs_src
                    )
                    for dsttype in hg.dsttypes:
                        etypes = [
                            etype for etype in gdata_utils.gdata(hg)['adj']
                            if etype[-1] == dsttype
                        ]
                        if len(etypes):
                            rel_ws = torch.stack(
                                [
                                    rel_w[self._etype_to_id[etype]]
                                    for etype in etypes
                                ]
                            )
                            rel_ws = rel_ws.sum(dim=0).unsqueeze(-1)
                            hg.dstnodes[dsttype].data['h'] *= (1 - rel_ws)

                    for etype in gdata_utils.gdata(hg)['adj']:
                        self.dense_update_all(
                            hg, etype, 'v', rel_w[self._etype_to_id[etype]],
                            'h'
                        )

            out = {
                ntype: h.view(-1, self.num_heads * self.head_size)
                for ntype, h in hg.dstdata['h'].items()
            }

            if self.norms:
                for ntype, h in out.items():
                    out[ntype] = self.norms[ntype](h)

            if self.residual:
                for ntype, h in out.items():
                    if ntype not in xs_dst:
                        continue
                    if h.shape[-1] != xs_dst[ntype].shape[-1]:
                        raise NotImplementedError
                        # out[ntype] = h + self.residual_fc[ntype](xs_dst[ntype])
                    elif self.residual is True:
                        out[ntype] = h + xs_dst[ntype]
                    else:
                        assert isinstance(self.residual,
                                          float) and (0. < self.residual < 1.)
                        out[ntype] = h * (1 - self.residual
                                          ) + xs_dst[ntype] * self.residual

            # activation
            if self.activation:
                for ntype, h in out.items():
                    out[ntype] = self.activation(h)
            return out


class RGCN(nn.Module):

    class SharedFeatProjArgs(TypedDict):
        in_feat_shapes: dict[NType, tuple[int, ...]] | int
        embedding_max_norm: float | None

    LinearByT = LinearByT

    def __init__(
        self,
        n_hidden: int,
        n_layers: int,
        n_heads: int,
        n_out: int,
        ntypes: list[NType],
        etypes: list[CEType],
        target_ntype: NType | list[NType],
        activation,
        linear_by: LinearByT | list[LinearByT],
        use_norm=True,
        residual: bool | float = True,
        feat_drop: float = 0.5,
        edge_drop: float = 0.5,
        proj_args: SharedFeatProjArgs | None = None,
        n_out_layers: int = 1,
        softmax_tau: float = 1.,
        aggregation: Literal['macro', 'micro'] = 'macro',
    ):
        super().__init__()
        in_size = n_hidden * n_heads
        if not isinstance(linear_by, str):
            assert len(linear_by) == n_layers
        else:
            linear_by = [linear_by] * n_layers
        assert aggregation in ('macro', 'micro')
        self.gcs: list[RGCNConv] = nn.ModuleList(
            [
                RGCNConv(
                    in_size=in_size,
                    rel_emb_size=in_size,
                    head_size=n_hidden,
                    num_heads=n_heads,
                    ntypes=ntypes,
                    cetypes=etypes,
                    use_norm=use_norm,
                    residual=residual if i > 0 else False,
                    feat_drop=feat_drop,
                    edge_drop=edge_drop,
                    activation=activation,
                    linear_by=linear_by[i],
                    aggregation=aggregation,
                    softmax_tau=softmax_tau,
                ) for i in range(n_layers)
            ]
        )
        self.n_hid = n_hidden
        self.n_out = n_out
        self.n_layers = n_layers
        self.proj = None
        if proj_args is not None:
            if isinstance(proj_args, int):
                self._proj = nn.Linear(proj_args, in_size)
                self.proj = {ntype: self._proj for ntype in ntypes}
                warnings.warn(
                    'Deprecated: Use {"in_feat_shapes": <dim>} instead.'
                )
            elif isinstance(proj_args['in_feat_shapes'], int):
                self._proj = MaxNormLinear(
                    proj_args['in_feat_shapes'], in_size,
                    max_norm=proj_args['embedding_max_norm']
                )
                self.proj = {ntype: self._proj for ntype in ntypes}
            else:
                self.proj = SharedSpaceProjection(
                    n_out=in_size,
                    **proj_args,
                )

        self.target_ntype = target_ntype
        if isinstance(target_ntype, str):
            self.out = self._construct_out_layers(
                n_out_layers,
                n_heads=n_heads,
                head_size=n_hidden,
                dropout=feat_drop,
                n_out=n_out,
            )
        else:
            self.out = nn.ModuleDict(
                {
                    ntype:
                    self._construct_out_layers(
                        n_out_layers,
                        n_heads=n_heads,
                        head_size=n_hidden,
                        dropout=feat_drop,
                        n_out=n_out,
                    )
                    for ntype in target_ntype
                }
            )
        self.in_feat_norm = None
        self.in_feat_norm = nn.ModuleDict(
            # { ntype: nn.LayerNorm(in_size)
            {ntype: nn.GroupNorm(n_heads, in_size)
             for ntype in ntypes}
        )
        self.reset_parameters()
        return

    def _construct_out_layers(
        self, n_out_layers: int, n_heads: int, head_size: int, dropout: float,
        n_out
    ):
        if n_out_layers == 0:
            return nn.Identity()
        _cur_size = n_heads * head_size
        out = nn.Sequential()
        for _ in range(n_out_layers - 1):
            out.extend(
                [
                    nn.Linear(_cur_size, max(head_size, n_out)),
                    nn.LayerNorm(max(head_size, n_out)),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                ]
            )
            _cur_size = max(head_size, n_out)
        out.append(nn.Linear(_cur_size, n_out))
        return out

    def reset_parameters(self):
        for ntype in self.proj:
            nn.init.normal_(self.proj[ntype].weight, std=0.01)
            nn.init.zeros_(self.proj[ntype].bias)
        pass

    def _in_feat_projection_(self, feat: dict[NType, torch.Tensor]):
        assert self.proj is not None

        # feat = self.proj(feat)
        for ntype, x_ in feat.items():
            feat[ntype] = self.proj[ntype](x_)
        if self.in_feat_norm is not None:
            feat = {
                ntype: self.in_feat_norm[ntype](h)
                for ntype, h in feat.items()
            }
        return feat

    def forward(
        self,
        hg: DGLHeteroGraph | list[DGLBlock],
        feat: dict[NType, torch.Tensor],
        edge_weight: dict[EType, torch.Tensor | str] = None,
    ) -> torch.Tensor:

        feat = self._in_feat_projection_(feat)

        def graph_forward(hg: DGLHeteroGraph):
            nonlocal feat
            for i, gc in enumerate(self.gcs):
                feat_ = gc.forward(hg, feat, edge_weight)
                feat = feat_

            return feat

        def minibatch_forward(blocks: list[DGLBlock]):
            nonlocal feat
            for i, gc in enumerate(self.gcs):
                feat = gc.forward(blocks[i], feat, edge_weight)
            return feat

        forward_fn = (
            minibatch_forward if isinstance(hg, Iterable) else graph_forward
        )

        h = forward_fn(hg)
        if isinstance(self.target_ntype, str):
            return self.out(h[self.target_ntype])
        return {
            ntype: self.out[ntype](h[ntype])
            for ntype in self.target_ntype
        }
