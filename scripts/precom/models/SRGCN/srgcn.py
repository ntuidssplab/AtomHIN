from __future__ import annotations

import math
from itertools import product
from typing import Literal, NamedTuple, TypedDict

import dgl
import torch
from dgl import function as fn
from torch import nn
from typing_extensions import deprecated

from dhgl.models.common import SharedSpaceProjection
from dhgl.type import CEType, NType
from dhgl.utils.precomputation import LowRankMatrix
from dhgl.utils.precomputation.metagraph import SelfLoop


def construct_metagraph(ntypes: list[NType], cetypes: list[CEType]):
    get_ntype_id = {ntype: nid for nid, ntype in enumerate(ntypes)}
    edges = [
        (get_ntype_id[srctype], get_ntype_id[dsttype])
        for srctype, _, dsttype in cetypes
    ]
    edges = torch.tensor(edges)
    return dgl.graph((edges[:, 0], edges[:, 1]))


class _BuildMetaAdjsReturnT(NamedTuple):
    meta_adjs: list[torch.Tensor]
    mp_list: list[tuple[CEType]]


def build_meta_adjs(
    ntypes: list[NType],
    cetypes: list[CEType],
    self_loop_etypes: list[SelfLoop],
    num_layers: int,
):

    class MPDomain:

        def __init__(self, self_loops: list[int]):
            self.mp_domain = {
                (sl, ): mp_id
                for mp_id, sl in enumerate(self_loops)
            }
            self.ptr = len(self_loops)
            self.self_loops = set(self_loops)
            return

        def _is_valid(self, mp: tuple[int]):
            cur = cetypes[mp[0]][-1]
            for eid in mp[1:]:
                s, _, d = cetypes[eid]
                if s != cur:
                    return False
                cur = d
            return True

        def _rm_self_loops(self, mp: tuple[int]):
            return tuple(
                [
                    mp[0],
                    *[eid for eid in mp[1:] if eid not in self.self_loops]
                ]
            )

        def update(self, mp: tuple[int]):
            if not self._is_valid(mp):
                return -1
            mp = self._rm_self_loops(mp)
            mp_id = self.mp_domain.get(mp, None)
            if mp_id is not None:
                return mp_id
            self.mp_domain[mp] = self.ptr
            self.ptr += 1
            return self.ptr - 1

        def get(self, item: tuple[int]):
            return self.mp_domain[self._rm_self_loops(item)]

        @property
        def mp_list(self) -> list[tuple[int]]:
            mps = [None] * len(self)
            for mp, mpid in self.mp_domain.items():
                mps[mpid] = mp
            return mps

        def __len__(self):
            return self.ptr

    assert len(self_loop_etypes) == len(ntypes)
    cetypes = cetypes.copy()
    for self_loop_etype in self_loop_etypes:
        assert isinstance(self_loop_etype, tuple)
        if not isinstance(self_loop_etype, SelfLoop):
            raise ValueError(
                f'Only accept self-loop etype wrapped with type {SelfLoop}'
            )
        if self_loop_etype not in cetypes:
            cetypes.append(self_loop_etype)
    self_loops = [
        cetypes.index(self_loop_etype) for self_loop_etype in self_loop_etypes
    ]
    mp_domain = MPDomain(self_loops)

    # meta_adj(r): MP -> MP
    meta_adjs = [[] for _ in cetypes]

    for l in range(num_layers + 1):
        if l == 0:
            continue

        for left_mp, r in product(mp_domain.mp_domain, range(len(cetypes))):
            mp = (*left_mp, r)

            r = mp[-1]
            mp_id = mp_domain.update(mp)
            if mp_id < 0:  # invalid metapath
                continue
            meta_adjs[r].append((mp_domain.get(mp[:-1]), mp_id))

    meta_adjs = [
        torch.sparse_coo_tensor(
            torch.tensor(adj).T,
            torch.ones(len(adj)).bool(), size=(len(mp_domain), len(mp_domain))
        ).coalesce().float() for adj in meta_adjs
    ]
    mp_list = mp_domain.mp_list
    mp_list = [tuple(cetypes[e] for e in mp) for mp in mp_list]
    return _BuildMetaAdjsReturnT(meta_adjs, mp_list)


class RGCNConvs(nn.Module):

    def __init__(
        self,
        rel_emb_size: int,
        num_heads: int,
        num_layers: int,
        ntypes: list[str],
        cetypes: list[CEType],
        residuals: list[bool],
        softmax_tau=1.,
    ):
        super().__init__()
        self.num_heads = num_heads

        self.etypes = cetypes
        self.tau = softmax_tau
        self._rel_alpha = (rel_emb_size / num_heads)**0.5
        self._rel_w = nn.Parameter(
            torch.Tensor(size=(len(cetypes), num_layers, num_heads))
        )
        self.residuals = residuals
        self._build = False
        self.reset_parameters()
        return

    @property
    def rel_w(self):
        return self._rel_w * self._rel_alpha

    def reset_parameters(self):
        nn.init.normal_(self._rel_w, std=(1 / self._rel_alpha))
        return

    def message(self, src, eweight, dst, meta_adjs: list[torch.Tensor]):

        def _message(edges):
            sids, dids, eids = edges.edges()
            m = torch.stack(
                [
                    (edges.src[src][i] @ meta_adjs[eid])
                    for i, eid in enumerate(eids)
                ]
            )
            # m *= edges.data[eweight].view(-1, self.num_heads, 1)
            m *= edges.data[eweight]
            return {dst: m}

        return _message

    def forward(self, mg: dgl.DGLGraph, meta_adjs: list[torch.Tensor]):

        x = torch.zeros(
            (mg.num_nodes(), self.num_heads, meta_adjs[0].shape[0]),
            device=mg.device
        )
        for i in range(mg.num_nodes()):
            x[i, :, i] = 1
        rel_w = dgl.ops.edge_softmax(mg, self.rel_w / self.tau)

        self.xs = [x]
        with mg.local_scope():
            for i in range(rel_w.shape[1]):
                mg.ndata['x'] = x
                mg.edata['rel_w'] = rel_w[:, i].unsqueeze(-1)
                mg.update_all(
                    self.message('x', 'rel_w', 'm', meta_adjs),
                    fn.sum('m', 'h'),
                )

                if self.residuals[i]:
                    x = mg.ndata.pop('h') + x
                else:
                    x = mg.ndata.pop('h')
                self.xs.append(x)
        return x

    def build(
        self,
        mg: dgl.DGLGraph,
        meta_adjs: list[torch.Tensor],
        target_ntype_idx: int,
    ):
        x = torch.zeros(
            (mg.num_nodes(), self.num_heads, meta_adjs[0].shape[0]),
            device=mg.device
        )
        for i in range(mg.num_nodes()):
            x[i, :, i] = 1
        adj_to_edge = torch.eye(mg.num_nodes(),
                                device=mg.device)[mg.edges()[0]].T
        # edge_to_node = torch.argsort(adj_to_edge.T, dim=0, stable=True)[-1]
        adj_reduce = torch.eye(mg.num_nodes(), device=mg.device)[mg.edges()[1]]
        adj_reduce_and_to_edge = adj_reduce @ adj_to_edge
        x0_on_edge = (x.transpose(0, 2) @ adj_to_edge).transpose(0, 2)
        # self.edge_to_node = edge_to_node.requires_grad_(False)
        self.to_target = (mg.edges()[0] == target_ntype_idx).nonzero()[0]
        self.adj_reduce_and_to_edge = adj_reduce_and_to_edge.requires_grad_(
            False
        )
        if meta_adjs[0].layout == torch.strided:
            self.meta_adjs = torch.stack(meta_adjs).requires_grad_(False)
        else:
            self.meta_adjs = meta_adjs
        self.x0_on_edge = x0_on_edge.requires_grad_(False)
        self.dst_masks = [mg.edges()[1] == i for i in range(mg.num_nodes())]
        self._build = True
        return

    def forward2(
        self,
        mg: dgl.DGLGraph,
        meta_adjs: list[torch.Tensor],
        target_ntype_idx: int,
    ):
        with torch.cuda.amp.autocast(enabled=False):
            if not self._build:
                self.build(mg, meta_adjs, target_ntype_idx)
            rel_w = dgl.ops.edge_softmax(mg, self.rel_w / self.tau)
            x_on_edge = self.x0_on_edge.clone()
            # x_on_edge: (|E|, *, |M|)
            for i, w in enumerate(rel_w.transpose(0, 1)):
                # w: (|E|, h)
                # h_on_edge = torch.einsum('chm,cmn -> chn', x_on_edge, meta_adjs_)
                if isinstance(self.meta_adjs, torch.Tensor):  # dense meta_adjs
                    h_on_edge = (x_on_edge * w.unsqueeze(-1)) @ self.meta_adjs
                else:  # sparse meta_adjs
                    x_on_edge_ = x_on_edge * w.unsqueeze(-1)
                    h_on_edge = torch.stack(
                        [
                            x_on_edge_[i] @ meta_adj
                            for i, meta_adj in enumerate(self.meta_adjs)
                        ]
                    )
                h_on_edge = (
                    h_on_edge.transpose(0, 2) @ self.adj_reduce_and_to_edge
                ).transpose(0, 2)
                if self.residuals[i]:
                    x_on_edge = h_on_edge + x_on_edge
                else:
                    x_on_edge = h_on_edge

            return x_on_edge[self.to_target].squeeze(dim=0)


class Conv1d1x1(nn.Module):

    def __init__(self, cin, cout, groups, bias=True, cformat='channel-first'):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.groups = groups
        self.cformat = cformat
        if not bias:
            self.bias = None
        if self.groups == 1:  # different keypoints share same kernel
            self.W = nn.Parameter(torch.randn(self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(1, self.cout))
        else:
            self.W = nn.Parameter(
                torch.randn(self.groups, self.cin, self.cout)
            )
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.groups, self.cout))
        self.reset_parameters()
        return

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(
                3.0
            ) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.groups == 1:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,mn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,mn->bnc', x, self.W) + self.bias.T
            else:
                assert False
        else:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,cmn->bnc', x, self.W) + self.bias.T
            else:
                assert False


@deprecated(
    'Will soon be replaced by SRGCN_ (use seperated model for label feat)'
)
class SRGCN(nn.Module):

    class SharedFeatProjArgs(TypedDict):
        in_feat_shapes: dict[NType, tuple[int, ...]]
        embedding_max_norm: float | None

    def __init__(
        self,
        n_hidden: int,
        n_layers: int,
        n_heads: int,
        n_out: int,
        ntypes: list[NType],
        etypes: list[CEType],
        input_drop: float,
        channel_drop: float,
        dropout: float,
        *,
        residual: bool | float = True,
        proj_args: SharedFeatProjArgs | int | None = None,
        out_norm_type: Literal['layer', 'batch'] = 'layer',
        n_in_layers: int = 1,
        n_out_layers: int = 1,
        softmax_tau: float = 1.,
        # relation_alpha: float | None = None,
        label_n_layers: int | None = None,
        label_n_heads: int | None = None,
        label_n_out_layers: int | None = None,
        label_emb_fc: nn.Module | None = None,
    ):
        super().__init__()
        in_size = n_hidden * n_heads
        # print('using first residual')
        # self.gcs: list[RGCNConv] = nn.ModuleList(
        #     [
        #         RGCNConv(
        #             in_size=in_size,
        #             rel_emb_size=in_size,
        #             num_heads=n_heads,
        #             ntypes=ntypes,
        #             cetypes=etypes,
        #             # residual=residual,
        #             residual=residual if i > 0 else False,
        #             softmax_tau=softmax_tau,
        #             # relation_alpha=relation_alpha or 1.,
        #         ) for i in range(n_layers)
        #     ]
        # )
        self.gcs = RGCNConvs(
            rel_emb_size=in_size,
            num_heads=n_heads,
            num_layers=n_layers,
            ntypes=ntypes,
            cetypes=etypes,
            # residual=residual,
            residuals=[residual if i > 0 else False for i in range(n_layers)],
            softmax_tau=softmax_tau,
            # relation_alpha=relation_alpha or 1.,
        )
        self.n_heads = n_heads
        self.n_hid = n_hidden
        self.n_out = n_out
        self.n_layers = n_layers
        self.proj = None
        self.ntypes = ntypes
        self.etypes = etypes
        if proj_args is not None:
            if isinstance(proj_args, int):
                self._proj = nn.Linear(proj_args, in_size)
                self.proj = {ntype: self._proj for ntype in ntypes}
                self._feat_project_layers = self._construct_in_layers(
                    n_in_layers=n_in_layers - 1,
                    n_heads=n_heads,
                    head_size=n_hidden,
                    dropout=input_drop,
                )
                self.feat_project_layers = {
                    ntype: self._feat_project_layers
                    for ntype in ntypes
                }
                if n_in_layers == 1:
                    self.feat_project_layers = None
            else:
                self.proj = SharedSpaceProjection(
                    n_out=in_size,
                    **proj_args,
                )
                self.feat_project_layers = nn.ModuleDict(
                    {
                        ntype:
                        self._construct_in_layers(
                            n_in_layers=n_in_layers - 1,
                            n_heads=n_heads,
                            head_size=n_hidden,
                            dropout=input_drop,
                        )
                        for ntype in ntypes
                    }
                )
                if n_in_layers == 1:
                    self.feat_project_layers = None
        self.input_drop = nn.Dropout(input_drop)
        self.channel_drop = nn.Dropout1d(channel_drop)
        self.in_feat_norm = None
        self.in_feat_norm = nn.ModuleDict(
            {
                ntype: nn.GroupNorm(n_heads, in_size)
                # {ntype: nn.LayerNorm(in_size)
                for ntype in ntypes
            }
        )
        if label_n_layers is not None:
            label_in_size = label_n_heads * n_hidden
            if isinstance(self.proj, SharedSpaceProjection):
                self.proj.proj['label'] = nn.Linear(n_out, label_in_size)
            else:
                self._label_proj = nn.Linear(n_out, label_in_size)
                self.proj['label'] = self._label_proj
            if self.feat_project_layers is not None:
                _label_proj_layers = self._construct_in_layers(
                    n_in_layers=n_in_layers - 1,
                    n_heads=label_n_heads,
                    head_size=n_hidden,
                    dropout=input_drop,
                )
                if not isinstance(self.feat_project_layers, nn.ModuleDict):
                    # normal dict
                    self._label_proj_layers = _label_proj_layers
                    self.feat_project_layers['label'] = _label_proj_layers
                self.feat_project_layers['label'] = _label_proj_layers
            if self.in_feat_norm is not None:
                self.in_feat_norm['label'] = nn.GroupNorm(
                    label_n_heads, label_in_size
                )
                # self.in_feat_norm['label'] = nn.LayerNorm(label_in_size)
            self.label_gcs = RGCNConvs(
                rel_emb_size=label_in_size,
                num_heads=label_n_heads,
                num_layers=label_n_layers,
                ntypes=ntypes,
                cetypes=etypes,
                # residual=residual,
                residuals=[
                    residual if i > 0 else False
                    for i in range(label_n_layers)
                ],
                softmax_tau=softmax_tau,
                # relation_alpha=relation_alpha or 1.,
            )
            self.label_out = self._construct_out_layers(
                label_n_out_layers,
                label_n_heads,
                head_size=n_hidden,
                dropout=dropout,
                n_out=n_out,
                norm_type=out_norm_type,
            )
            if label_emb_fc is not None:
                raise NotImplementedError

        self.label_emb_fc = label_emb_fc
        self.out = self._construct_out_layers(
            n_out_layers,
            n_heads=n_heads,
            # n_heads=n_heads + label_n_heads if label_n_layers else n_heads,
            head_size=n_hidden,
            dropout=dropout,
            n_out=n_out,
            norm_type=out_norm_type,
        )
        self.reset_parameters()
        return

    @classmethod
    def _construct_in_layers(
        cls, n_in_layers: int, n_heads: int, head_size: int, dropout: float
    ):
        layer = nn.Sequential()
        for _ in range(n_in_layers):
            layer.extend(
                [
                    Conv1d1x1(
                        head_size, head_size, n_heads, bias=True,
                        cformat='channel-first'
                    ),
                    nn.LayerNorm([n_heads, head_size]),
                    # nn.BatchNorm1d(n_heads),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                ]
            )
        return layer

    @classmethod
    def _construct_out_layers(
        cls, n_out_layers: int, n_heads: int, head_size: int, dropout: float,
        n_out, norm_type: Literal['layer', 'batch']
    ):
        _cur_size = n_heads * head_size
        assert norm_type in ('layer', 'batch')
        out = nn.Sequential()
        for _ in range(n_out_layers - 1):
            out.extend(
                [
                    nn.Linear(_cur_size, head_size),
                    nn.LayerNorm(head_size)
                    if norm_type == 'layer' else nn.BatchNorm1d(head_size),
                    # nn.BatchNorm1d(head_size),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                ]
            )
            _cur_size = head_size
        out.append(nn.Linear(_cur_size, n_out))
        return out

    def reset_parameters(self):
        for ntype in self.proj:
            nn.init.normal_(self.proj[ntype].weight, std=0.01)
            nn.init.zeros_(self.proj[ntype].bias)
        pass

    def forward(
        self,
        mg: dgl.DGLGraph,
        canonical_mp_indices: torch.Tensor,
        meta_adjs: list[torch.Tensor],
        feats: list[torch.Tensor],
        feat_fmt: Literal['strided', 'sparse_csr', 'sparse_coo'] | None = None,
        label_canonical_mp_indices: torch.Tensor | None = None,
        label_feats: list[torch.Tensor] | None = None,
        label_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            mg (dgl.DGLGraph): Metagraph
            canonical_mp_ids (torch.Tensor): list of triplet: (src_id, mp_id, dst_id).
                in shape (# of metapaths, 3)
            meta_adjs (list[torch.Tensor]): Meta adjacencies
            feats (list[torch.Tensor]): list of metapath-aggregated features

        Returns:
            torch.Tensor: _description_
        """

        assert len(canonical_mp_indices) == len(feats)
        tgt_ntype_id = canonical_mp_indices[0, -1]
        assert (canonical_mp_indices[:, -1] == tgt_ntype_id).all()

        def _feat_proj(srctypes: list[NType], xs: list[torch.Tensor]):
            for _i, (srctype, x) in enumerate(zip(srctypes, xs)):
                up = None
                if isinstance(x, LowRankMatrix):
                    x, up = x
                elif feat_fmt is not None:
                    x = x.to_sparse(layout=getattr(torch, feat_fmt))
                x = self.proj[srctype](self.input_drop(x))
                if self.in_feat_norm is not None:
                    x = self.in_feat_norm[srctype](x)
                x = self.input_drop(x)
                if self.feat_project_layers is not None:
                    x = self.feat_project_layers[srctype](
                        x.view(x.shape[0], -1, self.n_hid)
                    )
                if up is not None:
                    x = torch.einsum('nr,rhd->nhd', up, x)
                yield x

        srctypes = [self.ntypes[sid] for sid in canonical_mp_indices[:, 0]]
        feats = torch.stack(list(_feat_proj(srctypes, feats)), dim=0)
        # feats: |MPs ending with target_ntype| x |# target nodes| x dim

        if label_canonical_mp_indices is not None:
            label_feats = torch.stack(
                list(
                    _feat_proj(
                        ['label'] * len(label_canonical_mp_indices),
                        label_feats
                    )
                ), dim=0
            )
        # print(w_)
        # |HEAD| x |MP ending with target_ntype|
        # E.g. 8 x 1000
        # w_ = self.gcs.forward(mg, meta_adjs)
        # w_ = w_[tgt_ntype_id][:, canonical_mp_indices[:, 1]]
        w_ = self.gcs.forward2(mg, meta_adjs, tgt_ntype_id)
        # w_ = w_[tgt_ntype_id, :, canonical_mp_indices[:, 1]]
        # w_ = w_[tgt_ntype_id]
        w_ = w_[:, canonical_mp_indices[:, 1]]
        # print(f'{(w_ - w2).abs().max() = }')
        h = torch.einsum(
            'ch,cnhd->nhd', self.channel_drop(w_.T),
            feats.view(*feats.shape[:2], self.n_heads, self.n_hid)
        ).flatten(1)
        # |#nodes| x (|HEAD| x d)

        if label_emb is not None:
            assert self.label_emb_fc is not None
        out = self.out(h)
        if self.label_emb_fc is not None:
            out += self.label_emb_fc(label_emb)

        if label_canonical_mp_indices is not None:
            w__ = self.label_gcs.forward2(mg, meta_adjs, tgt_ntype_id)
            # w__ = w__[tgt_ntype_id][:, label_canonical_mp_indices[:, 1]]
            w__ = w__[:, label_canonical_mp_indices[:, 1]]
            label_h = torch.einsum(
                'ch,cnhd->nhd', w__.T,
                label_feats.view(*label_feats.shape[:2], -1, self.n_hid)
            ).flatten(1)
            # h = h + label_h
            out += self.label_out(label_h)
        return out

    # def build_meta_adjs(self):
    #     return build_meta_adjs(self.ntypes, self.etypes, self.n_layers)


class LabelEmb(nn.Module):

    def __init__(
        self,
        n_hidden: int,
        n_out: int,
        input_drop: float,
        dropout: float,
        *,
        out_norm_type: Literal['layer', 'batch'] = 'layer',
        n_out_layers: int = 1,
    ):
        super().__init__()
        self.input_drop = nn.Dropout(input_drop)
        self.fc = self._construct_out_layers(
            n_out_layers,
            n_hidden=n_hidden,
            n_out=n_out,
            dropout=dropout,
            norm_type=out_norm_type,
        )
        return

    def forward(self, x: torch.Tensor):
        return self.fc(self.input_drop(x))

    @classmethod
    def _construct_out_layers(
        cls, n_out_layers: int, n_hidden: int, n_out: int, dropout: float,
        norm_type: Literal['layer', 'batch']
    ):
        assert norm_type in ('layer', 'batch')
        assert n_out_layers >= 2
        out = nn.Sequential()
        _cur_size = n_out
        for _ in range(n_out_layers - 1):
            out.extend(
                [
                    nn.Linear(_cur_size, n_hidden),
                    nn.LayerNorm(n_hidden)
                    if norm_type == 'layer' else nn.BatchNorm1d(n_hidden),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                ]
            )
            _cur_size = n_hidden
        out.append(nn.Linear(_cur_size, n_out))
        return out


class SRGCN_(nn.Module):

    class SharedFeatProjArgs(TypedDict):
        in_feat_shapes: dict[NType, tuple[int, ...]]
        embedding_max_norm: float | None

    def __init__(
        self,
        n_hidden: int,
        n_layers: int,
        n_heads: int,
        n_out: int,
        ntypes: list[NType],
        etypes: list[CEType],
        input_drop: float,
        channel_drop: float,
        dropout: float,
        *,
        residual: bool | float = True,
        proj_args: SharedFeatProjArgs | int | None = None,
        out_norm_type: Literal['layer', 'batch'] = 'layer',
        n_in_layers: int = 1,
        n_out_layers: int = 1,
        softmax_tau: float = 1.,
        tgt_feat_residual: str | None = None,
        weight_scalar: float = 1,
    ):
        super().__init__()
        in_size = n_hidden * n_heads
        self.gcs = RGCNConvs(
            rel_emb_size=in_size,
            num_heads=n_heads,
            num_layers=n_layers,
            ntypes=ntypes,
            cetypes=etypes,
            residuals=[residual if i > 0 else False for i in range(n_layers)],
            softmax_tau=softmax_tau,
        )
        self.weight_scalar = weight_scalar
        self.n_heads = n_heads
        self.n_hid = n_hidden
        self.n_out = n_out
        self.n_layers = n_layers
        self.proj = None
        self.ntypes = ntypes
        self.etypes = etypes
        if proj_args is not None:
            if isinstance(proj_args, int):
                self._proj = nn.Linear(proj_args, in_size)
                self.proj = {ntype: self._proj for ntype in ntypes}
                self._feat_project_layers = self._construct_in_layers(
                    n_in_layers=n_in_layers - 1,
                    n_heads=n_heads,
                    head_size=n_hidden,
                    dropout=input_drop,
                )
                self.feat_project_layers = {
                    ntype: self._feat_project_layers
                    for ntype in ntypes
                }
                if n_in_layers == 1:
                    self.feat_project_layers = None
            else:
                self.proj = SharedSpaceProjection(
                    n_out=in_size,
                    **proj_args,
                )
                self.feat_project_layers = nn.ModuleDict(
                    {
                        ntype:
                        self._construct_in_layers(
                            n_in_layers=n_in_layers - 1,
                            n_heads=n_heads,
                            head_size=n_hidden,
                            dropout=input_drop,
                        )
                        for ntype in ntypes
                    }
                )
                if n_in_layers == 1:
                    self.feat_project_layers = None
        self.tgt_feat_res_proj = None
        if tgt_feat_residual:
            self.tgt_feat_res_proj = nn.Linear(
                self.proj[tgt_feat_residual].in_features, n_heads * n_hidden,
                bias=False
            )
        self.input_drop = nn.Dropout(input_drop)
        self.channel_drop = nn.Dropout1d(channel_drop)
        self.in_feat_norm = None
        self.in_feat_norm = nn.ModuleDict(
            {
                ntype: nn.GroupNorm(n_heads, in_size)
                # {ntype: nn.LayerNorm(in_size)
                for ntype in ntypes
            }
        )
        self.out = self._construct_out_layers(
            n_out_layers,
            n_heads=n_heads,
            head_size=n_hidden,
            dropout=dropout,
            n_out=n_out,
            norm_type=out_norm_type,
        )
        self.reset_parameters()
        return

    @classmethod
    def _construct_in_layers(
        cls, n_in_layers: int, n_heads: int, head_size: int, dropout: float
    ):
        layer = nn.Sequential()
        for _ in range(n_in_layers):
            layer.extend(
                [
                    Conv1d1x1(
                        head_size, head_size, n_heads, bias=True,
                        cformat='channel-first'
                    ),
                    nn.LayerNorm([n_heads, head_size]),
                    # nn.BatchNorm1d(n_heads),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                ]
            )
        return layer

    @classmethod
    def _construct_out_layers(
        cls, n_out_layers: int, n_heads: int, head_size: int, dropout: float,
        n_out, norm_type: Literal['layer', 'batch']
    ):
        _cur_size = n_heads * head_size
        assert norm_type in ('layer', 'batch')
        out = nn.Sequential()
        hidden_dim = max(head_size, n_out)
        for _ in range(n_out_layers - 1):
            out.extend(
                [
                    nn.Linear(_cur_size, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                    if norm_type == 'layer' else nn.BatchNorm1d(hidden_dim),
                    # nn.BatchNorm1d(hidden_dim),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                ]
            )
            _cur_size = hidden_dim
        out.append(nn.Linear(_cur_size, n_out))
        return out

    def reset_parameters(self):
        for ntype in self.proj:
            nn.init.normal_(self.proj[ntype].weight, std=0.01)
            nn.init.zeros_(self.proj[ntype].bias)
        if self.tgt_feat_res_proj is not None:
            nn.init.normal_(self.tgt_feat_res_proj.weight, std=0.01)
        pass

    def forward(
        self,
        mg: dgl.DGLGraph,
        canonical_mp_indices: torch.Tensor,
        meta_adjs: list[torch.Tensor],
        feats: list[torch.Tensor],
        feat_fmt: Literal['strided', 'sparse_csr', 'sparse_coo'] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            mg (dgl.DGLGraph): Metagraph
            canonical_mp_ids (torch.Tensor): list of triplet: (src_id, mp_id, dst_id).
                in shape (# of metapaths, 3)
            meta_adjs (list[torch.Tensor]): Meta adjacencies
            feats (list[torch.Tensor]): list of metapath-aggregated features

        Returns:
            torch.Tensor: _description_
        """

        assert len(canonical_mp_indices) == len(feats)
        tgt_ntype_id = canonical_mp_indices[0, -1]
        assert (canonical_mp_indices[:, -1] == tgt_ntype_id).all()

        def _feat_proj(srctypes: list[NType], xs: list[torch.Tensor]):
            for _i, (srctype, x) in enumerate(zip(srctypes, xs)):
                up = None
                if isinstance(x, LowRankMatrix):
                    x, up = x
                elif feat_fmt is not None:
                    x = x.to_sparse(layout=getattr(torch, feat_fmt))
                x = self.proj[srctype](self.input_drop(x))
                if self.in_feat_norm is not None:
                    x = self.in_feat_norm[srctype](x)
                x = self.input_drop(x)
                if self.feat_project_layers is not None:
                    x = self.feat_project_layers[srctype](
                        x.view(x.shape[0], -1, self.n_hid)
                    )
                if up is not None:
                    x = torch.einsum('nr,rhd->nhd', up, x)
                yield x

        srctypes = [self.ntypes[sid] for sid in canonical_mp_indices[:, 0]]
        if self.tgt_feat_res_proj is not None:
            tgt_feat = self.input_drop(feats[0])
        feats = torch.stack(list(_feat_proj(srctypes, feats)), dim=0)
        # feats: |MPs ending with target_ntype| x |# target nodes| x dim

        w_ = self.gcs.forward2(mg, meta_adjs, tgt_ntype_id)
        w_ = w_[:, canonical_mp_indices[:, 1]] * self.weight_scalar

        h = torch.einsum(
            'ch,cnhd->nhd', self.channel_drop(w_.T),
            feats.view(*feats.shape[:2], self.n_heads, self.n_hid)
        ).flatten(1)
        # |#nodes| x (|HEAD| x d)

        if self.tgt_feat_res_proj is not None:
            tgt_feat = self.tgt_feat_res_proj(tgt_feat).view(
                -1, self.n_heads, self.n_hid
            )
            h += (w_.mean(dim=1, keepdim=True) * tgt_feat).flatten(1)

        out = self.out(h)
        return out
