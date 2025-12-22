import math
from typing import Mapping
from typing_extensions import deprecated
import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl import DGLHeteroGraph
from dgl import function as fn
from ...type import NType, EType
from ..tools import scatter_edge_softmax


def dict_typed_linear(
    in_sizes: int | dict[int],
    out_sizes: int | dict[int],
    types: list[str],
):

    def int_to_dict(sizes: int | dict[int]):
        if not isinstance(sizes, dict):
            return {_type: sizes for _type in types}
        return sizes

    in_sizes = int_to_dict(in_sizes)
    out_sizes = int_to_dict(out_sizes)
    return nn.ModuleDict(
        {
            ntype: nn.Linear(in_sizes[ntype], out_sizes[ntype])
            for ntype in types
        }
    )


class HGTConv(nn.Module):
    r"""Heterogeneous graph transformer convolution from `Heterogeneous Graph Transformer
    Adapted from dgl.nn.pytorch.HGTConv
    """

    def __init__(
        self,
        in_size: int,
        head_size: int,
        num_heads: int,
        ntypes: list[str],
        etypes: list[str],
        dropout=0.5,
        use_norm=False,
        normsoftmax=False,
    ):
        super().__init__()
        self.in_size = in_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.sqrt_d = math.sqrt(head_size)
        self.use_norm = use_norm

        self.linear_k = dict_typed_linear(
            in_size, head_size * num_heads, ntypes
        )
        self.linear_q = dict_typed_linear(
            in_size, head_size * num_heads, ntypes
        )
        self.linear_v = dict_typed_linear(
            in_size, head_size * num_heads, ntypes
        )
        self.linear_a = dict_typed_linear(
            head_size * num_heads, head_size * num_heads, ntypes
        )

        self.relation_pri = nn.ParameterDict(
            {etype: torch.ones(num_heads)
             for etype in etypes}
        )
        self.relation_att = nn.ParameterDict(
            {
                etype: torch.Tensor(num_heads, head_size, head_size)
                for etype in etypes
            }
        )
        self.relation_msg = nn.ParameterDict(
            {
                etype: torch.Tensor(num_heads, head_size, head_size)
                for etype in etypes
            }
        )
        for etype in etypes:
            nn.init.xavier_uniform_(self.relation_att[etype])
            nn.init.xavier_uniform_(self.relation_msg[etype])
        self.skip = nn.ParameterDict(
            {ntype: torch.ones(1)
             for ntype in ntypes}
        )
        self.drop = nn.Dropout(dropout)
        if use_norm:
            self.norms = nn.ModuleDict(
                {
                    ntype: nn.LayerNorm(head_size * num_heads)
                    for ntype in ntypes
                }
            )

        if in_size != head_size * num_heads:
            self.residual_w = nn.Parameter(
                torch.Tensor(in_size, head_size * num_heads)
            )
            nn.init.xavier_uniform_(self.residual_w)
        self.tem_bn = nn.ModuleDict(
            {
                etype: nn.BatchNorm1d(num_heads, affine=False)
                for etype in etypes
            }
        )
        self.normsoftmax = normsoftmax
        return

    def forward(
        self,
        hg: DGLHeteroGraph,
        xs: dict[NType, torch.Tensor],
        edge_weight: dict[EType, torch.Tensor | str] | None,
    ):
        if hg.is_block:
            xs_src = {}
            xs_dst = {}
            for ntype, x in xs.items():
                xs_src[ntype] = x
                xs_dst[ntype] = x[:hg.num_dst_nodes(ntype)]

        else:
            xs_src = xs
            xs_dst = xs

        with hg.local_scope():
            for ntype in hg.srctypes:
                k = self.linear_k[ntype](xs_src[ntype])
                v = self.linear_v[ntype](xs_src[ntype])

                hg.srcnodes[ntype].data['k'] = k.view(
                    -1, self.num_heads, self.head_size
                )
                hg.srcnodes[ntype].data['v'] = v.view(
                    -1, self.num_heads, self.head_size
                )

            for ntype in hg.dsttypes:
                q = self.linear_q[ntype](xs_dst[ntype])
                hg.dstnodes[ntype].data['q'] = q.view(
                    -1, self.num_heads, self.head_size
                )

            if len(hg.etypes) > 1:
                for etype in hg.etypes:
                    if hg.num_edges(etype):
                        eweight_key = edge_weight and edge_weight.get(
                            etype,
                            edge_weight.
                            get(hg.to_canonical_etype(etype), None)
                        )
                        if isinstance(eweight_key, torch.Tensor):
                            hg.edges[etype].data['w'] = eweight_key
                            eweight_key = 'w'

                        hg.apply_edges(
                            self.message(etype, eweight_key), etype=etype
                        )

                if True:
                    # edge_softmax_fn = scatter_edge_softmax if hg.is_block else dgl.ops.edge_softmax
                    if hg.edata['a']:
                        edge_softmaxed = scatter_edge_softmax(
                            hg,
                            hg.edata['a'],
                            normsoftmax=self.normsoftmax,
                        )
                    else:
                        # while no edge in graph
                        edge_softmaxed = hg.edata['a']

                    for cetype, attn in edge_softmaxed.items():
                        hg.edata['m'][cetype] *= attn.unsqueeze(-1)
                        # hg.edata['m'] *= attn.unsqueeze(-1)
                    hg.multi_update_all(
                        {
                            etype: (fn.copy_e('m', 'm'), fn.sum('m', 'h'))
                            for etype in edge_softmaxed
                        }, cross_reducer='sum'
                    )
                else:
                    hg.multi_update_all(
                        {
                            etype: (fn.u, fn.sum('m', 'h'))
                            for etype in self.edict
                        }, cross_reducer='mean'
                    )
            else:
                for etype in hg.etypes:
                    if isinstance(edge_weight, Mapping):
                        eweight_key = edge_weight.get(etype, None)
                    else:
                        eweight_key = edge_weight
                    if isinstance(eweight_key, torch.Tensor):
                        hg.edges[etype].data['w'] = eweight_key
                        eweight_key = 'w'

                    hg.apply_edges(
                        self.message(etype, eweight_key), etype=etype
                    )

                attn = dgl.ops.edge_softmax(hg, hg.edata['a'])
                hg.edata['m'] *= attn.unsqueeze(-1)
                hg.update_all(
                    fn.copy_e('m', 'm'),
                    fn.sum('m', 'h'),
                    etype=hg.etypes[0],
                )

            return dict(self._target_specific_aggregation(hg, xs_dst))

    def _target_specific_aggregation(
        self, hg: DGLHeteroGraph, xs_dst: dict[str, torch.Tensor]
    ):
        for ntype, x in xs_dst.items():
            x = xs_dst[ntype]
            if ntype in hg.dsttypes and 'h' in hg.dstnodes[ntype].data:
                h = hg.dstnodes[ntype].data['h']
                h = h.view(-1, self.num_heads * self.head_size)
                h = F.gelu(self.linear_a[ntype](h))  # pylint: disable=not-callable
                # h = self.drop(h)
                alpha = torch.sigmoid(self.skip[ntype])
                if x.shape != h.shape:
                    h = h * alpha + (x @ self.residual_w) * (1 - alpha)
                else:
                    h = h * alpha + x * (1 - alpha)
                if self.use_norm:
                    h = self.norms[ntype](h)
                h = self.drop(h)
            else:
                if x.shape[-1] != self.head_size * self.num_heads:
                    h = x @ self.residual_w
                else:
                    h = x
            yield ntype, h

    def message(self, etype: str, eweight_key: str | None):
        """Message function."""

        def edge_attention(edges):
            relation_att = self.relation_att[etype]
            relation_pri = self.relation_pri[etype]
            relation_msg = self.relation_msg[etype]
            key = torch.bmm(edges.src['k'].transpose(1, 0),
                            relation_att).transpose(1, 0)
            att = (edges.dst['q'] *
                   key).sum(dim=-1) * relation_pri / self.sqrt_d
            val = torch.bmm(edges.src['v'].transpose(1, 0),
                            relation_msg).transpose(1, 0)
            if eweight_key is not None:
                att *= edges.data[eweight_key].view(-1, 1)
            return {'a': att, 'm': val}

        return edge_attention


class SwitchHGTConv(HGTConv):

    def _target_specific_aggregation(self, hg, xs_dst):
        assert hasattr(hg, 'gdata')
        rel_weights = {
            dsttype: weight
            for (_, __, dsttype), weight in hg.gdata['rel_weight'].items()
        }

        for ntype, x in xs_dst.items():
            if ntype in hg.dsttypes and 'h' in hg.dstnodes[ntype].data:
                x = xs_dst[ntype]
                h = hg.dstnodes[ntype].data['h'] * rel_weights[ntype]
                h = h.view(-1, self.num_heads * self.head_size)
                h = F.gelu(self.linear_a[ntype](h))  # pylint: disable=not-callable
                # h = self.drop(h)
                if x.shape != h.shape:
                    h = h + (x @ self.residual_w)
                else:
                    h = h + x
                assert self.use_norm
                if self.use_norm:
                    h = self.norms[ntype](h)
                h = self.drop(h)
                yield ntype, h
            else:
                assert x.shape[-1] == self.head_size * self.num_heads
                yield ntype, x


class LowRankRelationHGTConv(HGTConv):

    def __init__(
        self,
        in_size,
        head_size,
        num_heads,
        ntypes,
        etypes,
        dropout=0.5,
        use_norm=False,
        rank: int | None = None,
        relation_pri_alpha: float | None = None,
        normsoftmax=False,
    ):
        super().__init__(
            in_size,
            head_size,
            num_heads,
            ntypes,
            etypes,
            dropout,
            use_norm,
            normsoftmax,
        )

        assert rank is not None or relation_pri_alpha is not None

        self.rank = rank
        self.etypes = etypes

        if rank is not None:
            del self.relation_att
            del self.relation_msg

            self.relation_att_down = nn.ParameterDict(
                {
                    etype: torch.Tensor(num_heads, head_size, rank)
                    for etype in etypes
                }
            )
            self.relation_att_up = nn.ParameterDict(
                {
                    etype: torch.Tensor(num_heads, head_size, rank)
                    for etype in etypes
                }
            )
            self.relation_msg_down = nn.ParameterDict(
                {
                    etype: torch.Tensor(num_heads, head_size, rank)
                    for etype in etypes
                }
            )
            self.relation_msg_up = nn.ParameterDict(
                {
                    etype: torch.Tensor(num_heads, rank, head_size)
                    for etype in etypes
                }
            )

        self.prior_exp_alpha = relation_pri_alpha
        self.reset_parameters()
        return

    def reset_parameters(self):
        if self.prior_exp_alpha is not None:
            for etype in self.relation_pri:
                nn.init.constant_(self.relation_pri[etype], 0)

        if self.rank is not None:
            gain = nn.init.calculate_gain('relu')
            for etype in self.relation_att_down:
                nn.init.xavier_normal_(
                    self.relation_att_down[etype], gain=gain
                )
                nn.init.xavier_normal_(self.relation_att_up[etype], gain=gain)
                nn.init.xavier_normal_(
                    self.relation_msg_down[etype], gain=gain
                )
                nn.init.xavier_normal_(self.relation_msg_up[etype], gain=gain)
        return

    def _up_down_dot_product(self, etype: str, key, query):
        """
        K.T W_down W_up Q
        """
        relation_att_down = self.relation_att_down[etype]
        relation_att_up = self.relation_att_up[etype]
        key = torch.bmm(key.transpose(1, 0), relation_att_down).transpose(1, 0)
        query = torch.bmm(query.transpose(1, 0),
                          relation_att_up).transpose(1, 0)
        att = torch.einsum('ijk,ijk->ij', torch.relu(query), key)
        return att

    def _full_rank_dot_product(self, etype: str, key, query):
        """
        K.T W Q
        """
        relation_att = self.relation_att[etype]
        key = torch.bmm(key.transpose(1, 0), relation_att).transpose(1, 0)
        att = torch.einsum('ijk,ijk->ij', query, key)
        return att

    def _relation_msg_up_down_forward(self, etype: str, val):
        relation_msg_down = self.relation_msg_down[etype]
        relation_msg_up = self.relation_msg_up[etype]
        val = torch.bmm(val.transpose(1, 0), relation_msg_down)
        val = torch.bmm(F.relu(val), relation_msg_up).transpose(1, 0)
        return val

    def _relation_msg_full_rank_forward(self, etype: str, val):
        relation_msg = self.relation_msg[etype]
        return torch.bmm(val.transpose(1, 0), relation_msg).transpose(1, 0)

    def message(self, etype: str, eweight_key: str | None):
        """Message function."""

        def edge_attention(edges):
            relation_pri = self.relation_pri[etype]
            if self.prior_exp_alpha is not None:
                relation_pri = torch.exp(relation_pri * self.prior_exp_alpha)
            if self.rank is not None:
                att = self._up_down_dot_product(
                    etype, edges.src['k'], edges.dst['q']
                ) * relation_pri / self.sqrt_d
                val = self._relation_msg_up_down_forward(etype, edges.src['v'])
            else:
                att = self._full_rank_dot_product(
                    etype, edges.src['k'], edges.dst['q']
                ) * relation_pri / self.sqrt_d
                val = self._relation_msg_full_rank_forward(
                    etype, edges.src['v']
                )
            if eweight_key is not None:
                att *= edges.data[eweight_key].view(-1, 1)
            else:
                # warnings.warn('eweight_key not used')
                pass
            return {'a': att, 'm': val}

        # print(
        #     id(self), etype,
        #     f'pri={torch.exp(self.relation_pri[etype] * self.prior_exp_alpha).clone().detach().cpu()}'
        # )
        # print(id(self), etype, f'mu={self.tem_bn[etype].running_mean.cpu()}')
        # print(id(self), etype, f'var={self.tem_bn[etype].running_var.cpu()}')
        return edge_attention


class LowRankRelationSwitchHGTConv(SwitchHGTConv, LowRankRelationHGTConv):
    pass


@deprecated('HGTLayer is deprecated. Use HGTConv instead')
class HGTLayer(nn.Module):
    """This seems to be the original implementation of the original author
    However, this version suffers from several issue:

    1. redundant computation caculating q, k, v
    2. edge softmax performs first type-wise then mean pooling
        - According to the original paper, the softmax should perform cross-type
    3. the original code (pyG) use F.gelu after a_linear, while here the activation
        is misssing
    4. cannot forward with block (MFG)
    """

    def __init__(
        self, in_dim, out_dim, ntypes, etypes, n_heads, dropout=0.5,
        use_norm=False
    ):
        super(HGTLayer, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.ntypes = ntypes
        self.etypes = etypes
        self.ndict = {ntype: i for i, ntype in enumerate(ntypes)}
        self.edict = {etype: i for i, etype in enumerate(etypes)}
        self.num_types = len(ntypes)
        self.num_relations = len(etypes)
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)

        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        self.a_linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_norm = use_norm

        for _ in ntypes:
            self.k_linears.append(nn.Linear(in_dim, out_dim))
            self.q_linears.append(nn.Linear(in_dim, out_dim))
            self.v_linears.append(nn.Linear(in_dim, out_dim))
            self.a_linears.append(nn.Linear(out_dim, out_dim))
            if use_norm:
                self.norms.append(nn.LayerNorm(out_dim))

        self.relation_pri = nn.Parameter(torch.ones(len(etypes), self.n_heads))
        self.relation_att = nn.Parameter(
            torch.Tensor(len(etypes), n_heads, self.d_k, self.d_k)
        )
        self.relation_msg = nn.Parameter(
            torch.Tensor(len(etypes), n_heads, self.d_k, self.d_k)
        )
        self.skip = nn.Parameter(torch.ones(len(ntypes)))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def get_edge_attention_fn(self, e_id):

        def edge_attention(edges):
            relation_att = self.relation_att[e_id]
            relation_pri = self.relation_pri[e_id]
            relation_msg = self.relation_msg[e_id]
            key = torch.bmm(edges.src['k'].transpose(1, 0),
                            relation_att).transpose(1, 0)
            att = (edges.dst['q'] *
                   key).sum(dim=-1) * relation_pri / self.sqrt_dk
            val = torch.bmm(edges.src['v'].transpose(1, 0),
                            relation_msg).transpose(1, 0)
            return {'a': att, 'v': val}

        return edge_attention

    def message_func(self, edges):
        return {'v': edges.data['v'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        att = F.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(att.unsqueeze(dim=-1) * nodes.mailbox['v'], dim=1)
        return {'t': h.view(-1, self.out_dim)}

    def forward(self, G: DGLHeteroGraph, inp_key, out_key):
        for srctype, etype, dsttype in G.canonical_etypes:
            k_linear = self.k_linears[self.ndict[srctype]]
            v_linear = self.v_linears[self.ndict[srctype]]
            q_linear = self.q_linears[self.ndict[dsttype]]

            G.nodes[srctype].data['k'] =\
                k_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[srctype].data['v'] =\
                v_linear(G.nodes[srctype].data[inp_key]).view(-1, self.n_heads, self.d_k)
            G.nodes[dsttype].data['q'] =\
                q_linear(G.nodes[dsttype].data[inp_key]).view(-1, self.n_heads, self.d_k)

            G.apply_edges(
                func=self.get_edge_attention_fn(self.edict[etype]), etype=etype
            )
        G.multi_update_all({etype : (self.message_func, self.reduce_func) \
                            for etype in self.edict}, cross_reducer = 'mean')
        for ntype in G.ntypes:
            n_id = self.ndict[ntype]
            alpha = torch.sigmoid(self.skip[n_id])
            trans_out = self.a_linears[n_id](G.nodes[ntype].data['t'])
            trans_out = (
                trans_out * alpha + G.nodes[ntype].data[inp_key] * (1 - alpha)
            )
            if self.use_norm:
                G.nodes[ntype].data[out_key] = self.drop(
                    self.norms[n_id](trans_out)
                )
            else:
                G.nodes[ntype].data[out_key] = self.drop(trans_out)

    def __repr__(self):
        return '{}(in_dim={}, out_dim={}, num_types={}, num_types={})'.format(
            self.__class__.__name__, self.in_dim, self.out_dim, self.num_types,
            self.num_relations
        )
