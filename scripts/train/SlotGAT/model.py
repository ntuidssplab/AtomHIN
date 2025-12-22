# Ref: https://github.com/scottjiao/SlotGAT_ICML23/tree/new/LP/methods/slotGAT
"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
from __future__ import annotations

import torch
import torch as th
from dgl import function as fn
from dgl._ffi.base import DGLError
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from torch import nn


class slotGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """

    def __init__(
        self, edge_feats, num_etypes, in_feats, out_feats, num_heads,
        feat_drop=0., attn_drop=0., negative_slope=0.2, residual=False,
        activation=None, allow_zero_in_degree=False, bias=False, alpha=0.,
        num_ntype=None, eindexer=None, inputhead=False
    ):
        super().__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(
            num_etypes, edge_feats
        ) if edge_feats else None
        self.eindexer = eindexer
        self.num_ntype = num_ntype
        self.attentions = None

        if isinstance(in_feats, tuple):
            raise NotImplementedError()
        else:
            self.fc = nn.Parameter(
                th.FloatTensor(
                    size=(
                        self.num_ntype, self._in_src_feats,
                        out_feats * num_heads
                    )
                )
            )
            """else:
                self.fc =nn.ModuleList([nn.Linear(
                    self._in_src_feats, out_feats * num_heads, bias=False)  for _ in range(num_ntype)] )
                raise Exception("!!!")"""
        self.fc_e = nn.Linear(
            edge_feats, edge_feats * num_heads, bias=False
        ) if edge_feats else None

        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats * self.num_ntype))
        )
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats * self.num_ntype))
        )
        self.attn_e = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, edge_feats))
        ) if edge_feats else None
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Parameter(
                    th.FloatTensor(
                        size=(
                            self.num_ntype, self._in_src_feats,
                            out_feats * num_heads
                        )
                    )
                )

            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            raise NotImplementedError()
        self.alpha = alpha
        self.inputhead = inputhead

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc, gain=gain)

        else:
            raise Exception("!!!")
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

        if self._edge_feats:
            nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        elif isinstance(self.res_fc, Identity):
            pass
        elif isinstance(self.res_fc, nn.Parameter):
            nn.init.xavier_normal_(self.res_fc, gain=gain)
        if self._edge_feats:
            nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, get_out=[""], res_attn=None):
        with graph.local_scope():
            # node_idx_by_ntype = graph.node_idx_by_ntype
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        'There are 0-in-degree nodes in the graph, '
                        'output for those nodes will be invalid. '
                        'This is harmful for some applications, '
                        'causing silent performance regression. '
                        'Adding self-loop on the input graph by '
                        'calling `g = dgl.add_self_loop(g)` will resolve '
                        'the issue. Setting ``allow_zero_in_degree`` '
                        'to be `True` when constructing this module will '
                        'suppress the check and let the code run.'
                    )

            if isinstance(feat, tuple):
                raise Exception("!!!")
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                feat_dst = self.fc_dst(h_dst).view(
                    -1, self._num_heads, self._out_feats
                )
            else:
                #feature transformation first
                h_src = h_dst = self.feat_drop(
                    feat
                )  #num_nodes*(num_ntype*input_dim)

                if self.inputhead:
                    h_src = h_src.view(
                        -1, 1, self.num_ntype, self._in_src_feats
                    )
                else:
                    h_src = h_src.view(
                        -1, self._num_heads, self.num_ntype,
                        int(self._in_src_feats / self._num_heads)
                    )
                h_dst = h_src = h_src.permute(2, 0, 1, 3).flatten(
                    2
                )  #num_ntype*num_nodes*(in_feat_dim)
                if "getEmb" in get_out:
                    self.emb = h_dst.cpu().detach()
                #self.fc with num_ntype*(in_feat_dim)*(out_feats * num_heads)
                feat_dst = torch.bmm(
                    h_src, self.fc
                )  #num_ntype*num_nodes*(out_feats * num_heads)

                feat_src = feat_dst =feat_dst.permute(1,0,2).view(                 #num_nodes*num_heads*(num_ntype*hidden_dim)
                        -1,self.num_ntype ,self._num_heads, self._out_feats).permute(0,2,1,3).flatten(2)

                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            e_feat = self.edge_emb(e_feat) if self._edge_feats else None
            e_feat = self.fc_e(e_feat).view(
                -1, self._num_heads, self._edge_feats
            ) if self._edge_feats else None

            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(
                -1
            ) if self._edge_feats else 0  #(-1, self._num_heads, 1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee}) if self._edge_feats else None
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e_ = graph.edata.pop('e')
            ee = graph.edata.pop('ee') if self._edge_feats else 0
            e = e_ + ee

            e = self.leaky_relu(e)
            # compute softmax
            a = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                a = a * (1 - self.alpha) + res_attn * self.alpha

            graph.edata['a'] = a
            # then message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))

            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                if self._in_dst_feats != self._out_feats:
                    resval = torch.bmm(h_src, self.res_fc)
                    resval =resval.permute(1,0,2).view(                 #num_nodes*num_heads*(num_ntype*hidden_dim)
                        -1,self.num_ntype ,self._num_heads, self._out_feats).permute(0,2,1,3).flatten(2)
                    #resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                else:
                    resval = self.res_fc(h_src).view(
                        h_dst.shape[0], -1, self._out_feats * self.num_ntype
                    )  #Identity

                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            self.attentions = graph.edata.pop('a').detach()
            torch.cuda.empty_cache()
            return rst, self.attentions


# pylint: enable=W0235
class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """

    def __init__(
        self, edge_feats, num_etypes, in_feats, out_feats, num_heads,
        feat_drop=0., attn_drop=0., negative_slope=0.2, residual=False,
        activation=None, allow_zero_in_degree=False, bias=False, alpha=0.
    ):
        super().__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
        self.fc_e = nn.Linear(edge_feats, edge_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_e = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, edge_feats))
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False
                )
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, res_attn=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        'There are 0-in-degree nodes in the graph, '
                        'output for those nodes will be invalid. '
                        'This is harmful for some applications, '
                        'causing silent performance regression. '
                        'Adding self-loop on the input graph by '
                        'calling `g = dgl.add_self_loop(g)` will resolve '
                        'the issue. Setting ``allow_zero_in_degree`` '
                        'to be `True` when constructing this module will '
                        'suppress the check and let the code run.'
                    )

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                feat_dst = self.fc_dst(h_dst).view(
                    -1, self._num_heads, self._out_feats
                )
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            e_feat = self.edge_emb(e_feat)
            e_feat = self.fc_e(e_feat
                               ).view(-1, self._num_heads, self._edge_feats)
            ee = (e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e') + graph.edata.pop('ee'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata[
                    'a'] * (1 - self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(
                    h_dst.shape[0], -1, self._out_feats
                )
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()


import math

import dgl.function as fn
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl._ffi.base import DGLError
from dgl.nn.pytorch import edge_softmax

# from torch.profiler import ProfilerActivity, profile, record_function
"""
class DistMult(nn.Module):
    def __init__(self, num_rel, dim):
        super(DistMult, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rel, dim, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)

    def forward(self, left_emb, right_emb, r_id):
        thW = self.W[r_id]
        left_emb = torch.unsqueeze(left_emb, 1)
        right_emb = torch.unsqueeze(right_emb, 2)
        return torch.bmm(torch.bmm(left_emb, thW), right_emb).squeeze()"""


class DistMult(nn.Module):

    def __init__(self, num_rel, dim):
        super().__init__()
        self.W = nn.Parameter(torch.FloatTensor(size=(num_rel, dim, dim)))
        nn.init.xavier_normal_(self.W, gain=1.414)

    def forward(
        self, left_emb, right_emb, r_id, slot_num=None, prod_aggr=None,
        sigmoid="after"
    ):
        if not prod_aggr:
            thW = self.W[r_id]
            left_emb = torch.unsqueeze(left_emb, 1)
            right_emb = torch.unsqueeze(right_emb, 2)
            #return torch.bmm(torch.bmm(left_emb, thW), right_emb).squeeze()
            scores = torch.zeros(right_emb.shape[0]).to(right_emb.device)
            for i in range(int(max(r_id)) + 1):
                scores[r_id == i] = torch.bmm(
                    torch.matmul(left_emb[r_id == i], self.W[i]),
                    right_emb[r_id == i]
                ).squeeze()
            return scores
        else:
            raise Exception


class Dot(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self, left_emb, right_emb, r_id, slot_num=None, prod_aggr=None,
        sigmoid="after"
    ):
        if not prod_aggr:
            left_emb = torch.unsqueeze(left_emb, 1)
            right_emb = torch.unsqueeze(right_emb, 2)
            return torch.bmm(left_emb, right_emb).squeeze()
        else:
            left_emb = left_emb.view(
                -1, slot_num, int(left_emb.shape[1] / slot_num)
            )
            right_emb = right_emb.view(
                -1, int(right_emb.shape[1] / slot_num), slot_num
            )
            x = torch.bmm(
                left_emb, right_emb
            )  # num_sampled_edges* num_slot*num_slot
            if prod_aggr == "all":
                x = x.flatten(1)
                x = x.sum(1)
                return x
            x = torch.diagonal(x, 0, 1, 2)  # num_sampled_edges* num_slot
            if sigmoid == "before":
                x = F.sigmoid(x)

            if prod_aggr == "mean":
                x = x.mean(1)

            elif prod_aggr == "max":
                x = x.max(1)[0]
            elif prod_aggr == "sum":
                x = x.sum(1)
            else:
                raise Exception()
            return x


class myGAT(nn.Module):

    def __init__(
        self, g, edge_dim, num_etypes, in_dims, num_hidden, num_classes,
        num_layers, heads, activation, feat_drop, attn_drop, negative_slope,
        residual, alpha, decode='distmult', inProcessEmb="True", l2use="True",
        dataRecorder=None, get_out=None
    ):
        super().__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.inProcessEmb = inProcessEmb
        self.l2use = l2use
        self.dataRecorder = dataRecorder
        self.fc_list = nn.ModuleList(
            [nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims]
        )
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(
            myGATConv(
                edge_dim, num_etypes, num_hidden, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation,
                alpha=alpha
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                myGATConv(
                    edge_dim, num_etypes, num_hidden * heads[l - 1],
                    num_hidden, heads[l], feat_drop, attn_drop, negative_slope,
                    residual, self.activation, alpha=alpha
                )
            )
        # output projection
        self.gat_layers.append(
            myGATConv(
                edge_dim, num_etypes, num_hidden * heads[-2], num_classes,
                heads[-1], feat_drop, attn_drop, negative_slope, residual,
                None, alpha=alpha
            )
        )
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        if decode == 'distmult':
            self.decoder = DistMult(num_etypes, num_classes * (num_layers + 2))
        elif decode == 'dot':
            self.decoder = Dot()
        self.get_out = get_out

    def l2_norm(self, x):
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        return x / (
            torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon)
        )

    def forward(self, features_list, e_feat, left, right, mid):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        emb = [self.l2_norm(h)]
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](
                self.g, h, e_feat, res_attn=res_attn
            )
            emb.append(self.l2_norm(h.mean(1)))
            h = h.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](
            self.g, h, e_feat, res_attn=res_attn
        )  #None)
        logits = logits.mean(1)
        logits = self.l2_norm(logits)
        #emb.append(logits)
        if self.inProcessEmb == "True":
            emb.append(logits)
        else:
            emb = [logits]
        logits = torch.cat(emb, 1)
        left_emb = logits[left]
        right_emb = logits[right]
        return F.sigmoid(self.decoder(left_emb, right_emb, mid))


class slotGAT(nn.Module):

    def __init__(
        self, g, edge_dim, num_etypes, in_dims, num_hidden, num_classes,
        num_layers, heads, activation, feat_drop, attn_drop, negative_slope,
        residual, alpha, num_ntype, eindexer, aggregator="average",
        predicted_by_slot="None", get_out=[""], decode='distmult',
        inProcessEmb="True", l2BySlot="False", prod_aggr=None, sigmoid="after",
        l2use="True", SAattDim=128, dataRecorder=None
    ):
        super().__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.heads = heads
        self.activation = activation
        self.fc_list = nn.ModuleList(
            [nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims]
        )
        self.num_ntype = num_ntype
        self.num_classes = num_classes
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.predicted_by_slot = predicted_by_slot
        self.inProcessEmb = inProcessEmb
        self.l2BySlot = l2BySlot
        self.prod_aggr = prod_aggr
        self.sigmoid = sigmoid
        self.l2use = l2use
        self.SAattDim = SAattDim
        self.dataRecorder = dataRecorder

        self.get_out = get_out
        self.num_etypes = num_etypes
        self.num_hidden = num_hidden
        self.last_fc = nn.Parameter(
            th.FloatTensor(size=(num_classes * self.num_ntype, num_classes))
        )
        nn.init.xavier_normal_(self.last_fc, gain=1.414)

        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        # input projection (no residual)
        self.gat_layers.append(
            slotGATConv(
                edge_dim, num_etypes, num_hidden, num_hidden, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation,
                alpha=alpha, num_ntype=num_ntype, eindexer=eindexer,
                inputhead=True
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                slotGATConv(
                    edge_dim, num_etypes, num_hidden * heads[l - 1],
                    num_hidden, heads[l], feat_drop, attn_drop, negative_slope,
                    residual, self.activation, alpha=alpha,
                    num_ntype=num_ntype, eindexer=eindexer
                )
            )
        # output projection
        self.gat_layers.append(
            slotGATConv(
                edge_dim, num_etypes, num_hidden * heads[-2], num_classes,
                heads[-1], feat_drop, attn_drop, negative_slope, residual,
                None, alpha=alpha, num_ntype=num_ntype, eindexer=eindexer
            )
        )
        self.aggregator = aggregator
        if aggregator == "SA":
            if self.inProcessEmb == "True":
                last_dim = num_hidden * (2 + num_layers)
            else:
                last_dim = num_hidden

            self.macroLinear = nn.Linear(last_dim, self.SAattDim, bias=True)
            nn.init.xavier_normal_(self.macroLinear.weight, gain=1.414)
            nn.init.normal_(
                self.macroLinear.bias, std=1.414 *
                math.sqrt(1 / (self.macroLinear.bias.flatten().shape[0]))
            )
            self.macroSemanticVec = nn.Parameter(
                torch.FloatTensor(self.SAattDim, 1)
            )
            nn.init.normal_(self.macroSemanticVec, std=1)

        self.by_slot = [f"by_slot_{nt}" for nt in range(num_ntype)]
        assert aggregator in (
            ["average", "last_fc", "max", "None", "SA"] + self.by_slot
        )
        #self.get_out=get_out
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        if decode == 'distmult':
            if self.aggregator == "None":
                num_classes = num_classes * num_ntype
            self.decoder = DistMult(num_etypes, num_classes * (num_layers + 2))
        elif decode == 'dot':
            self.decoder = Dot()
        else:
            self.decoder = None

    # def forward(
    #     self, features_list, e_feat, left, right, mid, get_out="False"
    # ):
    def forward(self, features_list, e_feat, get_out="False"):
        encoded_embeddings = None

        h = []
        for nt_id, (fc,
                    feature) in enumerate(zip(self.fc_list, features_list)):
            nt_ft = fc(feature)
            emsen_ft = torch.zeros(
                [nt_ft.shape[0], nt_ft.shape[1] * self.num_ntype]
            ).to(feature.device)
            emsen_ft[:, nt_ft.shape[1] * nt_id:nt_ft.shape[1] *
                     (nt_id + 1)] = nt_ft
            h.append(emsen_ft)  # the id is decided by the node types
        h = torch.cat(h, 0)  #  num_nodes*(num_type*hidden_dim)

        emb = [self.aggr_func(self.l2_norm(h, l2BySlot=self.l2BySlot))]
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](
                self.g, h, e_feat, get_out=get_out, res_attn=res_attn
            )  #num_nodes*num_heads*(num_ntype*hidden_dim)
            emb.append(
                self.aggr_func(
                    self.l2_norm(h.mean(1), l2BySlot=self.l2BySlot)
                )
            )
            h = h.flatten(1)  #num_nodes*(num_heads*num_ntype*hidden_dim)

        # output projection
        logits, _ = self.gat_layers[-1](
            self.g, h, e_feat, get_out=get_out, res_attn=res_attn
        )  #None)   #num_nodes*num_heads*num_ntype*hidden_dim

        logits = logits.mean(1)
        if self.predicted_by_slot != "None" and self.training == False:
            logits = logits.view(-1, 1, self.num_ntype, self.num_classes)

            if self.predicted_by_slot == "max":
                if "getMaxSlot" in get_out:
                    maxSlotIndexesWithLabels = logits.max(2)[1].squeeze(1)
                    logits_indexer = logits.max(2)[0].max(2)[1]
                    self.maxSlotIndexes = torch.gather(
                        maxSlotIndexesWithLabels, 1, logits_indexer
                    )
                logits = logits.max(2)[0]
            elif self.predicted_by_slot == "all":
                if "getSlots" in get_out:
                    self.logits = logits.detach()
                logits = logits.view(-1, 1, self.num_ntype,
                                     self.num_classes).mean(2)

            else:
                target_slot = int(self.predicted_by_slot)
                logits = logits[:, :, target_slot, :].squeeze(2)
        else:
            logits = self.aggr_func(
                self.l2_norm(logits, l2BySlot=self.l2BySlot)
            )

        if self.inProcessEmb == "True":
            emb.append(logits)
        else:
            emb = [logits]
        if self.aggregator == "None" and self.inProcessEmb == "True":
            emb = [
                x.view(-1, self.num_ntype, int(x.shape[1] / self.num_ntype))
                for x in emb
            ]
            o = torch.cat(emb, 2).flatten(1)
        else:
            o = torch.cat(emb, 1)
        if self.aggregator == "SA":
            o = o.view(-1, self.num_ntype, int(o.shape[1] / self.num_ntype))

            slot_scores = (
                F.tanh(self.macroLinear(o)) @ self.macroSemanticVec
            ).mean(0, keepdim=True)  #num_slots
            self.slot_scores = F.softmax(slot_scores, dim=1)
            o = (o * self.slot_scores).sum(1)

        return o
        # left_emb = o[left]
        # right_emb = o[right]
        # if self.sigmoid == "after":
        #     logits = self.decoder(
        #         left_emb, right_emb, mid, slot_num=self.num_ntype,
        #         prod_aggr=self.prod_aggr
        #     )
        #     logits = F.sigmoid(logits)
        # elif self.sigmoid == "before":

        #     logits = self.decoder(
        #         left_emb, right_emb, mid, slot_num=self.num_ntype,
        #         prod_aggr=self.prod_aggr, sigmoid=self.sigmoid
        #     )
        # elif self.sigmoid == "None":
        #     left_emb = self.l2_norm(left_emb, l2BySlot=self.l2BySlot)
        #     right_emb = self.l2_norm(right_emb, l2BySlot=self.l2BySlot)
        #     logits = self.decoder(
        #         left_emb, right_emb, mid, slot_num=self.num_ntype,
        #         prod_aggr=self.prod_aggr
        #     )
        # else:
        #     raise Exception()
        # return logits

    def l2_norm(self, x, l2BySlot="False"):
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
        if self.l2use in ("True", True):
            if l2BySlot == "False":
                return x / (
                    torch.
                    max(torch.norm(x, dim=1, keepdim=True), self.epsilon)
                )
            elif l2BySlot == "True":
                x = x.view(
                    -1, self.num_ntype, int(x.shape[1] / self.num_ntype)
                )
                x = x / (
                    torch.
                    max(torch.norm(x, dim=2, keepdim=True), self.epsilon)
                )
                x = x.flatten(1)
                return x
        elif self.l2use == "False":
            return x
        else:
            raise Exception()

    def aggr_func(self, logits):
        if self.aggregator == "average":
            logits = logits.view(-1, self.num_ntype, self.num_classes).mean(1)
        elif self.aggregator == "last_fc":
            logits = logits.view(-1, self.num_ntype, self.num_classes)
            logits = logits.flatten(1)
            logits = logits.matmul(self.last_fc).unsqueeze(1)
        elif self.aggregator == "max":
            logits = logits.view(-1, self.num_ntype,
                                 self.num_classes).max(1)[0]

        elif self.aggregator == "None" or "SA":
            logits = logits.view(-1, self.num_ntype,
                                 self.num_classes).flatten(1)

        else:
            raise NotImplementedError()

        return logits
