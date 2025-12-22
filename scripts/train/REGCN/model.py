# Credit: https://github.com/bywmm/RE-GNN/tree/main
import torch
import torch as th
import torch.nn as nn
from dgl import function as fn
from torch import nn
from torch.nn import init


class REGraphConv(nn.Module):

    def __init__(
        self, num_etypes, scaling_factor, in_feats, out_feats, norm=True,
        bias=True, activation=None, weight=True, dropout=0.
    ):
        super(REGraphConv, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.norm = norm
        self.dropout = dropout

        # may add multi-head
        self.edge_weight = nn.Parameter(
            th.Tensor(num_etypes, 1), requires_grad=True
        )
        self.alpha = scaling_factor

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.feat_dropout = nn.Dropout(p=self.dropout)

        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

        init.constant_(self.edge_weight, 1.0 / self.alpha)

    def forward(
        self, graph, feat, e_feat, return_embedding=False, e_weight=None
    ):

        graph = graph.local_var()

        feat = self.feat_dropout(feat)

        edge_weight = self.edge_weight * self.alpha
        # edge_weight[6:] = 1.0
        edge_weight = nn.LeakyReLU()(edge_weight)
        ew = edge_weight[e_feat - 1]

        graph.edata.update({'ew_': ew.clone()})
        if e_weight is not None:
            ew *= e_weight.view(-1, 1)
        # ew = self.ew_dropout(ew)
        graph.edata.update({'ew': ew})
        # print(edge_weight.reshape(-1))

        if self.norm:
            num_nodes = graph.num_nodes()
            graph.ndata.update(
                {'nones': th.ones(num_nodes, 1).to(feat.device)}
            )
            # graph.update_all(
            #     fn.u_mul_e('nones', 'ew', 'm'),
            #     # graph.update_all(fn.copy_src(src='nones', out='m'),
            #     fn.sum('m', 'norm')
            # )
            graph.update_all(
                fn.u_mul_e('nones', 'ew_', 'm'),
                # graph.update_all(fn.copy_src(src='nones', out='m'),
                fn.sum('m', 'norm')
            )
            # norm = th.pow(graph.ndata['norm'].squeeze().clamp(min=1), -0.5)
            norm = th.pow(graph.ndata['norm'].squeeze().clamp(min=1), -0.5)
            shp = norm.shape + (1, ) * (feat.dim() - 1)
            norm = th.reshape(norm, shp).to(feat.device)
            feat = feat * norm

        if self.in_feats > self.out_feats:
            # mult W first to reduce the feature size for aggregation.
            if self.weight is not None:
                feat = th.matmul(feat, self.weight)
            graph.ndata['h'] = feat
            # graph.update_all(fn.copy_src(src='h', out='m'),
            graph.update_all(
                fn.u_mul_e('h', 'ew', 'm'), fn.sum(msg='m', out='h')
            )
            rst = graph.ndata['h']
        else:
            # aggregate first then mult W
            graph.ndata['h'] = feat
            # graph.update_all(fn.copy_src(src='h', out='m'),
            graph.update_all(
                fn.u_mul_e('h', 'ew', 'm'), fn.sum(msg='m', out='h')
            )
            rst = graph.ndata['h']
            if self.weight is not None:
                rst = th.matmul(rst, self.weight)

        if self.norm:
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self.activation is not None:
            rst = self.activation(rst)

        return rst


class REGCN(nn.Module):

    def __init__(
        self, g, num_etypes, R, in_feats, n_hidden, n_classes, n_layers,
        activation, dropout, feats_dim_list, use_sage=False
    ):
        super(REGCN, self).__init__()
        self.g = g
        self.num_layers = n_layers
        self.fc_list = nn.ModuleList(
            [
                nn.Linear(feats_dim, in_feats, bias=True)
                for feats_dim in feats_dim_list
            ]
        )
        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)

        self.layers = nn.ModuleList()
        GConv = RESAGEConv if use_sage else REGraphConv
        self.layers.append(
            GConv(
                num_etypes, R, in_feats, n_hidden, bias=False, activation=None,
                dropout=dropout, weight=False
            )
        )
        for i in range(1, n_layers - 1):
            self.layers.append(
                GConv(
                    num_etypes, R, n_hidden, n_hidden, activation=activation,
                    dropout=dropout
                )
            )
        self.layers.append(
            GConv(
                num_etypes, R, n_hidden, n_classes, bias=False,
                dropout=dropout, weight=False
            )
        )
        self.out_lin = nn.Linear(n_hidden, n_classes, bias=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features_list, e_feat, e_weight=None):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        h = self.layers[0](self.g, h, e_feat, e_weight=e_weight)

        for l in range(1, self.num_layers):
            h = self.dropout(h)
            h = self.layers[l](self.g, h, e_feat, e_weight=e_weight)
        out = self.out_lin(h)
        return out, h
