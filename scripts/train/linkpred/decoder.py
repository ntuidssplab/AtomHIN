from __future__ import annotations

from typing import Literal

import dgl
import torch
import torch.nn as nn
from torch.optim import Optimizer

from dhgl.data.link_prediction import LinkPredDatasetLike
from dhgl.script_utils import BaseConfig


class DistMult(nn.Module):

    def __init__(self, target_etypes, dim):
        super().__init__()
        etypes = [
            etype if isinstance(etype, str) else etype[1]
            for etype in target_etypes
        ]
        self.weights = nn.ParameterDict(
            {
                etype: nn.Parameter(torch.Tensor(size=(dim, dim)))
                for etype in etypes
            }
        )
        for etype in etypes:
            nn.init.xavier_normal_(self.weights[etype], gain=1.414)
        return

    def forward(self, graph: dgl.DGLHeteroGraph, xs: dict[str, torch.Tensor]):
        with graph.local_scope():
            for ntype in graph.ntypes:
                graph.nodes[ntype].data['x'] = xs[ntype]
            for etype in graph.canonical_etypes:
                e = self.weights[etype[1]]
                graph.nodes[etype[0]].data[etype] = xs[etype[0]] @ e
                graph.apply_edges(
                    dgl.function.u_dot_v(etype, 'x', 'score'), etype=etype
                )
            score = graph.edata['score']
            if isinstance(score, dict):
                score = torch.cat([v.sum(dim=1) for v in score.values()])
            else:
                score = torch.sum(score, dim=1)
            return score


class DistMultPlus(nn.Module):

    def __init__(self, target_etypes, dim, n_layers: int = 1, dropout=0.5):
        super().__init__()
        etypes = [
            etype if isinstance(etype, str) else etype[1]
            for etype in target_etypes
        ]
        self.weights = nn.ModuleDict(
            {
                etype: self._construct_layers(n_layers, dim, dropout=dropout)
                for etype in etypes
            }
        )
        # self.dropout = nn.Dropout(0.5)
        # self.weights = nn.ModuleDict(
        #     {etype: nn.Linear(dim, dim)
        #      for etype in etypes}
        # )
        return

    def _construct_layers(self, n_out_layers: int, dim: int, dropout: float):
        out = nn.Sequential()
        for _ in range(n_out_layers - 1):
            out.extend(
                [
                    nn.Linear(dim, dim),
                    nn.LayerNorm(dim),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                ]
            )
        out.append(nn.Linear(dim, dim, bias=False))
        return out

    def forward(self, graph: dgl.DGLHeteroGraph, xs: dict[str, torch.Tensor]):
        with graph.local_scope():
            for ntype in graph.ntypes:
                graph.nodes[ntype].data['x'] = xs[ntype]
            for etype in graph.canonical_etypes:
                graph.nodes[etype[0]].data[etype] = self.weights[etype[1]](
                    xs[etype[0]]
                )
                graph.apply_edges(
                    dgl.function.u_dot_v(etype, 'x', 'score'), etype=etype
                )

            score = graph.edata['score']
            if isinstance(score, dict):
                score = torch.cat([v.sum(dim=1) for v in score.values()])
            else:
                score = torch.sum(score, dim=1)
            return score


def embedding_dot(graph: dgl.DGLHeteroGraph, xs: dict[str, torch.Tensor]):
    with graph.local_scope():
        for ntype, x in xs.items():
            graph.nodes[ntype].data['x'] = x
        for etype in graph.canonical_etypes:
            graph.apply_edges(
                dgl.function.u_dot_v('x', 'x', 'score'), etype=etype
            )
        score = graph.edata['score']
        if isinstance(score, dict):
            score = torch.cat(list(score.values()))
        return score.squeeze()


# class Dot(nn.Module):

#     def __init__(self):
#         super().__init__()
#         return

#     def forward(self, graph: dgl.DGLHeteroGraph, xs: dict[str, torch.Tensor]):
#         with graph.local_scope():
#             for ntype in graph.ntypes:
#                 graph.nodes[ntype].data['x'] = xs[ntype]
#                 for etype in graph.canonical_etypes:
#                     graph.apply_edges(
#                         dgl.function.u_dot_v('x', 'x', 'score'), etype=etype
#                     )
#             score = graph.edata['score']
#             breakpoint()
#             if isinstance(score, dict):
#                 score = torch.cat(list(score.values()))
#             return score.squeeze()


class DecoderConfig(BaseConfig):

    name: Literal['dot', 'dist_mul', 'dist_mul+']
    dim: int
    l2_norm: bool | None = None
    num_layers: int | None = None
    dropout: float | None = None

    def add_l2_norm(self, forward_fn):

        def _l2_norm(logits: torch.Tensor):
            n: torch.Tensor = torch.norm(logits, dim=1, keepdim=True)
            return logits / n.clamp(1e-12)

        def wrap(graph, xs: dict):
            xs_ = {ntype: _l2_norm(x) for ntype, x in xs.items()}
            return forward_fn(graph, xs_)

        return wrap

    def init(self, dataset: LinkPredDatasetLike, optimizer: Optimizer):
        fn = self._init(dataset, optimizer)
        if self.l2_norm:
            if isinstance(fn, nn.Module):
                fn.forward = self.add_l2_norm(fn.forward)
            else:
                fn = self.add_l2_norm(fn)
            return fn
        return fn

    def _init(self, dataset: LinkPredDatasetLike, optimizer: Optimizer):
        if self.name == 'dist_mul':
            dist_mult = DistMult(dataset.target_etypes, self.dim)
            optimizer.add_param_group({'params': dist_mult.parameters()})
            return dist_mult
        elif self.name == 'dist_mul+':
            dist_mult = DistMultPlus(
                dataset.target_etypes, self.dim, self.num_layers or 1,
                self.dropout
            )
            optimizer.add_param_group({'params': dist_mult.parameters()})
            return dist_mult
        assert self.name == 'dot'
        assert len(dataset.target_etypes) == 1
        return embedding_dot
