from collections.abc import Iterable
from functools import partial
from typing import Iterable, TypedDict

import dgl
import torch
from dgl import DGLHeteroGraph
from dgl.heterograph import DGLBlock
from dgl.nn.pytorch import HGTConv as DGLHGTConv
from torch import nn

from ...type import EType, NType
from ..common import SharedSpaceProjection
from .hgtconv import (
    HGTConv,
    HGTLayer,
    LowRankRelationHGTConv,
    LowRankRelationSwitchHGTConv,
    SwitchHGTConv,
)


class LegacyHGT(nn.Module):

    def __init__(
        self, n_hidden: int, n_layers: int, n_heads: int, n_out: int,
        etypes: list, feat_dims: dict[NType, int], use_norm=True
    ):
        super().__init__()
        self.gcs: list[HGTConv] = nn.ModuleList(
            [
                HGTLayer(
                    in_dim=n_hidden,
                    out_dim=n_hidden,
                    ntypes=list(feat_dims.keys()),
                    etypes=etypes,
                    n_heads=n_heads,
                    use_norm=use_norm,
                ) for _ in range(n_layers)
            ]
        )
        self.n_hid = n_hidden
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleDict(
            {
                ntype: nn.Linear(feat_dim, n_hidden)
                for ntype, feat_dim in feat_dims.items()
            }
        )
        self.out = nn.Linear(n_hidden, n_out)
        return

    def forward(
        self,
        graph: DGLHeteroGraph,
        feat: dict[NType, torch.Tensor],
        target_ntype: NType,
    ) -> torch.Tensor:

        for ntype, linear in self.adapt_ws.items():
            graph.nodes[ntype].data['h'] = torch.tanh(linear(feat[ntype]))

        for gc in self.gcs:
            gc(graph, 'h', 'h')

        return self.out(graph.nodes[target_ntype].data['h'])

    def __repr__(self):
        return '{}(n_hid={}, n_out={}, n_layers={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out, self.n_layers
        )


class HGT(nn.Module):

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
        etypes: list[EType],
        head_size: int | None = None,
        use_norm=True,
        dropout: float = 0.5,
        normsoftmax: bool = False,
        shared_feat_proj_kwargs: SharedFeatProjArgs | None = None,
        relation_weights_rank: int | None = None,
        relation_pri_alpha: float | None = None,
        switching_mode: bool = False,
    ):
        super().__init__()
        conv_class = partial(
            LowRankRelationSwitchHGTConv
            if switching_mode else LowRankRelationHGTConv,
            rank=relation_weights_rank,
            relation_pri_alpha=relation_pri_alpha,
        )
        if (relation_weights_rank is None and relation_pri_alpha is None):
            conv_class = SwitchHGTConv if switching_mode else HGTConv
        head_size = head_size or (n_hidden // n_heads)
        n_hiddens = [n_hidden] + [n_heads * head_size] * (n_layers - 1)
        self.gcs: list[HGTConv] = nn.ModuleList(
            [
                conv_class(
                    in_size=n_hiddens[i],
                    head_size=head_size,
                    num_heads=n_heads,
                    ntypes=ntypes,
                    etypes=etypes,
                    use_norm=use_norm,
                    dropout=dropout,
                    normsoftmax=normsoftmax,
                ) for i in range(n_layers)
            ]
        )
        self.n_hid = n_hidden
        self.n_out = n_out
        self.n_layers = n_layers
        self.proj = None
        if shared_feat_proj_kwargs is not None:
            self.proj = SharedSpaceProjection(
                n_out=n_hidden,
                **shared_feat_proj_kwargs,
            )
        self.out = nn.Linear(head_size * n_heads, n_out)
        return

    def forward(
        self,
        hg: DGLHeteroGraph | list[DGLBlock],
        feat: dict[NType, torch.Tensor],
        target_ntype: NType | Iterable[NType],
        edge_weight: dict[EType, torch.Tensor | str] = None,
    ) -> torch.Tensor:

        if self.proj is not None:
            feat = {
                ntype: torch.tanh(h)
                for ntype, h in self.proj(feat).items()
            }

        def graph_forward(hg: DGLHeteroGraph):
            nonlocal feat
            for gc in self.gcs:
                feat = gc.forward(hg, feat, edge_weight)
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
        if isinstance(target_ntype, str):
            return self.out(h[target_ntype])

        return {ntype: self.out(h[ntype]) for ntype in target_ntype}

    def __repr__(self):
        return '{}(n_hid={}, n_out={}, n_layers={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out, self.n_layers
        )


class DGLHGT(nn.Module):

    def __init__(
        self,
        n_hidden: int,
        n_layers: int,
        n_heads: int,
        n_out: int,
        etypes: list,
        feat_dims: dict[NType, int],
        use_norm=True,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.gcs: list[DGLHGTConv] = nn.ModuleList(
            [
                DGLHGTConv(
                    in_size=n_hidden,
                    head_size=n_hidden // n_heads,
                    num_heads=n_heads,
                    num_ntypes=len(feat_dims),
                    num_etypes=len(etypes),
                    use_norm=use_norm,
                    dropout=dropout,
                ) for _ in range(n_layers)
            ]
        )
        self.n_hid = n_hidden
        self.n_out = n_out
        self.n_layers = n_layers
        self.adapt_ws = nn.ModuleDict(
            {
                ntype: nn.Linear(feat_dim, n_hidden)
                for ntype, feat_dim in feat_dims.items()
            }
        )
        self.out = nn.Linear(n_hidden, n_out)
        return

    def forward(
        self,
        hg: DGLHeteroGraph,
        feat: dict[NType, torch.Tensor],
        target_ntype: NType,
    ) -> torch.Tensor:

        def adaption():
            for ntype, linear in self.adapt_ws.items():
                yield torch.tanh(linear(feat[ntype]))

        def count2type(counts: list[int]):
            return torch.concat(
                [
                    torch.full(
                        (count, ), type_id, dtype=torch.long, device=g.device
                    ) for type_id, count in enumerate(counts)
                ]
            )

        g, ncounts, ecounts = dgl.to_homogeneous(
            hg, store_type=False, return_count=True
        )
        h = torch.concat(list(adaption()))
        ntype = count2type(ncounts)
        etype = count2type(ecounts)

        for gc in self.gcs:
            h = gc.forward(
                g,
                h,
                ntype=ntype,
                etype=etype,
                presorted=True,
            )

        return self.out(h[ntype == hg.get_ntype_id(target_ntype)])

    def __repr__(self):
        return '{}(n_hid={}, n_out={}, n_layers={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out, self.n_layers
        )
