from functools import partial
from typing import TypedDict

import dgl
import torch
from dgl.nn.pytorch.conv import RelGraphConv
from torch import nn
from torch.nn import functional as F

from dhgl.models.common import SharedSpaceProjection
from dhgl.type import CEType, EType, NType


class RGCN(nn.Module):

    def __init__(
        self,
        num_nodes,
        h_dim,
        out_dim,
        num_rels,
        num_bases=-1,
        num_hidden_layers=1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers

        # create rgcn layers
        self.build_model()

        # create initial features
        self.features = self.create_features()

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden
        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    # initialize feature for each node
    def create_features(self):
        features = torch.arange(self.num_nodes)
        return features

    def build_input_layer(self):
        return RelGraphConv(
            self.num_nodes,
            self.h_dim,
            self.num_rels,
            num_bases=self.num_bases,
            activation=F.relu,
            is_input_layer=True,
        )

    def build_hidden_layer(self):
        return RelGraphConv(
            self.h_dim,
            self.h_dim,
            self.num_rels,
            num_bases=self.num_bases,
            activation=F.relu,
        )

    def build_output_layer(self):
        return RelGraphConv(
            self.h_dim,
            self.out_dim,
            self.num_rels,
            num_bases=self.num_bases,
            # activation=partial(F.softmax, dim=1),
        )

    def forward(self, g):
        if self.features is not None:
            g.ndata["id"] = self.features
        for layer in self.layers:
            layer(g)
        return g.ndata.pop("h")


class RGCN(nn.Module):

    class SharedFeatProjArgs(TypedDict):
        in_feat_shapes: dict[NType, tuple[int, ...]] | int
        embedding_max_norm: float | None

    def __init__(
        self,
        n_hidden: int,
        n_layers: int,
        n_out: int,
        num_ntypes: int,
        num_etypes: int,
        use_norm=True,
        # residual: bool | float = True,
        dropout: float = 0.5,
        proj_args: SharedFeatProjArgs | None = None,
    ):
        super().__init__()
        self.gcs: list[RelGraphConv] = nn.ModuleList(
            [
                RelGraphConv(
                    in_feat=n_hidden,
                    out_feat=n_hidden,
                    num_rels=num_etypes,
                    # num_bases=
                    activation=F.elu,
                    dropout=dropout,
                    layer_norm=use_norm,
                ) for i in range(n_layers)
            ]
        )
        self.n_hid = n_hidden
        self.n_out = n_out
        self.n_layers = n_layers
        self.proj = SharedSpaceProjection(
            n_out=n_hidden,
            **proj_args,
        )
        self.out = nn.Linear(n_hidden, n_out)
        self.reset_parameters()
        return

    def reset_parameters(self):
        for ntype in self.proj:
            nn.init.normal_(self.proj[ntype].weight, std=0.01)
            nn.init.zeros_(self.proj[ntype].bias)
        pass

    def forward(
        self,
        g: dgl.DGLGraph,
        feat: dict[NType, torch.Tensor],
        etypes: torch.Tensor,
    ) -> torch.Tensor:

        hs = {ntype: self.proj[ntype](h) for ntype, h in feat.items()}
        h = torch.concatenate(list(hs.values()), dim=0)

        for i, gc in enumerate(self.gcs):
            h = gc.forward(g, h, etypes)

        return self.out(h)
