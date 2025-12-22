from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dgl

# from pydantic import Field
from torch.nn import functional as F
from torch.optim import Adam

import dhgl

# from dhgl import hgget as H
from dhgl.data.link_prediction import LinkPredDatasetLike
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.models import SimpleHGN
from dhgl.script_utils import BaseConfig

if TYPE_CHECKING:
    from ..linkpred import TrainerConfig
    from ..linkpred.base import HGNNReturnT


class SimpleConfig(BaseConfig):

    name: Literal['Simple'] = 'Simple'

    #######################
    # myGAT MODEL CONFIGS   #
    #######################

    edge_embedding_dim: int

    hidden_dim: int

    num_layers: int

    num_heads: int
    """Number of attention heads"""

    dropout_feat: float
    dropout_attn: float
    # negative_slope: float
    # edge_residual: bool = True
    # residual_att: float

    lr: float
    weight_decay: float
    # max_lr_scale: float
    # pct_start_epoch: int

    embedding_max_norm: float | None = None

    def init(
        self, dataset: LinkPredDatasetLike, global_conf: TrainerConfig
    ) -> HGNNReturnT:

        assert global_conf.decoder_config.dim == self.hidden_dim * (
            self.num_layers + 1
        ), ('SimpleHGN requires decoder dim == hidden_dim * (num_layers + 1)')

        hg = dataset.graph
        hg = dhgl.transforms.add_self_loop(hg)

        model = SimpleHGN(
            edge_dim=self.edge_embedding_dim,
            num_etypes=(len(hg.etypes) + 1),  # original etypes + self-loop
            num_hidden=self.hidden_dim,
            num_classes=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            activation=F.elu,
            feat_drop=self.dropout_feat,
            attn_drop=self.dropout_attn,
            negative_slope=0.01,
            residual=True,
            edge_residual=True,
            alpha=0.2,
            l2_norm=True,
            allow_zero_in_degree=True,
            shared_feat_proj_kwargs=SimpleHGN.SharedFeatProjArgs(
                in_feat_shapes={
                    ntype: data.shape
                    for ntype, data in hg.ndata['feat'].items()
                },
                embedding_max_norm=self.embedding_max_norm,
            ),
        )

        optimizer = Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        g = dgl.to_homogeneous(hg).to(global_conf.device)
        e_feat = g.edata[dgl.ETYPE]
        masks = {
            ntype:
            (g.ndata[dgl.NTYPE
                     ] == hg.get_ntype_id(ntype)).to(global_conf.device)
            for ntype in dataset.target_ntypes
        }

        def graph_forward(_: BaseHeteroGraphLike, feat: dict):
            res = model.forward_linkpred(g, feat, e_feat)
            res = {ntype: res[mask] for ntype, mask in masks.items()}
            return res

        dataset.graph = hg
        return {
            'dataset': dataset,
            'model': model,
            'optimizer': optimizer,
            'forward_fn': graph_forward
        }
