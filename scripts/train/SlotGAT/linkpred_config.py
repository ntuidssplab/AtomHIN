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
from dhgl.script_utils import BaseConfig

from .model import slotGAT

if TYPE_CHECKING:
    from ..linkpred import TrainerConfig
    from ..linkpred.base import HGNNReturnT


class SlotGATConfig(BaseConfig):

    name: Literal['SlotGAT'] = 'SlotGAT'

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
    residual_att: float

    lr: float
    weight_decay: float
    # max_lr_scale: float
    # pct_start_epoch: int

    # embedding_max_norm: float | None = Field(None)
    # """max_norm passed to the embedding layers"""

    # decoder: Literal['dot', 'distmult']
    # l2_norm: bool = True
    SAattDim: int

    def init(
        self, dataset: LinkPredDatasetLike, global_conf: TrainerConfig
    ) -> HGNNReturnT:

        hg = dataset.graph
        hg = dhgl.transforms.add_self_loop(hg)
        """Trainer for the MODEL"""

        assert global_conf.decoder_config.dim == self.hidden_dim * (
            self.num_layers + 2
        ), ('SlotGAT requires decoder dim == hidden_dim * (num_layers + 2)')

        # dataset_and_hypers = {
        #     ("PubMed_LP", 1, 5): {
        #         "hidden-dim": "[64]",
        #         "num-layers": "[4]",
        #         "lr": "[1e-3]",
        #         "weight-decay": "[1e-4]",
        #         "feats-type": [2],
        #         "num-heads": [2],
        #         "epoch": [1000],
        #         "decoder": ["distmult"],
        #         "batch-size": [
        #             8192,
        #         ],
        #         "dropout_feat": [0.5],
        #         "dropout_attn": [0.5],
        #         "residual_att": [0.2],
        #         "residual": ["True"],
        #         "SAattDim": [32]
        #     },
        #     ("LastFM", 1, 5): {
        #         "hidden-dim": "[64]",
        #         "num-layers": "[8]",
        #         "lr": "[5e-4]",
        #         "weight-decay": "[1e-4]",
        #         "feats-type": [2],
        #         "num-heads": [2],
        #         "epoch": [1000],
        #         "decoder": ["dot"],
        #         "batch-size": [
        #             8192,
        #         ],
        #         "SAattDim": [64],
        #         "dropout_feat": [0.2],
        #         "dropout_attn": [0.9],
        #         "residual_att": [0.5],
        #         "residual": ["True"]
        #     }
        # }
        g = dgl.to_homogeneous(hg).to(global_conf.device)
        model = slotGAT(
            g,
            self.edge_embedding_dim,
            num_etypes=len(hg.canonical_etypes) + 1,
            # len(dl.links['count']) * 2 + 1,
            # in_dims=in_dims,
            in_dims=[
                hg.nodes[ntype].data['feat'].shape[-1] for ntype in hg.ntypes
            ],
            num_hidden=self.hidden_dim,
            num_classes=self.hidden_dim,
            num_layers=self.num_layers,
            heads=[self.num_heads] * (self.num_layers + 1),
            activation=F.elu,
            feat_drop=self.dropout_feat,
            attn_drop=self.dropout_attn,
            # negative_slope=self.negative_slope,
            negative_slope=0.01,
            residual=True,
            alpha=self.residual_att,
            num_ntype=len(hg.ntypes),
            eindexer=None,
            # decode=self.decoder,  # distmult
            decode=None,  # distmult
            aggregator='SA',
            inProcessEmb='True',
            l2BySlot='True',
            prod_aggr='None',
            sigmoid='after',
            l2use='True',
            SAattDim=self.SAattDim,
            dataRecorder={
                "meta": {},
                "data": {},
                "status": "None"
            },
            get_out=[],
            predicted_by_slot='None',
        )

        optimizer = Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # if not (
        #     global_conf.batch_config.train.is_in_batch_mode
        #     and global_conf.batch_config.eval.is_in_batch_mode
        # ):
        # XXX: a little more memory consumption may introduce here.
        # the hg will be put to cuda somewhere else.
        # Things occupying cuda: hg.adj + hg.ndata + hg.edata + g.adj
        # the hg.adj is not used (but normally it only uses small amount of cuda memory.)
        # tgt_mask = g.ndata[dgl.NTYPE] == hg.get_ntype_id(H.tgt_ntype(hg))
        e_feat = g.edata[dgl.ETYPE]
        masks = {
            ntype:
            (g.ndata[dgl.NTYPE
                     ] == hg.get_ntype_id(ntype)).to(global_conf.device)
            for ntype in dataset.target_ntypes
        }

        def graph_forward(_: BaseHeteroGraphLike, feat: dict):
            res = model.forward(list(feat.values()), e_feat)
            res = {ntype: res[mask] for ntype, mask in masks.items()}
            return res

        dataset.graph = hg
        return {
            'dataset': dataset,
            'model': model,
            'optimizer': optimizer,
            'forward_fn': graph_forward
        }
