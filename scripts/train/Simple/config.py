from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dgl
import torch
from pydantic import Field, model_validator
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import dhgl
from dhgl import hgget as H
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.models import SimpleHGN
from dhgl.script_utils import BaseConfig

if TYPE_CHECKING:
    from ..trainer.config import TrainerConfig


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
    negative_slope: float
    edge_residual: bool = True

    lr: float
    weight_decay: float
    max_lr_scale: float
    pct_start_epoch: int

    embedding_max_norm: float | None = Field(None)
    """max_norm passed to the embedding layers"""
    l2_norm: bool = True

    @model_validator(mode='before')
    @classmethod
    def adapt_dropout(cls, data: dict):
        if 'dropout' in data:
            assert 'dropout_attn' not in data
            assert 'dropout_feat' not in data
            data['dropout_attn'] = data['dropout']
            data['dropout_feat'] = data.pop('dropout')
        return data

    def init(self, hg: BaseHeteroGraphLike, global_conf: TrainerConfig):

        unreachable_ntypes = [
            ntype for ntype in hg.ntypes if 'feat' not in hg.nodes[ntype].data
        ]
        data_dict = {}
        for etype in hg.canonical_etypes:
            s, _, d = etype
            if s in unreachable_ntypes or d in unreachable_ntypes:
                assert s == d
                continue
            data_dict[etype] = hg.edges(etype=etype)
        num_node_dict = {
            ntype: hg.num_nodes(ntype)
            for ntype in hg.ntypes if ntype not in unreachable_ntypes
        }
        hg_ = dhgl.transforms.update_graph_structure(
            hg,
            data_dict,
            num_nodes_dict=num_node_dict,
        )
        hg = hg_

        hg = dhgl.transforms.add_self_loop(hg)
        """Trainer for the MODEL"""

        model = SimpleHGN(
            edge_dim=self.edge_embedding_dim,
            num_etypes=(len(hg.etypes) + 1),  # original etypes + self-loop
            num_hidden=self.hidden_dim,
            num_classes=H.n_classes(hg),
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            activation=F.elu,
            feat_drop=self.dropout_feat,
            attn_drop=self.dropout_attn,
            negative_slope=self.negative_slope,
            residual=True,
            edge_residual=self.edge_residual,
            alpha=0.05,
            l2_norm=self.l2_norm,
            allow_zero_in_degree=True,
            shared_feat_proj_kwargs=SimpleHGN.SharedFeatProjArgs(
                in_feat_shapes={
                    ntype: data.shape
                    for ntype, data in hg.ndata['feat'].items()
                },
                embedding_max_norm=self.embedding_max_norm,
            ),
        )
        optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        iters_per_epoch = 1
        if global_conf.batch_config.train.is_in_batch_mode:
            n_samples = len(H.label(hg, 'train'))
            iters_per_epoch = n_samples // global_conf.batch_config.train.batch_size

        scheduler = OneCycleLR(
            optimizer,
            total_steps=(global_conf.epochs * iters_per_epoch),
            max_lr=self.lr * self.max_lr_scale,
            pct_start=self.pct_start_epoch / global_conf.epochs,
        )
        if not (
            global_conf.batch_config.train.is_in_batch_mode
            and global_conf.batch_config.eval.is_in_batch_mode
        ):
            edata = None
            if hg.edata['weight']:
                for etype in hg.canonical_etypes:
                    if etype not in hg.edata['weight']:
                        hg.edges[etype].data['weight'] = torch.ones(
                            hg.num_edges(etype=etype)
                        )
                edata = ['weight']
            g = dgl.to_homogeneous(hg, edata=edata).to(global_conf.device)
            # XXX: a little more memory consumption may introduce here.
            # the hg will be put to cuda somewhere else.
            # Things occupying cuda: hg.adj + hg.ndata + hg.edata + g.adj
            # the hg.adj is not used (but normally it only uses small amount of cuda memory.)
            tgt_mask = g.ndata[dgl.NTYPE] == hg.get_ntype_id(H.tgt_ntype(hg))
            e_feat = g.edata[dgl.ETYPE]

            def graph_forward(_: BaseHeteroGraphLike, feat: dict):
                return model.forward(
                    g, feat, e_feat, e_weight=g.edata.get('weight', None)
                )[tgt_mask]

        # mini-batch subgraph
        def mini_batch_forward(_hg: BaseHeteroGraphLike, feat: dict):

            g = dgl.to_homogeneous(_hg)
            tgt_mask = g.ndata[dgl.NTYPE] == _hg.get_ntype_id(H.tgt_ntype(hg))
            return model.forward(g, feat, g.edata[dgl.ETYPE])[tgt_mask]

        train_forward = (
            mini_batch_forward if
            global_conf.batch_config.train.is_in_batch_mode else graph_forward
        )
        eval_forward = (
            mini_batch_forward if
            global_conf.batch_config.eval.is_in_batch_mode else graph_forward
        )
        return hg, model, optimizer, scheduler, (train_forward, eval_forward)
