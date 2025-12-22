from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dgl
import torch
from pydantic import model_validator
from torch.optim import AdamW

import dhgl
from dhgl import hgget as H
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.script_utils import BaseConfig

from .model import RGCN

# from torch.optim.lr_scheduler import OneCycleLR

if TYPE_CHECKING:
    from ..trainer.config import TrainerConfig


class VanillaRGCNConfig(BaseConfig):

    name: Literal['VanillaRGCN'] = 'VanillaRGCN'

    #######################
    # MODEL CONFIGS   #
    #######################
    hidden_dim: int

    num_layers: int
    """Number of Layer"""

    dropout: float
    use_norm: bool

    lr: float
    weight_decay: float

    def init(self, hg: BaseHeteroGraphLike, global_conf: TrainerConfig):
        hg, model, optimizer, scheduler, forward_fn = self._init(
            hg,
            n_out=H.n_classes(hg),
            global_conf=global_conf,
        )

        tgt_mask =\
            dgl.to_homogeneous(hg).ndata[dgl.NTYPE] == hg.get_ntype_id(H.tgt_ntype(hg))
        tgt_mask = tgt_mask.to(global_conf.device)

        def forward(graph, feat):
            return forward_fn(graph, feat)[tgt_mask]

        return hg, model, optimizer, scheduler, forward

    def _init(
        self, hg: BaseHeteroGraphLike, n_out: int, global_conf: TrainerConfig
    ):

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

        model = RGCN(
            n_hidden=self.hidden_dim,
            n_layers=self.num_layers,
            n_out=n_out,
            num_ntypes=len(hg.ntypes),
            num_etypes=(len(hg.etypes) + 1),  # original etypes + self-loop
            use_norm=self.use_norm,
            dropout=self.dropout,
            proj_args=RGCN.SharedFeatProjArgs(
                in_feat_shapes={
                    ntype: data.shape
                    for ntype, data in hg.ndata['feat'].items()
                },
            ),
        )
        optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        if (
            global_conf.batch_config.train.is_in_batch_mode
            and global_conf.batch_config.eval.is_in_batch_mode
        ):
            raise NotImplementedError
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
        e_feat = g.edata[dgl.ETYPE]

        def graph_forward(_: BaseHeteroGraphLike, feat: dict):
            if 'weight' in g.edata:
                raise NotImplementedError
            return model.forward(g, feat, e_feat)

        return hg, model, optimizer, None, graph_forward
