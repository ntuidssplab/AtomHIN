from __future__ import annotations

import random
from typing import TYPE_CHECKING, Literal

import dgl
import torch

# from pydantic import Field from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import dhgl
from dhgl import hgget as H
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.script_utils import BaseConfig

from .model import HINormer

if TYPE_CHECKING:
    from ..trainer.config import TrainerConfig


class HINormerConfig(BaseConfig):

    name: Literal['HINormer'] = 'HINormer'

    hidden_dim: int

    num_layers: int

    num_heads: int
    """Number of attention heads"""

    seq_len: int

    num_gnns: int
    dropout: float
    temperature: float
    beta: float

    lr: float
    weight_decay: float
    max_lr_scale: float | None = None
    pct_start_epoch: int | None = None

    def init(self, hg: BaseHeteroGraphLike, global_conf: TrainerConfig):

        hg, model, optimizer, scheduler, forward_fn = self._init(
            hg,
            n_out=H.n_classes(hg),
            global_conf=global_conf,
        )

        g = dgl.to_homogeneous(hg)
        target_nids = (g.ndata[dgl.NTYPE
                               ] == hg.get_ntype_id(H.tgt_ntype(hg)
                                                    )).nonzero().squeeze()
        node_seq = self.get_node_seq(g, target_nids,
                                     self.seq_len).to(global_conf.device)

        def forward(graph, feat):
            return forward_fn(graph, feat, node_seq)

        return hg, model, optimizer, scheduler, forward
        # # tgt_mask = g.ndata[dgl.NTYPE] == hg.get_ntype_id(H.tgt_ntype(hg))
        # # e_feat = g.edata[dgl.ETYPE]
        # type_emb = torch.eye(len(hg.ntypes)).to(global_conf.device)
        # node_type = g.ndata[dgl.NTYPE]

        # def graph_forward(_: BaseHeteroGraphLike, feat: dict):
        #     res = model.forward(
        #         list(feat.values()), node_seq, type_emb, node_type, norm=True
        #     )
        #     return res

        # return hg, model, optimizer, scheduler, graph_forward

    def _init(
        self, hg: BaseHeteroGraphLike, n_out: int, global_conf: TrainerConfig
    ):

        hg = dhgl.transforms.add_self_loop(hg)
        """Trainer for the MODEL"""
        g = dgl.to_homogeneous(hg).to(global_conf.device)

        in_dims = [
            features.shape[-1] for features in hg.ndata['feat'].values()
        ]
        model = HINormer(
            g,
            n_out,
            in_dims,
            self.hidden_dim,
            self.num_layers,
            self.num_gnns,
            self.num_heads,
            self.dropout,
            temper=self.temperature,
            num_type=len(hg.ntypes),
            beta=self.beta,
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

        scheduler = None
        if self.max_lr_scale is not None:
            scheduler = OneCycleLR(
                optimizer,
                total_steps=(global_conf.epochs * iters_per_epoch),
                max_lr=self.lr * self.max_lr_scale,
                pct_start=self.pct_start_epoch / global_conf.epochs,
            )

        if (
            global_conf.batch_config.train.is_in_batch_mode
            and global_conf.batch_config.eval.is_in_batch_mode
        ):
            raise NotImplementedError
        # tgt_mask = g.ndata[dgl.NTYPE] == hg.get_ntype_id(H.tgt_ntype(hg))
        # e_feat = g.edata[dgl.ETYPE]
        type_emb = torch.eye(len(hg.ntypes)).to(global_conf.device)
        node_type = g.ndata[dgl.NTYPE]

        # target_nids = (node_type == hg.get_ntype_id(H.tgt_ntype(hg))).nonzero().squeeze()
        # node_seq = get_node_seq(g.cpu(), target_nids, self.seq_len).to(global_conf.device)

        def graph_forward(_: BaseHeteroGraphLike, feat: dict, node_seq):
            return model.forward(
                list(feat.values()), node_seq, type_emb, node_type, norm=True
            )

        # def graph_forward(_: BaseHeteroGraphLike, feat: dict):
        #     res = model.forward(
        #         list(feat.values()), node_seq, type_emb, node_type, norm=True
        #     )
        #     return res

        return hg, model, optimizer, scheduler, graph_forward

    @classmethod
    def get_node_seq(cls, g, target_nids: torch.Tensor, seq_len: int):
        # node_seq = torch.zeros(features_list[0].shape[0], args.len_seq).long()
        node_seq = torch.zeros(len(target_nids), seq_len).long()

        n = 0

        for x in target_nids:

            cnt = 0
            scnt = 0
            node_seq[n, cnt] = x
            cnt += 1
            start = node_seq[n, scnt].item()
            while (cnt < seq_len):
                sample_list = g.successors(start).numpy().tolist()
                nsampled = max(len(sample_list), 1)
                sampled_list = random.sample(sample_list, nsampled)
                for i in range(nsampled):
                    node_seq[n, cnt] = sampled_list[i]
                    cnt += 1
                    if cnt == seq_len:
                        break
                scnt += 1
                start = node_seq[n, scnt].item()
            n += 1
        return node_seq
