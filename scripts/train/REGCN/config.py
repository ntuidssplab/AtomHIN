from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dgl
import torch
from torch.nn import functional as F
from torch.optim import Adam

from dhgl import hgget as H
from dhgl import transforms
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.script_utils import BaseConfig
from dhgl.script_utils.trainer.base import HGNNReturnT

from .model import REGCN

if TYPE_CHECKING:
    from ..trainer.config import TrainerConfig


class REGCNConfig(BaseConfig):

    name: Literal['REGCN'] = 'REGCN'

    #######################
    # MODEL CONFIGS   #
    #######################
    hidden_dim: int

    num_layers: int
    """Number of Layer"""

    # num_out_layers: int
    # """Number of layers of output MLP"""

    num_heads: int
    """Number of attention heads"""
    activation: Literal['elu', 'identity'] = 'elu'

    lr: float
    weight_decay: float

    # max_lr_scale: float | None = None
    # pct_start_epoch: int | None = None

    dropout: float

    relation_scalar: float

    add_self_loop: bool = True

    # proj_by: Literal['ntype', 'shared'] = 'ntype'

    def init(
        self, hg: BaseHeteroGraphLike, global_conf: TrainerConfig
    ) -> HGNNReturnT:

        hg, model, optimizer, scheduler, forward = self._init(
            hg, H.tgt_ntype(hg), H.n_classes(hg), global_conf
        )
        data = {
            'hg': hg,
            'model': model,
            'optimizer': optimizer,
            'forward_fn': forward
        }
        if scheduler is not None:
            data['scheduler'] = scheduler
        return data

    def _init(
        self,
        hg: BaseHeteroGraphLike,
        target_ntype: str | list[str],
        n_out: int,
        global_conf: TrainerConfig,
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
        hg_ = transforms.update_graph_structure(
            hg,
            data_dict,
            num_nodes_dict=num_node_dict,
        )
        hg = hg_

        # for stype, etype, dtype in hg.canonical_etypes:
        #     # if 'weight' not in hg.edges[etype].data:
        #     #     continue
        #     with hg.local_scope():
        #         hg.nodes[stype].data['x'] = torch.ones(hg.num_nodes(stype))

        #         if 'weight' not in hg.edges[etype].data:
        #             hg.update_all(
        #                 dgl.function.copy_u('x', 'm'),
        #                 reduce_func=dgl.function.sum('m', 'h'), etype=etype
        #             )
        #         else:
        #             hg.update_all(
        #                 dgl.function.u_mul_e('x', 'weight', 'm'),
        #                 reduce_func=dgl.function.sum('m', 'h'), etype=etype
        #             )

        #         breakpoint()

        if self.add_self_loop:
            hg = transforms.add_self_loop(hg)

        assert not (
            global_conf.batch_config.train.is_in_batch_mode
            and global_conf.batch_config.eval.is_in_batch_mode
        )
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
        tgt_mask = g.ndata[dgl.NTYPE] == hg.get_ntype_id(target_ntype)
        e_feat = g.edata[dgl.ETYPE]

        model = REGCN(
            g,
            num_etypes=len(hg.etypes),
            R=self.relation_scalar,
            in_feats=self.hidden_dim,
            n_hidden=self.hidden_dim,
            n_classes=n_out,
            n_layers=self.num_layers,
            activation=F.elu if self.activation == 'elu' else lambda _: _,
            dropout=self.dropout,
            feats_dim_list=[
                data.shape[-1] for data in hg.ndata['feat'].values()
            ],
        )
        if global_conf.tracker_config.verbose:
            print(
                '#parameters=',
                sum(
                    torch.prod(torch.tensor(p.size())) for p in
                    filter(lambda p: p.requires_grad, model.parameters())
                )
            )
            num_params = [
                (n, torch.prod(torch.tensor(p.size())).item())
                for n, p in model.named_parameters() if p.requires_grad
            ]
            from collections import Counter
            counter = Counter()
            for name, n_parmas in num_params:
                prefix, _ = name.split('.', maxsplit=1)
                counter[prefix] += n_parmas
            print('#parameters=', counter)
        optimizer = Adam(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = None

        def forward(graph, feat):
            out, _ = model.forward(
                feat.values(), e_feat, e_weight=g.edata.get('weight', None)
            )
            return out[tgt_mask]

        return hg, model, optimizer, scheduler, forward
