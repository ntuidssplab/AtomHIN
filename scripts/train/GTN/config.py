from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dgl
import torch
from torch.nn import functional as F
from torch.optim import AdamW

import dhgl
from dhgl import hgget as H
from dhgl import transforms
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.script_utils import BaseConfig
from dhgl.script_utils.trainer.base import HGNNReturnT

from .model import fastGTN

if TYPE_CHECKING:
    from ..trainer.config import TrainerConfig


class GTNConfig(BaseConfig):

    name: Literal['GTN'] = 'GTN'

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
    # activation: Literal['elu', 'identity'] = 'elu'

    lr: float
    weight_decay: float

    # max_lr_scale: float | None = None
    # pct_start_epoch: int | None = None

    # dropout: float
    norm: bool = False
    identity: bool = False

    add_self_loop: bool = True

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
        g = dgl.to_homogeneous(hg, edata=edata)
        # XXX: a little more memory consumption may introduce here.
        # the hg will be put to cuda somewhere else.
        # Things occupying cuda: hg.adj + hg.ndata + hg.edata + g.adj
        # the hg.adj is not used (but normally it only uses small amount of cuda memory.)
        tgt_mask = (g.ndata[dgl.NTYPE] == hg.get_ntype_id(target_ntype)).to(
            global_conf.device
        )

        # model = GTN(
        #     num_edge=len(hg.etypes),
        #     num_channels=self.num_heads,
        #     in_shapes={
        #         ntype: feat.shape
        #         for ntype, feat in hg.ndata[dhgl.FEAT].items()
        #     },
        #     # w_in=node_features.shape[1],
        #     w_out=self.hidden_dim,
        #     num_class=n_out,
        #     num_layers=self.num_layers,
        #     num_nodes=g.num_nodes(),
        # )
        dims = [feat.shape[-1] for feat in hg.ndata[dhgl.FEAT].values()]
        assert max(dims) == dims[0]
        model = fastGTN(
            num_edge_type=len(hg.etypes),
            num_channels=self.num_heads,
            # in_shapes={
            #     ntype: feat.shape
            #     for ntype, feat in hg.ndata[dhgl.FEAT].items()
            # },
            # w_in=node_features.shape[1],
            in_dim=dims[0],
            hidden_dim=self.hidden_dim,
            num_class=n_out,
            num_layers=self.num_layers,
            category=target_ntype,
            norm=self.norm,
            identity=self.identity,
            # dropout=self.dropout,
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
        optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = None

        # adjs = adj_utils.adjs_to_homogeneous(
        #     hg, adj_utils.row_normalized_adjs(hg, hg.edata[dhgl.EWEIGHT])
        # )

        # adjs = {
        #     etype: adj.coalesce().to(global_conf.device)
        #     for etype, adj in adjs.items()
        # }
        # A = [(adj.indices(), adj.values()) for adj in adjs.values()]

        def forward(graph, feat):
            out = model.forward(graph, feat)
            return out[target_ntype]

        return hg, model, optimizer, scheduler, forward
