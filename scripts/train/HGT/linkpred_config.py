from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dgl

from dhgl.data.link_prediction import LinkPredDatasetLike

from .config import HGTConfig as BaseHGTConfig

if TYPE_CHECKING:
    from ..linkpred import TrainerConfig
    from ..linkpred.base import HGNNReturnT


class HGTConfig(BaseHGTConfig):

    name: Literal['HGT'] = 'HGT'

    def init(
        self, dataset: LinkPredDatasetLike, global_conf: TrainerConfig
    ) -> HGNNReturnT:

        dataset.graph, model, optimizer, scheduler = self._init(
            dataset.graph,
            n_out=global_conf.decoder_config.dim,
            global_conf=global_conf,
        )

        def forward(graph, feat):
            res = model.forward(graph, feat, dataset.target_ntypes)
            return res

        # ntype_ids = dgl.to_homogeneous(dataset.graph).ndata[dgl.NTYPE]

        # target_ntypes = set(
        #     sum(([s, d] for s, _, d in dataset.target_etypes), [])
        # )
        # masks = {
        #     ntype: (ntype_ids == dataset.graph.get_ntype_id(ntype)).to(
        #         global_conf.device
        #     )
        #     for ntype in target_ntypes
        # }

        # def forward(graph, feat):
        #     out = forward_fn(graph, feat)
        #     return {ntype: out[mask] for ntype, mask in masks.items()}

        return {
            'dataset': dataset,
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'forward_fn': forward,
        }
