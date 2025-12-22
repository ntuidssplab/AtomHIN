from __future__ import annotations

from typing import TYPE_CHECKING, Literal

# from dhgl import transforms
from dhgl.data.link_prediction import LinkPredDatasetLike

from .config import REGCNConfig as BaseREGCNConfig

if TYPE_CHECKING:
    from ..linkpred import TrainerConfig
    from ..linkpred.base import HGNNReturnT


class REGCNConfig(BaseREGCNConfig):

    name: Literal['REGCN'] = 'REGCN'

    #######################
    # MODEL CONFIGS   #
    #######################

    def init(
        self, dataset: LinkPredDatasetLike, global_conf: TrainerConfig
    ) -> HGNNReturnT:

        if len(dataset.target_ntypes) > 1:
            raise NotImplementedError
        else:
            target_ntype = list(dataset.target_ntypes)[0]
        dataset.graph, model, optimizer, scheduler, forward = self._init(
            dataset.graph, target_ntype, global_conf.decoder_config.dim,
            global_conf
        )

        def forward_fn(*args, **kwargs):
            return {target_ntype: forward(*args, **kwargs)}

        return {
            'dataset': dataset,
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'forward_fn': forward_fn,
        }
