from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from dhgl import transforms
from dhgl.data.link_prediction import LinkPredDatasetLike

from .config import RGCNConfig as BaseRGCNConfig

if TYPE_CHECKING:
    from ..linkpred import TrainerConfig
    from ..linkpred.base import HGNNReturnT


class RGCNConfig(BaseRGCNConfig):

    name: Literal['RGCN'] = 'RGCN'

    #######################
    # MODEL CONFIGS   #
    #######################

    def init(
        self, dataset: LinkPredDatasetLike, global_conf: TrainerConfig
    ) -> HGNNReturnT:

        dataset.graph, model, optimizer, scheduler, forward = self._init(
            dataset.graph, dataset.target_ntypes,
            global_conf.decoder_config.dim, global_conf
        )
        return {
            'dataset': dataset,
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'forward_fn': forward,
        }
