from __future__ import annotations

from typing import Protocol, overload

import torch
from naive_flow.tracker import TrackerConfig

from dhgl.script_utils.configs.batch import BatchConfig

from ...data.base.base_schema import BaseGraphSchema
from ..configs.batch import BatchConfig
from ..configs.cv import CVConfig
from ..configs.evalulator import BaseEvaluatorConfig
from ..configs.scheduler import SchedulerConfig
from .base import HGNNReturnT


class DatasetConfigProtocol(Protocol):

    def load(self) -> BaseGraphSchema:
        ...


class LossFn(Protocol):
    name: str

    @overload
    def __call__(
        self, pred_data: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        ...


class HGNNConfigProtocol(Protocol):

    num_layers: int

    @overload
    def init(self, hg: BaseGraphSchema, config: TrainerConfig) -> HGNNReturnT:
        ...


class TrainerConfig(Protocol):

    ###################
    # DATASET CONFIGS #
    ###################

    dataset: DatasetConfigProtocol

    cv_config: CVConfig | None

    hgnn_config: HGNNConfigProtocol
    scheduler_config: SchedulerConfig | None
    ####################
    # TRAINING CONFIGS #
    ####################
    device: str
    epochs: int

    loss_fn: LossFn

    avoid_underfitting_threshold: float | None

    tracker_config: TrackerConfig

    batch_config: BatchConfig

    evaluator_config: BaseEvaluatorConfig

    grad_max_value: float | None
    grad_max_norm: float | None
