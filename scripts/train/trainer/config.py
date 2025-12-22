from __future__ import annotations

from typing import Union

from lazy_imports import try_import
from pydantic import Field, field_validator

from dhgl.script_utils import BaseConfig, filter_env_private_fields, trainer
from dhgl.script_utils.configs.batch import BatchConfig
from dhgl.script_utils.configs.cv import CVConfig
from dhgl.script_utils.configs.dataset.hetero_dgl_dataset import HeteroDatasetConfig
from dhgl.script_utils.configs.evalulator import (
    HGBMultiLabelNodeClassificationEvaluatorConfig,
    HGBNodeClassificationEvaluatorConfig,
    OGBEvaluator,
)
from dhgl.script_utils.configs.loss import BCEWithLogits, CrossEntropy, SoftLabelCE
from dhgl.script_utils.configs.scheduler import SchedulerConfig
from naive_flow.tracker import TrackerConfig

from ..GAT.config import GATConfig
from ..GTN import GTNConfig
from ..HAN import HANConfig
from ..HGT import HGTConfig
from ..HINormer import HINormerConfig
from ..MLP import MLPConfig
from ..REGCN import REGCNConfig
from ..RGCN import RGCNConfig
from ..Simple import SimpleConfig
from ..Tree import TreeConfig
from ..VanillaRGCN import VanillaRGCNConfig

hgnn_types = (
    HGTConfig, SimpleConfig, TreeConfig, GATConfig, HANConfig, MLPConfig,
    RGCNConfig, HINormerConfig, VanillaRGCNConfig, REGCNConfig
)

lazy_hgnn_configs = {}
with try_import() as lazy_hgnn_configs['GTN']:
    from ..GTN import GTNConfig
    if lazy_hgnn_configs['GTN'].is_successful():
        hgnn_types += (GTNConfig, )
with try_import() as lazy_hgnn_configs['XGBoost']:
    from ..XGBoost import XGBoostConfig
    if lazy_hgnn_configs['XGBoost'].is_successful():
        hgnn_types += (XGBoostConfig, )
with try_import() as lazy_hgnn_configs['PSHGCN']:
    from ..PSHGCN import PSHGCNConfig
    if lazy_hgnn_configs['PSHGCN'].is_successful():
        hgnn_types += (PSHGCNConfig, )
with try_import() as lazy_hgnn_configs['NodeFormer']:
    from ..NodeFormer import NodeFormerConfig
    if lazy_hgnn_configs['NodeFormer'].is_successful():
        hgnn_types += (NodeFormerConfig, )
HGNNConfigT = Union[hgnn_types]


@filter_env_private_fields
class TrainerConfig(BaseConfig):

    ###################
    # DATASET CONFIGS #
    ###################

    dataset_config: HeteroDatasetConfig = Field(discriminator='name')

    hgnn_config: HGNNConfigT = Field( # pyright: ignore[reportInvalidTypeForm]
        discriminator='name'
    )
    scheduler_config: SchedulerConfig | None = None
    ####################
    # TRAINING CONFIGS #
    ####################
    device: str = 'cuda'
    epochs: int

    cv_config: CVConfig | None = None
    grad_max_norm: float | None = None
    grad_max_value: float | None = None

    loss_fn: CrossEntropy | BCEWithLogits | SoftLabelCE = Field(
        discriminator='name'
    )
    avoid_underfitting_threshold: float | None = None

    tracker_config: TrackerConfig

    batch_config: BatchConfig

    evaluator_config: HGBNodeClassificationEvaluatorConfig | HGBMultiLabelNodeClassificationEvaluatorConfig | OGBEvaluator

    @field_validator('hgnn_config', mode='before')
    @classmethod
    def check_hgnn_config(cls, v):
        if 'name' in v and v['name'] in lazy_hgnn_configs:
            lazy_hgnn_configs[v['name']].check()
        return v

    def run(self):
        return trainer.train(self)
