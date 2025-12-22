from typing import Literal
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from naive_flow.tracker import TrackerConfig
from ..configs.dataset import ACMConfig, DBLPConfig, IMDBConfig
from ..configs.loss import CrossEntropy, BCEWithLogits, SoftLabelCE


class XGBConfig(BaseSettings):

    n_estimators: int = 125
    gamma: float = 0.2
    min_child_weight: float = 1.
    colsample_bytree: float = 0.2
    max_depth: int = 2
    alpha: float = 0.2
    comment: str = '_xgb'


class TreeGNNConfig(BaseSettings):

    model_config = SettingsConfigDict(protected_namespaces=('settings_', ))
    model_gnn: Literal['mygat'] = 'mygat'
    leaf_fusion: bool = True
    pred_fusion: bool = True
    raw_fusion: bool = True
    edge_dim: int = 64
    """edge embedding dim"""
    num_hidden: int = 64
    n_estimators: int = XGBConfig.model_fields['n_estimators'].default
    num_layers: int = 3
    feat_drop: float = 0.5
    attn_drop: float = 0.5
    negative_slope: float = 0.05
    dim: int = 1  # Channel size of the embedding layer
    head_tree: int = 1
    fusion_type: Literal[0, 1, 2, 3, 4, 5] = 1
    beta: float = 0.5

    @field_validator('fusion_type', mode='before')
    @classmethod
    def handle_fusion_type(cls, v: str):
        if isinstance(v, str):
            return int(v)
        return v


class TreeConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_nested_delimiter='__', extra='forbid', frozen=True
    )

    name: Literal['Tree'] = 'Tree'

    ###################
    # DATASET CONFIGS #
    ###################

    dataset: ACMConfig | DBLPConfig | IMDBConfig = Field(discriminator='name')

    #######################
    #  MODEL CONFIGS      #
    #######################
    gnn_config: TreeGNNConfig

    ####################
    # TRAINING CONFIGS #
    ####################
    device: str = 'cuda:0'
    epochs: int = 500

    alpha: float = Field(0.5, ge=0., le=1.)
    """Loss ratio in [0, 1]"""

    lr: float = 5e-4
    weight_decay: float = 1e-4
    max_lr_scale: float
    pct_start_epoch: int

    loss_fn: CrossEntropy | BCEWithLogits | SoftLabelCE = Field(
        discriminator='name'
    )

    xgb_config: XGBConfig

    class DefaultTrackerConfig(TrackerConfig):
        early_stopping_rounds: int = 30
        comment: str = '.Simple'
        log_root_dir: str = 'runs'
        epochs_per_checkpoint: int = 0
        save_n_best: int = 0
        save_end: bool = False

    tracker_config: DefaultTrackerConfig
