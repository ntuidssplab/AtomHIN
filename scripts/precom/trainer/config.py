from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from typing import Literal

from pydantic import Field, field_validator, model_validator

from dhgl.script_utils import BaseConfig, filter_env_private_fields
from dhgl.script_utils.configs.evalulator import (
    HGBNodeClassificationEvaluatorConfig,
    OGBEvaluator,
)
from dhgl.script_utils.configs.loss import BCEWithLogits, CrossEntropy, SoftLabelCE
from dhgl.script_utils.configs.misc import EarlyBreakingConfig
from dhgl.type import Split
from naive_flow.tracker import TrackerConfig

from ..dataset import HeteroDatasetConfig
from ..models.LMSPS import LMSPSConfig
from ..models.PSGHCN import PSHGCNConfig
from ..models.SRGCN import SRGCNConfig


class BaseBatchConfig(BaseConfig):

    batch_size: int
    num_workers: int
    pin_memory: bool = False
    persistent_workers: bool = False


class TrainBatchConfig(BaseBatchConfig):
    chunk_size: int
    """
    TL;DR: Match this with `data_loading_config__chunk_size`.

    Description:
    Uses a random chunk loader instead of random shuffling for better speed.
    Workflow:
        1. Loads (batch_size // chunk_size) chunks
        2. Concatenates & shuffles them

    Rule: Should NOT be smaller than `data_loading_config__chunk_size`.
    Larger chunk_size may slightly improve speed.
    """


class BatchConfig(BaseConfig):

    train: TrainBatchConfig
    eval: BaseBatchConfig

    def __getitem__(self, split: Split) -> BaseBatchConfig | TrainBatchConfig:
        if split == 'train':
            return self.train
        return self.eval


class TransposingConfig(BaseConfig):

    name: Literal['disk'] = 'disk'

    # loader_axis: Literal['batch', 'channel'] = 'batch'
    chunk_size: int | None = None
    """
    TL;DR: Set as large as possible <= batch_size.

    Used when storing transposed cache chunks. See zarr for more details.
    """
    batch_size: int | None = None
    """TL;DR: Set as large as possible without causing OOM.

    Defines the shard size in the cache storage. See zarr for more details.
    """

    # dry_loading: int
    # """
    # TL;DR: Use 2 if dataset fits in memory, otherwise 0.

    # Dry loads the dataset for a few rounds before training, helping to keep it in memory for faster first epoch.
    # """

    persistent_cache: bool = False
    """If true, the cache will not be clean up."""
    cache_dir: str | None = None
    cache_idx_dtype: str | None = 'int32'
    cache_val_dtype: str | None = None
    memory_ratio: float | None = None
    """Further move transposed cache to memory. This is used when only transposing cache available,
    useful to the case that disk space is limited.
    Also, the dataloading can be faster using higher memory_ratio if available.
    """

    verbose: bool = True

    def mk_cache_dir(self):
        if self.persistent_cache:
            assert self.cache_dir is not None
            os.makedirs(self.cache_dir, exist_ok=True)
            return self.cache_dir
        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            if hasattr(self, '_temp_dir'):
                return self._temp_dir.name  # pylint: disable=access-member-before-definition
            self._temp_dir = TemporaryDirectory(dir=self.cache_dir)
            return self._temp_dir.name
        return None

    @model_validator(mode='after')
    @property
    def check(self):
        if self.cache_dir is not None:
            assert self.batch_size is not None
            assert self.chunk_size is not None
        return self

    def __del__(self):
        if hasattr(self, '_temp_dir'):
            self._temp_dir.cleanup()
        return

    @property
    def device(self):
        """For alignment"""
        return 'cpu'


class MemoryStorageConfig(BaseConfig):
    """Store the whole dataset in GPU or CPU(main memory)."""
    name: Literal['memory'] = 'memory'
    device: Literal['cpu', 'cuda'] | str
    feat_fmt: Literal['strided', 'sparse_csr'] | None = None
    """Define the format of features that stored in GPU or CPU.
    If None, sparse features will be converted to sparse_csr, dense will be kept dense.
    """
    verbose: bool = True

    @model_validator(mode='before')
    @classmethod
    def parse(cls, data: dict):
        if data['device'] != 'cpu':
            assert data['device'].startswith('cuda')
        return data


class DataLoadingConfig(BaseConfig):

    train: TransposingConfig | MemoryStorageConfig = Field(
        discriminator='name'
    )
    eval: TransposingConfig | MemoryStorageConfig = Field(discriminator='name')

    @model_validator(mode='before')
    @classmethod
    def load_shared_settings(cls, data):
        if isinstance(data, dict) and 'name' in data:
            assert 'train' not in data
            assert 'eval' not in data
            return {'train': data, 'eval': data}
        return data

    def __getitem__(self, split: Split):
        if split == 'train':
            return self.train
        return self.eval


class MultiStageConfig(BaseConfig):

    last_logits_path: str | None = None
    num_stages: int
    threshold: float
    gamma: float
    verbose: bool | None = None
    update_label_feats: bool | float | None = None
    """If float, threshold to filter out labels with confident lower than the threshold"""
    update_configs: dict[str, list] | None = None
    early_breaking: EarlyBreakingConfig | None = None

    @field_validator('update_configs', mode='before')
    @classmethod
    def parse_list(cls, v: str):
        if isinstance(v, str):
            import json
            return json.loads(v)
        return v

    @model_validator(mode='after')
    def check_update_configs(self):
        if self.update_configs is None:
            return self
        for key, vals in self.update_configs.items():
            expected_len = self.num_stages - 1 - int(
                self.last_logits_path is not None
            )
            if len(vals) != max(expected_len, 0):
                raise ValueError(
                    f'number of update configs should match (num_stages-1)={self.num_stages - 1}, '
                    f'but got {len(vals)} for key {key}'
                )
        return self


@filter_env_private_fields
class TrainerConfig(BaseConfig):

    ###################
    # DATASET CONFIGS #
    ###################

    dataset_config: HeteroDatasetConfig

    data_loading_config: DataLoadingConfig

    hgnn_config: SRGCNConfig | PSHGCNConfig | LMSPSConfig = Field(
        discriminator='name'
    )
    ####################
    # TRAINING CONFIGS #
    ####################
    amp: Literal['float16'] | None = None
    device: str = 'cuda'
    epochs: int

    early_breaking: EarlyBreakingConfig | None = None

    grad_max_norm: float | None = None
    grad_max_value: float | None = None

    loss_fn: CrossEntropy | BCEWithLogits | SoftLabelCE = Field(
        discriminator='name'
    )
    avoid_underfitting_threshold: float | None = None

    tracker_config: TrackerConfig

    batch_config: BatchConfig

    evaluator_config: HGBNodeClassificationEvaluatorConfig | OGBEvaluator

    multi_stage_config: MultiStageConfig | None = None

    def run(self):
        from .multi_stage import train_multi_stage
        from .trainer import train
        if self.multi_stage_config is None:
            return train(self)
        return train_multi_stage(self)
