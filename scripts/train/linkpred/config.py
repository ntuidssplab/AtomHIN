from __future__ import annotations

from typing import Literal, Union

import dgl
from lazy_imports import try_import
from pydantic import Field, model_validator

import dhgl
from dhgl.data.link_prediction import LinkPredDatasetLike
from dhgl.script_utils import BaseConfig, filter_env_private_fields
from dhgl.script_utils.configs.cv import CVConfig
from dhgl.script_utils.configs.dataset.hgb_link_pred_dataset import (
    HGBLinkPredDatasetConfig,
)
from dhgl.script_utils.configs.evalulator import HGBLinkPredEvaluator
from dhgl.script_utils.configs.loss import BCEWithLogits, CrossEntropy, SoftLabelCE
from dhgl.script_utils.configs.misc import EarlyBreakingConfig
from dhgl.script_utils.configs.scheduler import SchedulerConfig
from dhgl.type import Split
from naive_flow.tracker import TrackerConfig

from ..HGT.linkpred_config import HGTConfig
from ..HINormer.linkpred_config import HINormerConfig
from ..REGCN.linkpred_config import REGCNConfig
from ..Simple.linkpred_config import SimpleConfig
from ..SlotGAT.linkpred_config import SlotGATConfig
from ..VanillaRGCN.linkpred_config import VanillaRGCNConfig
from .decoder import DecoderConfig

hgnn_types = (
    SlotGATConfig, SimpleConfig, HGTConfig, VanillaRGCNConfig, HINormerConfig,
    REGCNConfig
)
lazy_hgnn_configs = {}
with try_import() as lazy_hgnn_configs['PSHGCN']:
    from ..PSHGCN.linkpred_config import PSHGCNConfig
    if lazy_hgnn_configs['PSHGCN'].is_successful():
        hgnn_types += (PSHGCNConfig, )
with try_import() as lazy_hgnn_configs['RGCN']:
    from ..RGCN.linkpred_config import RGCNConfig
    if lazy_hgnn_configs['RGCN'].is_successful():
        hgnn_types += (RGCNConfig, )
with try_import() as lazy_hgnn_configs['SeHGNN']:
    from ..SeHGNN.linkpred_config import SeHGNNConfig
    if lazy_hgnn_configs['SeHGNN'].is_successful():
        hgnn_types += (SeHGNNConfig, )
HGNNConfigT = Union[hgnn_types]


class BaseBatchConfig(BaseConfig):
    name: Literal['whole_graph']

    @property
    def is_in_batch_mode(self):
        return False


class BatchConfig(BaseConfig):

    train: BaseBatchConfig
    eval: BaseBatchConfig

    def __getitem__(self, split: Split) -> BaseBatchConfig:
        if split == 'train':
            return self.train
        return self.eval


class NHopNegativeSamplerConfig(BaseConfig):
    name: Literal['2hop']
    cache_dir: str | None = None
    """caching for storing precompuated 2hop graph"""
    verbose: bool | None = None

    @model_validator(mode='before')
    @classmethod
    def _load(cls, data):
        if isinstance(data, str):
            return {'name': data}
        return data

    def get_sampler(self, dataset: LinkPredDatasetLike, split: Split):
        from .sampler import NHopNegativeSampler
        device = dataset.graph.device

        def add_inverse_edges(g: dgl.DGLHeteroGraph):
            data_dict = {
                etype: g.edges(etype=etype)
                for etype in g.canonical_etypes
            }
            for etype in g.canonical_etypes:
                inv_etype = dataset.get_inverse_etype(etype)
                if inv_etype not in data_dict:
                    data_dict[inv_etype] = data_dict[etype][::-1]

            g_ = dhgl.transforms.update_graph_structure(g, data_dict)
            return g_

        hg = dataset.vanilla_graph.cpu()
        hgs = {
            'train': [hg],
            'val': [hg, add_inverse_edges(dataset.val_graph.cpu())],
            'test': [
                hg,
                add_inverse_edges(dataset.val_graph.cpu()),
                add_inverse_edges(dataset.test_graph.cpu())
            ],
        }[split]
        return NHopNegativeSampler(
            hgs,
            dataset.target_etypes,
            n_hops=2,
            k=1,
            cache_dir=self.cache_dir,
            verbose=self.verbose,
        ).to(device)


class NegativeSamplerConfig(BaseConfig):
    """
        - 'uniform': Sample negatives uniformly per source node.
        - '2hop': Sample from 2-hop neighbors excluding positive edges.
        - 'static': Use the negative graph provided in the dataset (if any).
    """
    train: Literal['uniform'] | NHopNegativeSamplerConfig
    val: Literal['uniform', 'static'] | NHopNegativeSamplerConfig
    test: Literal['uniform', 'static'] | NHopNegativeSamplerConfig = Field(
        'static', exclude=True
    )

    def __getitem__(self, split: Split):
        return getattr(self, split)

    def get_sampler(self, dataset: LinkPredDatasetLike, split: Split):
        if self[split] == 'static':
            return getattr(dataset, f'neg_{split}_graph')
        if self[split] == 'uniform':
            import dgl
            return dgl.dataloading.negative_sampler.PerSourceUniform(1)
        assert isinstance(self[split], NHopNegativeSamplerConfig)
        return self[split].get_sampler(dataset, split)


@filter_env_private_fields
class TrainerConfig(BaseConfig):

    ###################
    # DATASET CONFIGS #
    ###################

    dataset_config: HGBLinkPredDatasetConfig

    hgnn_config: HGNNConfigT = Field(# pyright: ignore[reportInvalidTypeForm]
        discriminator='name'
    )
    scheduler_config: SchedulerConfig | None = None
    decoder_config: DecoderConfig
    negative_sampler_config: NegativeSamplerConfig
    ####################
    # TRAINING CONFIGS #
    ####################
    device: str = 'cuda'
    epochs: int

    grad_max_norm: float | None = None
    grad_max_value: float | None = None

    loss_fn: CrossEntropy | BCEWithLogits | SoftLabelCE = Field(
        discriminator='name'
    )
    avoid_underfitting_threshold: float | None = None

    early_breaking: EarlyBreakingConfig | None = None

    cv_config: CVConfig | None = None
    tracker_config: TrackerConfig

    batch_config: BatchConfig

    decoder_config: DecoderConfig
    evaluator_config: HGBLinkPredEvaluator

    def run(self):
        from .trainer import train
        return train(self)
