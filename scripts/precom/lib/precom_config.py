from __future__ import annotations

import json

import dgl
import torch
from pydantic import field_validator

import dhgl
from dhgl import hgget as H
from dhgl.script_utils import BaseConfig
from dhgl.type import CEType
from dhgl.utils.precomputation.adj import row_normalized_adjs
from dhgl.utils.precomputation.feature_collector import (
    FeatureCollector,
    LabelFeatCollector,
)
from dhgl.utils.precomputation.metagraph import MetaGraph, MPAdaptor, SelfLoop

from .precom_dataset import PrecomDataset, SlotDataset


class MetaGraphConfig(BaseConfig):
    """Used to get all metapaths that
    1. Within num_hops
    2. Not include specified edge types
    3. Ending with the target node types
    """

    num_hops: int
    exclude_edge_types: list[str] | None = None

    @field_validator('exclude_edge_types', mode='before')
    def load_list(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    def init(self, root_hg: dgl.DGLHeteroGraph):
        mg = MetaGraph.from_hg(root_hg)
        if self.exclude_edge_types:
            mg = mg.remove_etypes(self.exclude_edge_types)
        self._mg = mg
        return mg

    def get_metapaths(self, root_hg: dgl.DGLHeteroGraph):
        if not hasattr(self, '_mg'):
            self.init(root_hg)
        return self._mg.metapaths(self.num_hops, dsttype=H.tgt_ntype(root_hg))


class LabelMetaGraphConfig(MetaGraphConfig):

    def get_metapaths(self, root_hg):
        mps = super().get_metapaths(root_hg)
        mps_ = [mp for mp in mps if len(mp) > 1 and mp[0][0] == mp[-1][-1]]
        return mps_


class MetapathConfig(BaseConfig):
    """Simple interface for specifying metapaths used."""

    metapaths: list[list[str]]
    """list of list of etype"""

    @field_validator('metapaths', mode='before')
    def load_list(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    def get_metapaths(self, root_hg: dhgl.BaseHeteroGraphLike):
        adaptor = MPAdaptor.from_hg(root_hg)
        return list(map(adaptor.to_canonical_metapath, self.metapaths))


class PrecomputationConfig(BaseConfig):

    drop_unlabeled: bool = False
    """
    If True, removes unlabeled target nodes after precomputation.
    Reduces storage and speeds up loading when many nodes are unused.
    """
    verbose: int | bool | None = None

    store_dir: str
    """Dir to store the outputs of precomputation."""

    # cache_dir: str | None = None
    # """Dir to cache the outputs of precomputation. If unspecified, use a tmp dir in store_dir"""

    cache_idx_dtype: str | None = 'int32'
    """Dtype of index in saved cache. int32 is recommended, which reduces
    cache size significantly
    """

    cache_val_dtype: str | None = None
    """Dtype of value in saved cache"""

    readonly: bool | None = None
    "Default to True. Whether to disable precomputation on the fly"

    def get_precom_dataset(
        self,
        hg: dhgl.BaseHeteroGraphLike,
        feats: dict[str, torch.Tensor | dict[str, torch.Tensor]],
        required_mps,
        label_required_mps=None,
    ):
        mg = MetaGraph.from_hg(hg)

        def check(mps: list[tuple[CEType, ...]]):
            for mp in mps:
                assert isinstance(
                    mp[0], SelfLoop
                ), f'Metapaths are required to startwith {SelfLoop}'

        check(required_mps)
        if label_required_mps:
            check(label_required_mps)

        all_cetypes = set(
            sum(required_mps + (label_required_mps or []), tuple())
        )
        collector = FeatureCollector(
            mg,
            row_normalized_adjs(hg, hg.edata['weight'], etypes=all_cetypes),
            cache_dir=self.store_dir,
            cache_idx_dtype=self.cache_idx_dtype,
            cache_val_dtype=self.cache_val_dtype,
            verbose=self.verbose,
            readonly=True if self.readonly is None else self.readonly,
        )
        if isinstance(list(feats.values())[0], dict):

            feat_dataset = SlotDataset(
                collector=collector,
                required_mps=required_mps,
                feats=feats,
                drop_unlabeled=self.drop_unlabeled,
                label_mask=H.mask(hg),
            )

        else:
            feat_dataset = PrecomDataset(
                collector=collector,
                required_mps=required_mps,
                feats=feats,
                drop_unlabeled=self.drop_unlabeled,
                label_mask=H.mask(hg),
            )
        if label_required_mps is not None:
            train_label = LabelFeatCollector.get_masksed_label(
                H.label(hg), H.index(hg, 'train')
            )
            lpa_dataset = PrecomDataset(
                LabelFeatCollector.from_collector(collector),
                required_mps=label_required_mps,
                feats={H.tgt_ntype(hg): train_label},
                drop_unlabeled=self.drop_unlabeled,
                label_mask=H.mask(hg),
            )
            return {'feat': feat_dataset, 'label_feat': lpa_dataset}
        return feat_dataset

    def get_label_precom_dataset(
        self,
        hg: dhgl.BaseHeteroGraphLike,
        train_label: torch.Tensor,
        required_mps,
    ):
        mg = MetaGraph.from_hg(hg)

        def check(mps: list[tuple[CEType, ...]]):
            for mp in mps:
                assert isinstance(
                    mp[0], SelfLoop
                ), f'Metapaths are required to startwith {SelfLoop}'

        check(required_mps)

        all_cetypes = set(sum(required_mps, tuple()))
        collector = LabelFeatCollector(
            mg,
            row_normalized_adjs(hg, hg.edata['weight'], etypes=all_cetypes),
            cache_dir=self.store_dir,
            cache_idx_dtype=self.cache_idx_dtype,
            cache_val_dtype=self.cache_val_dtype,
            verbose=self.verbose,
            readonly=True if self.readonly is None else self.readonly,
        )
        # train_label = LabelFeatCollector.get_masksed_label(
        #     H.label(hg), H.index(hg, 'train')
        # )
        return PrecomDataset(
            collector,
            required_mps=required_mps,
            feats={H.tgt_ntype(hg): train_label},
            drop_unlabeled=self.drop_unlabeled,
            label_mask=H.mask(hg),
        )
