from __future__ import annotations

from typing import Annotated, ClassVar, Literal

import dgl
import torch
from pydantic import Field

from dhgl import transforms

from ....data.link_prediction import LastFMDataset, LinkPredDatasetLike, PubMedDataset
from ....data.link_prediction.amazon import AmazonDataset, NormalizedAmazonDataset
from ....data.link_prediction.pubmed import NormalizedPubMedDataset
from .hetero_dgl_dataset import BaseDatasetConfig


def fix_validation_split(dataset: LinkPredDatasetLike):
    """Fix the problem that HGB split validation with edge types other than target edge type.
    This merges all edges except for the target edges in val_graph into training graph
    """

    data_dict = {
        etype: dataset.val_graph.edges(etype=etype)
        for etype in dataset.val_graph.canonical_etypes
    }
    data_dict_recp = {}
    for etype in dataset.target_etypes:
        inv_etype = dataset.get_inverse_etype(etype)
        data_dict_recp[etype] = data_dict[etype]
        data_dict[etype] = ([], [])
        data_dict[inv_etype] = ([], [])
    num_nodes_dict = {
        ntype: dataset.graph.num_nodes(ntype)
        for ntype in dataset.graph.ntypes
    }
    vanilla_hg_fixed = dgl.merge(
        [
            dataset.vanilla_graph,
            transforms.update_graph_structure(
                dataset.val_graph, data_dict, num_nodes_dict, copy_ndata=False,
                copy_edata=False
            )
        ]
    )
    for etype in dataset.graph.canonical_etypes:
        if etype not in data_dict:
            # Add edge type for normalized datasets
            data_dict[etype] = ([], [])
    val_graph = transforms.update_graph_structure(
        dataset.val_graph, data_dict, num_nodes_dict, copy_ndata=False,
        copy_edata=False
    )
    if dataset.graph.edata['weight']:
        for etype in dataset.graph.edata['weight']:
            val_graph.edges[etype].data['weight'] = torch.zeros((0, ))
    hg_ = dgl.merge([dataset.graph, val_graph])
    val_graph = transforms.update_graph_structure(
        dataset.val_graph, data_dict_recp, num_nodes_dict, copy_ndata=False,
        copy_edata=False
    )
    dataset._vanilla_graph = vanilla_hg_fixed
    dataset.graph = hg_
    dataset.val_graph = val_graph
    return dataset


class BaseLinkPredDatasetConfig(BaseDatasetConfig):
    pass


class LastFMDatasetConfig(BaseLinkPredDatasetConfig):

    name: Literal['lastfm'] = 'lastfm'
    raw_path: str
    use_symmetric_user_user: bool = False
    """If true edge-type user-user-inv will be merged into user-user"""
    _ntypes: ClassVar[set[str]] = set(LastFMDataset.ntypes)
    fix_valid: bool | None = None

    @property
    def cache_dir(self):
        return self.save_dir if self.save_dir is not None else self.raw_path

    def load(self, verbose: bool | None = None, **kwargs):
        dataset = LastFMDataset(self.raw_path, save_dir=self.save_dir)
        if self.fix_valid:
            dataset = fix_validation_split(dataset)
        if self.use_symmetric_user_user:
            dataset.graph = transforms.merge_etypes(
                dataset.graph,
                'user-user',
                etype_to_drop='user-user-inv',
            )
            if 'user-user' in dataset.val_graph.canonical_etypes:
                dataset.val_graph = transforms.merge_etypes(
                    dataset.graph,
                    'user-user',
                    etype_to_drop='user-user-inv',
                )
            dataset.canonical_etypes = [
                e for e in dataset.canonical_etypes if e[1] != 'user-user-inv'
            ]
            if self.exclude_edge_types:
                CUSERUSERINV = dataset.vanilla_graph.to_canonical_etype(
                    'user-user-inv'
                )
                if CUSERUSERINV in self.exclude_edge_types:
                    self.exclude_edge_types.pop(
                        self.exclude_edge_types.index(CUSERUSERINV)
                    )
                if 'user-user-inv' in self.exclude_edge_types:
                    self.exclude_edge_types.pop(
                        self.exclude_edge_types.index('user-user-inv')
                    )

        dataset.graph = self._process_feats(
            dataset.graph, verbose=verbose, **kwargs
        )
        return dataset


class AmazonDatasetConfig(BaseLinkPredDatasetConfig):

    name: Literal['amazon'] = 'amazon'
    raw_path: str
    use_symmetric_etypes: bool = False
    """If true edge-type co-purchase-inv, co-view-inv will be merged"""
    _ntypes: ClassVar[set[str]] = set(AmazonDataset.ntypes)

    @property
    def cache_dir(self):
        return self.save_dir if self.save_dir is not None else self.raw_path

    @classmethod
    def _merge(cls, hg):
        hg = transforms.merge_etypes(
            hg,
            'co-purchase',
            etype_to_drop='co-purchase-inv',
        )
        hg = transforms.merge_etypes(
            hg,
            'co-view',
            etype_to_drop='co-view-inv',
        )
        return hg

    def load(self, verbose: bool | None = None, **kwargs):
        dataset = AmazonDataset(self.raw_path, save_dir=self.save_dir)
        if self.use_symmetric_etypes:
            dataset.graph = self._merge(dataset.graph)
            dataset.val_graph = self._merge(dataset.val_graph)
            dataset.canonical_etypes = [
                e for e in dataset.canonical_etypes
                if e[1] not in ('co-view-inv', 'co-purchase-inv')
            ]

        dataset.graph = self._process_feats(
            dataset.graph, verbose=verbose, **kwargs
        )
        return dataset


class NAmazonDatasetConfig(BaseLinkPredDatasetConfig):

    name: Literal['namazon', 'atomic-amazon'] = 'atomic-amazon'
    raw_path: str
    use_symmetric_etypes: bool = False
    """If true edge-type co-purchase-inv, co-view-inv will be merged"""
    _ntypes: ClassVar[set[str]] = set(NormalizedAmazonDataset.ntypes)

    @property
    def cache_dir(self):
        return self.save_dir if self.save_dir is not None else self.raw_path

    def load(self, verbose: bool | None = None, **kwargs):
        dataset = NormalizedAmazonDataset(
            self.raw_path, save_dir=self.save_dir
        )
        if self.use_symmetric_etypes:
            dataset.graph = AmazonDatasetConfig._merge(dataset.graph)
            dataset.val_graph = AmazonDatasetConfig._merge(dataset.val_graph)
            dataset.canonical_etypes = [
                e for e in dataset.canonical_etypes
                if e[1] not in ('co-view-inv', 'co-purchase-inv')
            ]
        dataset.graph = self._process_feats(
            dataset.graph, verbose=verbose, **kwargs
        )
        return dataset


class PubMedDatasetConfig(BaseLinkPredDatasetConfig):

    name: Literal['pubmed'] = 'pubmed'
    raw_path: str
    use_symmetric_etypes: bool = False
    _ntypes: ClassVar[set[str]] = set(PubMedDataset.ntypes)
    fix_valid: bool | None = None

    @classmethod
    def _merge(cls, hg):
        if 'species-species-inv' in hg.etypes:
            hg = transforms.merge_etypes(
                hg,
                'species-species',
                etype_to_drop='species-species-inv',
            )
        if 'chemical-chemical-inv' in hg.etypes:
            hg = transforms.merge_etypes(
                hg,
                'chemical-chemical',
                etype_to_drop='chemical-chemical-inv',
            )
        if 'gene-gene-inv' in hg.etypes:
            hg = transforms.merge_etypes(
                hg,
                'gene-gene',
                etype_to_drop='gene-gene-inv',
            )
        if 'disease-disease-inv' in hg.etypes:
            hg = transforms.merge_etypes(
                hg,
                'disease-disease',
                etype_to_drop='disease-disease-inv',
            )
        return hg

    @property
    def cache_dir(self):
        return self.save_dir if self.save_dir is not None else self.raw_path

    def load(self, verbose: bool | None = None, **kwargs):
        dataset = PubMedDataset(self.raw_path, save_dir=self.save_dir)
        if self.fix_valid:
            dataset = fix_validation_split(dataset)
        if self.use_symmetric_etypes:
            dataset.graph = self._merge(dataset.graph)
            dataset.val_graph = self._merge(dataset.val_graph)

            dataset.canonical_etypes = [
                e for e in dataset.canonical_etypes if e[1] not in (
                    'species-species-inv', 'disease-disease-inv',
                    'chemical-chemical-inv', 'gene-gene-inv'
                )
            ]

        dataset.graph = self._process_feats(
            dataset.graph, verbose=verbose, **kwargs
        )
        return dataset


class NPubMedDatasetConfig(BaseLinkPredDatasetConfig):

    name: Literal['npubmed', 'atomic-pubmed'] = 'atomic-pubmed'
    raw_path: str
    _ntypes: ClassVar[set[str]] = set(NormalizedPubMedDataset.ntypes)
    fix_valid: bool | None = None

    @property
    def cache_dir(self):
        return self.save_dir if self.save_dir is not None else self.raw_path

    def load(self, verbose: bool | None = None, **kwargs):
        dataset = NormalizedPubMedDataset(
            self.raw_path, save_dir=self.save_dir
        )
        if self.fix_valid:
            dataset = fix_validation_split(dataset)
        dataset.graph = self._process_feats(
            dataset.graph, verbose=verbose, **kwargs
        )
        return dataset


HGBLinkPredDatasetConfig = Annotated[
    LastFMDatasetConfig | AmazonDatasetConfig
    | NAmazonDatasetConfig
    | PubMedDatasetConfig
    | NPubMedDatasetConfig,
    Field(discriminator='name'),
]
