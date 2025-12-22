from __future__ import annotations

from functools import cached_property
from typing import Annotated, ClassVar, Literal, Union

from pydantic import ConfigDict, Tag, TypeAdapter, model_validator

import naive_flow as nf

from ...data import (
    ACMDataset,
    AtomicAmazonDataset,
    AtomicDBLPDataset,
    AtomicIMDBDataset,
    AtomicMAGDataset,
    AtomicPubMedDataset,
    FreebaseDataset,
    LastFMDataset,
)
from ...script_utils.configs.dataset import (
    HeteroDatasetConfig,
    HGBLinkPredDatasetConfig,
    PrepropagationConfig,
)
from ...script_utils.misc import BaseConfig
from ...type import CEType, EType, NType

DEFAULT_PREPROP = PrepropagationConfig(cache_dir='~/.dgl/atomic-preprop')


class UndirectedCanonicalEtype:
    etype1: CEType
    etype2: CEType

    def __init__(self, etype1, etype2):
        self.etype1, self.etype2 = sorted([etype1, etype2])
        return

    def __getitem__(self, i):
        return [self.etype1, self.etype2][i]

    def __iter__(self):
        yield self.etype1
        yield self.etype2

    def __eq__(self, rval):
        return (self.etype1, self.etype2) == (rval.etype1, rval.etype2)

    def __hash__(self):
        return hash(self.etype1 + self.etype2)

    @classmethod
    def build(
        cls, canonical_etypes: list[CEType], inverse_etypes: list[CEType]
    ) -> tuple[set[UndirectedCanonicalEtype], dict[EType,
                                                   UndirectedCanonicalEtype]]:
        assert len(canonical_etypes) == len(
            inverse_etypes
        ), f'({len(canonical_etypes)=}) != ({len(inverse_etypes)=})'
        udcetypes = set()
        to_udcetype = {}
        for _, (etype1,
                etype2) in enumerate(zip(canonical_etypes, inverse_etypes)):
            udcetype = UndirectedCanonicalEtype(etype1, etype2)
            udcetypes.add(udcetype)
            to_udcetype[etype1] = udcetype
            to_udcetype[etype2] = udcetype
            to_udcetype[etype1[1]] = udcetype
            to_udcetype[etype2[1]] = udcetype
        return udcetypes, to_udcetype


DatasetConfigType = Union[
    Annotated[HeteroDatasetConfig,
              Tag('NodeClassificationDataset')],
    Annotated[HGBLinkPredDatasetConfig,
              Tag('LinkPredictionDataset')],
]
ADAPTER = TypeAdapter(DatasetConfigType)


class BaseAtomicDatasetConfig(BaseConfig):

    model_config = ConfigDict(extra='allow')
    name: str
    selection: dict[str, bool]
    prepropagation: bool | PrepropagationConfig
    _feat_types: ClassVar[dict[NType, str]]
    _ntypes: ClassVar[set[NType]]
    _canonical_etypes: ClassVar[list[CEType]]
    _inverse_etypes: ClassVar[list[CEType]]
    _default_preprop: ClassVar[PrepropagationConfig] = DEFAULT_PREPROP

    @model_validator(mode='before')
    @classmethod
    def parse(cls, data: dict):
        _, to_udcetype = UndirectedCanonicalEtype.build(
            cls._canonical_etypes, cls._inverse_etypes
        )
        data['selection'] = data.get('selection', {})
        etype_selections = {}
        for etype in data.get('selection', {}).copy():
            if etype in to_udcetype:
                etype_selections[to_udcetype[etype]
                                 ] = data['selection'].pop(etype)
        for key in data.copy():
            if key in cls._feat_types:
                data['selection'][key] = data.pop(key)
            elif key in to_udcetype:
                etype_selections[to_udcetype[key]] = data.pop(key)
        for udcetype, selected in etype_selections.items():
            data['selection'][udcetype.etype1[1]] = selected
        return data

    @property
    def _prepropagation_config(self) -> PrepropagationConfig | None:
        if self.prepropagation is False:
            return None
        if self.prepropagation is True:
            return self._default_preprop
        return self.prepropagation

    def update(self, **kwargs):
        data = self.model_dump()
        selection = data.pop('selection')
        data.update(**{**selection, **kwargs})
        return self.__class__.model_validate(data)

    @cached_property
    def _config(self) -> HeteroDatasetConfig:

        feat_types = {
            ntype:
            self._feat_types[ntype][int(self.selection.get(ntype, False))]
            for ntype in self._ntypes
        }

        _, to_udcetype = UndirectedCanonicalEtype.build(
            self._canonical_etypes, self._inverse_etypes
        )
        etype_selection = {}
        for etype, selected in self.selection.items():
            if etype in to_udcetype:
                etype_selection[to_udcetype[etype]] = selected

        config = ADAPTER.validate_python(
            dict(
                name=self.name,
                feat_types=feat_types,
                exclude_edge_types=sum(
                    (
                        [e1[1], e2[1]]
                        for (e1, e2), selected in etype_selection.items()
                        if not selected
                    ), []
                ),
                prepropagation_config=self._prepropagation_config,
                **self.model_extra,
            )
        )
        return config

    def load(self, verbose: bool | None = None, **kwargs):
        if verbose:
            print(nf.strfconfig(self._config))
        return self._config.load(verbose=verbose, **kwargs)


class AtomicFreebaseConfig(BaseAtomicDatasetConfig):

    name: Literal['freebase']
    _feat_types: ClassVar[dict[str, str]] = {
        ntype: ['none', 'nid_coo']
        for ntype in FreebaseDataset.ntypes
    }
    _ntypes: ClassVar[set[str]] = FreebaseDataset.ntypes
    _canonical_etypes: ClassVar[list[CEType]
                                ] = FreebaseDataset.canonical_etypes
    _inverse_etypes: ClassVar[list[CEType]] = FreebaseDataset.inverse_etypes


class AtomicDBLPConfig(BaseAtomicDatasetConfig):

    name: Literal['atomic-dblp']
    _feat_types: ClassVar[dict[str, str]] = {
        ntype: ['none', 'nid']
        for ntype in AtomicDBLPDataset.ntypes
    }
    _ntypes: ClassVar[set[str]] = AtomicDBLPDataset.ntypes
    _canonical_etypes: ClassVar[list[CEType]
                                ] = AtomicDBLPDataset.canonical_etypes
    _inverse_etypes: ClassVar[list[CEType]] = AtomicDBLPDataset.inverse_etypes


class AtomicACMConfig(BaseAtomicDatasetConfig):

    name: Literal['acm']
    _feat_types: ClassVar[dict[str, str]] = {
        ntype: ['none', 'nid']
        for ntype in ACMDataset.ntypes
    }
    _ntypes: ClassVar[set[str]] = ACMDataset.ntypes
    _canonical_etypes: ClassVar[list[CEType]] = ACMDataset.canonical_etypes
    _inverse_etypes: ClassVar[list[CEType]] = ACMDataset.inverse_etypes


class AtomicIMDBConfig(BaseAtomicDatasetConfig):
    name: Literal['atomic-imdb']
    _feat_types: ClassVar[dict[str, str]] = {
        ntype: ['none', 'nid']
        for ntype in AtomicIMDBDataset.ntypes
    }
    _ntypes: ClassVar[set[str]] = AtomicIMDBDataset.ntypes
    _canonical_etypes: ClassVar[list[CEType]
                                ] = AtomicIMDBDataset.canonical_etypes
    _inverse_etypes: ClassVar[list[CEType]] = AtomicIMDBDataset.inverse_etypes


class AtomicMAGConfig(BaseAtomicDatasetConfig):

    name: Literal['atomic-mag']
    _feat_types: ClassVar[dict[str, str]] = {
        'paper': ['none', 'u(256, -0.5, 0.5, 0xAA9)'],
        'institution': ['none', 'u(256, -0.5, 0.5, 0xAAA)'],
        'field_of_study': ['none', 'u(256, -0.5, 0.5, 0xAAB)'],
        'author': ['none', 'u(256, -0.5, 0.5, 0xAAC)'],
        "numerical": ['none', 'nid'],
        'year': ['none', 'nid'],
    }
    _ntypes: ClassVar[set[str]] = AtomicMAGDataset.ntypes
    _canonical_etypes: ClassVar[list[CEType]
                                ] = AtomicMAGDataset.canonical_etypes
    _inverse_etypes: ClassVar[list[CEType]] = AtomicMAGDataset.inverse_etypes


class LastFMConfig(BaseAtomicDatasetConfig):
    name: Literal['lastfm']
    _feat_types: ClassVar[dict[str, str]] = {
        ntype: ['none', 'nid_coo']
        for ntype in LastFMDataset.ntypes
    }
    _ntypes: ClassVar[set[str]] = LastFMDataset.ntypes
    _canonical_etypes: ClassVar[list[CEType]] = LastFMDataset.canonical_etypes
    _inverse_etypes: ClassVar[list[CEType]] = LastFMDataset.inverse_etypes


class AtomicPubMedConfig(BaseAtomicDatasetConfig):
    name: Literal['atomic-pubmed']
    _feat_types: ClassVar[dict[str, str]] = {
        ntype: ['none', 'nid']
        for ntype in AtomicPubMedDataset.ntypes
    }
    _ntypes: ClassVar[set[str]] = AtomicPubMedDataset.ntypes
    _canonical_etypes: ClassVar[list[CEType]
                                ] = AtomicPubMedDataset.canonical_etypes
    _inverse_etypes: ClassVar[list[CEType]
                              ] = AtomicPubMedDataset.inverse_etypes
    _default_preprop: ClassVar[PrepropagationConfig] = PrepropagationConfig(
        cache_dir='~/.dgl/atomic-preprop',
        to_sparse_threshold=0.001,
        # NOTE: pubmed use dense to prepropagation. convert to sparse afterward.
    )


class AtomicAmazonConfig(BaseAtomicDatasetConfig):
    name: Literal['atomic-amazon']
    _feat_types: ClassVar[dict[str, str]] = {
        'product': ['none', 'nid_coo'],
        'price': ['none', 'nid'],
        'sales_rank': ['none', 'nid'],
        'brand': ['none', 'nid'],
        'category': ['none', 'nid'],
    }
    _ntypes: ClassVar[set[str]] = AtomicAmazonDataset.ntypes
    _canonical_etypes: ClassVar[list[CEType]
                                ] = AtomicAmazonDataset.canonical_etypes
    _inverse_etypes: ClassVar[list[CEType]
                              ] = AtomicAmazonDataset.inverse_etypes
