from __future__ import annotations

import abc
import json
from typing import Annotated, ClassVar, Literal

import torch
from pydantic import Field, ValidationError, field_validator, model_validator

import dhgl

from .... import BaseHeteroGraphLike, transforms
from ....data import (
    FreebaseDataset,
    HeteroACMDataset,
    HeteroDBLPDataset,
    HeteroIMDBDataset,
    MAGDataset,
    NormalizedIMDBDataset,
)
from ....data.dblp.dblp_dataset import NormalizedDBLPDataset
from ....data.mag.mag_dataset import NormalizedMAGDataset
from ....script_utils import BaseConfig
from ._feat_types import BasicFeatType, FeatTypes, RandomFeatType
from ._prepropagation import PrepropagationConfig

# FeatTypeLiteral = Literal['original', 'ntype', 'id', 'zero', 'nid',
#                           'ntype_emb', 'nid_emb', 'none', 'nid_coo']


class BaseDatasetConfig(BaseConfig):

    name: str

    save_dir: str | None = None

    feat_types: FeatTypes[BasicFeatType]

    non_tgt_feat: BasicFeatType | None = Field(None, exclude=True)
    # NOTE: will be adapted to feat_types in before validator

    prepropagation_config: PrepropagationConfig | None = None
    exclude_edge_types: list[str] | None = None

    verbose: bool | None = None

    _tgt_ntype: ClassVar[str]
    _ntypes: ClassVar[set[str]]

    @field_validator('feat_types', mode='before')
    @classmethod
    def lazy_feat_types(cls, v: str):
        """Lazy loading
        E.g.
        feat_types="original"
        """
        if isinstance(v, str):
            try:
                v = BasicFeatType.model_validate(v)
                return {ntype: v for ntype in cls._ntypes}
            except ValidationError:
                pass
            return json.loads(v)
        return v

    @field_validator('exclude_edge_types', mode='before')
    @classmethod
    def load_json(cls, v: str):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @model_validator(mode='before')
    @classmethod
    def adaption(cls, data: dict):
        if data.get('non_tgt_feat', None) is not None:
            assert 'feat_types' not in data, (
                'Expect one of "feat_types" or "non_tgt_feat" is set.'
            )
            data['feat_types'] = {
                ntype: data['non_tgt_feat']
                for ntype in cls._ntypes if ntype != cls._tgt_ntype
            }
            data['feat_types'][cls._tgt_ntype] = 'original'
        return data

    @model_validator(mode='after')
    def check_valid(self):

        if self.feat_types is not None:
            assert set(self.feat_types) == set(self._ntypes), (
                'If feat_type is used, it must include all node types '
                f'{self._ntypes}, but got {set(self.feat_types)}'
            )
        return self

    @property
    @abc.abstractmethod
    def cache_dir(self) -> str:
        ...

    def _process_feats(
        self,
        hg: BaseHeteroGraphLike,
        verbose: bool | None = None,
        **kwargs,
    ):

        hg = self.feat_types.apply_(hg)

        if self.prepropagation_config is not None:
            hg = self.prepropagation_config.propagate(hg)
        if self.exclude_edge_types:
            if len(self.exclude_edge_types) < len(hg.etypes):
                hg = transforms.remove_etypes(hg, self.exclude_edge_types)
            else:
                assert set(
                    map(hg.to_canonical_etype, self.exclude_edge_types)
                ) == set(hg.canonical_etypes)
                # XXX: This is an UNSAFE workaround, which assume later dhgl.add_self_loop will be called
                # (and the self-loop edge type follow the format {NTYPE}-self)
                left_ntypes = (
                    [self._tgt_ntype]
                    if hasattr(self, '_tgt_ntype') else hg.ntypes
                )
                hg = transforms.update_graph_structure(
                    hg, {
                        (ntype, f'{ntype}-self', ntype): ([], [])
                        for ntype in left_ntypes
                    }, copy_ndata=True, copy_edata=True
                )
                for ntype in hg.ntypes:
                    if ntype in left_ntypes:
                        continue
                    if dhgl.FEAT in hg.nodes[ntype].data:
                        hg.nodes[ntype].data.pop(dhgl.FEAT)
        if (verbose if verbose is not None else self.verbose):
            print(dhgl.info(hg))
        return hg

    @abc.abstractmethod
    def load(self, verbose: bool | None = None, **kwargs):
        ...


class IMDBConfig(BaseDatasetConfig):

    name: Literal['imdb'] = 'imdb'
    raw_path: str

    _tgt_ntype: ClassVar[str] = 'movie'
    _ntypes: ClassVar[set[str]] = {'movie', 'actor', 'director', 'keyword'}

    @property
    def cache_dir(self):
        return self.save_dir if self.save_dir is not None else self.raw_path

    def load(self, verbose: bool | None = None, **kwargs):
        hg = HeteroIMDBDataset(self.raw_path, save_dir=self.save_dir)[0]
        hg = self._process_feats(hg, verbose=verbose, **kwargs)
        return hg


class NIMDBConfig(BaseDatasetConfig):

    name: Literal['atomic-imdb', 'nimdb'] = 'atomic-imdb'
    raw_path: str

    _tgt_ntype: ClassVar[str] = 'movie'
    _ntypes: ClassVar[set[str]] = {
        'movie', 'actor', 'director', 'keyword', 'color', 'content_rating',
        'country', 'language', 'numerical', 'word'
    }
    movie_feat: str | None = None
    """Could be list of ntype with '+' as separator. E.g. 'actor+color+country'
    """
    _MOVIE_NDATA: ClassVar[dict] = {
        'numerical': slice(16),
        'color': slice(16, 19),
        'language': slice(19, 67),
        'country': slice(67, 132),
        'content_rating': slice(132, 148),
        'word': slice(148, None),
    }

    @field_validator('movie_feat', mode='after')
    @classmethod
    def check_is_valid(cls, value: str | None):
        if value is None:
            return value

        for nt in value.split('+'):
            assert nt in cls._MOVIE_NDATA, f'The value: {value} is not valid'

        return value

    @property
    def cache_dir(self):
        return self.save_dir if self.save_dir is not None else self.raw_path

    def load(self, verbose: bool | None = None, **kwargs):
        hg = NormalizedIMDBDataset(self.raw_path, save_dir=self.save_dir)[0]
        movie_ndata = hg.nodes['movie'].data['feat']
        hg = self._process_feats(hg, verbose=verbose, **kwargs)

        if self.movie_feat is not None:
            hg.nodes['movie'].data['feat'] = torch.concat(
                [
                    movie_ndata[:, self._MOVIE_NDATA[ntype]]
                    for ntype in self.movie_feat.split('+')
                ], dim=1
            )
        return hg


class ACMConfig(BaseDatasetConfig):

    name: Literal['acm'] = 'acm'
    raw_path: str

    use_citing_only: bool | None = None
    """Default to False. Whether to remove the 'cited' edges which are identical to 'citing'."""

    _tgt_ntype: ClassVar[str] = 'paper'
    _ntypes: ClassVar[set[str]] = {'paper', 'author', 'subject', 'term'}

    @property
    def cache_dir(self):
        return self.save_dir if self.save_dir is not None else self.raw_path

    def load(self, verbose: bool | None = None, **kwargs):
        hg = HeteroACMDataset(self.raw_path, save_dir=self.save_dir)[0]
        if self.use_citing_only is True:
            hg = transforms.remove_etypes(hg, ['cited'])

        hg = self._process_feats(hg, verbose=verbose, **kwargs)
        return hg


class DBLPConfig(BaseDatasetConfig):

    name: Literal['dblp'] = 'dblp'
    raw_path: str

    _tgt_ntype: ClassVar[str] = 'author'
    _ntypes: ClassVar[set[str]] = {'author', 'conference', 'paper', 'term'}

    @property
    def cache_dir(self):
        return self.save_dir if self.save_dir is not None else self.raw_path

    def load(self, verbose: bool | None = None, **kwargs):
        hg = HeteroDBLPDataset(self.raw_path, save_dir=self.save_dir)[0]

        hg = self._process_feats(hg, verbose=verbose, **kwargs)
        return hg


class NDBLPConfig(BaseDatasetConfig):

    name: Literal['atomic-dblp', 'ndblp'] = 'atomic-dblp'
    raw_path: str

    _tgt_ntype: ClassVar[str] = 'author'
    _ntypes: ClassVar[set[str]] = {
        'author', 'conference', 'paper', 'term', 'numerical', 'paperfeat',
        'authorfeat'
    }

    @property
    def cache_dir(self):
        return self.save_dir if self.save_dir is not None else self.raw_path

    def load(self, verbose: bool | None = None, **kwargs):
        hg = NormalizedDBLPDataset(self.raw_path, save_dir=self.save_dir)[0]

        hg = self._process_feats(hg, verbose=verbose, **kwargs)
        return hg


# FreebaseFeatTypeLiteral = Literal[get_args(FeatTypeLiteral) +
#                                   ('nid_coo_unlabeled', 'nid_unlabeled')]


class FreebaseConfig(BaseDatasetConfig):

    name: Literal['freebase'] = 'freebase'
    raw_path: str

    _tgt_ntype: ClassVar[str] = 'book'
    _ntypes: ClassVar[set[str]] = {
        'book', 'business', 'film', 'location', 'music', 'organization',
        'people', 'sports'
    }
    use_symmetric_inv: bool | None = None

    @property
    def cache_dir(self):
        return self.save_dir if self.save_dir is not None else self.raw_path

    def _to_symmetric(self, hg: BaseHeteroGraphLike):
        symmetrics = [
            [
                ('book', 'book-and-book', 'book'),
                ('book', 'book-and-book-inv', 'book')
            ],
            [
                ('business', 'business-and-business', 'business'),
                ('business', 'business-and-business-inv', 'business')
            ],
            [
                ('film', 'film-and-film', 'film'),
                ('film', 'film-and-film-inv', 'film')
            ],
            [
                ('location', 'location-and-location', 'location'),
                ('location', 'location-and-location-inv', 'location')
            ],
            [
                ('music', 'music-and-music', 'music'),
                ('music', 'music-and-music-inv', 'music')
            ],
            [
                (
                    'organization', 'organization-and-organization',
                    'organization'
                ),
                (
                    'organization', 'organization-and-organization-inv',
                    'organization'
                )
            ],
            [
                ('people', 'people-and-people', 'people'),
                ('people', 'people-and-people-inv', 'people')
            ],
            [
                ('sports', 'sports-and-sports', 'sports'),
                ('sports', 'sports-and-sports-inv', 'sports')
            ],
        ]
        for e1, e2 in symmetrics:
            hg = transforms.merge_etypes(hg, etype=e1, etype_to_drop=e2)
        return hg

    def load(self, verbose: bool | None = None, **kwargs):
        hg = FreebaseDataset(self.raw_path, save_dir=self.save_dir)[0]
        if self.use_symmetric_inv:
            hg = self._to_symmetric(hg)
        hg = self._process_feats(hg, verbose=verbose, **kwargs)
        return hg


class MAGConfig(BaseDatasetConfig):

    name: Literal['mag', 'ogbn-mag'] = 'ogbn-mag'
    _tgt_ntype: ClassVar[str] = 'paper'
    _ntypes: ClassVar[set[str]
                      ] = {'paper', 'author', 'field_of_study', 'institution'}

    symmetric_citing: bool = False
    """Whether to merge the edges type 'cites' and 'cited' into one"""

    @property
    def cache_dir(self):
        return self.save_dir

    def _etype_to_symmetric(
        self,
        hg: BaseHeteroGraphLike,
        etype: str,
        etype_to_drop: str,
    ):

        assert dhgl.EWEIGHT not in hg.edges[etype].data

        etype = hg.to_canonical_etype(etype)
        etype_to_drop = hg.to_canonical_etype(etype_to_drop)
        indices = (
            hg.adj_external(etype=etype) +
            hg.adj_external(etype=etype_to_drop)
        ).coalesce().indices()

        data_dict = {e: hg.edges(etype=e) for e in hg.canonical_etypes}
        assert data_dict.pop(
            hg.to_canonical_etype(etype_to_drop), None
        ) is not None
        data_dict[etype] = (indices[0], indices[1])

        hg = transforms.update_graph_structure(hg, data_dict)
        return hg

    def load(self, verbose: bool | None = None, **kwargs):
        hg = MAGDataset(self.save_dir, save_dir=self.save_dir)[0]

        if self.symmetric_citing:
            hg = self._etype_to_symmetric(hg, 'cites', 'cited')
        hg = self._process_feats(hg, verbose=verbose, **kwargs)
        return hg


class NMAGConfig(MAGConfig):
    name: Literal['nmag', 'ogbn-nmag', 'atomic-mag'] = 'atomic-mag'
    feat_types: FeatTypes[BasicFeatType | RandomFeatType]
    _ntypes: ClassVar[set[str]] = {
        'paper', 'author', 'field_of_study', 'institution', 'year', 'numerical'
    }

    def load(self, verbose: bool | None = None, **kwargs):
        hg = NormalizedMAGDataset(self.save_dir, save_dir=self.save_dir)[0]

        if self.symmetric_citing:
            hg = self._etype_to_symmetric(hg, 'cites', 'cited')
            if 'cited' in self.exclude_edge_types:
                self.exclude_edge_types.pop(
                    self.exclude_edge_types.index('cited')
                )
        hg = self._process_feats(hg, verbose=verbose, **kwargs)
        return hg


HeteroDatasetConfig = Annotated[ACMConfig | IMDBConfig
                                | NIMDBConfig
                                | DBLPConfig | NDBLPConfig | FreebaseConfig
                                | MAGConfig | NMAGConfig,
                                Field(discriminator='name')]
