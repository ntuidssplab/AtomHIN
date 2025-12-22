from __future__ import annotations

import os
from typing import ClassVar

import dgl
import torch

from ...type import CEType, NType
from ..base.base_hetero_dataset import BaseHeteroDGLDataset
from ..raw_data_parsers import process_raw_data
from ..shared.utils import load_graphs, mat2tensor, save_graphs
from .imdb_schema import IMDBGraphSchema
from .nimdb_schema import NIMDBGraphSchema

VARIANTS = {
    'HGB':
    'https://www.dropbox.com/scl/fi/c0a62tcixg2sqh8fxh5mm/IMDB.zip?rlkey=zhlofgvs1xhqee6wpb62915bj&st=ien0l0s0&dl=0'
}


class IMDBDataset(BaseHeteroDGLDataset):
    """ Heterogeneous IMDB Dataset

    Parameters
    ----------
    raw_path : str
        Specifying the directory that stores raw data
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """

    name: ClassVar[str] = 'imdb'
    graph: IMDBGraphSchema
    _saved_graph_name: ClassVar[str] = 'graph.bin'
    variants: ClassVar[list[str]] = list(VARIANTS)
    ntypes: ClassVar[list[NType]] = [
        'movie',
        'keyword',
        'director',
        'actor',
    ]
    canonical_etypes: ClassVar[list[CEType]] = [
        ('actor', 'acts', 'movie'),
        ('movie', 'stars', 'actor'),
        ('director', 'directed', 'movie'),
        ('movie', 'directed-by', 'director'),
        ('keyword', 'is-in', 'movie'),
        ('movie', 'contains', 'keyword'),
    ]
    inverse_etypes: ClassVar[list[CEType]] = [
        ('movie', 'stars', 'actor'),
        ('actor', 'acts', 'movie'),
        ('movie', 'directed-by', 'director'),
        ('director', 'directed', 'movie'),
        ('movie', 'contains', 'keyword'),
        ('keyword', 'is-in', 'movie'),
    ]

    def __init__(
        self,
        raw_path: str = None,
        raw_dir: str = None,
        save_dir: str = None,
        force_reload: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            raw_path=VARIANTS.get(raw_path or list(VARIANTS)[0], raw_path),
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def __getitem__(self, idx: int):
        if idx != 0:
            raise ValueError(f'index: {idx} is out of bound of {len(self)}.')

        return self.graph

    def __len__(self):
        return 1

    def save(self):
        # save processed data to directory `self.save_path`

        path = os.path.join(self.save_path, self._saved_graph_name)
        save_graphs(path, [self.graph])
        return

    def load(self):
        # load processed data from directory `self.save_path`
        path = os.path.join(self.save_path, self._saved_graph_name)
        [g], _ = load_graphs(path)

        g: IMDBGraphSchema
        g.nodes['movie'].data['train_mask'] =\
            g.nodes['movie'].data['train_mask'].bool()
        g.nodes['movie'].data['val_mask'] =\
            g.nodes['movie'].data['val_mask'].bool()
        g.nodes['movie'].data['test_mask'] =\
            g.nodes['movie'].data['test_mask'].bool()

        self.graph = g

        return

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        path = os.path.join(self.save_path, self._saved_graph_name)
        has = os.path.exists(path)
        return has

    def process(self):
        """process raw data to graphs, labels, splitting masks
        """

        graph_data = process_raw_data(self.raw_path)

        features = graph_data['features']
        adj = graph_data['adj']
        ptr = graph_data['ntype_idx_ptr']
        label_masks = graph_data['label_masks']
        labels = graph_data['labels']

        features = [mat2tensor(features) for features in features]

        labels = torch.Tensor(labels)

        # yapf: disable
        ################################
        node_dict = {
            'movie'    : 0,  # 0     ~ 4572
            'director' : 1,  # 4573  ~ 6779
            'actor'    : 2,  # 6780  ~ 12368
            'keyword'  : 3   # 12369 ~ 19932
        }
        # Link_meta
        # {0: (0, 1), 1: (1, 0), 2: (0, 2), 3: (2, 0), 4: (0, 3), 5: (3, 0)}
        edge_dict = { #pylint: disable=unused-variable
            'directed-by' : 0,  # (movie, director)
            'directed'    : 1,  # (director, movie)
            'stars'       : 2,  # (movie, actor)
            'acts'        : 3,  # (actor, movie)
            'contains'    : 4,  # (movie, keyword)
            'is-in'       : 5   # (keyword, movie)
        }

        data_dict = {
            ('movie',    'directed-by', 'director') : adj[ptr[0] : ptr[1], ptr[1] : ptr[2]].nonzero(),
            ('director', 'directed',    'movie')    : adj[ptr[0] : ptr[1], ptr[1] : ptr[2]].transpose().nonzero(),
            ('movie',    'stars',       'actor')    : adj[ptr[0] : ptr[1], ptr[2] : ptr[3]].nonzero(),
            ('actor',    'acts',        'movie')    : adj[ptr[0] : ptr[1], ptr[2] : ptr[3]].transpose().nonzero(),
            ('movie',    'contains',    'keyword')  : adj[ptr[0] : ptr[1], ptr[3] :       ].nonzero(),
            ('keyword',  'is-in',       'movie')    : adj[ptr[0] : ptr[1], ptr[3] :       ].transpose().nonzero(),
        }
        # yapf: enable

        hg: IMDBGraphSchema = dgl.heterograph(data_dict)

        # assign node features
        for idx, ntype in enumerate(node_dict.keys()):
            hg.nodes[ntype].data['feat'] = features[idx]

        hg.nodes['movie'].data['train_mask'] =\
            torch.from_numpy(label_masks['train']).bool()
        hg.nodes['movie'].data['val_mask'] =\
            torch.from_numpy(label_masks['val']).bool()
        hg.nodes['movie'].data['test_mask'] =\
            torch.from_numpy(label_masks['test']).bool()

        hg.nodes['movie'].data['label'] = labels

        self.graph = hg
        return


# Alias
HeteroIMDBDataset = IMDBDataset


class AtomicIMDBDataset(HeteroIMDBDataset):
    """ Normalized IMDB Dataset

    Parameters
    ----------
    raw_path : str
        Specifying the directory that stores raw data
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """

    name: ClassVar[str] = 'imdb-n'
    graph: NIMDBGraphSchema
    _saved_graph_name: ClassVar[str] = 'graph-imdb-n.bin'
    ntypes: ClassVar[list[NType]] = [
        'movie',
        'keyword',
        'director',
        'actor',
        'color',
        'language',
        'country',
        'content_rating',
        'numerical',
        'word',
    ]
    canonical_etypes: ClassVar[list[CEType]] = [
        ('actor', 'acts', 'movie'),
        ('movie', 'stars', 'actor'),
        ('color', 'is-type-of', 'movie'),
        ('movie', 'has-color', 'color'),
        ('content_rating', 'is-rating-for', 'movie'),
        ('movie', 'has-rating', 'content_rating'),
        ('country', 'is-country-of', 'movie'),
        ('movie', 'is-from-country', 'country'),
        ('director', 'directed', 'movie'),
        ('movie', 'directed-by', 'director'),
        ('keyword', 'is-in', 'movie'),
        ('movie', 'contains', 'keyword'),
        ('language', 'is-language-of', 'movie'),
        ('movie', 'is-in-language', 'language'),
        ('movie', 'contains-word', 'word'),
        ('word', 'is-word-of', 'movie'),
        ('movie', 'has-numerical', 'numerical'),
        ('numerical', 'is-numerical-of', 'movie'),
    ]
    inverse_etypes: ClassVar[list[CEType]] = [
        ('movie', 'stars', 'actor'),
        ('actor', 'acts', 'movie'),
        ('movie', 'has-color', 'color'),
        ('color', 'is-type-of', 'movie'),
        ('movie', 'has-rating', 'content_rating'),
        ('content_rating', 'is-rating-for', 'movie'),
        ('movie', 'is-from-country', 'country'),
        ('country', 'is-country-of', 'movie'),
        ('movie', 'directed-by', 'director'),
        ('director', 'directed', 'movie'),
        ('movie', 'contains', 'keyword'),
        ('keyword', 'is-in', 'movie'),
        ('movie', 'is-in-language', 'language'),
        ('language', 'is-language-of', 'movie'),
        ('word', 'is-word-of', 'movie'),
        ('movie', 'contains-word', 'word'),
        ('numerical', 'is-numerical-of', 'movie'),
        ('movie', 'has-numerical', 'numerical'),
    ]

    def __getitem__(self, idx: int):
        if idx != 0:
            raise ValueError(f'index: {idx} is out of bound of {len(self)}.')

        return self.graph

    def __len__(self):
        return 1

    def save(self):
        # save processed data to directory `self.save_path`

        path = os.path.join(self.save_path, self._saved_graph_name)
        save_graphs(path, [self.graph])
        return

    def load(self):
        # load processed data from directory `self.save_path`
        path = os.path.join(self.save_path, self._saved_graph_name)
        [g], _ = load_graphs(path)

        g: NIMDBGraphSchema
        g.nodes['movie'].data['train_mask'] =\
            g.nodes['movie'].data['train_mask'].bool()
        g.nodes['movie'].data['val_mask'] =\
            g.nodes['movie'].data['val_mask'].bool()
        g.nodes['movie'].data['test_mask'] =\
            g.nodes['movie'].data['test_mask'].bool()

        self.graph = g

        return

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        path = os.path.join(self.save_path, self._saved_graph_name)
        has = os.path.exists(path)
        return has

    def process(self):
        """process raw data to graphs, labels, splitting masks
        """
        super().process()
        hg = self.graph
        movie_ndata = hg.nodes['movie'].data['feat']
        numerical_data = movie_ndata[:, :16].clone()
        color = movie_ndata[:, 16:19].to_sparse_coo()
        assert (color.values() == 1.).all()
        language = movie_ndata[:, 19:67].to_sparse_coo()
        assert (language.values() == 1.).all()
        country = movie_ndata[:, 67:132].to_sparse_coo()
        assert (country.values() == 1.).all()
        content_rating = movie_ndata[:, 132:148].to_sparse_coo()
        assert (content_rating.values() == 1.).all()

        def normalize_bow(x: torch.Tensor):
            """ Normalize bag-of-words tensor by row"""
            # NOTE: this is to align with the other adjacencies
            # so that mean reduce function can be used for all etypes.
            # This won't harm if someone perform the row-normalization again if needed.
            scale = (x > 0).sum(dim=1, keepdim=True) /\
                x.sum(dim=1, keepdim=True).clamp_min(1e-6)
            return x * scale

        raw_bow = movie_ndata[:, 148:].clone()
        bow = normalize_bow(raw_bow).to_sparse_coo()
        bow_t = normalize_bow(raw_bow.T).to_sparse_coo()

        def normalize_numerical(x: torch.Tensor):
            """ Normalize numerical tensor by row"""
            # NOTE: this is to align with the other adjacencies
            # so that mean reduce function can be used for all etypes.
            return x * x.shape[1] / x.abs().sum(dim=1, keepdim=True)

        # numerical_d = numerical_data.abs().sum(dim=1, keepdim=True)
        # numerical_data = numerical_data * numerical_data.shape[1] / numerical_d
        numerical = normalize_numerical(numerical_data).to_sparse_coo()
        numerical_t = normalize_numerical(numerical_data.T).to_sparse_coo()

        data_dict = {
            cetype: hg.edges(etype=cetype)
            for cetype in hg.canonical_etypes
        }
        # yapf: disable
        data_dict[('movie', 'has-color', 'color')] = tuple(color.indices())
        data_dict[('color', 'is-type-of', 'movie')] = tuple(color.T.coalesce().indices())
        data_dict[('movie', 'is-in-language', 'language')] = tuple(language.indices())
        data_dict[('language', 'is-language-of', 'movie')] = tuple(language.T.coalesce().indices())
        data_dict[('movie', 'is-from-country', 'country')] = tuple(country.indices())
        data_dict[('country', 'is-country-of', 'movie')] = tuple(country.T.coalesce().indices())
        data_dict[('movie', 'has-rating', 'content_rating')] = tuple(content_rating.indices())
        data_dict[('content_rating', 'is-rating-for', 'movie')] = tuple(content_rating.T.coalesce().indices())
        data_dict[('movie', 'has-numerical', 'numerical')] = tuple(numerical_t.T.coalesce().indices())
        data_dict[('numerical', 'is-numerical-of', 'movie')] = tuple(numerical.T.coalesce().indices())
        data_dict[('movie', 'contains-word', 'word')] = tuple(bow_t.T.coalesce().indices())
        data_dict[('word', 'is-word-of', 'movie')] = tuple(bow.T.coalesce().indices())
        num_nodes = {ntype: hg.num_nodes(ntype=ntype) for ntype in hg.ntypes}
        num_nodes['color'] = color.shape[-1]
        num_nodes['language'] = language.shape[-1]
        num_nodes['country'] = country.shape[-1]
        num_nodes['content_rating'] = content_rating.shape[-1]
        new_hg = dgl.heterograph(data_dict)
        for data_key, ndata in hg.nodes['movie'].data.items():
            new_hg.nodes['movie'].data[data_key] = ndata
        new_hg.edges['has-numerical'].data['weight'] = numerical_t.T.coalesce().values()
        new_hg.edges['is-numerical-of'].data['weight'] = numerical.T.coalesce().values()
        new_hg.edges['contains-word'].data['weight'] = bow_t.T.coalesce().values()
        new_hg.edges['is-word-of'].data['weight'] = bow.T.coalesce().values()
        # new_hg.nodes['movie'].data['feat'] = numerical_data
        self.graph = new_hg
        return

# Alias
NormalizedIMDBDataset = AtomicIMDBDataset
