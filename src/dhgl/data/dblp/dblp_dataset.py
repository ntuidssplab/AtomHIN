from __future__ import annotations

import json
import os
from typing import ClassVar

import dgl
import torch

from ...type import CEType, NType
from ..base.base_hetero_dataset import BaseHeteroDGLDataset
from ..raw_data_parsers import process_raw_data
from ..shared.utils import load_graphs, mat2tensor, save_graphs
from .dblp_schema import DBLPGraphSchema

VARIANTS = {
    'HGB':
    'https://www.dropbox.com/scl/fi/8x361cq0xmspwcijvskmk/DBLP.zip?rlkey=stcapehu550xu0pg9g87uhy6n&st=srlmscb8&dl=0'
}


class DBLPDataset(BaseHeteroDGLDataset):
    """ Heterogeneous DBLP Dataset

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

    name: ClassVar[str] = 'dblp'
    graph: DBLPGraphSchema
    _saved_graph_name: ClassVar[str] = 'graph.bin'
    variants: ClassVar[list[str]] = list(VARIANTS)

    ntypes: ClassVar[list[NType]] = ['author', 'conference', 'paper', 'term']
    canonical_etypes: ClassVar[list[CEType]] = [
        ('conference', 'has', 'paper'),
        ('paper', 'pubs-in', 'conference'),
        ('paper', 'contains', 'term'),
        ('term', 'is-in', 'paper'),
        ('author', 'writing', 'paper'),
        ('paper', 'written-by', 'author'),
    ]
    inverse_etypes: ClassVar[list[CEType]] = [
        ('paper', 'pubs-in', 'conference'),
        ('conference', 'has', 'paper'),
        ('term', 'is-in', 'paper'),
        ('paper', 'contains', 'term'),
        ('paper', 'written-by', 'author'),
        ('author', 'writing', 'paper'),
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
        save_graphs(path, self.graph)

        return

    def load(self):
        # load processed data from directory `self.save_path`

        path = os.path.join(self.save_path, self._saved_graph_name)
        [g], _ = load_graphs(path)

        g: DBLPGraphSchema

        g.nodes['author'].data['train_mask'] =\
            g.nodes['author'].data['train_mask'].bool()
        g.nodes['author'].data['val_mask'] =\
            g.nodes['author'].data['val_mask'].bool()
        g.nodes['author'].data['test_mask'] =\
            g.nodes['author'].data['test_mask'].bool()

        self.graph: DBLPGraphSchema = g
        return

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        path = os.path.join(self.save_path, self._saved_graph_name)
        return os.path.exists(path)

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

        labels = torch.LongTensor(labels)
        labels = torch.argmax(labels, dim=1)

        ################################
        if os.path.isfile(os.path.join(self.raw_path, 'info.dat')):
            with open(
                os.path.join(self.raw_path, 'info.dat'), encoding='utf8'
            ) as fin:
                info = json.load(fin)
            nid = {
                ntype: int(ni)
                for ni, ntype in info['node.dat']['node type'].items()
            }
            nid['conference'] = nid.pop('venue')  # backward compatibility
            TARGET_NODE_TYPE = 'author'

        else:
            # backward compatibility
            # Node type
            nid = {'author': 0, 'paper': 1, 'conference': 2, 'term': 3}
            TARGET_NODE_TYPE = 'author'
            # Link_meta
            # {2: (0, 1), 3: (1, 0), 4: (1, 2), 5: (2, 1), 6: (1, 3), 7: (3, 1)}
            edge_dict = { #pylint: disable=unused-variable
                'cited'      : 0, # (paper, paper)
                'citing'     : 1, # (paper, paper)
                'writing'    : 2, # (author, paper)
                'written-by' : 3, # (paper, author)
                'pubs-in'    : 4, # (paper, conference)
                'has'        : 5, # (conference, paper)
                'contains'   : 6, # (paper, term)
                'is-in'      : 7  # (term, paper)
            }

        def get_slice(ntype):
            end_id = nid[ntype] + 1
            if end_id >= len(ptr):
                return slice(ptr[nid[ntype]], None)
            return slice(ptr[nid[ntype]], ptr[nid[ntype] + 1])

        # yapf: disable
        data_dict = {
            # ('paper',      'citing',     'paper')     : adj[ptr[1] : ptr[2], ptr[1] : ptr[2]].nonzero(),
            # ('paper',      'cited',      'paper')     : adj[ptr[1] : ptr[2], ptr[1] : ptr[2]].transpose().nonzero(),
            ('paper',      'written-by', 'author')    : adj[get_slice('paper'), get_slice('author')    ].nonzero(),
            ('author',     'writing',    'paper')     : adj[get_slice('paper'), get_slice('author')    ].transpose().nonzero(),
            ('paper',      'pubs-in',    'conference'): adj[get_slice('paper'), get_slice('conference')].nonzero(),
            ('conference', 'has',        'paper')     : adj[get_slice('paper'), get_slice('conference')].transpose().nonzero(),
            ('paper',      'contains',    'term')     : adj[get_slice('paper'), get_slice('term')      ].nonzero(),
            ('term',       'is-in',      'paper')     : adj[get_slice('paper'), get_slice('term')      ].transpose().nonzero(),
        }
        # yapf: enable

        hg: DBLPGraphSchema = dgl.heterograph(data_dict)

        for idx, ntype in enumerate(nid.keys()):
            hg.nodes[ntype].data['feat'] = features[idx]

        hg.nodes[TARGET_NODE_TYPE].data['train_mask'] = torch.from_numpy(
            label_masks['train']
        ).bool()
        hg.nodes[TARGET_NODE_TYPE].data['val_mask'] = torch.from_numpy(
            label_masks['val']
        ).bool()
        hg.nodes[TARGET_NODE_TYPE].data['test_mask'] = torch.from_numpy(
            label_masks['test']
        ).bool()
        hg.nodes[TARGET_NODE_TYPE].data['label'] = labels

        self.graph = hg
        return


HeteroDBLPDataset = DBLPDataset


class AtomicDBLPDataset(HeteroDBLPDataset):
    name: ClassVar[str] = 'ndblp'
    _saved_graph_name: ClassVar[str] = 'ngraph.bin'

    ntypes: ClassVar[list[NType]] = [
        'author', 'authorfeat', 'conference', 'numerical', 'paper',
        'paperfeat', 'term'
    ]
    canonical_etypes: ClassVar[list[CEType]] = [
        ('author', 'has-authorfeat', 'authorfeat'),
        ('authorfeat', 'is-authorfeat-of', 'author'),
        ('conference', 'has', 'paper'), ('paper', 'pubs-in', 'conference'),
        ('paper', 'contains', 'term'), ('term', 'is-in', 'paper'),
        ('paper', 'has-paperfeat', 'paperfeat'),
        ('paperfeat', 'is-paperfeat-of', 'paper'),
        ('author', 'writing', 'paper'), ('paper', 'written-by', 'author'),
        ('term', 'has-numerical', 'numerical'),
        ('numerical', 'is-numerical-of', 'term')
    ]
    inverse_etypes: ClassVar[list[CEType]] = [
        ('authorfeat', 'is-authorfeat-of', 'author'),
        ('author', 'has-authorfeat', 'authorfeat'),
        ('paper', 'pubs-in', 'conference'), ('conference', 'has', 'paper'),
        ('term', 'is-in', 'paper'), ('paper', 'contains', 'term'),
        ('paperfeat', 'is-paperfeat-of', 'paper'),
        ('paper', 'has-paperfeat', 'paperfeat'),
        ('paper', 'written-by', 'author'), ('author', 'writing', 'paper'),
        ('numerical', 'is-numerical-of', 'term'),
        ('term', 'has-numerical', 'numerical')
    ]

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

        labels = torch.LongTensor(labels)
        labels = torch.argmax(labels, dim=1)

        ################################
        if os.path.isfile(os.path.join(self.raw_path, 'info.dat')):
            with open(
                os.path.join(self.raw_path, 'info.dat'), encoding='utf8'
            ) as fin:
                info = json.load(fin)
            nid = {
                ntype: int(ni)
                for ni, ntype in info['node.dat']['node type'].items()
            }
            nid['conference'] = nid.pop('venue')  # backward compatibility
            TARGET_NODE_TYPE = 'author'

        else:
            raise ValueError(
                f'{self.__class__.__name__} requires term embedding features'
            )
            # Node type
            nid = {
                'author': 0,
                'paper': 1,
                'conference': 2,
                'term': 3,
                'numerical': 4
            }
            TARGET_NODE_TYPE = 'author'
            # Link_meta
            # {2: (0, 1), 3: (1, 0), 4: (1, 2), 5: (2, 1), 6: (1, 3), 7: (3, 1)}
            edge_dict = { #pylint: disable=unused-variable
                'cited'      : 0, # (paper, paper)
                'citing'     : 1, # (paper, paper)
                'writing'    : 2, # (author, paper)
                'written-by' : 3, # (paper, author)
                'pubs-in'    : 4, # (paper, conference)
                'has'        : 5, # (conference, paper)
                'contains'   : 6, # (paper, term)
                'is-in'      : 7  # (term, paper)
            }

        def get_slice(ntype):
            end_id = nid[ntype] + 1
            if end_id >= len(ptr):
                return slice(ptr[nid[ntype]], None)
            return slice(ptr[nid[ntype]], ptr[nid[ntype] + 1])

        def normalize_numerical(x: torch.Tensor):
            """ Normalize numerical tensor by row"""
            # NOTE: this is to align with the other adjacencies
            # so that mean reduce function can be used for all etypes.
            return x * x.shape[1] / x.abs().sum(dim=1, keepdim=True)

        # numerical_d = numerical_data.abs().sum(dim=1, keepdim=True)
        # numerical_data = numerical_data * numerical_data.shape[1] / numerical_d
        numerical = normalize_numerical(features[nid['term']]).to_sparse_coo()
        numerical_t = normalize_numerical(features[nid['term']].T
                                          ).to_sparse_coo()

        def normalize_bow(x: torch.Tensor):
            """ Normalize bag-of-words tensor by row"""
            # NOTE: this is to align with the other adjacencies
            # so that mean reduce function can be used for all etypes.
            # This won't harm if someone perform the row-normalization again if needed.
            scale = (x > 0).sum(dim=1, keepdim=True) /\
                x.sum(dim=1, keepdim=True).clamp_min(1e-6)
            return x * scale

        raw_bow = features[nid['paper']].clone()
        paper_feat = normalize_bow(raw_bow).to_sparse_coo()
        paper_feat_t = normalize_bow(raw_bow.T).to_sparse_coo()
        author_feat = features[nid['author']].to_sparse_coo()

        # yapf: disable
        data_dict = {
            # ('paper',      'citing',     'paper')     : adj[ptr[1] : ptr[2], ptr[1] : ptr[2]].nonzero(),
            # ('paper',      'cited',      'paper')     : adj[ptr[1] : ptr[2], ptr[1] : ptr[2]].transpose().nonzero(),
            ('paper',      'written-by', 'author')    : adj[get_slice('paper'), get_slice('author')    ].nonzero(),
            ('author',     'writing',    'paper')     : adj[get_slice('paper'), get_slice('author')    ].transpose().nonzero(),
            ('paper',      'pubs-in',    'conference'): adj[get_slice('paper'), get_slice('conference')].nonzero(),
            ('conference', 'has',        'paper')     : adj[get_slice('paper'), get_slice('conference')].transpose().nonzero(),
            ('paper',      'contains',    'term')     : adj[get_slice('paper'), get_slice('term')      ].nonzero(),
            ('term',       'is-in',      'paper')     : adj[get_slice('paper'), get_slice('term')      ].transpose().nonzero(),
            ('term', 'has-numerical', 'numerical')    : tuple(numerical_t.T.coalesce().indices()),
            ('numerical', 'is-numerical-of', 'term')  : tuple(numerical.T.coalesce().indices()),
            ('paper', 'has-paperfeat', 'paperfeat')  : tuple(paper_feat_t.T.coalesce().indices()),
            ('paperfeat', 'is-paperfeat-of', 'paper'):  tuple(paper_feat.T.coalesce().indices()),
            ('author', 'has-authorfeat', 'authorfeat')  : tuple(author_feat.indices()),
            ('authorfeat', 'is-authorfeat-of', 'author'): tuple(author_feat.T.coalesce().indices()),
        }
        # yapf: enable

        hg: DBLPGraphSchema = dgl.heterograph(data_dict)

        for idx, ntype in enumerate(nid.keys()):
            hg.nodes[ntype].data['feat'] = features[idx]

        hg.edges['has-numerical'].data['weight'] = numerical_t.T.coalesce(
        ).values()
        hg.edges['is-numerical-of'].data['weight'] = numerical.T.coalesce(
        ).values()
        hg.edges['has-paperfeat'].data['weight'] = paper_feat_t.T.coalesce(
        ).values()
        hg.edges['is-paperfeat-of'].data['weight'] = paper_feat.T.coalesce(
        ).values()
        hg.nodes[TARGET_NODE_TYPE].data['train_mask'] = torch.from_numpy(
            label_masks['train']
        ).bool()
        hg.nodes[TARGET_NODE_TYPE].data['val_mask'] = torch.from_numpy(
            label_masks['val']
        ).bool()
        hg.nodes[TARGET_NODE_TYPE].data['test_mask'] = torch.from_numpy(
            label_masks['test']
        ).bool()
        hg.nodes[TARGET_NODE_TYPE].data['label'] = labels

        self.graph = hg
        return


NormalizedDBLPDataset = AtomicDBLPDataset
