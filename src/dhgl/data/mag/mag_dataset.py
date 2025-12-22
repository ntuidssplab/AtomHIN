from __future__ import annotations

import os
from typing import ClassVar

import dgl
import torch
from lazy_imports import try_import
from sklearn.preprocessing import OneHotEncoder

from ...type import CEType, NType

with try_import() as ogb:
    from ogb.nodeproppred import DglNodePropPredDataset

from ...transforms import merge_etypes
from ..base.base_hetero_dataset import BaseHeteroDGLDataset
from ..shared.utils import load_graphs, save_graphs
from .mag_schema import MAGGraphSchema


class MAGDataset(BaseHeteroDGLDataset):
    """ Heterogeneous IMDB Dataset

    Parameters
    ----------
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

    name: ClassVar[str] = 'ogbn-mag'
    graph: MAGGraphSchema
    _saved_graph_name: ClassVar[str] = 'graph.bin'

    ntypes: ClassVar[list[NType]
                     ] = ['author', 'field_of_study', 'institution', 'paper']
    canonical_etypes: ClassVar[list[CEType]] = [
        ('author', 'affiliated_with', 'institution'),
        ('author', 'writes', 'paper'),
        ('field_of_study', 'contains', 'paper'),
        ('institution', 'affiliates', 'author'),
        ('paper', 'cited', 'paper'),
        ('paper', 'cites', 'paper'),
        ('paper', 'has_topic', 'field_of_study'),
        ('paper', 'written_by', 'author'),
    ]
    inverse_etypes: ClassVar[list[CEType]] = [
        ('institution', 'affiliates', 'author'),
        ('paper', 'written_by', 'author'),
        ('paper', 'has_topic', 'field_of_study'),
        ('author', 'affiliated_with', 'institution'),
        ('paper', 'cites', 'paper'),
        ('paper', 'cited', 'paper'),
        ('field_of_study', 'contains', 'paper'),
        ('author', 'writes', 'paper'),
    ]

    def __init__(
        self,
        raw_dir: str = None,
        save_dir: str = None,
        force_reload: bool = False,
        verbose: bool = False,
    ):

        super().__init__(
            raw_path=None,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    @property
    def symmetric_(self):
        """Merge cited into cites"""
        self.graph = merge_etypes(self.graph, 'cites', 'cited')
        return self

    def __getitem__(self, idx: int):
        if idx != 0:
            raise ValueError(f'index: {idx} is out of bound of {len(self)}.')

        return self.graph

    def __len__(self):
        return 1

    def download(self):
        # download raw data to local disk
        ogb.check()
        DglNodePropPredDataset(name='ogbn-mag', root=self.raw_dir)
        return

    def save(self):
        # save processed data to directory `self.save_path`

        path = os.path.join(self.save_path, self._saved_graph_name)
        save_graphs(path, [self.graph])
        return

    def load(self):
        # load processed data from directory `self.save_path`
        path = os.path.join(self.save_path, self._saved_graph_name)
        [g], _ = load_graphs(path)

        g: MAGGraphSchema
        g.nodes['paper'].data['train_mask'] =\
            g.nodes['paper'].data['train_mask'].bool()
        g.nodes['paper'].data['val_mask'] =\
            g.nodes['paper'].data['val_mask'].bool()
        g.nodes['paper'].data['test_mask'] =\
            g.nodes['paper'].data['test_mask'].bool()

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

        ogb.check()
        dataset = DglNodePropPredDataset(name='ogbn-mag', root=self.raw_dir)

        graph, label = dataset[0]
        graph: dgl.DGLGraph

        # yapf: disable
        etype_to_add = {
            ('author', 'affiliated_with', 'institution'): ('institution', 'affiliates', 'author'),
            ('paper', 'cites', 'paper'):                  ('paper', 'cited', 'paper'),
            ('author', 'writes', 'paper'):                ('paper', 'written_by', 'author'),
            ('paper', 'has_topic', 'field_of_study'):     ('field_of_study', 'contains', 'paper'),
        }
        # yapf: enable
        data_dict = {}
        for existing_etype, etype_to_add in etype_to_add.items():
            adj = graph.adj(existing_etype)
            assert (adj.val == 1.).all()
            u, v = adj.indices()
            data_dict[existing_etype] = (u, v)
            data_dict[etype_to_add] = (v, u)

        hg: dgl.DGLHeteroGraph = dgl.heterograph(data_dict)

        # NOTE: only papers have features, and the "year" is not used (following ogb example)
        TGT_NTYPE = 'paper'
        hg.nodes[TGT_NTYPE].data['feat'] = graph.nodes[TGT_NTYPE].data['feat']
        hg.nodes[TGT_NTYPE].data['label'] = label[TGT_NTYPE].squeeze()
        split_idx = dataset.get_idx_split()

        for split in ('train', 'val', 'test'):
            hg.nodes[TGT_NTYPE].data[f'{split}_mask'] = torch.zeros(
                (hg.num_nodes(TGT_NTYPE), ), dtype=torch.bool
            )
            key = split if split != 'val' else 'valid'
            hg.nodes[TGT_NTYPE].data[f'{split}_mask'][split_idx[key][TGT_NTYPE]
                                                      ] = 1

        self.graph = hg
        return


class AtomicMAGDataset(MAGDataset):
    name: ClassVar[str] = 'nmag'
    _saved_graph_name: ClassVar[str] = 'ngraph.bin'

    ntypes: ClassVar[list[NType]] = [
        'author', 'field_of_study', 'institution', 'numerical', 'paper', 'year'
    ]
    canonical_etypes: ClassVar[list[CEType]] = [
        ('author', 'affiliated_with', 'institution'),
        ('author', 'writes', 'paper'),
        ('field_of_study', 'contains', 'paper'),
        ('institution', 'affiliates', 'author'),
        ('numerical', 'is-numerical-of', 'paper'),
        ('paper', 'cited', 'paper'),
        ('paper', 'cites', 'paper'),
        ('paper', 'has-numerical', 'numerical'),
        ('paper', 'has_topic', 'field_of_study'),
        ('paper', 'published-in-year', 'year'),
        ('paper', 'written_by', 'author'),
        ('year', 'year-of-publication', 'paper'),
    ]
    inverse_etypes: ClassVar[list[CEType]] = [
        ('institution', 'affiliates', 'author'),
        ('paper', 'written_by', 'author'),
        ('paper', 'has_topic', 'field_of_study'),
        ('author', 'affiliated_with', 'institution'),
        ('paper', 'has-numerical', 'numerical'),
        ('paper', 'cites', 'paper'),
        ('paper', 'cited', 'paper'),
        ('numerical', 'is-numerical-of', 'paper'),
        ('field_of_study', 'contains', 'paper'),
        ('year', 'year-of-publication', 'paper'),
        ('author', 'writes', 'paper'),
        ('paper', 'published-in-year', 'year'),
    ]

    def process(self):
        """process raw data to graphs, labels, splitting masks
        """

        ogb.check()
        dataset = DglNodePropPredDataset(name='ogbn-mag', root=self.raw_dir)
        split_idx = dataset.get_idx_split()

        graph, label = dataset[0]
        graph: dgl.DGLGraph

        TGT_NTYPE = 'paper'
        # yapf: disable
        etype_to_add = {
            ('author', 'affiliated_with', 'institution'): ('institution', 'affiliates', 'author'),
            ('paper', 'cites', 'paper'):                  ('paper', 'cited', 'paper'),
            ('author', 'writes', 'paper'):                ('paper', 'written_by', 'author'),
            ('paper', 'has_topic', 'field_of_study'):     ('field_of_study', 'contains', 'paper'),
        }
        # yapf: enable
        data_dict = {}
        for existing_etype, etype_to_add in etype_to_add.items():
            adj = graph.adj(existing_etype)
            assert (adj.val == 1.).all()
            u, v = adj.indices()
            data_dict[existing_etype] = (u, v)
            data_dict[etype_to_add] = (v, u)

        def normalize_numerical(x: torch.Tensor):
            """ Normalize numerical tensor by row"""
            # NOTE: this is to align with the other adjacencies
            # so that mean reduce function can be used for all etypes.
            return x * x.shape[1] / x.abs().sum(dim=1, keepdim=True)

        def to_one_hot(x: torch.Tensor):
            train_x = x[split_idx['train']['paper']]
            # cats = np.arange(train_x.max().item() + 1).reshape(-1, 1).tolist()
            cats = [
                list(range(train_x.min().item(),
                           train_x.max().item() + 1))
            ]

            x = torch.full_like(x, -1)
            x[split_idx['train']['paper']] = train_x
            return OneHotEncoder(categories=cats, handle_unknown='ignore'
                                 ).fit_transform(x).tocoo()

        # venue = to_one_hot(label['paper'])
        year = to_one_hot(graph.nodes[TGT_NTYPE].data['year'])
        numerical_feat = graph.nodes[TGT_NTYPE].data['feat']
        numerical = normalize_numerical(numerical_feat).to_sparse_coo()
        numerical_t = normalize_numerical(numerical_feat.T).to_sparse_coo()
        data_dict.update(
            {
                ('paper', 'published-in-year', 'year'): (year.row, year.col),
                ('year', 'year-of-publication', 'paper'): (year.col, year.row),
                # ('paper', 'published-in-venue', 'venue'): (venue.row, venue.col),
                # ('venue', 'venue-of-publication', 'paper'): (venue.col, venue.row),
                ('paper', 'has-numerical', 'numerical'):
                tuple(numerical_t.T.coalesce().indices()),
                ('numerical', 'is-numerical-of', 'paper'):
                tuple(numerical.T.coalesce().indices())
            }
        )
        hg: dgl.DGLHeteroGraph = dgl.heterograph(data_dict)

        # NOTE: only papers have features, and the "year" is not used (following ogb example)
        hg.edges['has-numerical'].data['weight'] = numerical_t.T.coalesce(
        ).values()
        hg.edges['is-numerical-of'].data['weight'] = numerical.T.coalesce(
        ).values()
        # hg.nodes[TGT_NTYPE].data['feat'] = graph.nodes[TGT_NTYPE].data['feat']
        hg.nodes[TGT_NTYPE].data['label'] = label[TGT_NTYPE].squeeze()

        for split in ('train', 'val', 'test'):
            hg.nodes[TGT_NTYPE].data[f'{split}_mask'] = torch.zeros(
                (hg.num_nodes(TGT_NTYPE), ), dtype=torch.bool
            )
            key = split if split != 'val' else 'valid'
            hg.nodes[TGT_NTYPE].data[f'{split}_mask'][split_idx[key][TGT_NTYPE]
                                                      ] = 1

        self.graph = hg
        return


NormalizedMAGDataset = AtomicMAGDataset
