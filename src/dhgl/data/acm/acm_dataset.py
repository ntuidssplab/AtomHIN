from __future__ import annotations

import os
from typing import ClassVar

import dgl
import torch

from ...type import CEType, NType
from ..base.base_hetero_dataset import BaseHeteroDGLDataset
from ..raw_data_parsers import process_raw_data
from ..shared.utils import load_graphs, mat2tensor, save_graphs
from .acm_schema import ACMGraphSchema

VARIANTS = {
    'HGB':
    'https://www.dropbox.com/scl/fi/005py5oyfs15juq4leyxp/ACM.zip?rlkey=flpxp62bj264in2m07gnsviwk&st=2ipz5nug&dl=0'
}


class ACMDataset(BaseHeteroDGLDataset):
    """ Heterogeneous ACM Dataset

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

    name: ClassVar[str] = 'acm'
    graph: ACMGraphSchema
    _saved_graph_name: ClassVar[str] = 'graph.bin'
    variants: ClassVar[list[str]] = list(VARIANTS)
    ntypes: ClassVar[list[NType]] = ['author', 'paper', 'subject', 'term']
    canonical_etypes: ClassVar[list[CEType]] = [
        ('author', 'writing', 'paper'),
        ('paper', 'cited', 'paper'),
        ('paper', 'citing', 'paper'),
        ('paper', 'contains', 'term'),
        ('paper', 'is-about', 'subject'),
        ('paper', 'written-by', 'author'),
        ('subject', 'has', 'paper'),
        ('term', 'is-in', 'paper'),
    ]
    inverse_etypes: ClassVar[list[CEType]] = [
        ('paper', 'written-by', 'author'),
        ('paper', 'citing', 'paper'),
        ('paper', 'cited', 'paper'),
        ('term', 'is-in', 'paper'),
        ('subject', 'has', 'paper'),
        ('author', 'writing', 'paper'),
        ('paper', 'is-about', 'subject'),
        ('paper', 'contains', 'term'),
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

        g: ACMGraphSchema

        g.nodes['paper'].data['train_mask'] =\
            g.nodes['paper'].data['train_mask'].bool()
        g.nodes['paper'].data['val_mask'] =\
            g.nodes['paper'].data['val_mask'].bool()
        g.nodes['paper'].data['test_mask'] =\
            g.nodes['paper'].data['test_mask'].bool()

        self.graph: ACMGraphSchema = g
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
        # Node type
        node_dict = {'paper': 0, 'author': 1, 'subject': 2, 'term': 3}
        # Link_meta
        # {0: (0, 0), 1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (0, 2), 5: (2, 0), 6: (0, 3), 7: (3, 0)}
        edge_dict = { # pylint: disable=unused-variable
            'cited'     : 0,  # (paper, paper)
            'citing'    : 1,  # (paper, paper)
            'written-by': 2,  # (paper, author)
            'writing'   : 3,  # (author, paper)
            'is-about'  : 4,  # (paper, subject)
            'has'       : 5,  # (subject, paper)
            'contains'  : 6,  # (paper, term)
            'is-in'     : 7   # (term, paper)
        }

        # yapf: disable
        # pylint: disable=line-too-long
        data_dict = {
            ('paper',   'written-by', 'author') : adj[: ptr[1], ptr[1] : ptr[2]].nonzero(),
            ('author',  'writing',    'paper')  : adj[: ptr[1], ptr[1] : ptr[2]].transpose().nonzero(),
            ('paper',   'citing',     'paper')  : adj[: ptr[1], ptr[0] : ptr[1]].nonzero(),
            ('paper',   'cited',      'paper')  : adj[: ptr[1], ptr[0] : ptr[1]].transpose().nonzero(),
            ('paper',   'is-about',   'subject'): adj[: ptr[1], ptr[2] : ptr[3]].nonzero(),
            ('subject', 'has',        'paper')  : adj[: ptr[1], ptr[2] : ptr[3]].transpose().nonzero(),
            ('paper',   'contains',    'term')  : adj[: ptr[1], ptr[3] :       ].nonzero(),
            ('term',    'is-in',      'paper')  : adj[: ptr[1], ptr[3] :       ].transpose().nonzero(),
        }
        # yapf: enable
        hg: ACMGraphSchema = dgl.heterograph(data_dict)
        TARGET_NODE_TYPE = 'paper'

        for idx, ntype in enumerate(node_dict.keys()):
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


HeteroACMDataset = ACMDataset
