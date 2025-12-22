from __future__ import annotations

import os
from typing import ClassVar

import dgl
import torch

from ...transforms import merge_etypes
from ...type import CEType, NType
from ..base.base_hetero_dataset import BaseHeteroDGLDataset
from ..raw_data_parsers import process_raw_data
from ..shared.utils import load_graphs, mat2tensor, save_graphs
from .freebase_schema import FreebaseSchema

VARIANTS = {
    'HGB':
    'https://www.dropbox.com/scl/fi/q6xluabh1m0a5vb9arvwq/Freebase.zip?rlkey=50f5305g4ah0phg0ktm3osckq&st=m4oppz36&dl=0'
}


class FreebaseDataset(BaseHeteroDGLDataset):
    """ Heterogeneous Freebase Dataset

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

    name: ClassVar[str] = 'freebase'
    ntypes: ClassVar[list[NType]] = [
        'book', 'business', 'film', 'location', 'music', 'organization',
        'people', 'sports'
    ]
    canonical_etypes: ClassVar[list[CEType]] = [
        ('book', 'book-about-organization', 'organization'),
        ('organization', 'book-about-organization-inv', 'book'),
        ('book', 'book-and-book', 'book'),
        ('book', 'book-and-book-inv', 'book'),
        ('book', 'book-on-location', 'location'),
        ('location', 'book-on-location-inv', 'book'),
        ('book', 'book-on-sports', 'sports'),
        ('sports', 'book-on-sports-inv', 'book'),
        ('book', 'book-to-film', 'film'), ('film', 'book-to-film-inv', 'book'),
        ('business', 'business-about-book', 'book'),
        ('book', 'business-about-book-inv', 'business'),
        ('business', 'business-about-film', 'film'),
        ('film', 'business-about-film-inv', 'business'),
        ('business', 'business-about-music', 'music'),
        ('music', 'business-about-music-inv', 'business'),
        ('business', 'business-about-sports', 'sports'),
        ('sports', 'business-about-sports-inv', 'business'),
        ('business', 'business-and-business', 'business'),
        ('business', 'business-and-business-inv', 'business'),
        ('business', 'business-on-location', 'location'),
        ('location', 'business-on-location-inv', 'business'),
        ('film', 'film-and-film', 'film'),
        ('film', 'film-and-film-inv', 'film'),
        ('location', 'location-and-location', 'location'),
        ('location', 'location-and-location-inv', 'location'),
        ('location', 'location-in-film', 'film'),
        ('film', 'location-in-film-inv', 'location'),
        ('music', 'music-and-music', 'music'),
        ('music', 'music-and-music-inv', 'music'),
        ('music', 'music-for-sports', 'sports'),
        ('sports', 'music-for-sports-inv', 'music'),
        ('music', 'music-in-book', 'book'),
        ('book', 'music-in-book-inv', 'music'),
        ('music', 'music-in-film', 'film'),
        ('film', 'music-in-film-inv', 'music'),
        ('music', 'music-on-location', 'location'),
        ('location', 'music-on-location-inv', 'music'),
        ('organization', 'organization-and-organization', 'organization'),
        ('organization', 'organization-and-organization-inv', 'organization'),
        ('organization', 'organization-for-business', 'business'),
        ('business', 'organization-for-business-inv', 'organization'),
        ('organization', 'organization-in-film', 'film'),
        ('film', 'organization-in-film-inv', 'organization'),
        ('organization', 'organization-on-location', 'location'),
        ('location', 'organization-on-location-inv', 'organization'),
        ('organization', 'organization-to-music', 'music'),
        ('music', 'organization-to-music-inv', 'organization'),
        ('organization', 'organization-to-sports', 'sports'),
        ('sports', 'organization-to-sports-inv', 'organization'),
        ('people', 'people-and-people', 'people'),
        ('people', 'people-and-people-inv', 'people'),
        ('people', 'people-in-business', 'business'),
        ('business', 'people-in-business-inv', 'people'),
        ('people', 'people-in-organization', 'organization'),
        ('organization', 'people-in-organization-inv', 'people'),
        ('people', 'people-on-location', 'location'),
        ('location', 'people-on-location-inv', 'people'),
        ('people', 'people-to-book', 'book'),
        ('book', 'people-to-book-inv', 'people'),
        ('people', 'people-to-film', 'film'),
        ('film', 'people-to-film-inv', 'people'),
        ('people', 'people-to-music', 'music'),
        ('music', 'people-to-music-inv', 'people'),
        ('people', 'people-to-sports', 'sports'),
        ('sports', 'people-to-sports-inv', 'people'),
        ('sports', 'sports-and-sports', 'sports'),
        ('sports', 'sports-and-sports-inv', 'sports'),
        ('sports', 'sports-in-film', 'film'),
        ('film', 'sports-in-film-inv', 'sports'),
        ('sports', 'sports-on-location', 'location'),
        ('location', 'sports-on-location-inv', 'sports')
    ]
    inverse_etypes: ClassVar[list[CEType]] = [
        ('organization', 'book-about-organization-inv', 'book'),
        ('book', 'book-about-organization', 'organization'),
        ('book', 'book-and-book-inv', 'book'),
        ('book', 'book-and-book', 'book'),
        ('location', 'book-on-location-inv', 'book'),
        ('book', 'book-on-location', 'location'),
        ('sports', 'book-on-sports-inv', 'book'),
        ('book', 'book-on-sports', 'sports'),
        ('film', 'book-to-film-inv', 'book'), ('book', 'book-to-film', 'film'),
        ('book', 'business-about-book-inv', 'business'),
        ('business', 'business-about-book', 'book'),
        ('film', 'business-about-film-inv', 'business'),
        ('business', 'business-about-film', 'film'),
        ('music', 'business-about-music-inv', 'business'),
        ('business', 'business-about-music', 'music'),
        ('sports', 'business-about-sports-inv', 'business'),
        ('business', 'business-about-sports', 'sports'),
        ('business', 'business-and-business-inv', 'business'),
        ('business', 'business-and-business', 'business'),
        ('location', 'business-on-location-inv', 'business'),
        ('business', 'business-on-location', 'location'),
        ('film', 'film-and-film-inv', 'film'),
        ('film', 'film-and-film', 'film'),
        ('location', 'location-and-location-inv', 'location'),
        ('location', 'location-and-location', 'location'),
        ('film', 'location-in-film-inv', 'location'),
        ('location', 'location-in-film', 'film'),
        ('music', 'music-and-music-inv', 'music'),
        ('music', 'music-and-music', 'music'),
        ('sports', 'music-for-sports-inv', 'music'),
        ('music', 'music-for-sports', 'sports'),
        ('book', 'music-in-book-inv', 'music'),
        ('music', 'music-in-book', 'book'),
        ('film', 'music-in-film-inv', 'music'),
        ('music', 'music-in-film', 'film'),
        ('location', 'music-on-location-inv', 'music'),
        ('music', 'music-on-location', 'location'),
        ('organization', 'organization-and-organization-inv', 'organization'),
        ('organization', 'organization-and-organization', 'organization'),
        ('business', 'organization-for-business-inv', 'organization'),
        ('organization', 'organization-for-business', 'business'),
        ('film', 'organization-in-film-inv', 'organization'),
        ('organization', 'organization-in-film', 'film'),
        ('location', 'organization-on-location-inv', 'organization'),
        ('organization', 'organization-on-location', 'location'),
        ('music', 'organization-to-music-inv', 'organization'),
        ('organization', 'organization-to-music', 'music'),
        ('sports', 'organization-to-sports-inv', 'organization'),
        ('organization', 'organization-to-sports', 'sports'),
        ('people', 'people-and-people-inv', 'people'),
        ('people', 'people-and-people', 'people'),
        ('business', 'people-in-business-inv', 'people'),
        ('people', 'people-in-business', 'business'),
        ('organization', 'people-in-organization-inv', 'people'),
        ('people', 'people-in-organization', 'organization'),
        ('location', 'people-on-location-inv', 'people'),
        ('people', 'people-on-location', 'location'),
        ('book', 'people-to-book-inv', 'people'),
        ('people', 'people-to-book', 'book'),
        ('film', 'people-to-film-inv', 'people'),
        ('people', 'people-to-film', 'film'),
        ('music', 'people-to-music-inv', 'people'),
        ('people', 'people-to-music', 'music'),
        ('sports', 'people-to-sports-inv', 'people'),
        ('people', 'people-to-sports', 'sports'),
        ('sports', 'sports-and-sports-inv', 'sports'),
        ('sports', 'sports-and-sports', 'sports'),
        ('film', 'sports-in-film-inv', 'sports'),
        ('sports', 'sports-in-film', 'film'),
        ('location', 'sports-on-location-inv', 'sports'),
        ('sports', 'sports-on-location', 'location')
    ]

    graph: FreebaseSchema
    _saved_graph_name: ClassVar[str] = 'graph.bin'
    variants: ClassVar[list[str]] = list(VARIANTS)

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

    @property
    def symmetric_(self):
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
            self.graph = merge_etypes(self.graph, etype=e1, etype_to_drop=e2)
        return self

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

        g: FreebaseSchema

        g.nodes['book'].data['train_mask'] =\
            g.nodes['book'].data['train_mask'].bool()
        g.nodes['book'].data['val_mask'] =\
            g.nodes['book'].data['val_mask'].bool()
        g.nodes['book'].data['test_mask'] =\
            g.nodes['book'].data['test_mask'].bool()

        self.graph: FreebaseSchema = g
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
        unlabeled_mask = labels.sum(1) == 0
        labels = torch.argmax(labels, dim=1)
        labels[unlabeled_mask] = -1  # set unlabeled nodes to -1

        ################################
        # Node type
        node_dict = {
            'book': 0,
            'film': 1,
            'music': 2,
            'sports': 3,
            'people': 4,
            'location': 5,
            'organization': 6,
            'business': 7,
        }
        # yapf: disable
        edge_dict = { # pylint: disable=unused-variable,line-too-long
            'book-and-book': ('book', 'book-and-book', 'book'),
            'book-to-film': ('book', 'book-to-film', 'film'),
            'book-on-sports': ('book', 'book-on-sports', 'sports'),
            'book-on-location': ('book', 'book-on-location', 'location'),
            'book-about-organization': ('book', 'book-about-organization', 'organization'),
            'film-and-film': ('film', 'film-and-film', 'film'),
            'music-in-book': ('music', 'music-in-book', 'book'),
            'music-in-film': ('music', 'music-in-film', 'film'),
            'music-and-music': ('music', 'music-and-music', 'music'),
            'music-for-sports': ('music', 'music-for-sports', 'sports'),
            'music-on-location': ('music', 'music-on-location', 'location'),
            'sports-in-film': ('sports', 'sports-in-film', 'film'),
            'sports-and-sports': ('sports', 'sports-and-sports', 'sports'),
            'sports-on-location': ('sports', 'sports-on-location', 'location'),
            'people-to-book': ('people', 'people-to-book', 'book'),
            'people-to-film': ('people', 'people-to-film', 'film'),
            'people-to-music': ('people', 'people-to-music', 'music'),
            'people-to-sports': ('people', 'people-to-sports', 'sports'),
            'people-and-people': ('people', 'people-and-people', 'people'),
            'people-on-location': ('people', 'people-on-location', 'location'),
            'people-in-organization': ('people', 'people-in-organization', 'organization'),
            'people-in-business': ('people', 'people-in-business', 'business'),
            'location-in-film': ('location', 'location-in-film', 'film'),
            'location-and-location': ('location', 'location-and-location', 'location'),
            'organization-in-film': ('organization', 'organization-in-film', 'film'),
            'organization-to-music': ('organization', 'organization-to-music', 'music'),
            'organization-to-sports': ('organization', 'organization-to-sports', 'sports'),
            'organization-on-location': ('organization', 'organization-on-location', 'location'),
            'organization-and-organization': ('organization', 'organization-and-organization', 'organization'),
            'organization-for-business': ('organization', 'organization-for-business', 'business'),
            'business-about-book': ('business', 'business-about-book', 'book'),
            'business-about-film': ('business', 'business-about-film', 'film'),
            'business-about-music': ('business', 'business-about-music', 'music'),
            'business-about-sports': ('business', 'business-about-sports', 'sports'),
            'business-on-location': ('business', 'business-on-location', 'location'),
            'business-and-business': ('business', 'business-and-business', 'business'),
        }
        # yapf: enable

        def get_slice(ntype):
            end_id = node_dict[ntype] + 1
            if end_id >= len(ptr):
                return slice(ptr[node_dict[ntype]], None)
            return slice(ptr[node_dict[ntype]], ptr[node_dict[ntype] + 1])

        data_dict = {
            (stype, etype, dtype):
            adj[get_slice(stype), get_slice(dtype)].nonzero()
            for (stype, etype, dtype) in edge_dict.values()
        }
        # Add inverse edges
        for stype, etype, dtype in edge_dict.values():
            data_dict[(dtype, f'{etype}-inv', stype)] = \
                adj[get_slice(stype), get_slice(dtype)].transpose().nonzero()

        hg: FreebaseSchema = dgl.heterograph(data_dict)
        TARGET_NODE_TYPE = 'book'

        # NO feature in Freebase
        # for idx, ntype in enumerate(node_dict.keys()):
        #     hg.nodes[ntype].data['feat'] = features[idx]

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
