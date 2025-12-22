from __future__ import annotations

import os
from typing import ClassVar, TypeVar

import dgl
import numpy as np
import torch
from scipy.sparse import coo_matrix, spmatrix

from ...type import CEType, NType
from ..base.base_hetero_dataset import BaseHeteroDGLDataset
from ..shared.utils import load_graphs, save_graphs
from .hgb_data_loader import HGBDataLoader


class BaseLinkPredictionDataset(BaseHeteroDGLDataset):
    """

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

    name: ClassVar[str]
    ntypes: ClassVar[list[NType]]
    canonical_etypes: ClassVar[list[CEType]]
    target_etypes: ClassVar[list[CEType]]
    graph: dgl.DGLHeteroGraph
    val_graph: dgl.DGLHeteroGraph
    neg_val_graph: dgl.DGLHeteroGraph | None
    test_graph: dgl.DGLHeteroGraph
    neg_test_graph: dgl.DGLHeteroGraph | None

    @property
    def vanilla_graph(self) -> dgl.DGLHeteroGraph:
        """A copy of (training) graph that kept for the sake of alignment of negative sampling.
        For example, the negative sampling may sample negative pairs for two hop neighbors, but the
        (training) graph is mutable that resulting in change of negative sampling.
        """
        raise NotImplementedError

    @property
    def target_ntypes(self) -> set[NType]:
        return set(sum(([s, d] for s, _, d in self.target_etypes), []))

    _saved_graph_name: ClassVar[str] = 'graph.bin'

    def __init__(
        self,
        raw_path: str = None,
        raw_dir: str = None,
        save_dir: str = None,
        force_reload: bool = False,
        verbose: bool = False,
    ):

        super().__init__(
            raw_path=raw_path,
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

    @classmethod
    def get_inverse_etype(cls, etype: CEType) -> CEType:
        raise NotImplementedError

    def save(self):
        # save processed data to directory `self.save_path`
        raise NotImplementedError

    def load(self):
        # load processed data from directory `self.save_path`
        raise NotImplementedError

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        path = os.path.join(self.save_path, self._saved_graph_name)
        return os.path.exists(path)


LinkPredDatasetLike = TypeVar(
    'LinkPredDatasetLike', bound=BaseLinkPredictionDataset
)


class BaseHGBLinkPredictionDataset(BaseLinkPredictionDataset):
    """

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

    name: ClassVar[str]
    ntypes: ClassVar[list[NType]]
    canonical_etypes: ClassVar[list[CEType]]
    target_etypes: ClassVar[list[CEType]]
    _saved_graph_name: ClassVar[str] = 'graph.bin'

    def __init__(
        self,
        raw_path: str = None,
        raw_dir: str = None,
        save_dir: str = None,
        force_reload: bool = False,
        verbose: bool = False,
    ):

        super().__init__(
            raw_path=raw_path,
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
        self._vanilla_graph = self.graph.clone()
        save_graphs(
            path,
            [self.graph, self.val_graph, self.test_graph, self.neg_test_graph]
        )
        return

    def load(self):
        # load processed data from directory `self.save_path`

        path = os.path.join(self.save_path, self._saved_graph_name)
        [self.graph, self.val_graph, self.test_graph,
         self.neg_test_graph], _ = load_graphs(path)
        self.neg_val_graph = None
        self._vanilla_graph = self.graph.clone()
        return

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        path = os.path.join(self.save_path, self._saved_graph_name)
        return os.path.exists(path)

    @property
    def vanilla_graph(self):
        return self._vanilla_graph

    @classmethod
    def get_inverse_etype(cls, etype: CEType):
        n = len(cls.canonical_etypes) // 2
        i = (cls.canonical_etypes.index(etype) + n) % len(cls.canonical_etypes)
        return cls.canonical_etypes[i]

    @classmethod
    def _process(cls, raw_path: str):
        """process raw data to graphs, labels, splitting masks
        """

        data = HGBDataLoader(raw_path)

        ETYPES = cls.canonical_etypes[:len(data.links['meta'])]
        ETYPES_INV = cls.canonical_etypes[len(data.links['meta']):]

        def check_meta(meta: dict[int, tuple[int, int]]):
            for eid, (sid, did) in meta.items():
                srctype = cls.ntypes[sid]
                dsttype = cls.ntypes[did]
                if sid == did:
                    assert cls.canonical_etypes[eid][
                        0] == cls.canonical_etypes[eid][-1] == srctype
                    cls.canonical_etypes[eid] in (
                        (cls.ntypes[sid], cls.ntypes[did])
                    )
                else:
                    assert cls.canonical_etypes[eid] == (
                        srctype, f'{srctype}-{dsttype}', dsttype
                    )

        check_meta(data.links['meta'])

        def adj_transform(eid, adj):
            if isinstance(adj, spmatrix):
                adj: coo_matrix = adj.tocoo()
                edges = (adj.row, adj.col)
            else:
                edges = np.array(adj)
            stype_id, dtype_id = data.links['meta'][eid]
            edges = (
                edges[0] - data.nodes['shift'][stype_id],
                edges[1] - data.nodes['shift'][dtype_id]
            )
            return edges

        def add_inverse(data_dict: dict):
            for inv_etype, etype in zip(ETYPES_INV, ETYPES):
                data_dict[inv_etype] = data_dict[etype][::-1]
            return data_dict

        data_dict = {
            etype: adj_transform(eid, data.links['data'][eid])
            for eid, etype in enumerate(ETYPES)
        }

        num_nodes_dict = {
            ntype: data.nodes['count'][nid]
            for nid, ntype in enumerate(cls.ntypes)
        }
        hg: dgl.DGLHeteroGraph = dgl.heterograph(
            add_inverse(data_dict), num_nodes_dict
        )
        assert set(hg.canonical_etypes) == set(cls.canonical_etypes)

        for i, ntype in enumerate(cls.ntypes):
            feat = data.nodes['attr'][i]
            if feat is not None:
                hg.nodes[ntype].data['feat'] = torch.from_numpy(feat).float()

        # target_links = [
        #     cls.canonical_etypes[i] for i in data.links_test['meta']
        # ]
        # XXX: Why non-target edges are also splited into valid?????????????????
        # ????????????????????????????????????????????????????????????????????????????
        # ????????????????????????????????????????????????????????????????????????????
        # ????????????????????????????????????????????????????????????????????????????
        # ????????????????????????????????????????????????????????????????????????????
        # ????????????????????????????????????????????????????????????????????????????
        # ????????????????????????????????????????????????????????????????????????????
        valid_graph = dgl.heterograph(
            add_inverse(
                {
                    ETYPES[eid]: adj_transform(eid, adj)
                    for eid, adj in data.valid_pos.items()
                }
            ), num_nodes_dict
        )
        test_graph = dgl.heterograph(
            {
                ETYPES[eid]: adj_transform(eid, adj)
                for eid, adj in data.links_test['data'].items()
            }, num_nodes_dict
        )

        test_neg, test_label = data.get_test_neigh()
        neg_test_graph = dgl.heterograph(
            {
                ETYPES[eid]:
                adj_transform(
                    eid,
                    np.array(test_neg[eid])[:,
                                            np.array(test_label[eid]) == 0]
                )
                for eid in data.links_test['meta']
            }, num_nodes_dict
        )

        return (hg, valid_graph, None, test_graph, neg_test_graph)

    def process(self):
        """process raw data to graphs, labels, splitting masks
        """
        [
            self.graph, self.val_graph, self.neg_val_graph, self.test_graph,
            self.neg_test_graph
        ] = self._process(self.raw_path)
        self._vanilla_graph = self.graph.clone()
        return
