from __future__ import annotations

import os
from typing import ClassVar

import dgl
import torch

from ....transforms import merge_etypes
from ....type import CEType, NType
from ...shared.utils import load_graphs, save_graphs
from ..base import BaseHGBLinkPredictionDataset

VARIANTS = {
    'HGB':
    'https://www.dropbox.com/scl/fi/gknw5xejphhtks5xa9jff/amazon.zip?rlkey=l3vzbkomndt157o1owt9719t6&st=dcxbo2ut&dl=0'
}


class AmazonDataset(BaseHGBLinkPredictionDataset):
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

    name: ClassVar[str] = 'amazon'
    ntypes: ClassVar[list[NType]] = ['product']
    canonical_etypes: ClassVar[list[CEType]] = [
        ('product', 'co-view', 'product'),
        ('product', 'co-purchase', 'product'),
        # inv etypes
        ('product', 'co-view-inv', 'product'),
        ('product', 'co-purchase-inv', 'product'),
    ]
    inverse_etypes: ClassVar[list[CEType]] = [
        ('product', 'co-view-inv', 'product'),
        ('product', 'co-purchase-inv', 'product'),
        # inv etypes
        ('product', 'co-view', 'product'),
        ('product', 'co-purchase', 'product'),
    ]
    target_etypes: ClassVar[list[CEType]] = [
        ('product', 'co-view', 'product'),
        ('product', 'co-purchase', 'product')
    ]
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
        """Merge co-view-inv into co-view; co-purchase-inv into co-purchase.
        This is an inplace operation"""

        def _merge(hg):
            hg = merge_etypes(
                hg,
                'co-purchase',
                etype_to_drop='co-purchase-inv',
            )
            hg = merge_etypes(
                hg,
                'co-view',
                etype_to_drop='co-view-inv',
            )
            return hg

        self.graph = _merge(self.graph)
        self.val_graph = _merge(self.val_graph)
        self.canonical_etypes = [
            e for e in self.canonical_etypes
            if e[1] not in ('co-view-inv', 'co-purchase-inv')
        ]
        return self

    @classmethod
    def _add_dummy_ntype(cls, hg):
        if hg is None:
            return hg
        num_nodes_dict = {'product': hg.num_nodes(), 'dummy': 1}
        data_dict = {
            etype: hg.edges(etype=etype)
            for etype in hg.canonical_etypes
        }
        feat = hg.ndata.pop('feat', None)
        assert not bool(hg.edata)
        assert not bool(hg.ndata)
        new_hg = dgl.heterograph(data_dict, num_nodes_dict)
        if feat is not None:
            new_hg.nodes['product'].data['feat'] = feat
        return new_hg

    def process(self):
        """process raw data to graphs, labels, splitting masks
        """
        graphs = AmazonDataset._process(self.raw_path)

        [
            self.graph, self.val_graph, self.neg_val_graph, self.test_graph,
            self.neg_test_graph
        ] = map(self._add_dummy_ntype, graphs)
        self.graph.nodes['dummy'].data['feat'] = torch.ones((1, 1))
        return


class AtomicAmazonDataset(AmazonDataset):
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

    name: ClassVar[str] = 'namazon'
    ntypes: ClassVar[list[NType]] = [
        'product', 'price', 'sales_rank', 'brand', 'category'
    ]
    canonical_etypes: ClassVar[list[CEType]] = [
        ('product', 'co-view', 'product'),
        ('product', 'co-purchase', 'product'),
        ('product', 'product-price', 'price'),
        ('product', 'product-sales_rank', 'sales_rank'),
        ('product', 'product-brand', 'brand'),
        ('product', 'product-category', 'category'),
        # inv etypes
        ('product', 'co-view-inv', 'product'),
        ('product', 'co-purchase-inv', 'product'),
        ('price', 'price-product', 'product'),
        ('sales_rank', 'sales_rank-product', 'product'),
        ('brand', 'brand-product', 'brand'),
        ('category', 'category-product', 'product'),
    ]
    inverse_etypes: ClassVar[list[CEType]] = [
        ('product', 'co-view-inv', 'product'),
        ('product', 'co-purchase-inv', 'product'),
        ('price', 'price-product', 'product'),
        ('sales_rank', 'sales_rank-product', 'product'),
        ('brand', 'brand-product', 'brand'),
        ('category', 'category-product', 'product'),
        #
        ('product', 'co-view', 'product'),
        ('product', 'co-purchase', 'product'),
        ('product', 'product-price', 'price'),
        ('product', 'product-sales_rank', 'sales_rank'),
        ('product', 'product-brand', 'brand'),
        ('product', 'product-category', 'category'),
    ]
    target_etypes: ClassVar[list[CEType]] = [
        ('product', 'co-view', 'product'),
        ('product', 'co-purchase', 'product')
    ]
    _saved_graph_name: ClassVar[str] = 'ngraph.bin'
    vanilla_graph: dgl.DGLHeteroGraph

    def save(self):
        # save processed data to directory `self.save_path`
        path = os.path.join(self.save_path, self._saved_graph_name)
        save_graphs(
            path, [
                self.graph, self.val_graph, self.test_graph,
                self.neg_test_graph, self._vanilla_graph
            ]
        )
        return

    def load(self):
        # load processed data from directory `self.save_path`

        path = os.path.join(self.save_path, self._saved_graph_name)
        [
            self.graph, self.val_graph, self.test_graph, self.neg_test_graph,
            self._vanilla_graph
        ], _ = load_graphs(path)
        self.neg_val_graph = None
        return

    def process(self):
        """process raw data to graphs, labels, splitting masks
        """
        graphs = list(
            map(
                AmazonDataset._add_dummy_ntype,
                AmazonDataset._process(self.raw_path)
            )
        )
        [
            self.graph, self.val_graph, self.neg_val_graph, self.test_graph,
            self.neg_test_graph
        ] = graphs
        hg = self.graph
        self._vanilla_graph = hg

        # price: 0-1
        # sales-rank: 1~811
        # brand: 811~814
        # category: 814:

        feat = hg.nodes['product'].data['feat']
        price = torch.concat(
            [
                feat[:, :1],
                torch.full_like(feat[:, :1], feat[:, :1].abs().mean())
            ], dim=1
        ) + 1e-7  # add a dummy ones to avoid normalization problem

        def normalize_numerical(x: torch.Tensor):
            """ Normalize numerical tensor by row"""
            # NOTE: this is to align with the other adjacencies
            # so that mean reduce function can be used for all etypes.
            return x * x.shape[1] / x.abs().sum(dim=1, keepdim=True)

        data_dict = {
            cetype: hg.edges(etype=cetype)
            for cetype in hg.canonical_etypes
        }

        edata = {}
        num_nodes = {
            ntype: hg.num_nodes(ntype)
            for ntype in hg.ntypes if ntype != 'dummy'
        }

        numerical = normalize_numerical(price).to_sparse_coo()
        numerical_t = normalize_numerical(price.T).to_sparse_coo()
        data_dict['product', 'product-price',
                  'price'] = tuple(numerical_t.T.coalesce().indices())
        data_dict['price', 'price-product',
                  'product'] = tuple(numerical.T.coalesce().indices())
        edata['product', 'product-price',
              'price'] = numerical_t.T.coalesce().values()
        edata['price', 'price-product',
              'product'] = numerical.T.coalesce().values()
        num_nodes['price'] = price.shape[1]

        def add_onehot_feat(ntype, onehot_feat: torch.Tensor):
            etypes = [
                ('product', f'product-{ntype}', ntype),
                (ntype, f'{ntype}-product', 'product'),
            ]
            data_dict[etypes[0]] = tuple(onehot_feat.to_sparse_coo().indices())
            data_dict[etypes[1]
                      ] = tuple(onehot_feat.T.to_sparse_coo().indices())
            num_nodes[ntype] = onehot_feat.shape[1]
            return

        add_onehot_feat('sales_rank', feat[:, 1:811])
        add_onehot_feat('brand', feat[:, 811:814])
        add_onehot_feat('category', feat[:, 814:])
        new_hg = dgl.heterograph(data_dict, num_nodes)
        for etype, weight in edata.items():
            new_hg.edges[etype].data['weight'] = weight
        self.graph = new_hg
        return


NormalizedAmazonDataset = AtomicAmazonDataset
