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
    'HGBRe':
    'https://www.dropbox.com/scl/fi/odyt33yvfglpprahjar27/PubMedRe.zip?rlkey=dla60x8ujowkqfb3281b62zix&st=8tije33g&dl=0',
    'HGB':
    'https://www.dropbox.com/scl/fi/cuczmrux4n6sais68tvfh/PubMed.zip?rlkey=gx9ev8qxevq6mpnzdytlrlmjf&st=43dl8lvj&dl=0',
}


class PubMedDataset(BaseHGBLinkPredictionDataset):
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

    name: ClassVar[str] = 'pubmed'
    ntypes: ClassVar[list[NType]] = ['species', 'disease', 'chemical', 'gene']
    canonical_etypes: ClassVar[list[CEType]] = [
        ('species', 'species-species', 'species'),
        ('species', 'species-disease', 'disease'),
        ('disease', 'disease-disease', 'disease'),
        ('chemical', 'chemical-species', 'species'),
        ('chemical', 'chemical-disease', 'disease'),
        ('chemical', 'chemical-chemical', 'chemical'),
        ('chemical', 'chemical-gene', 'gene'),
        ('gene', 'gene-species', 'species'),
        ('gene', 'gene-disease', 'disease'),
        ('gene', 'gene-gene', 'gene'),
        # # inv etypes
        ('species', 'species-species-inv', 'species'),
        ('disease', 'disease-species', 'species'),
        ('disease', 'disease-disease-inv', 'disease'),
        ('species', 'species-chemical', 'chemical'),
        ('disease', 'disease-chemical', 'chemical'),
        ('chemical', 'chemical-chemical-inv', 'chemical'),
        ('gene', 'gene-chemical', 'chemical'),
        ('species', 'species-gene', 'gene'),
        ('disease', 'disease-gene', 'gene'),
        ('gene', 'gene-gene-inv', 'gene'),
    ]
    inverse_etypes: ClassVar[list[CEType]] = [
        ('species', 'species-species-inv', 'species'),
        ('disease', 'disease-species', 'species'),
        ('disease', 'disease-disease-inv', 'disease'),
        ('species', 'species-chemical', 'chemical'),
        ('disease', 'disease-chemical', 'chemical'),
        ('chemical', 'chemical-chemical-inv', 'chemical'),
        ('gene', 'gene-chemical', 'chemical'),
        ('species', 'species-gene', 'gene'),
        ('disease', 'disease-gene', 'gene'),
        ('gene', 'gene-gene-inv', 'gene'),
        #
        ('species', 'species-species', 'species'),
        ('species', 'species-disease', 'disease'),
        ('disease', 'disease-disease', 'disease'),
        ('chemical', 'chemical-species', 'species'),
        ('chemical', 'chemical-disease', 'disease'),
        ('chemical', 'chemical-chemical', 'chemical'),
        ('chemical', 'chemical-gene', 'gene'),
        ('gene', 'gene-species', 'species'),
        ('gene', 'gene-disease', 'disease'),
        ('gene', 'gene-gene', 'gene'),
    ]
    target_etypes: ClassVar[list[CEType]] = [
        ('disease', 'disease-disease', 'disease')
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

    def process(self):
        if self.has_cache():
            return self.load()
        return super().process()

    @property
    def symmetric_(self):
        """Merge:
        - chemical-chemical-inv into chemical-chemical
        - disease-disease-inv into disease-disease
        - gene-gene-inv into gene-gene
        - species-species-inv into species-species

        This is an inplace operation"""

        def _merge(hg: dgl.DGLHeteroGraph):
            if 'chemical-chemical-inv' in hg.etypes:
                hg = merge_etypes(
                    hg,
                    'chemical-chemical',
                    etype_to_drop='chemical-chemical-inv',
                )
            if 'disease-disease-inv' in hg.etypes:
                hg = merge_etypes(
                    hg,
                    'disease-disease',
                    etype_to_drop='disease-disease-inv',
                )
            if 'gene-gene-inv' in hg.etypes:
                hg = merge_etypes(
                    hg,
                    'gene-gene',
                    etype_to_drop='gene-gene-inv',
                )
            if 'species-species-inv' in hg.etypes:
                hg = merge_etypes(
                    hg,
                    'species-species',
                    etype_to_drop='species-species-inv',
                )
            return hg

        self.graph = _merge(self.graph)
        self.val_graph = _merge(self.val_graph)
        self.canonical_etypes = [
            e for e in self.canonical_etypes if e[1] not in (
                'chemical-chemical-inv', 'disease-disease-inv',
                'gene-gene-inv', 'species-species-inv'
            )
        ]
        return self


class AtomicPubMedDataset(PubMedDataset):
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

    name: ClassVar[str] = 'npubmed'
    ntypes: ClassVar[list[NType]] = [
        'species', 'disease', 'chemical', 'gene', 'species_feat',
        'disease_feat', 'chemical_feat', 'gene_feat'
    ]
    canonical_etypes: ClassVar[list[CEType]] = [
        ('species', 'species-species', 'species'),
        ('species', 'species-disease', 'disease'),
        ('disease', 'disease-disease', 'disease'),
        ('chemical', 'chemical-species', 'species'),
        ('chemical', 'chemical-disease', 'disease'),
        ('chemical', 'chemical-chemical', 'chemical'),
        ('chemical', 'chemical-gene', 'gene'),
        ('gene', 'gene-species', 'species'),
        ('gene', 'gene-disease', 'disease'),
        ('gene', 'gene-gene', 'gene'),
        ('species', 'species-has-feat', 'species_feat'),
        ('disease', 'disease-has-feat', 'disease_feat'),
        ('chemical', 'chemical-has-feat', 'chemical_feat'),
        ('gene', 'gene-has-feat', 'gene_feat'),
        # # inv etypes
        ('species', 'species-species-inv', 'species'),
        ('disease', 'disease-species', 'species'),
        ('disease', 'disease-disease-inv', 'disease'),
        ('species', 'species-chemical', 'chemical'),
        ('disease', 'disease-chemical', 'chemical'),
        ('chemical', 'chemical-chemical-inv', 'chemical'),
        ('gene', 'gene-chemical', 'chemical'),
        ('species', 'species-gene', 'gene'),
        ('disease', 'disease-gene', 'gene'),
        ('gene', 'gene-gene-inv', 'gene'),
        ('species_feat', 'feat-of-species', 'species'),
        ('disease_feat', 'feat-of-disease', 'disease'),
        ('chemical_feat', 'feat-of-chemical', 'chemical'),
        ('gene_feat', 'feat-of-gene', 'gene'),
    ]
    inverse_etypes: ClassVar[list[CEType]] = [
        # # inv etypes
        ('species', 'species-species-inv', 'species'),
        ('disease', 'disease-species', 'species'),
        ('disease', 'disease-disease-inv', 'disease'),
        ('species', 'species-chemical', 'chemical'),
        ('disease', 'disease-chemical', 'chemical'),
        ('chemical', 'chemical-chemical-inv', 'chemical'),
        ('gene', 'gene-chemical', 'chemical'),
        ('species', 'species-gene', 'gene'),
        ('disease', 'disease-gene', 'gene'),
        ('gene', 'gene-gene-inv', 'gene'),
        ('species_feat', 'feat-of-species', 'species'),
        ('disease_feat', 'feat-of-disease', 'disease'),
        ('chemical_feat', 'feat-of-chemical', 'chemical'),
        ('gene_feat', 'feat-of-gene', 'gene'),
        #
        ('species', 'species-species', 'species'),
        ('species', 'species-disease', 'disease'),
        ('disease', 'disease-disease', 'disease'),
        ('chemical', 'chemical-species', 'species'),
        ('chemical', 'chemical-disease', 'disease'),
        ('chemical', 'chemical-chemical', 'chemical'),
        ('chemical', 'chemical-gene', 'gene'),
        ('gene', 'gene-species', 'species'),
        ('gene', 'gene-disease', 'disease'),
        ('gene', 'gene-gene', 'gene'),
        ('species', 'species-has-feat', 'species_feat'),
        ('disease', 'disease-has-feat', 'disease_feat'),
        ('chemical', 'chemical-has-feat', 'chemical_feat'),
        ('gene', 'gene-has-feat', 'gene_feat'),
    ]
    target_etypes: ClassVar[list[CEType]] = [
        ('disease', 'disease-disease', 'disease')
    ]
    _saved_graph_name: ClassVar[str] = 'ngraph.bin'
    dense_etypes: ClassVar[list[CEType]] = [
        ('chemical', 'chemical-has-feat', 'chemical_feat'),
        ('chemical_feat', 'feat-of-chemical', 'chemical'),
        ('disease', 'disease-has-feat', 'disease_feat'),
        ('disease_feat', 'feat-of-disease', 'disease'),
        ('gene', 'gene-has-feat', 'gene_feat'),
        ('gene_feat', 'feat-of-gene', 'gene'),
        ('species', 'species-has-feat', 'species_feat'),
        ('species_feat', 'feat-of-species', 'species'),
    ]

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

    def _try_load_super(self):
        path = os.path.join(self.save_path, PubMedDataset._saved_graph_name)
        if not os.path.exists(path):
            return None
        [graph, val_graph, test_graph, neg_test_graph], _ = load_graphs(path)
        return [graph, val_graph, None, test_graph, neg_test_graph]

    def process(self):
        """process raw data to graphs, labels, splitting masks
        """
        graphs = self._try_load_super()
        if graphs is None:
            graphs = PubMedDataset._process(self.raw_path)
        [
            self.graph, self.val_graph, self.neg_val_graph, self.test_graph,
            self.neg_test_graph
        ] = graphs
        self._vanilla_graph = self.graph
        hg = self.graph

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
        num_nodes = {ntype: hg.num_nodes(ntype) for ntype in hg.ntypes}

        def add(ntype):
            numerical_data = hg.nodes[ntype].data['feat']
            numerical = normalize_numerical(numerical_data).to_sparse_coo()
            numerical_t = normalize_numerical(numerical_data.T).to_sparse_coo()
            etypes = [
                (ntype, f'{ntype}-has-feat', f'{ntype}_feat'),
                (f'{ntype}_feat', f'feat-of-{ntype}', ntype),
            ]
            data_dict[etypes[0]] = tuple(numerical_t.T.coalesce().indices())
            data_dict[etypes[1]] = tuple(numerical.T.coalesce().indices())
            edata[etypes[0]] = numerical_t.T.coalesce().values()
            edata[etypes[1]] = numerical.T.coalesce().values()
            num_nodes[f'{ntype}_feat'] = numerical_data.shape[1]
            return

        for ntype in hg.ntypes:
            add(ntype)
        new_hg = dgl.heterograph(data_dict)
        for etype, weight in edata.items():
            new_hg.edges[etype].data['weight'] = weight
        self.graph = new_hg
        return


NormalizedPubMedDataset = AtomicPubMedDataset
