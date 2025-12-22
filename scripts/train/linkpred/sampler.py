from __future__ import annotations

import os
import warnings

import dgl
import torch
from dgl.dataloading.negative_sampler import _BaseNegativeSampler

from dhgl import transforms
from dhgl.type import CEType


class NHopNegativeSampler(_BaseNegativeSampler):

    def __init__(
        self,
        hgs: list[dgl.DGLHeteroGraph],
        target_etypes: list[CEType],
        n_hops: int,
        k: int,
        cache_dir: str | None = None,
        verbose=False,
    ):
        device = hgs[0].device
        hgs = [hg.cpu() for hg in hgs]
        self.cache_dir = cache_dir
        self.data = self._init_data(
            hgs, target_etypes, n_hops, cache_dir, verbose
        )
        self.k = k
        self.to(device)
        return

    @classmethod
    def _init_data(
        cls,
        hgs: list[dgl.DGLHeteroGraph],
        target_etypes: list[CEType],
        n_hops: int,
        cache_dir: str | None = None,
        verbose=False,
    ):
        if cache_dir is not None:
            data = cls._find_cache(cache_dir, hgs, target_etypes, n_hops)
            if data is not None:
                if verbose:
                    print(f'Cache found in {cache_dir}')
                return data
        data = cls._build(hgs, target_etypes, n_hops)
        if cache_dir is not None:
            cls._save_cache(cache_dir, hgs, target_etypes, n_hops, data)
            if verbose:
                print(f'Cache saved in {cache_dir}')
        return data

    @classmethod
    def _cache_name(
        cls,
        hgs: list[dgl.DGLHeteroGraph],
        target_etypes: list[CEType],
        n_hops: int,
    ):
        import hashlib
        sha1 = hashlib.sha1(''.join(map(str, hgs)).encode()).hexdigest()
        etypes_str = '-'.join(e[1] for e in target_etypes)
        return f'{etypes_str}_{n_hops}_{sha1[:10]}'

    @classmethod
    def _find_cache(
        cls,
        cache_dir: str,
        hgs: list[dgl.DGLHeteroGraph],
        target_etypes: list[CEType],
        n_hops: int,
    ):
        cache_path = os.path.join(
            cache_dir, f'{cls._cache_name(hgs, target_etypes, n_hops)}.pt'
        )
        if os.path.exists(cache_path):
            return torch.load(cache_path)
        return None

    @classmethod
    def _save_cache(
        cls,
        cache_dir: str,
        hgs: list[dgl.DGLHeteroGraph],
        target_etypes: list[CEType],
        n_hops: int,
        data: dict,
    ):
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir, f'{cls._cache_name(hgs, target_etypes, n_hops)}.pt'
        )
        return torch.save(data, cache_path)

    @classmethod
    def _build(
        cls,
        hgs: list[dgl.DGLHeteroGraph],
        target_etypes: list[CEType],
        n_hops: int,
    ):

        def etype_align(hg_: dgl.DGLHeteroGraph):
            data_dict = {
                etype: hg_.edges(etype=etype)
                for etype in hg_.canonical_etypes
            }
            if len(hg_.canonical_etypes) == 1:
                warnings.warn(
                    f'Only one edge type: {hg_.canonical_etypes} detected. '
                    'Graph should include inverse edges and their edge type'
                )
            for etype in hgs[0].canonical_etypes:
                if etype not in data_dict:
                    data_dict[etype] = ([], [])
            return transforms.update_graph_structure(
                hg_, data_dict, copy_ndata=False, copy_edata=False
            )

        hgs = [hgs[0], *[etype_align(hg) for hg in hgs[1:]]]

        def filter_ndata(hg):
            if hg.edata['weight']:
                # NOTE: should use vanilla graph for nhop calculation to align with testing set
                raise NotImplementedError
            hg = hg.clone()
            for k in hg.ndata:
                hg.ndata.pop(k)
            return hg

        # if any(set(hg.canonical_etypes) != set(hgs[0].canonical_etypes) for hg in hgs):
        #     raise NotImplementedError('Only support')
        # hg: dgl.DGLHeteroGraph = dgl.merge(list(map(filter_ndata, hgs)))
        hgs = list(map(filter_ndata, hgs))
        hg: dgl.DGLHeteroGraph = dgl.merge(hgs) if len(hgs) > 1 else hgs[0]

        # hg: dgl.DGLHeteroGraph = dgl.merge(hgs)

        def get(target_etype: CEType) -> tuple[torch.Tensor, torch.Tensor]:
            """only support small graph now"""
            with hg.local_scope():
                dsttype = target_etype[-1]
                hs = {
                    dsttype:
                    torch.eye(hg.num_nodes(dsttype), device=hg.device)
                }
                for i in range(n_hops):
                    hg.ndata['x'] = hs
                    hg.multi_update_all(
                        {
                            etype: (
                                dgl.function.copy_u('x', 'm'),
                                dgl.function.sum('m', 'h')
                            )
                            for etype in hg.canonical_etypes if etype[0] in hs
                        }, cross_reducer='sum'
                    )
                    for ntype, h in hg.ndata.pop('h').items():
                        hs[ntype] = hs.get(ntype, 0) + h
                nhop_adj = hs[target_etype[0]] > 0
                adj = hg.adj_external(etype=target_etype).to_dense().bool()
                assert (nhop_adj | adj).sum() == nhop_adj.sum()
                nhop_adj = nhop_adj ^ adj.to_dense().bool()
                return nhop_adj.to_sparse_csr(), nhop_adj.sum(1)

        return {
            target_etype: get(target_etype)
            for target_etype in target_etypes
        }

    def to(self, device):
        self.data = {
            etype: (adj.to(device), length.to(device))
            for etype, (adj, length) in self.data.items()
        }
        return self

    def _generate(
        self, g: dgl.DGLHeteroGraph, eids, canonical_etype, seed=None
    ):
        adj, lengths = self.data[canonical_etype]
        src, _ = g.find_edges(eids, etype=canonical_etype)
        src = src.repeat(self.k)
        indptr = adj.crow_indices()[src]
        gen = None if seed is None else torch.Generator().manual_seed(seed)
        rand = torch.rand(len(src), device=src.device, generator=gen)
        increment = (lengths[src] * rand).long()
        dst = adj.col_indices()[indptr + increment]
        has_nhop_mask = lengths[src] > 0
        return src[has_nhop_mask], dst[has_nhop_mask]

    @classmethod
    def sample(
        cls,
        positive_hg: dgl.DGLHeteroGraph,
        seed: int,
        hgs: list[dgl.DGLHeteroGraph],
        target_etypes: list[CEType],
        n_hops: int,
        k: int,
        cache_dir: str | None = None,
        verbose=False,
    ):
        """Sample a static negative graph.
        This is slow and only supposed to be used to sample a static negative graph.

        Args:
            positive_hg (dgl.DGLHeteroGraph): _description_
            hgs (list[dgl.DGLHeteroGraph]): _description_
            target_etypes (list[CEType]): _description_
            n_hops (int): _description_
            k (int): _description_
            cache_dir (str | None, optional): _description_. Defaults to None.
            verbose (bool, optional): _description_. Defaults to False.

        Returns:
            The negative graph
        """
        sampler = cls(hgs, target_etypes, n_hops, k, cache_dir, verbose)

        # positive_hg = positive_hg.edge_type_subgraph(target_etypes)
        def generate(etype):
            return sampler._generate(
                positive_hg, positive_hg.edges('eid', etype=etype), etype,
                seed=seed
            )

        neg_edge_dict = {
            etype: generate(etype)
            for etype in positive_hg.canonical_etypes
        }
        num_nodes_dict = {
            ntype: positive_hg.num_nodes(ntype)
            for ntype in positive_hg.ntypes
        }
        negative_hg = dgl.heterograph(neg_edge_dict, num_nodes_dict)
        return negative_hg
