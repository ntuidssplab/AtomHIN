from __future__ import annotations

from typing import overload

import dgl
import torch
from torch.utils.data import Dataset

from dhgl.type import CEType


class LinkPredTaskDataset(Dataset):

    @overload
    def __init__(
        self,
        target_etypes: list[CEType],
        positive_hg: dgl.DGLHeteroGraph,
        negative_sampler,
    ):
        ...

    def __init__(
        self,
        target_etypes: list[CEType],
        positive_hg: dgl.DGLHeteroGraph,
        negative_hg: dgl.DGLHeteroGraph,
    ):
        positive_hg = positive_hg.edge_type_subgraph(target_etypes)
        self.positive_hg: dgl.DGLHeteroGraph = positive_hg
        if isinstance(negative_hg, dgl.DGLGraph):
            self.negative_hg = negative_hg.edge_type_subgraph(target_etypes)
        else:
            self.negative_hg = None
            self.neg_sampler = negative_hg
            eids = [
                positive_hg.edges(etype=etype, form='eid')
                for etype in positive_hg.canonical_etypes
            ]
            etypes = [
                torch.full_like(eids, etype)
                for etype, eids in enumerate(eids)
            ]
            self.to_eids = torch.stack(list(map(torch.concat, [etypes, eids])))
            self.num_nodes_dict = {
                ntype: positive_hg.num_nodes(ntype)
                for ntype in positive_hg.ntypes
            }
        return

    def __getitems__(self, indices):
        assert not isinstance(indices, int)
        if self.negative_hg is not None:
            return self.positive_hg, self.negative_hg
        indices = torch.tensor(indices, device=self.to_eids.device)
        etype_ids, eids = self.to_eids[:, indices]
        neg_edge_dict = self.neg_sampler(
            self.positive_hg, {
                etype: eids[etype_ids == etype_id]
                for etype_id, etype in
                enumerate(self.positive_hg.canonical_etypes)
            }
        )
        return (
            self.positive_hg,
            dgl.heterograph(neg_edge_dict, self.num_nodes_dict)
        )

    def __len__(self):
        if self.negative_hg is not None:
            return sum(
                self.negative_hg.num_edges(etype)
                for etype in self.negative_hg.canonical_etypes
            )
        return self.to_eids.shape[1]
