from __future__ import annotations

from itertools import accumulate

import dgl
import torch


def scatter_edge_softmax(
    hg: dgl.DGLHeteroGraph,
    logits: dict[tuple[str, str, str], torch.Tensor],
    normsoftmax=False,
):
    from torch_scatter import scatter_softmax
    ntypes = {dst_ntype for _, _, dst_ntype in logits}
    src = torch.concat(list(logits.values()))
    ncounts = {ntype: hg.num_dst_nodes(ntype) for ntype in ntypes}
    offsets = list(accumulate(ncounts.values(), initial=0))[:len(ntypes)]
    offsets = dict(zip(ntypes, offsets))
    idx = torch.concat(
        [
            hg.all_edges(etype=etype)[1] + offsets[dst_ntype]
            for _, etype, dst_ntype in logits
        ]
    )
    if normsoftmax:
        src = (src - src.mean(0)) / src.std(0, unbiased=False).clamp_min(1e-10)
    a = scatter_softmax(src, idx, dim=0)

    def gen():
        s = 0
        for cetype in logits:
            e = s + hg.num_edges(etype=cetype)
            yield cetype, a[s:e]
            s = e

    return dict(gen())
