from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dgl

from dhgl.data.link_prediction import LinkPredDatasetLike

from .config import HINormerConfig as BaseHINormerConfig

if TYPE_CHECKING:
    from ..linkpred import TrainerConfig
    from ..linkpred.base import HGNNReturnT


class HINormerConfig(BaseHINormerConfig):

    name: Literal['HINormer'] = 'HINormer'

    hidden_dim: int

    num_layers: int

    num_heads: int
    """Number of attention heads"""

    seq_len: int

    num_gnns: int
    dropout: float
    temperature: float
    beta: float

    lr: float
    weight_decay: float
    max_lr_scale: float | None = None
    pct_start_epoch: int | None = None

    def init(
        self, dataset: LinkPredDatasetLike, global_conf: TrainerConfig
    ) -> HGNNReturnT:

        dataset.graph, model, optimizer, scheduler, forward_fn = self._init(
            dataset.graph,
            n_out=global_conf.decoder_config.dim,
            global_conf=global_conf,
        )

        g = dgl.to_homogeneous(dataset.graph)
        # tgt_ntype_id = hg.get_ntype_id(H.tgt_ntype(hg))
        # target_mask = torch.zeros_like(g.ndata[dgl.NTYPE]).bool()
        node_seqs = {}

        for target_ntype in dataset.target_ntypes:
            ntype_id = dataset.graph.get_ntype_id(target_ntype)
            target_nids = (g.ndata[dgl.NTYPE] == ntype_id).nonzero().squeeze()
            node_seq = self.get_node_seq(g, target_nids,
                                         self.seq_len).to(global_conf.device)
            node_seqs[target_ntype] = node_seq

        def forward(graph, feat):
            return forward_fn(graph, feat, node_seqs)

        return {
            'dataset': dataset,
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'forward_fn': forward,
        }
        # # tgt_mask = g.ndata[dgl.NTYPE] == hg.get_ntype_id(H.tgt_ntype(hg))
        # # e_feat = g.edata[dgl.ETYPE]
        # type_emb = torch.eye(len(hg.ntypes)).to(global_conf.device)
        # node_type = g.ndata[dgl.NTYPE]

        # def graph_forward(_: BaseHeteroGraphLike, feat: dict):
        #     res = model.forward(
        #         list(feat.values()), node_seq, type_emb, node_type, norm=True
        #     )
        #     return res

        # return hg, model, optimizer, scheduler, graph_forward
