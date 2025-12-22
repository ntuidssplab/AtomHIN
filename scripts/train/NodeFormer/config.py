from __future__ import annotations
from typing import Literal, TYPE_CHECKING
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import remove_self_loops, add_self_loops, from_dgl

import dgl
from dhgl import hgget as H
from dhgl.models import NodeFormer
from dhgl.script_utils import BaseConfig
from dhgl.script_utils.trainer import BasePredData

if TYPE_CHECKING:
    from ..trainer.config import TrainerConfig


class NodeFormerConfig(BaseConfig):

    name: Literal['NodeFormer'] = 'NodeFormer'

    #######################
    # myGAT MODEL CONFIGS   #
    #######################

    hidden_channels: int
    num_layers: int
    dropout: float
    num_heads: int
    use_bn: bool
    """use layernorm"""

    n_random_features: int
    """(M) number of random features"""

    use_gumbel: bool
    use_residual: bool
    use_act: bool
    use_jk: bool
    """concat the layer-wise results in the final layer"""

    n_gumbel_sample: int
    """(K) num of samples for gumbel softmax sampling"""

    rb_order: int
    """order for relational bias, 0 for not use"""
    rb_trans: Literal['sigmoid', 'identity']
    """non-linearity for relational bias"""

    tau: float
    """temperature for gumbel softmax"""

    loss_lambda: float
    """weight for edge reg loss"""

    # LR Configs
    lr: float
    weight_decay: float
    max_lr_scale: float
    pct_start_epoch: int

    def init(
        self,
        hgraph: HeteroData | dgl.DGLHeteroGraph,
        global_conf: TrainerConfig,
    ):
        hg = (
            hgraph
            if isinstance(hgraph, dgl.DGLHeteroGraph) else to_dgl_hg(hgraph)
        )
        g = dgl.to_homogeneous(hg)
        hdata = (
            hgraph if isinstance(hgraph, HeteroData) else from_dgl(hgraph)
        )
        _data = hdata.to_homogeneous(node_attrs=[], edge_attrs=[])

        model = NodeFormer(
            in_channels={
                ntype: data.shape[-1]
                for ntype, data in hg.ndata['feat'].items()
            },
            hidden_channels=self.hidden_channels,
            out_channels=H.n_classes(hdata),
            num_layers=self.num_layers,
            dropout=self.dropout,
            num_heads=self.num_heads,
            use_bn=self.use_bn,
            nb_random_features=self.n_random_features,
            use_gumbel=self.use_gumbel,
            use_residual=self.use_residual,
            use_act=self.use_act,
            use_jk=self.use_jk,
            nb_gumbel_sample=self.n_gumbel_sample,
            rb_order=self.rb_order,
            rb_trans=self.rb_trans,
        )
        optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        iters_per_epoch = 1
        assert global_conf.batch_config.train.is_in_batch_mode is False

        scheduler = OneCycleLR(
            optimizer,
            total_steps=(global_conf.epochs * iters_per_epoch),
            max_lr=self.lr * self.max_lr_scale,
            pct_start=self.pct_start_epoch / global_conf.epochs,
        )

        def loss_fn(pred_data: NodeFormerPredData, labels: torch.Tensor):
            loss = global_conf.loss_fn(pred_data.logits, labels)
            edge_loss = self.loss_lambda * sum(pred_data.link_loss
                                               ) / len(pred_data.link_loss)
            loss -= edge_loss
            return loss - edge_loss

        adjs = [adj.to(global_conf.device) for adj in self._get_adj(_data)]
        tgt_mask = g.ndata[dgl.NTYPE] == hg.get_ntype_id(H.tgt_ntype(hg))

        def graph_forward(_, feat: dict):
            # feat = (
            #     feat
            #     if isinstance(feat, torch.Tensor) else feat[H.tgt_ntype(hdata)]
            # )
            out, link_loss_ = model.forward(feat, adjs, self.tau)
            return NodeFormerPredData(out[tgt_mask], link_loss_)

        hg = to_dgl_hg(hdata)
        return hg, model, optimizer, scheduler, graph_forward, loss_fn

    def _get_adj(self, data: Data):
        """Adj storage for relational bias"""

        def adj_mul(adj_i, adj, N):
            adj_i_sp = torch.sparse_coo_tensor(
                adj_i,
                torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device),
                (N, N)
            )
            adj_sp = torch.sparse_coo_tensor(
                adj,
                torch.ones(adj.shape[1], dtype=torch.float).to(adj.device),
                (N, N)
            )
            adj_j = torch.sparse.mm(adj_i_sp, adj_sp)  # pylint: disable=not-callable
            adj_j = adj_j.coalesce().indices()
            return adj_j

        adjs = []
        adj, _ = remove_self_loops(data.edge_index)
        adj, _ = add_self_loops(adj, num_nodes=data.num_nodes)
        adjs.append(adj)
        for _ in range(
            self.rb_order - 1
        ):  # edge_index of high order adjacency
            adj = adj_mul(adj, adj, data.num_nodes)
            adjs.append(adj)
        return adjs


class NodeFormerPredData(BasePredData):

    def __init__(self, logits: torch.Tensor, link_loss: torch.Tensor):
        self._logits = logits
        self.link_loss = link_loss
        return

    def __getitem__(self, item):
        return self.__class__(self._logits[item], self.link_loss)

    @property
    def logits(self):
        return self._logits


def to_dgl_hg(hdata: HeteroData) -> dgl.DGLHeteroGraph:
    data_dict = {}
    for edge_type, edge_store in hdata.edge_items():
        if edge_store.get('edge_index') is not None:
            row, col = edge_store.edge_index
        else:
            row, col, _ = edge_store['adj_t'].t().coo()

        data_dict[edge_type] = (row, col)

    num_nodes_dict = {
        ntype: len(next(iter(ndata.values())))
        for ntype, ndata in hdata.node_items()
    }

    if len(hdata.node_types) == 1:
        # add dummy node types to avoid the graph created as homogeneous
        g: dgl.DGLGraph = dgl.heterograph(
            data_dict | {('dummy', 'd', 'dummy'): ([], [])},
            num_nodes_dict=num_nodes_dict | {'dummy': 0},
        )
    else:
        g: dgl.DGLGraph = dgl.heterograph(
            data_dict, num_nodes_dict=num_nodes_dict
        )

    for node_type, node_store in hdata.node_items():
        for attr, value in node_store.items():
            g.nodes[node_type].data[attr] = value

    for edge_type, edge_store in hdata.edge_items():
        for attr, value in edge_store.items():
            if attr in ['edge_index', 'adj_t']:
                continue
            g.edges[edge_type].data[attr] = value

    return g
