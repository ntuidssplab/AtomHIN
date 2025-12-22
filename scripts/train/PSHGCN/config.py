from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dgl
import torch
from pydantic import model_validator
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch_sparse import SparseTensor

import dhgl
from dhgl import hgget as H
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.script_utils import BaseConfig
from dhgl.utils.precomputation import adj as adj_utils

from .model import PSHGCN, PSHGCNPlus
from .static import ETYPE_MAPPERS

if TYPE_CHECKING:
    from ..trainer.config import TrainerConfig

# def _get_base_adjs2(hg: BaseHeteroGraphLike):
#     g = dgl.to_homogeneous(hg)
#     src, dst = g.edges()
#     adjs = {
#         etype:
#         SparseTensor(
#             # NOTE: adj should in shape dst<-src as model uses adj @ feat
#             # They did this wrong in there original code,
#             # although this doesn't not affect performance
#             # (as for example, AP works as PA, vice versa)
#             # However, this may be a pitfall in extensions
#             row=dst[g.edata[dgl.ETYPE] == eid],
#             col=src[g.edata[dgl.ETYPE] == eid],
#             value=hg.edges[etype].data.get(
#                 dhgl.EWEIGHT, torch.ones(hg.num_edges(etype))
#             ),
#             sparse_sizes=(g.num_nodes(), ) * 2,
#         )
#         for eid, etype in enumerate(hg.etypes)
#     }
#     adj_abs = {
#         etype:
#         SparseTensor(
#             # NOTE: adj should in shape dst<-src as model uses adj @ feat
#             # They did this wrong in there original code,
#             # although this doesn't not affect performance
#             # (as for example, AP works as PA, vice versa)
#             # However, this may be a pitfall in extensions
#             row=dst[g.edata[dgl.ETYPE] == eid],
#             col=src[g.edata[dgl.ETYPE] == eid],
#             value=hg.edges[etype].data.get(
#                 dhgl.EWEIGHT, torch.ones(hg.num_edges(etype))
#             ).abs(),
#             sparse_sizes=(g.num_nodes(), ) * 2,
#         )
#         for eid, etype in enumerate(hg.etypes)
#     }
#     for etype, adj in adjs.items():
#         deg_inv_sqrt = adj_abs[etype].sum(dim=1).pow_(-1.0)
#         deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
#         adj = torch_sparse.mul(adj, deg_inv_sqrt.view(-1, 1))  #D^-1A
#         adjs[etype] = adj
#         # adjs[etype] = adj.to(global_conf.device)
#     return adjs


class SchedulerConfig(BaseConfig):
    max_lr_scale: float
    pct_start_epoch: int

    def init(self, optimizer, global_conf: TrainerConfig):
        assert not global_conf.batch_config.train.is_in_batch_mode
        return OneCycleLR(
            optimizer,
            total_steps=global_conf.epochs,
            max_lr=[
                pg['lr'] * self.max_lr_scale for pg in optimizer.param_groups
            ],
            pct_start=self.pct_start_epoch / global_conf.epochs,
        )


class PSHGCNConfig(BaseConfig):

    name: Literal['PSHGCN', 'PSHGCN+'] = 'PSHGCN'

    #######################
    # MODEL CONFIGS   #
    #######################
    hidden_dim: int
    emb_dim: int

    num_layers: int
    """Number of Layer"""

    num_out_layers: int | None = None
    """Number of layers of output MLP"""

    num_heads: int | None = None
    """Number of attention heads"""

    lr: float
    weight_decay: float
    prop_lr: float
    prop_weight_decay: float
    scheduler_config: SchedulerConfig | None = None

    input_drop: float
    dropout: float

    @model_validator(mode='after')
    def check_num_heads(self):
        if self.name == 'PSHGCN+':
            assert self.num_heads is not None
            assert self.emb_dim * self.num_heads == self.hidden_dim
            assert self.num_out_layers is not None
        else:
            assert self.num_heads is None
            assert self.num_out_layers is None
            assert self.name == 'PSHGCN'
        return self

    # embedding_max_norm: float | None = Field(None)
    # """max_norm passed to the embedding layers"""

    def init(self, hg: BaseHeteroGraphLike, global_conf: TrainerConfig):
        hg, model, optimizer, scheduler, forward_fn = self._init(
            hg,
            n_out=H.n_classes(hg),
            global_conf=global_conf,
        )

        tgt_mask =\
            dgl.to_homogeneous(hg).ndata[dgl.NTYPE] == hg.get_ntype_id(H.tgt_ntype(hg))
        tgt_mask = tgt_mask.to(global_conf.device)

        def forward(graph, feat):
            return forward_fn(graph, feat)[tgt_mask]

        return hg, model, optimizer, scheduler, forward

    def _init(
        self,
        hg: BaseHeteroGraphLike,
        n_out: int,
        global_conf: TrainerConfig,
    ):

        assert all(
            len(x.shape) == 2 for x in hg.ndata[dhgl.FEAT].values()
        ), 'not accept ids as features'
        etype_mapper = ETYPE_MAPPERS[global_conf.dataset_config.name]
        lis, lis_t = _get_metapaths(
            [etype_mapper[etype] for etype in hg.etypes],
            k_hops=self.num_layers,
        )
        # NOTE: For unreachable ntypes, set zero features to ensure correct offset
        for ntype in hg.ntypes:
            if dhgl.FEAT in hg.nodes[ntype].data:
                continue
            hg.nodes[ntype].data[dhgl.FEAT] = torch.zeros(
                (hg.num_nodes(ntype), 1)
            )
        if self.name == 'PSHGCN':
            model = PSHGCN(
                in_dims=[x.shape[-1] for x in hg.ndata[dhgl.FEAT].values()],
                num_classes=n_out,
                lis=lis,
                lis_t=lis_t,
                emb_dim=self.emb_dim,
                hidden=self.hidden_dim,
                K=self.num_layers,
                input_drop=self.input_drop,
                dropout=self.dropout,
                bns=False,
            )
        else:
            assert self.name == 'PSHGCN+'
            model = PSHGCNPlus(
                in_dims=[x.shape[-1] for x in hg.ndata[dhgl.FEAT].values()],
                num_classes=n_out,
                num_heads=self.num_heads,
                num_out_layers=self.num_out_layers,
                lis=lis,
                lis_t=lis_t,
                emb_dim=self.emb_dim,
                hidden=self.hidden_dim,
                K=self.num_layers,
                input_drop=self.input_drop,
                dropout=self.dropout,
                bns=False,
            )
        if global_conf.tracker_config.verbose:
            print(
                '#parameters=',
                sum(
                    torch.prod(torch.tensor(p.size())) for p in
                    filter(lambda p: p.requires_grad, model.parameters())
                )
            )
            num_params = [
                (n, torch.prod(torch.tensor(p.size())).item())
                for n, p in model.named_parameters() if p.requires_grad
            ]
            from collections import Counter
            counter = Counter()
            for name, n_parmas in num_params:
                if '.' in name:
                    prefix, _ = name.split('.', maxsplit=1)
                else:
                    prefix = name
                counter[prefix] += n_parmas
            print('#parameters=', counter)
        optimizer = AdamW(
            [
                {
                    'params': model.feat_project.parameters(),
                    'weight_decay': self.weight_decay,
                    'lr': self.lr,
                }, {
                    'params': model.lin1.parameters(),
                    'weight_decay': self.weight_decay,
                    'lr': self.lr,
                }, {
                    'params': model.lin2.parameters(),
                    'weight_decay': self.weight_decay,
                    'lr': self.lr,
                }, {
                    'params': model.W,
                    'weight_decay': self.prop_weight_decay,
                    'lr': self.prop_lr
                }
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = None
        if self.scheduler_config is not None:
            scheduler = self.scheduler_config.init(optimizer, global_conf)

        adjs = adj_utils.adjs_to_homogeneous(
            hg, adj_utils.row_normalized_adjs(hg, hg.edata[dhgl.EWEIGHT])
        )
        if global_conf.dataset_config.name == 'acm':
            # NOTE: this is the special prepropcessing used in official PSHGCN on ACM
            # note that the diagnoal is added globally instead of only on paper nodes
            a = adjs['paper', 'citing', 'paper'].to_dense().bool().float()
            a = (a + torch.eye(len(a))).bool().float()
            a /= a.sum(dim=1, keepdim=True)
            adjs['paper', 'citing', 'paper'] = a.to_sparse_coo()
        adjs = {
            etype_mapper[cetype[1]][::-1]:  # dst <- src
            SparseTensor.from_torch_sparse_coo_tensor(adj).to(
                global_conf.device
            )
            for cetype, adj in adjs.items()
        }

        def forward(graph, feat):
            logits = model.forward(adjs, feat.values())
            return logits

        return hg, model, optimizer, scheduler, forward


# https://github.com/ivam-he/PSHGCN/blob/f87406ce56ba3780d19aa540d3d1081ff085cca5/hgb/processing.py#L109
def _get_metapaths(etypes: list[str], k_hops: int):
    """Etype here required to a pair of characters which denote the node types
    E.g. PA, PP, PT, etc.
    """
    lis = {}
    for k in range(1, k_hops + 1):
        lis[k] = []
        if k == 1:
            for i in etypes:
                lis[k].append(i)
        else:
            for i in lis[k - 1]:
                for j in etypes:
                    if i[-1] == j[0]:
                        lis[k].append(i + j)
    lis_t = {}
    for k, v in lis.items():
        lis_t[k] = []
        for i in v:
            temp = i[-2:][::-1]
            i = i[:-2]
            while (len(i)):
                temp += i[-2:][::-1]
                i = i[:-2]
            lis_t[k].append(temp)
    return lis, lis_t
