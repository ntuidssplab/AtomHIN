from __future__ import annotations
from typing import Literal, TYPE_CHECKING
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import functional as F

import dgl
from dhgl import hgget as H
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.models import AdaptedGAT
from dhgl.script_utils import BaseConfig
import dhgl

if TYPE_CHECKING:
    from ..trainer.config import TrainerConfig


class GATConfig(BaseConfig):

    name: Literal['GAT'] = 'GAT'

    #######################
    # myGAT MODEL CONFIGS   #
    #######################

    edge_weights_alpha: float

    hidden_dim: int

    num_layers: int

    num_heads: int
    """Number of attention heads"""

    dropout: float
    negative_slope: float
    use_layer_norm: bool

    lr: float
    weight_decay: float
    max_lr_scale: float
    pct_start_epoch: int

    embedding_max_norm: float
    """max_norm passed to the embedding layers"""

    @H.use_cache()
    def init(self, hg: BaseHeteroGraphLike, global_conf: TrainerConfig):

        hg = dhgl.transforms.add_self_loop(hg)
        """Trainer for the MODEL"""
        model = AdaptedGAT(
            etypes=hg.etypes,
            num_hidden=self.hidden_dim,
            num_classes=H.n_classes(hg),
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            activation=F.elu,
            feat_drop=self.dropout,
            attn_drop=self.dropout,
            negative_slope=self.negative_slope,
            residual=True,
            use_layer_norm=self.use_layer_norm,
            edge_weights_alpha=self.edge_weights_alpha,
            shared_feat_proj_kwargs=AdaptedGAT.SharedFeatProjArgs(
                in_feat_shapes={
                    ntype: data.shape
                    for ntype, data in hg.ndata['feat'].items()
                },
                embedding_max_norm=self.embedding_max_norm,
            ),
            allow_zero_in_degree=(
                global_conf.batch_config.train.is_in_batch_mode
            ),
            # When training with mini-batch,
            # some of the self loop of target nodes of the sampled subgraph will be
            # drop, resulting in some node having zero in-degree.
            # However, those nodes are not dstnodes and thus is safe to turn this off.
        )
        print(
            f'#parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}'
        )
        optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        iters_per_epoch = 1
        if global_conf.batch_config.train.is_in_batch_mode:
            n_samples = len(H.label(hg, 'train'))
            iters_per_epoch = n_samples // global_conf.batch_config.train.batch_size

        scheduler = OneCycleLR(
            optimizer,
            total_steps=((global_conf.epochs + 1) * iters_per_epoch),
            max_lr=self.lr * self.max_lr_scale,
            pct_start=self.pct_start_epoch / global_conf.epochs,
        )

        if not (
            global_conf.batch_config.train.is_in_batch_mode
            and global_conf.batch_config.eval.is_in_batch_mode
        ):
            g = dgl.to_homogeneous(hg).to(global_conf.device)
            # XXX: a little more memeory consumption may introduce here.
            # the hg will be put to cuda somewhere else.
            # Things occupying cuda: hg.adj + hg.ndata + hg.edata + g.adj
            # the hg.adj is not used (but normally it only uses small amount of cuda memory.)
            tgt_mask = g.ndata[dgl.NTYPE] == hg.get_ntype_id(H.tgt_ntype(hg))
            etypes = g.edata[dgl.ETYPE
                             ] if self.edge_weights_alpha > 0 else None

            def graph_forward(_: BaseHeteroGraphLike, feat: dict):
                return model.forward(g, feat, etypes)[tgt_mask]

        # mfgs
        def mini_batch_forward_mfgs(blocks: list[dgl.DGLGraph], feat: dict):
            etypes = [
                block.edata[dgl.ETYPE] if self.edge_weights_alpha > 0 else None
                for block in blocks
            ]
            return model.forward(blocks, feat, etypes)

        # mini-batch subgraph
        def mini_batch_forward_subgraph(_hg: BaseHeteroGraphLike, feat: dict):

            g = dgl.to_homogeneous(_hg)
            tgt_mask = g.ndata[dgl.NTYPE] == _hg.get_ntype_id(H.tgt_ntype(hg))
            etypes = g.edata[dgl.ETYPE
                             ] if self.edge_weights_alpha > 0 else None
            return model.forward(g, feat, etypes)[tgt_mask]

        def get_forward_fn(conf):
            if not conf.is_in_batch_mode:
                return graph_forward
            if 'subgraph' in conf.name:
                return mini_batch_forward_subgraph
            assert 'block' in conf.name
            return mini_batch_forward_mfgs

        train_forward = get_forward_fn(global_conf.batch_config.train)
        eval_forward = get_forward_fn(global_conf.batch_config.eval)
        return hg, model, optimizer, scheduler, (train_forward, eval_forward)
