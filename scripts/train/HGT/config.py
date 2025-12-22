from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from dhgl import hgget as H
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.models.HGT import HGT

if TYPE_CHECKING:
    from ..trainer.config import TrainerConfig


class HGTConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_nested_delimiter='__', extra='forbid', frozen=True
    )

    name: Literal['HGT'] = 'HGT'

    #######################
    # MODEL CONFIGS   #
    #######################
    hidden_dim: int

    num_layers: int
    """Number of Layer"""

    num_heads: int
    """Number of attention heads"""

    lr: float
    weight_decay: float
    max_lr_scale: float | None = None
    pct_start_epoch: int | None = None

    dropout: float

    embedding_max_norm: float | None = Field(None)
    """max_norm passed to the embedding layers"""

    relation_weights_rank: int | None = None
    relation_pri_alpha: float | None = None

    def init(self, hg: BaseHeteroGraphLike, global_conf: TrainerConfig):
        hg, model, optimizer, scheduler = self._init(
            hg,
            n_out=H.n_classes(hg),
            global_conf=global_conf,
        )

        def forward(graph, feat):
            return model.forward(graph, feat, H.tgt_ntype(hg))

        return hg, model, optimizer, scheduler, forward

    def _init(
        self, hg: BaseHeteroGraphLike, n_out: int, global_conf: TrainerConfig
    ):
        model = HGT(
            n_hidden=self.hidden_dim,
            ntypes=hg.ntypes,
            etypes=hg.etypes,
            n_out=n_out,
            n_layers=self.num_layers,
            n_heads=self.num_heads,
            use_norm=True,
            dropout=self.dropout,
            shared_feat_proj_kwargs=HGT.SharedFeatProjArgs(
                in_feat_shapes={
                    ntype: data.shape
                    for ntype, data in hg.ndata['feat'].items()
                },
                embedding_max_norm=self.embedding_max_norm,
            ),
            relation_weights_rank=self.relation_weights_rank,
            relation_pri_alpha=self.relation_pri_alpha,
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

        scheduler = None
        if self.max_lr_scale is not None:
            scheduler = OneCycleLR(
                optimizer,
                total_steps=global_conf.epochs * iters_per_epoch,
                max_lr=self.lr * self.max_lr_scale,
                pct_start=self.pct_start_epoch / global_conf.epochs,
            )

        # def forward(graph, feat):
        #     return model.forward(graph, feat, H.tgt_ntype(hg))

        return hg, model, optimizer, scheduler  #, forward
