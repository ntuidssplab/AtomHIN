from __future__ import annotations
from typing import TYPE_CHECKING, Literal
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dhgl import hgget as H
from dhgl.models.mlp import MLP
from dhgl.data.schema import BaseHeteroGraphLike

if TYPE_CHECKING:
    from ..trainer.config import TrainerConfig


class MLPConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_nested_delimiter='__', extra='forbid', frozen=True
    )

    name: Literal['MLP'] = 'MLP'

    #######################
    # MODEL CONFIGS   #
    #######################
    hidden_dim: int
    num_layers: int

    lr: float
    weight_decay: float
    max_lr_scale: float
    pct_start_epoch: int

    dropout: float

    embedding_max_norm: float | None = Field(None)
    """max_norm passed to the embedding layers"""

    def init(self, hg: BaseHeteroGraphLike, global_conf: TrainerConfig):
        model = MLP(
            n_hidden=self.hidden_dim,
            n_out=H.n_classes(hg),
            n_layers=self.num_layers,
            use_norm=True,
            dropout=self.dropout,
            in_feat_shape=H.tgt_feat(hg).shape,
            embedding_max_norm=self.embedding_max_norm,
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
            total_steps=global_conf.epochs * iters_per_epoch,
            max_lr=self.lr * self.max_lr_scale,
            pct_start=self.pct_start_epoch / global_conf.epochs,
        )

        def forward(_, feat):
            return model.forward(feat[H.tgt_ntype(hg)])

        return hg, model, optimizer, scheduler, forward
