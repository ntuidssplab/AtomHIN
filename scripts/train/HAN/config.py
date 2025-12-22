from __future__ import annotations
import json
from typing import TYPE_CHECKING, Literal
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dhgl import transforms
from dhgl import hgget as H
from dhgl.models.HAN import HAN
from dhgl.data.schema import BaseHeteroGraphLike

if TYPE_CHECKING:
    from ..trainer.config import TrainerConfig


class HANConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_nested_delimiter='__', extra='forbid', frozen=True
    )

    name: Literal['HAN'] = 'HAN'

    #######################
    # MODEL CONFIGS   #
    #######################
    hidden_dim: int

    num_layers: int
    """Number of Layer"""

    num_heads: int
    """Number of attention heads"""

    metapaths: list[list[str]]

    lr: float
    weight_decay: float
    max_lr_scale: float
    pct_start_epoch: int

    use_self_loop: bool

    dropout: float

    @field_validator('metapaths', mode='before')
    @classmethod
    def load_list(cls, v: str):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @H.use_cache()
    def init(self, hg: BaseHeteroGraphLike, global_conf: TrainerConfig):
        if self.use_self_loop:
            hg = transforms.add_self_loop(
                hg, H.tgt_ntype(hg), {H.tgt_ntype(hg): 'self'}
            )

        model = HAN(
            meta_paths=self.metapaths +
            ([['self']] if self.use_self_loop else []),
            in_size=H.tgt_feat(hg).shape[-1],
            hidden_size=self.hidden_dim,
            out_size=H.n_classes(hg),
            num_heads=[self.num_heads] * self.num_layers,
            dropout=self.dropout,
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

        def forward(graph, feat):
            return model.forward(graph, feat[H.tgt_ntype(hg)])

        return hg, model, optimizer, scheduler, forward
