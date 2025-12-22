from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Annotated, Literal

import torch
from pydantic import Field, field_validator, model_validator
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from dhgl import hgget as H
from dhgl import transforms
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.models.sRGCN.rgcn import RGCN
from dhgl.script_utils import BaseConfig
from dhgl.script_utils.trainer.base import HGNNReturnT

if TYPE_CHECKING:
    from ..trainer.config import TrainerConfig


class RGCNConfig(BaseConfig):

    name: Literal['RGCN'] = 'RGCN'

    #######################
    # MODEL CONFIGS   #
    #######################
    hidden_dim: int

    num_layers: int
    """Number of Layer"""

    num_out_layers: int
    """Number of layers of output MLP"""

    num_heads: int
    """Number of attention heads"""
    activation: Literal['elu', 'identity'] = 'elu'

    lr: float
    weight_decay: float
    max_lr_scale: float | None = None
    pct_start_epoch: int | None = None

    class DropoutT(BaseConfig):
        feat: float
        edge: float

        @model_validator(mode='before')
        @classmethod
        def load(cls, data):
            if isinstance(data, float):
                return {'feat': data, 'edge': data}
            return data

    dropout: DropoutT
    # feat_drop: float edge_drop: float
    use_norm: bool
    use_residual: bool | Annotated[float, Field(gt=0, lt=1)]

    embedding_max_norm: float | None = Field(None)
    """max_norm passed to the embedding layers"""

    add_self_loop: bool
    proj_by: Literal['ntype', 'shared'] = 'ntype'
    linear_by: RGCN.LinearByT | list[RGCN.LinearByT]

    softmax_tau: float
    aggregation: Literal['macro', 'micro']
    relation_weight_mode: str | None = Field(
        None, deprecated='unused.', exclude=True
    )

    @field_validator('linear_by', 'dropout', mode='before')
    @classmethod
    def load_list(cls, v: str):
        if isinstance(v, str):
            try:
                return json.loads(v)
            except ValueError:
                return v
        return v

    @model_validator(mode='after')
    def check_proj_by(self):
        if self.proj_by == 'shared':
            assert self.embedding_max_norm is None, (
                'embedding_max_norm has no effect while proj_by=="shared"'
            )
        return self

    @model_validator(mode='after')
    def check_list_length(self):

        def check_field(field_name):
            v = getattr(self, field_name)
            if isinstance(v, list):
                assert len(
                    v
                ) == self.num_layers, f'Length of "{field_name}" does not match num_layers'
                return v
            return v

        assert check_field('linear_by') == self.linear_by
        return self

    def init(
        self, hg: BaseHeteroGraphLike, global_conf: TrainerConfig
    ) -> HGNNReturnT:

        hg, model, optimizer, scheduler, forward = self._init(
            hg, H.tgt_ntype(hg), H.n_classes(hg), global_conf
        )
        data = {
            'hg': hg,
            'model': model,
            'optimizer': optimizer,
            'forward_fn': forward
        }
        if scheduler is not None:
            data['scheduler'] = scheduler
        return data

    def _init(
        self,
        hg: BaseHeteroGraphLike,
        target_ntype: str | list[str],
        n_out: int,
        global_conf: TrainerConfig,
    ):
        if self.add_self_loop:
            hg = transforms.add_self_loop(hg)
        if self.proj_by == 'ntype':
            proj_args = RGCN.SharedFeatProjArgs(
                in_feat_shapes={
                    ntype: data.shape
                    for ntype, data in hg.ndata['feat'].items()
                },
                embedding_max_norm=self.embedding_max_norm,
            )
        else:
            assert self.proj_by == 'shared'
            dim = max(
                data.shape[-1] for data in hg.ndata['feat'].values()
                if len(data.shape) == 2
            )
            proj_args = RGCN.SharedFeatProjArgs(
                in_feat_shapes=dim,
                embedding_max_norm=self.embedding_max_norm,
            )
        model = RGCN(
            n_hidden=self.hidden_dim,
            ntypes=hg.ntypes,
            etypes=hg.canonical_etypes,
            target_ntype=target_ntype,
            n_out=n_out,
            n_layers=self.num_layers,
            n_heads=self.num_heads,
            activation=F.elu if self.activation == 'elu' else lambda _: _,
            linear_by=self.linear_by,
            use_norm=self.use_norm,
            residual=self.use_residual,
            feat_drop=self.dropout.feat,
            edge_drop=self.dropout.edge,
            proj_args=proj_args,
            n_out_layers=self.num_out_layers,
            softmax_tau=self.softmax_tau,
            aggregation=self.aggregation,
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
                prefix, _ = name.split('.', maxsplit=1)
                counter[prefix] += n_parmas
            print('#parameters=', counter)
        optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = None
        if self.pct_start_epoch is not None:
            warnings.warn('Use scheduler_config instead.', DeprecationWarning)
            # XXX: scheduler initialization has moved outside hgnn_config
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
            return model.forward(
                graph, feat, edge_weight=graph.edata['weight']
            )

        return hg, model, optimizer, scheduler, forward
