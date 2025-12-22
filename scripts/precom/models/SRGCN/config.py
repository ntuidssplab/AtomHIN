from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Annotated, Literal

import torch
from pydantic import Field, model_validator
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from dhgl import hgget as H
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.script_utils import BaseConfig
from dhgl.type import NType
from dhgl.utils.precomputation.metagraph import MPAdaptor
from scripts.precom.lib.base import HGNNReturnT
from scripts.precom.lib.precom_config import (
    LabelMetaGraphConfig,
    MetaGraphConfig,
    PrecomputationConfig,
)

from .srgcn import SRGCN_, LabelEmb, build_meta_adjs, construct_metagraph

if TYPE_CHECKING:
    from ...trainer.config import TrainerConfig


class _LPAConfig(BaseConfig):

    name: Literal['label_feats'] = 'label_feats'

    # num_hops: int
    # exclude_edge_types: list[str] | None = None
    hidden_dim: int | None = None
    num_in_layers: int | None = None
    input_drop: float | None = None
    dropout: float | None = None
    mp_config: LabelMetaGraphConfig
    num_out_layers: int
    num_pseudo_layers: int
    num_heads: int

    @model_validator(mode='before')
    @classmethod
    def backward_adaption(cls, data: dict):
        if 'num_hops' in data:
            warnings.warn(
                '"num_hops" has deprecated. Use mp_config__num_hops instead.',
                DeprecationWarning
            )
            if 'mp_config' not in data:
                data['mp_config'] = {}
            data['mp_config']['num_hops'] = data.pop('num_hops')
        if 'exclude_edge_types' in data:
            warnings.warn(
                '"exclude_edge_types" has deprecated. Use mp_config__exclude_edge_types instead.',
                DeprecationWarning
            )
            if 'mp_config' not in data:
                data['mp_config'] = {}
            data['mp_config']['exclude_edge_types'] = data.pop(
                'exclude_edge_types'
            )
        return data


class _LabelEmbConfig(BaseConfig):

    name: Literal['label_emb'] = 'label_emb'
    # num_hops: int
    cache_path: str
    n_hidden: int
    input_drop: float
    dropout: float
    n_out_layers: int = Field(ge=2)

    def init(self, hg: BaseHeteroGraphLike):
        return LabelEmb(
            n_hidden=self.n_hidden,
            n_out=H.n_classes(hg),
            input_drop=self.input_drop,
            dropout=self.dropout,
            out_norm_type='batch',
            n_out_layers=self.n_out_layers,
        )


class _SchedulerConfig(BaseConfig):
    max_lr_scale: float
    pct_start_epoch: int


class SRGCNConfig(BaseConfig):

    name: Literal['SRGCN'] = 'SRGCN'

    #######################
    # MODEL CONFIGS   #
    #######################
    hidden_dim: int

    mp_config: MetaGraphConfig

    @property
    def num_layers(self):
        return self.mp_config.num_hops

    num_pseudo_layers: int
    """num_pseudo_layers >= num_hops"""

    num_in_layers: int
    """Number of layers of input MLP"""

    num_out_layers: int
    """Number of layers of output MLP"""

    out_norm_type: Literal['batch', 'layer'] = 'layer'  # XXX

    num_heads: int
    """Number of attention heads"""
    # activation: Literal['identity'] = 'identity'

    lr: float
    weight_decay: float
    scheduler: _SchedulerConfig | None = None

    input_drop: float
    channel_drop: float
    dropout: float
    use_residual: bool | Annotated[float, Field(gt=0, lt=1)]

    weight_scalar: float | None = None
    """A simple scalar that used to scale all weights to avoid overflow in half training"""
    tgt_feat_residual: bool | None = None

    embedding_max_norm: float | None = Field(None)
    """max_norm passed to the embedding layers"""

    proj_by: Literal['ntype', 'shared'] = 'ntype'

    softmax_tau: float

    relation_weight_drop: float | None = Field(
        None,
        deprecated='relation_weight_drop has deprecated',
        exclude=True,
        le=0,
        ge=0,
    )

    relation_weight_mode: Literal['re-gnn'] = Field(
        're-gnn', deprecated='relation_weight_mode has deprecated',
        exclude=True
    )
    # relation_alpha: float | None = None
    precomputation_config: PrecomputationConfig

    feat_fmt: Literal['strided', 'sparse_csr', 'sparse_coo'] | None = None
    """Efficiency: stride >> coo > csr"""

    lpa_config: _LPAConfig | _LabelEmbConfig | None = Field(
        None, discriminator='name'
    )

    @model_validator(mode='after')
    def check_proj_by(self):
        if self.proj_by == 'shared':
            assert self.embedding_max_norm is None, (
                'embedding_max_norm has no effect while proj_by=="shared"'
            )
        return self

    @model_validator(mode='before')
    @classmethod
    def backward_adaption(cls, data: dict):
        if 'num_layers' in data:
            warnings.warn(
                '"num_layers" has deprecated. Use mp_config__num_hops instead.',
                DeprecationWarning
            )
            if 'mp_config' not in data:
                data['mp_config'] = {}
            data['mp_config']['num_hops'] = data.pop('num_layers')
        return data

    def init(
        self,
        hg: BaseHeteroGraphLike,
        feats: dict[NType, torch.Tensor | dict[NType, torch.Tensor]],
        global_conf: TrainerConfig,
        require_dataset:
        bool = True,  # NOTE: set False to avoid memory peak in multi-stage training
    ):

        def feat_filter_fn(mp):
            return (
                mp[-1][-1] == H.tgt_ntype(hg)
                and len(mp) <= self.mp_config.num_hops + 1
            )

        model = WrapSRGCN(
            self, self, hg, proj_args=self._get_proj_args(feats),
            mp_filter_fn=feat_filter_fn
        )
        net = nn.ModuleDict({'feat': model.net})
        model.to(global_conf.device)
        label_model = None
        if isinstance(self.lpa_config, _LPAConfig):

            def label_filter_fn(mp):
                hops = self.lpa_config.mp_config.num_hops
                return mp[0][0] == mp[-1][
                    -1] == H.tgt_ntype(hg) and 1 < len(mp) <= hops + 1

            label_model = WrapSRGCN(
                self,
                hgnn_config=self.lpa_config,
                hg=hg,
                proj_args=H.n_classes(hg),
                mp_filter_fn=label_filter_fn,
            )
            label_model.to(global_conf.device)
            net['label_feat'] = label_model.net
        elif isinstance(self.lpa_config, LabelEmb):
            net['label_emb'] = self.lpa_config.init(hg)
            label_emb = torch.load(
                self.lpa_config.cache_path, map_location=global_conf.device
            )

        optimizer = AdamW(
            net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        scheduler = None
        if self.scheduler is not None:
            # XXX: Consider move scheduler to global config
            iters_per_epoch = len(
                H.label(hg, 'train')
            ) // global_conf.batch_config.train.batch_size

            scheduler = OneCycleLR(
                optimizer,
                total_steps=(global_conf.epochs * iters_per_epoch),
                max_lr=self.lr * self.scheduler.max_lr_scale,
                pct_start=self.scheduler.pct_start_epoch / global_conf.epochs,
            )

        if global_conf.tracker_config.verbose:
            print(
                f'required_mps = {list(map(MPAdaptor.from_hg(hg).canonical_to_short, model.required_mps))}',
                f'total: {len(model.canonical_mp_indices)}',
            )
            if isinstance(self.lpa_config, _LPAConfig):
                print(
                    f'label_required_mps = {list(map(MPAdaptor.from_hg(hg).canonical_to_short, label_model.required_mps))}',
                    f'total: {len(label_model.canonical_mp_indices)}',
                )

        def forward(indices, xs: list | dict[str, list], *args):
            if isinstance(xs, dict):
                out = model.forward(xs['feat']
                                    ) + label_model.forward(xs['label_feat'])
            else:
                out = model.forward(xs)
            if 'label_emb' in net:
                out += net['label_emb'].forward(label_emb[indices])
            return out

        return HGNNReturnT(
            model=net,
            dataset=self.precomputation_config.get_precom_dataset(
                hg,
                feats,
                required_mps=model.required_mps,
                label_required_mps=label_model.required_mps
                if label_model else None,
            ) if require_dataset else None,
            optimizer=optimizer,
            scheduler=scheduler,
            forward_fn=forward,
        )

    def _get_proj_args(
        self, feats: dict[str, torch.Tensor | dict[str, torch.Tensor]]
    ):

        def equal(args):
            a = set(args)
            if len(a) != 1:
                raise ValueError(f'Sequence {args} not equal')
            return list(a)[0]

        if isinstance(list(feats.values())[0], dict):
            if self.proj_by == 'ntype':
                return SRGCN_.SharedFeatProjArgs(
                    in_feat_shapes={
                        ntype: (
                            equal(t.shape[0] for t in data.values()),
                            sum(t.shape[-1] for t in data.values())
                        )
                        for ntype, data in feats.items()
                    },
                    embedding_max_norm=self.embedding_max_norm,
                )
            else:
                assert self.proj_by == 'shared'
                return equal(
                    sum(t.shape[-1] for t in data.values())
                    for data in feats.values()
                )
        if self.proj_by == 'ntype':
            return SRGCN_.SharedFeatProjArgs(
                in_feat_shapes={
                    ntype: feat.shape
                    for ntype, feat in feats.items()
                },
                embedding_max_norm=self.embedding_max_norm,
            )
        else:
            assert self.proj_by == 'shared'
            return equal(feat.shape[-1] for feat in feats.values())


class WrapSRGCN:

    def __init__(
        self,
        shared_config: SRGCNConfig,
        hgnn_config: SRGCNConfig | _LPAConfig,
        hg: BaseHeteroGraphLike,
        proj_args,
        mp_filter_fn,
    ):
        self.mg = hgnn_config.mp_config.init(hg).add_self_loops()
        self.metagraph = construct_metagraph(
            self.mg.ntypes, self.mg.canonical_etypes
        )
        self.meta_adjs, self.mp_list = build_meta_adjs(
            self.mg.ntypes,
            self.mg.canonical_etypes,
            self.mg.self_loop_etypes,
            num_layers=hgnn_config.num_pseudo_layers,
        )

        def get_arg(hgnn_val, shared_val):
            return hgnn_val if hgnn_val is not None else shared_val

        self.net = SRGCN_(
            n_hidden=get_arg(hgnn_config.hidden_dim, shared_config.hidden_dim),
            n_layers=hgnn_config.num_pseudo_layers,
            n_heads=hgnn_config.num_heads,
            n_out=H.n_classes(hg),
            ntypes=self.mg.ntypes,
            etypes=self.mg.canonical_etypes,
            residual=shared_config.use_residual,
            input_drop=get_arg(
                hgnn_config.input_drop, shared_config.input_drop
            ),
            channel_drop=shared_config.channel_drop,
            dropout=get_arg(hgnn_config.dropout, shared_config.dropout),
            proj_args=proj_args,
            n_in_layers=get_arg(
                hgnn_config.num_in_layers, shared_config.num_in_layers
            ),
            n_out_layers=hgnn_config.num_out_layers,
            out_norm_type=shared_config.out_norm_type,
            softmax_tau=shared_config.softmax_tau,
            tgt_feat_residual=(
                H.tgt_ntype(hg)
                if getattr(hgnn_config, 'tgt_feat_residual', False) else None
            ),
            weight_scalar=shared_config.weight_scalar or 1.,
        )
        self.feat_fmt = shared_config.feat_fmt
        self.canonical_mp_indices = torch.tensor(
            [
                [
                    hg.get_ntype_id(mp[0][0]), mp_id,
                    hg.get_ntype_id(mp[-1][-1])
                ] for mp_id, mp in enumerate(self.mp_list) if mp_filter_fn(mp)
            ]
        )
        return

    def to(self, device):
        self.metagraph = self.metagraph.to(device)
        self.meta_adjs = [madj.to(device) for madj in self.meta_adjs]
        self.canonical_mp_indices = self.canonical_mp_indices.to(device)
        return self

    def forward(self, xs: list):
        xs = [x.to(self.metagraph.device) for x in xs]
        return self.net.forward(
            self.metagraph,
            self.canonical_mp_indices,
            self.meta_adjs,
            xs,
            feat_fmt=self.feat_fmt,
        )

    @property
    def required_mps(self):
        return [self.mp_list[i] for i in self.canonical_mp_indices[:, 1]]
