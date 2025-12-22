# Credit: https://github.com/ivam-he/PSHGCN
from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Literal

import torch
from pydantic import model_validator
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from dhgl import hgget as H
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.script_utils import BaseConfig
from dhgl.type import CEType, NType
from dhgl.utils.precomputation.metagraph import MPAdaptor
from scripts.precom.lib.precom_config import LabelMetaGraphConfig, MetaGraphConfig
from scripts.precom.lib.precom_config import MetapathConfig as BaseMetapathConfig
from scripts.precom.lib.precom_config import PrecomputationConfig
from scripts.precom.trainer.base import HGNNReturnT

from .pshgcn import PSHGCN

if TYPE_CHECKING:
    from ...trainer.config import TrainerConfig


def expander(K, etypes=["AI", "AP", "PP", "PF", "IA", "PA", "FP"]):
    keys = etypes
    lis = {}
    for k in range(1, K + 1):
        lis[k] = []
        if k == 1:
            for i in keys:
                lis[k].append(i)
        else:
            for i in lis[k - 1]:
                for j in keys:
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

    gi = ["10"]
    index = 11
    for v in lis.values():
        for i in v:
            gi.append(str(index) + i)
            index += 1

    gi_t = ["10"]
    index = 11
    for v in lis_t.values():
        for i in v:
            gi_t.append(str(index) + i)
            index += 1
    res = []
    for i in gi_t:
        for j in gi:
            if len(i) == 2:
                if len(j) == 2 or j[2] == "P":
                    res.append(i + j)
            else:
                coe = i[:2] + j[:2]
                if len(j) == 2 and i[2] == "P":
                    res.append(coe + i[2:])
                elif len(j) > 2 and i[-1] == j[2] and i[2] == "P":
                    res.append(coe + i[2:] + j[2:])
    num_coe = len(gi)
    return res, num_coe


class MetapathConfig(BaseMetapathConfig):

    metapaths: list[str]

    def get_metapaths(self, root_hg: BaseHeteroGraphLike):
        adaptor = MPAdaptor.from_hg(root_hg)

        def pshgcn_to_short(mp: str):
            if len(mp) == 1:
                return mp
            # PAAP -> PAP
            # PA AP PP -> PAPP
            assert len(mp) % 2 == 0
            return mp[::2] + mp[-1]

        keys = [pshgcn_to_short(key) for key in self.metapaths]
        mps = list(map(adaptor.short_to_canonical, keys))
        return mps


class PSHGCNConfig(BaseConfig):

    name: Literal['PSHGCN'] = 'PSHGCN'

    #######################
    # MODEL CONFIGS   #
    #######################

    mp_config: MetaGraphConfig
    label_mp_config: MetapathConfig | LabelMetaGraphConfig | None = None
    emb_dim: int
    hidden_dim: int

    # num_layers: int
    # """Number of Layer"""
    @property
    def num_layers(self):
        return self.mp_config.num_hops

    num_out_layers: int
    """Number of layers of output MLP"""

    label_num_out_layers: int
    label_hidden_dim: int

    # parser.add_argument("--dropout", type=float, default=0.5)
    # parser.add_argument("--input_drop", type=float, default=0.1)
    # parser.add_argument("--K", type=int, default=2)
    # parser.add_argument("--epochs", type=int, default=500)
    # parser.add_argument("--stage", type=int, default=1)
    # parser.add_argument("--lr", type=float, default=0.001)
    # parser.add_argument("--weight_decay", type=float, default=5e-5)
    # parser.add_argument("--batch-size", type=int, default=10000)
    # parser.add_argument("--patience", type=int, default=100)
    # parser.add_argument("--gama", type=float, default=10)
    # parser.add_argument("--threshold", type=float, default=0.75)
    # parser.add_argument("--bias", action='store_true', default=False)
    # parser.add_argument("--extra_emb", action='store_true', default=False)

    lr: float
    weight_decay: float
    # max_lr_scale: float
    # pct_start_epoch: int

    input_drop: float
    dropout: float

    bias: bool

    precomputation_config: PrecomputationConfig

    max_lr_scale: float | None = None
    pct_start_epoch: int | None = None
    # label_cache_dir: str | None = None
    # label_keys: list[str] | None = None
    # label_emb: str | None = None

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
        if 'label_keys' in data:
            assert 'label_mp_config' not in data
            warnings.warn(
                '"label_keys" has deprecated. Use label_mp_config__metapaths instead.',
                DeprecationWarning
            )
            data['label_mp_config'] = {'metapaths': data.pop('label_keys')}
        return data

    def init(
        self,
        hg: BaseHeteroGraphLike,
        feats: dict[NType, torch.Tensor | dict[NType, torch.Tensor]],
        global_conf: TrainerConfig,
    ):
        adaptor = MPAdaptor.from_hg(hg)

        def mp_to_pshgcn_format(mp: tuple[CEType]):
            short = adaptor.canonical_to_short(mp)
            if len(short) == 1:
                return short
            return ''.join(s + d for s, d in zip(short, short[1:]))

        mg = self.mp_config.init(hg)

        short_etypes = [
            f'{d[0]}{s[0]}'.upper() for s, _, d in mg.canonical_etypes
        ]
        expander_, num_coe = expander(int(self.num_layers / 2), short_etypes)

        label_mps = label_required_mps = None
        if self.label_mp_config is not None:
            label_required_mps = self.label_mp_config.get_metapaths(hg)
            label_mps = list(map(mp_to_pshgcn_format, label_required_mps))
        model = PSHGCN(
            emb_dim=self.emb_dim,
            hidden_x=self.hidden_dim,
            hidden_l=self.label_hidden_dim,
            nclass=H.n_classes(hg),
            in_dims={
                ntype[0].upper():
                sum(x.shape[-1] for x in feat.values())
                if isinstance(feat, dict) else feat.shape[-1]
                for ntype, feat in feats.items()
            },
            label_keys=(label_mps or []),
            layers_x=self.num_out_layers,
            layers_l=self.label_num_out_layers,
            coe_num=num_coe,
            expander=expander_,
            dropout=self.dropout,
            input_drop=self.input_drop,
            bias=self.bias,
        )

        if global_conf.tracker_config.verbose:
            print(
                '#parameters=',
                sum(
                    torch.prod(torch.tensor(p.size())) for p in
                    filter(lambda p: p.requires_grad, model.parameters())
                )
            )
        optimizer = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = None
        if self.pct_start_epoch is not None:
            # # XXX: Consider move scheduler to global config
            iters_per_epoch = len(
                H.label(hg, 'train')
            ) // global_conf.batch_config.train.batch_size

            scheduler = OneCycleLR(
                optimizer,
                total_steps=(global_conf.epochs * iters_per_epoch),
                max_lr=self.lr * self.max_lr_scale,
                pct_start=self.pct_start_epoch / global_conf.epochs,
            )

        required_mps = mg.metapaths(
            self.mp_config.num_hops, dsttype=H.tgt_ntype(hg)
        )
        mps = list(map(mp_to_pshgcn_format, required_mps))
        if global_conf.tracker_config.verbose:
            print(f'{mps = }')
            if label_required_mps:
                print(f'{label_mps = }')

        def forward(batch_indices: torch.Tensor, xs: list | dict[str, list]):
            label_feats_ = None
            if label_required_mps is not None:
                label_feats_ = xs.get('label_feat')
                xs = xs.get('feat')
            xs = dict(
                zip(mps, [x.to(global_conf.device).to_dense() for x in xs])
            )
            if label_required_mps is not None:
                label_feats_ = dict(
                    zip(
                        label_mps, [
                            x.to(global_conf.device).to_dense()
                            for x in label_feats_
                        ]
                    )
                )
            return model.forward(xs, label_feats=label_feats_)

        return HGNNReturnT(
            model=model,
            dataset=self.precomputation_config.get_precom_dataset(
                hg, feats, required_mps, label_required_mps
            ),
            optimizer=optimizer,
            scheduler=scheduler,
            forward_fn=forward,
        )
