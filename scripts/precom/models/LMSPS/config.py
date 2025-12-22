# Credit: https://github.com/JHL-HUST/LMSPS/tree/main/ogbn
from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Literal

import torch
from pydantic import Field, field_validator
from pydantic_settings import SettingsConfigDict
from torch.optim import AdamW

from dhgl import hgget as H
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.script_utils import BaseConfig
from dhgl.type import NType
from dhgl.utils.precomputation.metagraph import SelfLoop
from scripts.precom.lib.precom_config import (
    LabelMetaGraphConfig,
    MPAdaptor,
    PrecomputationConfig,
)
from scripts.precom.trainer.base import HGNNReturnT

from .arch import archs
from .lmsps import LMSPS

if TYPE_CHECKING:
    from ...trainer.config import TrainerConfig


class LabelEmbConfig(LabelMetaGraphConfig):

    # num_hops: int
    # exclude_edge_types: list[str] | None = None

    @classmethod
    def _null_label_emb(cls, hg: BaseHeteroGraphLike):
        return torch.zeros((hg.num_nodes(H.tgt_ntype(hg)), H.n_classes(hg)))

    def _get_label_emb(self, hg: BaseHeteroGraphLike, verbose):

        import hashlib
        mps = self.get_metapaths(hg)
        # self_loop_cetypes = [
        #     (ntype, f'self-{ntype}', ntype) for ntype in hg.ntypes
        # ]
        # exclude_mps = {(e, ) for e in self_loop_cetypes}
        # label_mps = list(
        #     filter(
        #         lambda mp: mp[0][0] == mp[-1][-1] == H.tgt_ntype(hg) and mp
        #         not in exclude_mps,
        #         get_metapaths(
        #             hg.ntypes, hg.canonical_etypes, self_loop_cetypes,
        #             self.num_hops
        #         )
        #     )
        # )
        # XXX: allowed using custom labeled emb
        label_emb_hash = hashlib.sha1(str(tuple(mps)).encode()).hexdigest()
        label_emb_cache_path = os.path.join(
            '/tmp', f'label_emb_{label_emb_hash[:10]}.pt'
        )
        if verbose:
            adaptor = MPAdaptor.from_hg(hg)
            print(f'{len(mps) = }')
            print(f'mps = {list(map(adaptor.canonical_to_short, mps))}')
            print(f'Loading label_emb from {label_emb_cache_path}')
        assert os.path.exists(label_emb_cache_path)
        return torch.load(label_emb_cache_path)


class LabelFeatConfig(BaseConfig):
    arch: list[str]

    @field_validator('arch', mode='before')
    @classmethod
    def _load_arch(cls, v: str):
        if not isinstance(v, str):
            return v
        if v in archs:
            return archs[v][0]
        try:
            v = json.loads(v)
        except json.decoder.JSONDecodeError:
            return v
        return v

    def label_mps(self, mp_adaptor: MPAdaptor):
        mps = list(map(mp_adaptor.short_to_canonical, self.arch))
        return mps


class LMSPSConfig(BaseConfig):

    model_config = SettingsConfigDict(validate_by_name=True)

    name: Literal['LMSPS'] = 'LMSPS'

    #######################
    # MODEL CONFIGS   #
    #######################

    # parser.add_argument("--dataset", type=str, default="ogbn-mag")
    # parser.add_argument("--gpu", type=int, default=0)
    # parser.add_argument("--cpu", action='store_true', default=False)
    # parser.add_argument("--root", type=str, default='../data/')
    # parser.add_argument(
    #     "--stages", nargs='+', type=int,
    #     default=[200, 200, 200, 200, 200,
    #              200], help="The epoch setting for each stage."
    # )
    # ## For pre-processing
    # parser.add_argument("--emb_path", type=str, default='../data/')
    # parser.add_argument(
    #     "--extra-embedding", type=str, default='',
    #     help="the name of extra embeddings"
    # )
    # parser.add_argument(
    #     "--embed-size", type=int, default=256,
    #     help="initial embedding size of nodes with no attributes"
    # )
    # parser.add_argument(
    #     "--num-hops", type=int, default=6,
    #     help="number of hops for propagation of raw labels"
    # )
    # parser.add_argument(
    #     "--label-feats", action='store_true', default=False,
    #     help="whether to use the label propagated features"
    # )
    # parser.add_argument(
    #     "--num-label-hops", type=int, default=4,
    #     help="number of hops for propagation of raw features"
    # )
    # ## For network structure
    # parser.add_argument("--hidden", type=int, default=512)
    # parser.add_argument(
    #     "--dropout", type=float, default=0.5, help="dropout on activation"
    # )
    # parser.add_argument(
    #     "--n-layers-2", type=int, default=2,
    #     help="number of layers of the downstream task"
    # )
    # parser.add_argument(
    #     "--n-layers-3", type=int, default=2,
    #     help="number of layers of residual label connection"
    # )
    # parser.add_argument(
    #     "--input-drop", type=float, default=0.1,
    #     help="input dropout of input features"
    # )
    # parser.add_argument(
    #     "--att-drop", type=float, default=0., help="attention dropout of model"
    # )
    # parser.add_argument(
    #     "--label-drop", type=float, default=0.,
    #     help="label feature dropout of model"
    # )
    # parser.add_argument(
    #     "--residual", action='store_true', default=False,
    #     help="whether to connect the input features"
    # )
    # parser.add_argument(
    #     "--bns", action='store_true', default=False,
    #     help="whether to process the input features"
    # )
    # parser.add_argument(
    #     "--label-bns", action='store_true', default=False,
    #     help="whether to process the input label features"
    # )
    # ## for training
    # parser.add_argument(
    #     "--amp", action='store_true', default=False, help=
    #     "whether to amp to accelerate training with float16(half) calculation"
    # )
    # parser.add_argument("--lr", type=float, default=3e-3)
    # parser.add_argument("--weight-decay", type=float, default=0)
    # parser.add_argument("--eval-every", type=int, default=1)
    # parser.add_argument("--batch-size", type=int, default=10000)
    # parser.add_argument(
    #     "--patience",
    #     type=int,
    #     default=50,  # original 100
    #     help="early stop of times of the experiment"
    # )
    # parser.add_argument(
    #     "--threshold", type=float, default=0.6,
    #     help="the threshold of multi-stage learning, confident nodes " +
    #     "whose score above this threshold would be added into the training set"
    # )
    # parser.add_argument(
    #     "--gama", type=float, default=5, help="parameter for the KL loss"
    # )
    # parser.add_argument("--start-stage", type=int, default=0)
    # parser.add_argument("--reload", type=str, default='')
    # parser.add_argument("--identity", action='store_true', default=False)
    # parser.add_argument('--arch', type=str, default='DBLP')
    # parser.add_argument("--eps", type=float, default=0)  #1e-12
    # parser.add_argument("--edge_mask_ratio", type=float, default=0)
    # parser.add_argument("--mask_seed", type=int, default=1)
    # parser.add_argument("--max_mask_deg", type=int, default=None)
    # parser.add_argument("--in_max_deg", type=int, default=None)
    # parser.add_argument("--out_max_deg", type=int, default=None)

    # embed_size: int
    hidden_dim: int

    # num_layers: int
    # """Number of Layer"""

    num_out_layers: int
    """Number of layers of output MLP"""

    # label_num_layers: int
    # label_bns: bool
    label_num_out_layers: int
    label_drop: float

    lr: float
    weight_decay: float
    # max_lr_scale: float
    # pct_start_epoch: int

    input_drop: float
    dropout: float

    residual: bool
    bns: bool
    eps: float

    arch: list[str]
    precomputation_config: PrecomputationConfig
    label_emb_config: LabelEmbConfig | None = Field(None, alias='lpa_config')
    label_feat_config: LabelFeatConfig | None = None

    # feat_fmt: Literal['strided', 'sparse_csr', 'sparse_coo'] | None = None
    # """Efficiency: stride >> coo > csr"""
    @field_validator('arch', mode='before')
    @classmethod
    def _load_arch(cls, v: str):
        if not isinstance(v, str):
            return v
        if v in archs:
            return archs[v][0]
        try:
            v = json.loads(v)
        except json.decoder.JSONDecodeError:
            return v
        return v

    @property
    def feat_arch(self):
        return self.arch

    # @property
    # def label_arch(self):
    #     return self.arch[1]

    def init(
        self,
        hg: BaseHeteroGraphLike,
        feats: dict[NType, torch.Tensor | dict[NType, torch.Tensor]],
        global_conf: TrainerConfig,
    ):

        adaptor = MPAdaptor.from_hg(hg)

        required_mps = [
            (SelfLoop.from_ntype(H.tgt_ntype(hg)), )
        ] + list(map(adaptor.short_to_canonical, self.feat_arch))
        feat_arch = [H.tgt_ntype(hg)[0].upper()] + self.feat_arch

        def _feat_size(feat: torch.Tensor | dict[NType, torch.Tensor]):
            return sum(x.shape[-1] for x in feat.values()
                       ) if isinstance(feat, dict) else feat.shape[-1]

        data_size = {
            mp: _feat_size(feats[cmp[0][0]])
            for mp, cmp in zip(feat_arch, required_mps)
        }
        data_size[H.tgt_ntype(hg)[0].upper()
                  ] = _feat_size(feats[H.tgt_ntype(hg)])
        label_required_mps = []
        if self.label_feat_config is not None:
            label_required_mps = self.label_feat_config.label_mps(adaptor)
        model = LMSPS(
            dataset=global_conf.dataset_config.name,
            data_size=data_size,
            # nfeat=self.embed_size,
            nfeat=max(data_size.values()),
            hidden=self.hidden_dim,
            nclass=H.n_classes(hg),
            num_feats=len(required_mps),
            num_label_feats=len(label_required_mps),
            tgt_key=H.tgt_ntype(hg)[0].upper(),
            dropout=self.dropout,
            input_drop=self.input_drop,
            att_drop=...,
            label_drop=self.label_drop,
            n_layers_2=self.num_out_layers,
            n_layers_3=self.label_num_out_layers,
            residual=self.residual,
            bns=self.bns,
            # label_bns=self.label_bns,
            path=feat_arch,
            label_path=label_required_mps,  # [] at stage0
            eps=self.eps,
            device=global_conf.device,
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

        if self.label_emb_config is not None:
            label_emb = self.label_emb_config._get_label_emb(
                hg, verbose=self.precomputation_config.verbose
            ).to(global_conf.device)
        else:
            label_emb = LabelEmbConfig._null_label_emb(hg).to(
                global_conf.device
            )

        # label_emb = torch.zeros(
        #     (hg.num_nodes(H.tgt_ntype(hg)), H.n_classes(hg))
        # ).to(global_conf.device)

        def forward(batch_indices: torch.Tensor, xs: list | dict[str, list]):
            if label_required_mps:
                label_feats = dict(
                    zip(
                        self.label_feat_config.arch,
                        [x.to(global_conf.device) for x in xs['label_feat']]
                    )
                )
                xs = dict(
                    zip(
                        feat_arch,
                        [x.to(global_conf.device) for x in xs['feat']]
                    )
                )
            else:
                xs = dict(
                    zip(feat_arch, [x.to(global_conf.device) for x in xs])
                )
                label_feats = {}
            # label_emb = torch.zeros((bs, H.n_classes(hg))).to(global_conf.device)
            return model.forward(xs, label_feats, label_emb[batch_indices])

        return HGNNReturnT(
            model=model,
            dataset=self.precomputation_config.get_precom_dataset(
                hg, feats, required_mps, label_required_mps
            ),
            optimizer=optimizer,
            # scheduler=scheduler,
            forward_fn=forward,
        )
