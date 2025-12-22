from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import dgl
from torch.optim import AdamW

from dhgl.data.link_prediction import LinkPredDatasetLike
from dhgl.script_utils import BaseConfig
from dhgl.utils.precomputation import (
    FeatureCollector,
    MetaGraph,
    MPAdaptor,
    row_normalized_adjs,
)
from scripts.precom.lib.precom_config import PrecomputationConfig

from .model import SeHGNN

# from .config import PSHGCNConfig as BasePSHGCNConfig

# from dhgl.utils.precomputation. import MetaGraph

if TYPE_CHECKING:
    from ..linkpred import TrainerConfig
    from ..linkpred.base import HGNNReturnT


class SeHGNNConfig(BaseConfig):

    name: Literal['SeHGNN'] = 'SeHGNN'

    precom_config: PrecomputationConfig

    hidden_dim: int
    emb_dim: int

    num_layers: int
    """Number of Layer"""

    num_out_layers: int
    """Number of layers of output MLP"""
    num_in_layers: int

    lr: float
    weight_decay: float

    att_drop: float
    input_drop: float
    dropout: float
    residual: bool

    def init(
        self, dataset: LinkPredDatasetLike, global_conf: TrainerConfig
    ) -> HGNNReturnT:

        # NOTE: only support PubMed for now
        mg = MetaGraph.from_hg(dataset.graph)
        if len(dataset.target_ntypes) > 1:
            raise NotImplementedError
        target_ntype = list(dataset.target_ntypes)[0]
        required_mps = mg.metapaths(
            self.num_layers,
            dsttype=target_ntype,
        )
        collector = FeatureCollector(
            mg,
            row_normalized_adjs(
                dataset.graph, dataset.graph.edata['weight'],
                etypes=dataset.graph.canonical_etypes
            ),
            cache_dir=self.precom_config.store_dir,
            cache_idx_dtype=self.precom_config.cache_idx_dtype,
            cache_val_dtype=self.precom_config.cache_val_dtype,
            verbose=self.precom_config.verbose,
            readonly=False,
        )
        adaptor = MPAdaptor.from_metagraph(mg)
        feats = {}
        for mp_id, feat in collector.precompute_features(
            required_mps, dataset.graph.ndata['feat']
        ):
            mp = required_mps[mp_id]
            short = adaptor.canonical_to_short(mp)
            feats[short] = feat.to_sparse_coo().to(global_conf.device)

        # ntype_ids = dgl.to_homogeneous(dataset.graph).ndata[dgl.NTYPE]
        # masks = {
        #     ntype: (ntype_ids == dataset.graph.get_ntype_id(ntype)).to(
        #         global_conf.device
        #     )
        #     for ntype in dataset.target_ntypes
        # }
        data_size = {k: v.size(-1) for k, v in feats.items()}
        model = SeHGNN(
            None,
            self.emb_dim,
            self.hidden_dim,
            nclass=global_conf.decoder_config.dim,
            feat_keys=feats.keys(),
            label_feat_keys=[],
            tgt_type=adaptor.to_ntype_alias[target_ntype],
            dropout=self.dropout,
            input_drop=self.input_drop,
            att_drop=self.att_drop,
            n_fp_layers=self.num_in_layers,
            n_task_layers=self.num_out_layers,
            act='none',
            residual=self.residual,
            data_size=data_size,
        )
        optimizer = AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        def forward(_, __):
            res = model.forward(None, feats, {}, None)
            return {target_ntype: res}

        return {
            'dataset': dataset,
            'model': model,
            'optimizer': optimizer,
            # 'scheduler': scheduler,
            'forward_fn': forward,
        }
