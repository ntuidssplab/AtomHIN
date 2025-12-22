from __future__ import annotations
from typing import Literal, TYPE_CHECKING
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from pydantic import field_validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from sklearn.metrics import f1_score
from xgboost import XGBClassifier

import naive_flow as nf

import dgl
from dhgl import hgget as H
from dhgl.models import TreeGNN
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.script_utils.trainer import BasePredData
import dhgl

if TYPE_CHECKING:
    from ..trainer.config import TrainerConfig

Split = Literal['train', 'val', 'test']


def _train_xgb(
    config: XGBConfig,
    features: np.ndarray,
    labels: np.ndarray,
    masks: dict[Split, np.ndarray],
):

    xgb_clf = XGBClassifier(
        n_estimators=config.n_estimators,
        gamma=config.gamma,
        min_child_weight=config.min_child_weight,
        colsample_bytree=config.colsample_bytree,
        max_depth=config.max_depth,
        alpha=config.alpha,
        n_jobs=-1,
        verbosity=0,
    )
    # Train & Fit the model
    xgb_clf.fit(
        features[masks['train']], labels[masks['train']], verbose=False
    )

    # tree_pred = xgb_clf.predict_proba(feats_list_np[0])
    # tree_prob = mat2tensor(device, tree_pred)
    # X_leaves = xgb_clf.apply(feats_list_np[0])

    writer = SummaryWriter(comment=config.comment)
    writer.add_text('config', nf.strfconfig(config, strformat='markdown'))
    """XGB classifer evaluation"""

    def evaluate(split: Split):
        mask = masks[split]
        feature = features[mask]
        label = labels[mask]
        pred = xgb_clf.predict(feature)
        micro_f1 = f1_score(label, pred, average='micro')
        macro_f1 = f1_score(label, pred, average='macro')
        writer.add_scalar(f'micro_f1/{split}', micro_f1)
        writer.add_scalar(f'macro_f1/{split}', macro_f1)

    evaluate('train')
    evaluate('test')
    return xgb_clf


def _reindex_leaves(x_leaves):
    leaves = x_leaves.copy()
    new_leaf_index = dict()  # Dictionary to store leaf index
    total_leaves = 0
    for c in range(
        x_leaves.shape[1]
    ):  # Iterate for each column (i.e. # trees)
        column = x_leaves[:, c]
        unique_vals = list(sorted(set(column)))
        new_idx = {v: (i + total_leaves) for i, v in enumerate(unique_vals)}
        for i, v in enumerate(unique_vals):
            leaf_id = i + total_leaves
            new_leaf_index[leaf_id] = {c: v}
        leaves[:, c] = [new_idx[v] for v in column]
        total_leaves += len(unique_vals)

    assert leaves.ravel().max() == total_leaves - 1
    return leaves, total_leaves, new_leaf_index


class XGBConfig(BaseSettings):

    n_estimators: int = 125
    gamma: float = 0.2
    min_child_weight: float = 1.
    colsample_bytree: float = 0.2
    max_depth: int = 2
    alpha: float = 0.2
    comment: str = '_xgb'


class _TreeGNNConfig(BaseSettings):

    model_config = SettingsConfigDict(protected_namespaces=('settings_', ))
    model_gnn: Literal['mygat'] = 'mygat'
    leaf_fusion: bool = True
    pred_fusion: bool = True
    raw_fusion: bool = True
    edge_dim: int = 64
    """edge embedding dim"""
    num_hidden: int = 64
    n_estimators: int = XGBConfig.model_fields['n_estimators'].default
    num_layers: int = 3
    feat_drop: float = 0.5
    attn_drop: float = 0.5
    negative_slope: float = 0.05
    dim: int = 1  # Channel size of the embedding layer
    head_tree: int = 1
    fusion_type: Literal[0, 1, 2, 3, 4, 5] = 1
    beta: float = 0.5

    @field_validator('fusion_type', mode='before')
    @classmethod
    def handle_fusion_type(cls, v: str):
        if isinstance(v, str):
            return int(v)
        return v


class TreeConfig(BaseSettings):

    model_config = SettingsConfigDict(
        env_nested_delimiter='__', extra='forbid', frozen=True
    )

    name: Literal['Tree'] = 'Tree'

    #######################
    #  MODEL CONFIGS      #
    #######################
    backbone_config: _TreeGNNConfig

    alpha: float = Field(0.5, ge=0., le=1.)
    """Loss ratio in [0, 1]"""

    lr: float = 5e-4
    weight_decay: float = 1e-4
    max_lr_scale: float
    pct_start_epoch: int

    xgb_config: XGBConfig

    @property
    def num_layers(self):
        return self.backbone_config.num_layers

    @H.use_cache()
    def init(self, hg: BaseHeteroGraphLike, global_conf: TrainerConfig):

        hg = dhgl.transforms.add_self_loop(hg)
        order = [
            H.tgt_ntype(hg),
            *(ntype for ntype in hg.ntypes if ntype != H.tgt_ntype(hg))
        ]
        features = [hg.ndata['feat'][ntype] for ntype in order]
        """Trainer for the MODEL"""
        xgb_clf = _train_xgb(
            self.xgb_config,
            features[0],
            H.label(hg),
            {
                'train': H.mask(hg, 'train'),
                'test': H.mask(hg, 'test')
            },
        )
        x_leaves, leaf_num, _ = _reindex_leaves(xgb_clf.apply(features[0]))
        tree_prob = xgb_clf.predict_proba(features[0])
        x_leaves = torch.from_numpy(x_leaves)
        tree_prob = torch.from_numpy(tree_prob)
        model = TreeGNN(
            **self.backbone_config.model_dump(),
            num_etypes=len(hg.etypes) + 1,  # +1 is for self-loop
            in_dims=[data.shape[-1] for data in features],
            num_classes=H.n_classes(hg),
            num_heads=H.n_classes(hg),
            activation=F.elu,
            residual=True,
            alpha=0.05,
            max_leaf=leaf_num,
            device=global_conf.device,
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
            total_steps=(global_conf.epochs * iters_per_epoch),
            max_lr=self.lr * self.max_lr_scale,
            pct_start=self.pct_start_epoch / global_conf.epochs,
        )

        def loss_fn(pred_data: TreePredData, labels: torch.Tensor):
            loss1 = global_conf.loss_fn(pred_data.logits, labels)
            loss2 = global_conf.loss_fn(pred_data.pred_tree, labels)
            return (1. - self.alpha) * loss1 + self.alpha * loss2

        assert not global_conf.batch_config.train.is_in_batch_mode
        assert not global_conf.batch_config.eval.is_in_batch_mode
        g = dhgl.transforms.to_homogeneous(hg,
                                           order=order).to(global_conf.device)
        tgt_mask = g.ndata['_TYPE'] == 0
        e_feat = g.edata[dgl.ETYPE]
        tree_prob = tree_prob.to(global_conf.device)
        x_leaves = x_leaves.to(global_conf.device).long()

        def graph_forward(_: BaseHeteroGraphLike, feat: dict):
            features = [feat[ntype] for ntype in order]
            logits, pred_tree =\
                model.forward(g, x_leaves, tree_prob, features, e_feat)

            return TreePredData(logits[tgt_mask], pred_tree)

        return hg, model, optimizer, scheduler, graph_forward, loss_fn


class TreePredData(BasePredData):

    def __init__(self, logits: torch.Tensor, pred_tree: torch.Tensor):
        self._logits = logits
        self.pred_tree = pred_tree
        return

    def __getitem__(self, item):
        return self.__class__(self._logits[item], self.pred_tree[item])

    @property
    def logits(self):
        return self._logits
