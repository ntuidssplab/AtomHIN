from __future__ import annotations

import random

# import time
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from pydantic import Field
from torch import nn
from torch.optim import AdamW
from xgboost import XGBClassifier

import dhgl
from dhgl import hgget as H
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.script_utils import BaseConfig
from dhgl.script_utils.trainer.base import HGNNReturnT

if TYPE_CHECKING:
    from ..trainer.config import TrainerConfig


class Dummy(nn.Module):

    def __init__(self, n_in, n_out):
        super().__init__()
        self.dummy = nn.Linear(n_in, n_out)
        return

    def forward(self, feat, **kwargs):
        return self.dummy(feat)


class XGBoostConfig(BaseConfig):

    name: Literal['XGBoost'] = 'XGBoost'

    #######################
    # MODEL CONFIGS   #
    #######################
    num_layers: Literal[1] = Field(1, exclude=True)
    n_estimators: int = 125
    gamma: float = 0.2
    min_child_weight: float = 1.
    colsample_bytree: float = 0.2
    max_depth: int = 2
    alpha: float = 0.2
    verbose: bool = False

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
        feat = hg.ndata[dhgl.FEAT][target_ntype].to_dense().cpu().numpy()
        clf = self._train_xgb(
            feat[H.mask(hg, 'train')],
            H.label(hg, 'train').numpy()
        )
        model = Dummy(feat.shape[-1], n_out)
        optimizer = AdamW(model.parameters())

        def forward(graph, x):
            dummy = model(x[target_ntype]).float()
            assert len(x[target_ntype]) == len(feat)
            return dummy * 1e-9 + torch.from_numpy(
                clf.predict_proba(feat)
            ).float().to(global_conf.device)

        return hg, model, optimizer, None, forward

    def _train_xgb(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
    ):

        xgb_clf = XGBClassifier(
            n_estimators=self.n_estimators,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            colsample_bytree=self.colsample_bytree,
            max_depth=self.max_depth,
            alpha=self.alpha,
            n_jobs=-1,
            verbosity=0,
            seed=random.randint(0, 16384),
        )
        # Train & Fit the model
        xgb_clf.fit(train_features, train_labels, verbose=self.verbose)

        return xgb_clf
