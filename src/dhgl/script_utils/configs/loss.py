from __future__ import annotations

import math
from typing import Literal

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch.nn import functional as F

__all__ = ['BCEWithLogits', 'SoftLabelCE', 'CrossEntropy']


class BaseLoss(BaseSettings):

    model_config = SettingsConfigDict(env_nested_delimiter='__', frozen=True)

    def _l2_norm(self, logits: torch.Tensor):
        n: torch.Tensor = torch.norm(logits, dim=1, keepdim=True)
        return logits / n.clamp(1e-12)


class BCEWithLogits(BaseLoss):

    name: Literal['bce', 'bce_with_logits'] = 'bce'
    l2_norm: bool | float | None = None
    r"""
    In the case of binary classification l2_norm=eps where eps should be a float.
    The norm value will be calculated as:
    norm = \sqrt{logit^2 + eps^2}

    if eps -> 0, the norm -> |logit|, which may result in gradient vanishment.
    """
    confidence_alpha: float | None = None

    def _l2_norm(self, logits):
        if self.l2_norm is True:
            if len(logits.shape) == 1:
                raise ValueError(
                    'l2_norm=True is not supported for binary classification. Use l2_norm=<float> instead.'
                )
            return super()._l2_norm(logits)

        if self.l2_norm <= 0:
            raise ValueError('l2_norm must be a positive number')
        return logits / (logits.pow(2) + (self.l2_norm**2)).sqrt()

    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, *args, **kwargs
    ):
        if self.l2_norm is True:
            l = F.binary_cross_entropy_with_logits(
                self._l2_norm(logits), labels, *args, **kwargs
            )
        elif isinstance(self.l2_norm, float):
            l = F.binary_cross_entropy_with_logits(
                self._l2_norm(logits), labels, *args, **kwargs
            )
        else:
            l = F.binary_cross_entropy_with_logits(
                logits, labels, *args, **kwargs
            )
        if self.confidence_alpha:
            # assert isinstance(self.confidence_alpha, float)
            proba = torch.sigmoid(logits)
            l_ = (torch.log(proba) + torch.log(1 - proba)).mean() + math.log(4)
            l -= self.confidence_alpha * l_
        return l


class SoftLabelCE(BaseLoss):
    """This is a loss for IMDB (BCE performs poor on IMDB)"""

    name: Literal['soft_label_ce'] = 'soft_label_ce'
    l2_norm: bool | None = None
    confidence_alpha: float | None = None

    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, *args, **kwargs
    ):
        if self.l2_norm:
            l = F.cross_entropy(
                self._l2_norm(logits),
                labels / labels.sum(dim=-1, keepdim=True), *args, **kwargs
            )
        else:
            l = F.cross_entropy(
                logits, labels / labels.sum(dim=-1, keepdim=True), *args,
                **kwargs
            )

        if self.confidence_alpha:
            confidence_loss = self.confidence_alpha * (
                torch.log_softmax(logits, -1).mean() +
                torch.log(torch.tensor(logits.shape[-1], dtype=logits.dtype))
            )
            l -= confidence_loss
        return l


class CrossEntropy(BaseLoss):

    name: Literal['ce'] = 'ce'
    l2_norm: bool | None = None
    confidence_alpha: float | None = None

    def __call__(
        self, logits: torch.Tensor, labels: torch.Tensor, *args, **kwargs
    ):
        if self.l2_norm:
            l = F.cross_entropy(self._l2_norm(logits), labels, *args, **kwargs)
        else:
            l = F.cross_entropy(logits, labels, *args, **kwargs)

        if self.confidence_alpha:
            # assert isinstance(self.confidence_alpha, float)
            confidence_loss = self.confidence_alpha * (
                torch.log_softmax(logits, -1).mean() +
                torch.log(torch.tensor(logits.shape[-1], dtype=logits.dtype))
            )
            l -= confidence_loss
        return l
