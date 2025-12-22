from __future__ import annotations

import torch
from torch import nn

from .common import EmbeddingLinear


class MLP(nn.Module):

    def __init__(
        self,
        n_hidden: int,
        n_layers: int,
        n_out: int,
        use_norm=True,
        dropout: float = 0.5,
        in_feat_shape: tuple[int, ...] | None = None,
        embedding_max_norm: float | None = None,
    ):
        super().__init__()

        self.layers = nn.Sequential()
        for _ in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
            if use_norm:
                self.layers.append(nn.LayerNorm(n_hidden))
            self.layers.append(nn.PReLU())
            self.layers.append(nn.Dropout(dropout))

        self.layers.append(nn.Linear(n_hidden, n_out))

        self.proj = None
        if in_feat_shape is not None:
            self.proj = EmbeddingLinear(in_feat_shape[0], n_hidden, max_norm=embedding_max_norm)\
                  if len(in_feat_shape) == 1 else\
                    nn.Linear(in_feat_shape[-1], n_hidden)
        return

    def forward(
        self,
        feat: torch.Tensor,
    ) -> torch.Tensor:

        if self.proj is not None:
            feat = self.proj(feat)

        return self.layers(feat)
