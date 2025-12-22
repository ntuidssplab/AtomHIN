from __future__ import annotations

import math

import torch
from torch import nn

from ..type import NType


class SharedSpaceProjection(nn.Module):

    def __init__(
        self,
        in_feat_shapes: dict[NType, tuple[int, ...]],
        n_out: int,
        embedding_max_norm: float = None,
    ):
        """Projection layers map heterogeneous features into shared feature space.

        Basically, this module consists of linear layers for every node types.

        It's possible to specify which ntypes should use embedding+bias instead of nn.Linear,
            which could be more efficient computation- and memory-wise.
        """

        super().__init__()
        self.proj = nn.ModuleDict(
            {
                ntype:
                EmbeddingLinear(shape[0], n_out, max_norm=embedding_max_norm)
                if len(shape) == 1 else
                MaxNormLinear(shape[-1], n_out, max_norm=embedding_max_norm)
                for ntype, shape in in_feat_shapes.items()
            }
        )
        return

    def __iter__(self):
        yield from self.proj

    def __getitem__(self, ntype: NType):
        return self.proj[ntype]

    def forward(self, x: dict[NType, torch.Tensor]):

        for ntype, x_ in x.items():
            x[ntype] = self.proj[ntype](x_)
        return x


class MaxNormLinear(nn.Linear):
    """Linear layer that renorms weights if max_norm specified."""

    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype=None,
        max_norm=None
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.max_norm = max_norm
        return

    def forward(self, x):
        if self.max_norm is not None:
            with torch.no_grad():
                torch.renorm(
                    self.weight, p=2, dim=1, maxnorm=0.2, out=self.weight
                )
        return super().forward(x)


class EmbeddingLinear(nn.Module):

    def __init__(self, in_features: int, out_featuers: int, max_norm=None):
        super().__init__()
        self.bias = nn.Parameter(torch.Tensor(out_featuers))
        self.w = nn.Embedding(in_features, out_featuers, max_norm=max_norm)
        self.out_features = out_featuers
        self.max_norm = max_norm
        self.reset_parameters()
        return

    @property
    def weight(self):
        return self.w.weight

    def reset_parameters(self):
        # NOTE: following the weights initialization of nn.Linear.
        with torch.no_grad():
            l = nn.Linear(self.w.num_embeddings, self.w.embedding_dim)
            self.w.weight.copy_(l.weight.T)
            self.bias.copy_(l.bias)
        return

    @torch.no_grad()
    def reset_parameters_as_normal_embedding(self):
        if self.max_norm is not None:
            std = self.max_norm / math.sqrt(self.out_features)
        else:
            std = 1
        bound = math.sqrt(3) * std
        nn.init.uniform_(self.w.weight, -bound, bound)
        nn.init.zeros_(self.bias)
        return

    def forward(self, x):
        return self.w(x) + self.bias
