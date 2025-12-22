from __future__ import annotations

import warnings
from typing import Iterable, TypedDict

import torch
import torch.nn as nn

from ..common import NType, SharedSpaceProjection
from .conv import SimpleHGNConv


class SimpleHGN(nn.Module):

    class SharedFeatProjArgs(TypedDict):
        in_feat_shapes: dict[NType, tuple[int, ...]]
        embedding_max_norm: float | None

    def __init__(
        self,
        edge_dim,
        num_etypes,
        num_hidden,
        num_classes,  # dataset.num_classes
        num_layers,
        num_heads: int,
        # heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
        edge_residual: bool,
        alpha,
        l2_norm: bool,
        allow_zero_in_degree: bool = False,
        shared_feat_proj_kwargs: SharedFeatProjArgs | None = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.edge_rediual = edge_residual
        heads = [num_heads] * (num_layers - 1) + [1]
        if num_layers == 1 and num_heads != 1:
            warnings.warn(
                'When num_layers==1, only one head will be use. Thus the argument '
                f'num_heads={num_heads} will be ignored.'
            )
        if l2_norm is True:
            warnings.warn(
                'Consider to have the l2 normalization with the loss function, not with the model'
            )
        self.l2_norm = l2_norm

        self.proj = None
        if shared_feat_proj_kwargs is not None:
            self.proj = SharedSpaceProjection(
                n_out=num_hidden,
                **shared_feat_proj_kwargs,
            )
        # hidden layers
        for i in range(num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                SimpleHGNConv(
                    edge_feats=edge_dim,
                    num_etypes=num_etypes,
                    in_feats=num_hidden * heads[i - 1],
                    out_feats=(
                        num_hidden if i < num_layers - 1 else num_classes
                    ),
                    num_heads=heads[i],
                    feat_drop=feat_drop,
                    attn_drop=attn_drop,
                    negative_slope=negative_slope,
                    residual=(bool(i)
                              and residual),  # no residual on input layer
                    activation=(
                        self.activation if i < num_layers - 1 else None
                    ),  # no act on output layer
                    allow_zero_in_degree=allow_zero_in_degree,
                    alpha=alpha,
                )
            )
        self.epsilon = torch.FloatTensor([1e-12]).cuda()
        #self.register_buffer("epsilon", torch.FloatTensor([1e-12]))

    def forward(
        self,
        g,
        x,
        e_feat,
        e_weight: torch.Tensor | str | None = None,
    ):
        if self.proj is not None:
            hs = self.proj(x)
        else:
            hs = x
        h = torch.concatenate(list(hs.values()), dim=0)

        def graph_forward(h):
            res_attn = None
            for l in range(self.num_layers - 1):
                h, res_attn = self.gat_layers[l](
                    g, h, e_feat, res_attn=res_attn, e_weight=e_weight
                )
                if not self.edge_rediual:
                    res_attn = None
                h = h.flatten(1)

            # output projection
            logits, _ = self.gat_layers[-1](
                g, h, e_feat, res_attn=None, e_weight=e_weight
            )
            return logits

        def batch_forward(h):
            assert self.edge_rediual is False
            assert isinstance(g, Iterable)
            assert isinstance(e_feat, Iterable)
            assert isinstance(e_weight, Iterable)
            for l in range(self.num_layers - 1):
                h, _ = self.gat_layers[l](
                    g[l], h, e_feat[l], res_attn=None, e_weight=e_weight[l]
                )
                h = h.flatten(1)

            # output projection
            logits, _ = self.gat_layers[-1](
                g[-1], h, e_feat[-1], res_attn=None, e_weight=e_weight[-1]
            )
            return logits

        logits = (batch_forward
                  if isinstance(g, Iterable) else graph_forward)(h)
        logits = logits.mean(1)
        if self.l2_norm:
            # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
            logits = logits / \
                (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        return logits

    def forward_linkpred(
        self,
        g,
        x,
        e_feat,
        e_weight: torch.Tensor | str | None = None,
    ):
        if g.is_block:
            raise NotImplementedError
        if self.proj is not None:
            hs = self.proj(x)
        else:
            hs = x
        h = torch.concatenate(list(hs.values()), dim=0)

        assert self.l2_norm

        def l2_norm(x):
            # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
            return x / (
                torch.max(torch.norm(x, dim=1, keepdim=True), self.epsilon)
            )

        emb = [l2_norm(h)]
        res_attn = None
        for l in range(self.num_layers - 1):
            h, res_attn = self.gat_layers[l](
                g, h, e_feat, res_attn=res_attn, e_weight=e_weight
            )
            emb.append(l2_norm(h.mean(1)))
            h = h.flatten(1)
        # output projection
        logits, _ = self.gat_layers[-1](
            g, h, e_feat, res_attn=res_attn
        )  #None)
        logits = logits.mean(1)
        logits = l2_norm(logits)
        emb.append(logits)
        logits = torch.cat(emb, 1)
        return logits
