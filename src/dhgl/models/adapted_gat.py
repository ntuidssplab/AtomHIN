from __future__ import annotations

from typing import TypedDict

import dgl
import torch
from dgl import function as fn
from dgl.base import DGLError
# from dgl.nn.pytorch import GATConv
from dgl.heterograph import DGLBlock, DGLGraph
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair
from torch import nn

from ..type import NType
from .common import SharedSpaceProjection


# pylint: enable=W0235
class GATConv(nn.Module):

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        allow_zero_in_degree=False,
        bias=True,
        use_layer_norm=False,
    ):
        super().__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False
            )
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False
            )
        self.attn_l = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.attn_r = nn.Parameter(
            torch.FloatTensor(size=(1, num_heads, out_feats))
        )
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.has_linear_res = False
        self.has_explicit_bias = False
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=bias
                )
                self.has_linear_res = True
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)

        if bias and not self.has_linear_res:
            self.bias = nn.Parameter(
                torch.FloatTensor(size=(num_heads * out_feats, ))
            )
            self.has_explicit_bias = True
        else:
            self.register_buffer("bias", None)

        self.layer_norm = None
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm((num_heads, out_feats))
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The fc weights :math:`W^{(l)}` are initialized using Glorot uniform initialization.
        The attention weights are using xavier initialization method.
        """
        gain = nn.init.calculate_gain("relu")
        if hasattr(self, "fc"):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self.has_explicit_bias:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
            if self.res_fc.bias is not None:
                nn.init.constant_(self.res_fc.bias, 0)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            A 1D tensor of edge weight values.  Shape: :math:`(|E|,)`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),
                                        ) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = graph.edata.pop("e")
            if edge_weight is not None:
                e += edge_weight.tile(1, self._num_heads, 1).transpose(0, 2)
            e = self.leaky_relu(e)
            # compute softmax
            graph.edata["a"] = self.attn_drop(dgl.ops.edge_softmax(graph, e))
            # if edge_weight is not None:
            #     graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
            #         1, self._num_heads, 1
            #     ).transpose(0, 2)
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    *((1, ) * len(dst_prefix_shape)), self._num_heads,
                    self._out_feats
                )

            if self.layer_norm is not None:
                rst = self.layer_norm(rst)

            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class SwitchGATConv(GATConv):

    def forward(self, graph, feat, edge_weight=None, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, *, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, *, D_{in_{src}})` and :math:`(N_{out}, *, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            A 1D tensor of edge weight values.  Shape: :math:`(|E|,)`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, *, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        torch.Tensor, optional
            The attention values of shape :math:`(E, *, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, "fc_src"):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats
                    )
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats
                    )
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats
                )
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),
                                        ) + dst_prefix_shape[1:]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({"ft": feat_src, "el": el})
            graph.dstdata.update({"er": er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v("el", "er", "e"))
            e = graph.edata.pop("e")
            if edge_weight is not None:
                e *= edge_weight.tile(1, self._num_heads, 1).transpose(0, 2)
            e = self.leaky_relu(e)
            # compute softmax
            graph.edata["a"] = self.attn_drop(dgl.ops.edge_softmax(graph, e))
            # if edge_weight is not None:
            #     graph.edata["a"] = graph.edata["a"] * edge_weight.tile(
            #         1, self._num_heads, 1
            #     ).transpose(0, 2)
            # message passing
            graph.update_all(fn.u_mul_e("ft", "a", "m"), fn.sum("m", "ft"))
            rst = graph.dstdata["ft"]
            if "rel_weight" in graph.dstdata:
                rst *= graph.dstdata["rel_weight"].view(-1, 1, 1)
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats
                )
                rst = rst + resval
            # bias
            if self.has_explicit_bias:
                rst = rst + self.bias.view(
                    *((1, ) * len(dst_prefix_shape)), self._num_heads,
                    self._out_feats
                )

            if self.layer_norm is not None:
                rst = self.layer_norm(rst)

            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata["a"]
            else:
                return rst


class AdaptedGAT(nn.Module):
    """Re-implementation of https://arxiv.org/pdf/2209.11414.

    If the edge_weights_alpha == 0, then this is just normal GAT having projection layers
    for heterogeneous node features.
    If the edge_weights_alpha > 0, the edge_weights (relation embeddings) is used. There are two
        differences compared with the official RE-GAT.
        1) the activatation LeakedReLU is not used.
        2) In the official code, the RE-GAT has the relation embeddings "added to" the original
        attention values instead of "scaling". In this implementation,
        it uses "scaling" since it's easier to implement using edge_weight of
        DGL built-in GATConv (the paper implies the "scaling" instead of "adding" as well).

    """

    class SharedFeatProjArgs(TypedDict):
        in_feat_shapes: dict[NType, tuple[int, ...]]
        embedding_max_norm: float | None

    # TODO: add argument to determine whether "scaling" or "adding".
    # TODO: add argument to set the activatations of relation embeddings

    def __init__(
        self,
        etypes: list[str],
        num_hidden: int,
        num_classes: int,
        num_heads,
        num_layers: int,
        activation,
        edge_weights_alpha: float,
        feat_drop=0,
        attn_drop=0,
        negative_slope=0.2,
        residual=False,
        use_layer_norm=False,
        shared_feat_proj_kwargs: SharedFeatProjArgs | None = None,
        allow_zero_in_degree=False,
    ):
        super().__init__()
        heads = [num_heads] * (num_layers - 1) + [1]
        self.convs = nn.ModuleList(
            GATConv(
                in_feats=num_hidden * heads[i - 1],
                out_feats=(num_hidden if i < num_layers - 1 else num_classes),
                num_heads=heads[i],
                feat_drop=feat_drop,
                attn_drop=attn_drop,
                negative_slope=negative_slope,
                residual=(bool(i) and residual),  # no residual on input layer
                use_layer_norm=use_layer_norm,
                allow_zero_in_degree=allow_zero_in_degree,
            ) for i in range(num_layers)
        )
        self.activation = activation

        self.proj = None
        if shared_feat_proj_kwargs is not None:
            self.proj = SharedSpaceProjection(
                n_out=num_hidden,
                **shared_feat_proj_kwargs,
            )

        self.alpha = edge_weights_alpha
        self.edge_weights = None
        if self.alpha > 0.:
            self.edge_weights = nn.ModuleList(
                nn.Embedding(len(etypes), 1) for _ in range(num_layers)
            )
            for edge_weight in self.edge_weights:
                nn.init.constant_(edge_weight.weight, 1. / self.alpha)
        return

    def forward(
        self,
        g: DGLGraph | list[DGLBlock],
        x: dict[NType, torch.Tensor],
        etypes: torch.Tensor | list[torch.Tensor] = None,
        e_weight: torch.Tensor | str | None = None,
    ):
        """

        Args:
            g (DGLGraph | list[DGLBlock]): graph or blocks
            x (dict[NType, torch.Tensor]): node features
            etypes (torch.Tensor | list[torch.Tensor], optional):
            If None, the edge_weight is disabled.
            If the g is list of blocks, the etypes should be list of etypes.
                E.g. etypes = [block.edata[dgl.ETYPE] for block in g]
            If the g is dgl graph, the etypes should be etypes. E.g. etypes = g.edata[dgl.ETYPE]
        """

        if self.proj is not None:
            hs = self.proj(x)
            hs = torch.concatenate(list(hs.values()), dim=0)
        else:
            hs = torch.concatenate(list(x.values()), dim=0)

        if isinstance(g, DGLGraph):
            g = [g] * len(self.convs)
            etypes = [etypes] * len(self.convs)
            e_weight = [e_weight] * len(self.convs)
        elif g[0].is_block:
            hs = hs[g[0].srcdata['hetero->homo']]
            # NOTE: The homogeneous mfg stores node data in format differing from homogeneous
            # whole graph. In the case of homogeneous graph converted from heterogeneous graph,
            # the nodes are organized in the (internal) order of node types.
            # For example,
            # >>> g = dgl.to_homogeneous(hg)
            # >>> g.ndata[dgl.NTYPE] == [0, 0, ..., 1, 1, ..., 2, 2, ...]
            # However, in the case of mfg, the dst nodes are included as the prefix of src nodes.
            # Therefore, reindexing is necessary here.

        for i, (g_, etypes_, ew_, conv) in \
            enumerate(zip(g, etypes, e_weight, self.convs)):
            g_: DGLGraph
            if etypes_ is not None:
                edge_weight = self.edge_weights[i](etypes_
                                                   ).squeeze() * self.alpha
            else:
                edge_weight = None
            if ew_ is not None:
                if isinstance(ew_, str):
                    ew_ = g_.edata[ew_]
                if edge_weight is not None:
                    edge_weight *= ew_
                else:
                    edge_weight = ew_
            hs: torch.Tensor = conv(
                g_, self.activation(hs), edge_weight=edge_weight
            ).flatten(1)
        return hs
