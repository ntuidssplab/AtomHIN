# pylint: disable=all
import warnings
import torch
import torch.nn as nn
from ..SimpleHGN.conv import SimpleHGNConv


# ref: https://github.com/cmsflash/efficient-attention/blob/master/efficient_attention.py
class Attention(nn.Module):

    def __init__(self, dim_Q, dim_K, head, in_dim):
        super(Attention, self).__init__()
        self.in_dim = in_dim
        self.head = head

        self.linearQ = nn.ModuleList(
            [nn.Linear(dim_Q, self.in_dim) for _ in range(self.head)]
        )
        self.linearK = nn.ModuleList(
            [nn.Linear(dim_K, self.in_dim) for _ in range(self.head)]
        )
        self.linearV = nn.ModuleList(
            [nn.Linear(dim_K, self.in_dim) for _ in range(self.head)]
        )
        self.layernorm = nn.LayerNorm([self.in_dim, self.head])
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputQ, inputK, inputV, fusion_type):
        attended_values = []

        for i in range(self.head):
            query = torch.softmax(
                self.linearQ[i](inputQ), dim=0
            )  # Softmax along row
            key = torch.softmax(
                self.linearK[i](inputK), dim=1
            )  # Softmax along col
            value = self.linearV[i](inputV)
            context = key.transpose(0, 1) @ value
            attention_value = query @ context
            attended_values.append(attention_value)
        attention = torch.stack(attended_values, dim=2)
        if fusion_type == 3:
            attention = torch.flatten(attention, start_dim=1)
        elif fusion_type == 4:
            attention, _ = torch.max(attention, 2)
        elif fusion_type == 5:
            attention = torch.mean(attention, 2)

        attention = self.dropout(attention)
        return attention


class TreeGNN(nn.Module):

    def __init__(
        self,
        model_gnn,
        leaf_fusion,
        pred_fusion,
        raw_fusion,
        edge_dim,
        num_etypes,
        in_dims,  #dataset.num_features
        num_hidden,
        num_classes,
        n_estimators,
        num_layers,
        num_heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
        alpha,
        max_leaf,
        dim,
        head_tree=4,
        fusion_type=0,
        beta=0.5,
        act="relu",
        device="cpu",
        use_self=False,
    ):
        super(TreeGNN, self).__init__()
        self.d = dim
        self.device = device
        if act == "relu":
            self.act = nn.LeakyReLU()
        self.fusion_type = fusion_type
        self.use_self = use_self
        self.tree_num = n_estimators
        self.leaf_num = self.tree_num * num_classes * dim
        self.in_dims = [int(in_dim) for in_dim in in_dims]

        # Embedding Layers
        self.leaf_embedding = nn.Embedding(max_leaf, dim, max_norm=True)

        # Attention Layer
        self.model_gnn = model_gnn
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        # Fusion Candidates
        self.leaf_fusion = leaf_fusion
        self.pred_fusion = pred_fusion
        self.raw_fusion = raw_fusion

        self.beta = beta

        # Change linear layers' dim due to fusion permutations
        self.rawMLP1 = nn.Linear(self.in_dims[0], num_hidden * 2)
        self.rawMLP2 = nn.Linear(
            num_hidden * 2, int(num_hidden * self.beta * 2)
        )
        self.leafMLP1 = nn.Linear(self.leaf_num, num_hidden * 2)
        self.leafMLP2 = nn.Linear(
            num_hidden * 2, int(num_hidden * (1 - self.beta) * 2)
        )

        self.in_dims[0] = int(num_hidden * self.beta * 2)
        if self.fusion_type == 1 or self.fusion_type == 2:
            if not self.raw_fusion is True and not self.leaf_fusion is True:
                self.in_dims[0] = 0
        else:
            if not self.raw_fusion is True:
                self.in_dims[0] = 0
            if self.leaf_fusion is True:
                self.in_dims[0] += int(
                    num_hidden * (1 - self.beta) * 2
                )  # self.leaf_num #
        self.attention_bolck = Attention(self.in_dims[0], self.in_dims[0],\
                                        head_tree, self.in_dims[0]).to(device)

        if self.fusion_type == 3:
            self.in_dims[0] *= head_tree
        if self.pred_fusion is True:
            self.in_dims[0] += num_classes

        self.leaf2cls = nn.Linear(self.in_dims[0], num_classes)

        # HGN setting
        self.fc_list = nn.ModuleList(
            [
                nn.Linear(in_dim, num_hidden, bias=True)
                for in_dim in self.in_dims
            ]
        )

        # Input Projection (NO Residual)
        if self.model_gnn == 'mygat':
            if num_layers == 1 and num_heads != 1:
                warnings.warn(
                    'When num_layers==1, only one head will be use. Thus the argument '
                    f'num_heads={num_heads} will be ignored.'
                )
            heads = [num_heads] * (num_layers - 1) + [1]
            for i in range(num_layers):  # noqa E741
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(
                    SimpleHGNConv(
                        edge_dim,
                        num_etypes,
                        num_hidden * heads[i - 1],
                        num_hidden if i < num_layers - 1 else num_classes,
                        heads[i],
                        feat_drop,
                        attn_drop,
                        negative_slope,
                        bool(i) and residual,  # no residual on input layer
                        self.activation if i < num_layers -
                        1 else None,  # no act on output layer
                        alpha=alpha,
                    )
                )
        self.dropout = nn.Dropout(p=feat_drop)
        self.epsilon = torch.FloatTensor([1e-12]).cuda()

    def forward(self, g, leaf, pred_tree, features_list, e_feat, dl=None):
        # Embedding of leaf_id
        if dl is not None:
            all_nodes = dl.labels['all_nodes']
        if self.leaf_fusion is True:
            leaf_vectors = self.leaf_embedding(leaf)
            leaf_vectors = torch.flatten(leaf_vectors, start_dim=1)
            leaf_vectors = self.act(self.leafMLP1(leaf_vectors))
            leaf_vectors = self.dropout(leaf_vectors)
            leaf_vectors = self.act(self.leafMLP2(leaf_vectors))
            leaf_vectors = self.dropout(leaf_vectors)
        else:
            leaf_vectors = None

        if self.use_self:
            # Apply multi-head attention
            leaf_vectors = self.attention_bolck(
                features_list[0], leaf_vectors, leaf_vectors
            )

            # Normalization
            #leaf_vectors = self.layer_norm(leaf_vectors)

        if self.raw_fusion is True:
            raw_features = self.act(self.rawMLP1(features_list[0]))
            raw_features = self.dropout(raw_features)
            raw_features = self.act(self.rawMLP2(raw_features))
            raw_features = self.dropout(raw_features)
        else:
            raw_features = None

        if leaf_vectors is None and raw_features is None:
            feat_target = pred_tree
        elif leaf_vectors is None:
            feat_target = raw_features
        elif raw_features is None:
            feat_target = leaf_vectors
        else:
            if self.fusion_type == 1 or self.fusion_type == 2:
                if dl is None:
                    feat_target = torch.stack(
                        (leaf_vectors, raw_features), dim=2
                    )
                    if self.fusion_type == 1:
                        feat_target, _ = torch.max(feat_target, 2)
                    elif self.fusion_type == 2:
                        feat_target = torch.mean(feat_target, 2)
                else:
                    feat_target = torch.stack(
                        (leaf_vectors[all_nodes[0]], raw_features), dim=2
                    )
                    if self.fusion_type == 1:
                        feat_target, _ = torch.max(feat_target, 2)
                    elif self.fusion_type == 2:
                        feat_target = torch.mean(feat_target, 2)
                    pass
            else:
                feat_target = torch.cat((leaf_vectors, raw_features), dim=-1)
                if self.fusion_type >= 3:
                    feat_target = self.attention_bolck(
                        feat_target, feat_target, feat_target, self.fusion_type
                    )

        if self.pred_fusion is True and feat_target is not pred_tree:
            if dl is None:
                feat_target = torch.cat((feat_target, pred_tree), dim=-1)
            else:
                feat_target = torch.cat(
                    (feat_target, pred_tree[all_nodes[0]]), dim=-1
                )

        pred_tree = self.act(self.leaf2cls(feat_target))
        pred_tree = self.dropout(pred_tree)
        pred_tree = pred_tree / (
            torch.
            max(torch.norm(pred_tree, dim=1, keepdim=True), self.epsilon)
        )

        h = []

        feats = [feat_target] + features_list[1:]
        for fc, feature in zip(self.fc_list, feats):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        res_attn = None
        if self.model_gnn == 'mygat':
            for l in range(self.num_layers - 1):
                h, res_attn = self.gat_layers[l](
                    g, h, e_feat, res_attn=res_attn
                )
                h = h.flatten(1)
            # Output Projection
            logits, _ = self.gat_layers[-1](g, h, e_feat, res_attn=None)
            logits = logits.mean(1)

            # This is an equivalent replacement for tf.l2_normalize,
            # See https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
            logits = logits / (
                torch.
                max(torch.norm(logits, dim=1, keepdim=True), self.epsilon)
            )

        return logits, pred_tree
