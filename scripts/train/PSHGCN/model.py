from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import matmul as torch_sparse_matmul


class PSHGCN(nn.Module):

    def __init__(
        self,
        in_dims: list[int],
        num_classes,
        lis,
        lis_t,
        emb_dim,
        hidden,
        K,
        input_drop,
        dropout,
        bns=False,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.h_dim = hidden
        self.K = K
        self.lis = lis
        self.lis_t = lis_t
        self.feat_project = nn.ModuleList(
            [
                nn.Linear(in_dim, self.emb_dim, bias=bns)
                for in_dim in self.in_dims
            ]
        )
        self.lin1 = nn.Linear(self.emb_dim, self.h_dim)
        self.lin2 = nn.Linear(self.h_dim, self.num_classes)

        self.W = nn.Parameter(torch.tensor(self.init_coe()))
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        return

    def init_coe(self):
        coe_num = 1 + len(self.lis[1])
        for i in range(2, self.K + 1):
            coe_num += len(self.lis[i])

        ###Random
        bound = np.sqrt(3.0 / (coe_num))
        TEMP = np.random.uniform(-bound, bound, coe_num)
        TEMP = TEMP / np.sum(np.abs(TEMP))
        return TEMP[:coe_num]

    def normalize(self, x):
        means = x.mean(1, keepdim=True)
        deviations = x.std(1, keepdim=True)
        x = (x - means) / deviations
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        return x

    def forward(self, adjs, features_list):
        output = []
        for lin, feature in zip(self.feat_project, features_list):
            feature = self.input_drop(lin(feature))
            output.append(feature)

        x = torch.cat(output, 0)
        x = F.relu(self.lin1(x))
        x = self.normalize(x)
        x = self.dropout(x)
        #g propgation
        coe_index = 0
        res = self.W[coe_index] * x
        coe_index += 1
        for k in range(1, self.K + 1):
            temp_now = {}
            if k == 1:
                for i in self.lis[k]:
                    out = torch_sparse_matmul(adjs[i], x)
                    temp_now[i] = out
                    res += self.W[coe_index] * out
                    coe_index += 1
                temp_lst = temp_now
            else:
                for i, j in enumerate(self.lis[k]):
                    out = torch_sparse_matmul(adjs[j[:2]], temp_lst[j[2:]])
                    temp_now[j] = out
                    res += self.W[coe_index] * out
                    coe_index += 1
                temp_lst = temp_now

        #g^t propagation
        x = res
        coe_index = 0
        res = self.W[coe_index] * x
        coe_index += 1
        for k in range(1, self.K + 1):
            temp_now = {}
            if k == 1:
                for i in self.lis_t[k]:
                    out = torch_sparse_matmul(adjs[i], x)
                    temp_now[i] = out
                    res += self.W[coe_index] * out
                    coe_index += 1
                temp_lst = temp_now
            else:
                for i, j in enumerate(self.lis_t[k]):
                    out = torch_sparse_matmul(adjs[j[:2]], temp_lst[j[2:]])
                    temp_now[j] = out
                    res += self.W[coe_index] * out
                    coe_index += 1
                temp_lst = temp_now

        res = self.lin2(res)
        return res


class PSHGCNPlus(nn.Module):

    def __init__(
        self,
        in_dims: list[int],
        num_classes,
        lis,
        lis_t,
        emb_dim,
        hidden,
        num_heads: int,
        K,
        input_drop,
        dropout,
        num_out_layers: int,
        bns=False,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.h_dim = hidden
        self.num_heads = num_heads
        self.K = K
        self.lis = lis
        self.lis_t = lis_t
        self.feat_project = nn.ModuleList(
            [
                nn.Linear(in_dim, self.emb_dim, bias=bns)
                for in_dim in self.in_dims
            ]
        )
        self.lin1 = nn.Linear(self.emb_dim, self.h_dim)
        # self.lin2 = nn.Linear(self.h_dim, self.num_classes)
        self.lin2 = self._construct_out_layers(
            num_out_layers,
            n_heads=num_heads,
            head_size=emb_dim,
            dropout=dropout,
            n_out=num_classes,
        )

        self.W = nn.Parameter(torch.tensor(self.init_coe()).float())
        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        return

    def _construct_out_layers(
        self, n_out_layers: int, n_heads: int, head_size: int, dropout: float,
        n_out
    ):
        _cur_size = n_heads * head_size
        out = nn.Sequential()
        for _ in range(n_out_layers - 1):
            out.extend(
                [
                    nn.Linear(_cur_size, head_size),
                    nn.LayerNorm(head_size),
                    nn.PReLU(),
                    nn.Dropout(dropout),
                ]
            )
            _cur_size = head_size
        out.append(nn.Linear(_cur_size, n_out))
        return out

    def init_coe(self):
        coe_num = 1 + len(self.lis[1])
        for i in range(2, self.K + 1):
            coe_num += len(self.lis[i])

        ###Random
        bound = np.sqrt(3.0 / (coe_num))
        TEMP = np.random.uniform(-bound, bound, (self.num_heads, coe_num))
        TEMP = TEMP / np.sum(np.abs(TEMP), axis=1).reshape(-1, 1)
        return TEMP.T.reshape(-1, 1)

    def normalize(self, x):
        means = x.mean(1, keepdim=True)
        deviations = x.std(1, keepdim=True)
        x = (x - means) / deviations
        x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)
        return x

    def forward(self, adjs, features_list):
        output = []
        for lin, feature in zip(self.feat_project, features_list):
            feature = self.input_drop(lin(feature))
            output.append(feature)

        x = torch.cat(output, 0)
        x = F.relu(self.lin1(x))
        x = self.normalize(x)
        x = self.dropout(x)
        #g propgation
        coe_index = 0
        res = self.W[coe_index] * x.view(
            -1, self.num_heads, self.h_dim // self.num_heads
        )
        coe_index += 1
        for k in range(1, self.K + 1):
            temp_now = {}
            if k == 1:
                for i in self.lis[k]:
                    out = torch_sparse_matmul(adjs[i], x)
                    temp_now[i] = out
                    res += self.W[coe_index] * out.view(
                        -1, self.num_heads, self.h_dim // self.num_heads
                    )
                    coe_index += 1
                temp_lst = temp_now
            else:
                for i, j in enumerate(self.lis[k]):
                    out = torch_sparse_matmul(adjs[j[:2]], temp_lst[j[2:]])
                    temp_now[j] = out
                    res += self.W[coe_index] * out.view(
                        -1, self.num_heads, self.h_dim // self.num_heads
                    )
                    coe_index += 1
                temp_lst = temp_now

        #g^t propagation
        x = res.view(-1, self.h_dim)
        coe_index = 0
        res = self.W[coe_index] * x.view(
            -1, self.num_heads, self.h_dim // self.num_heads
        )
        coe_index += 1
        for k in range(1, self.K + 1):
            temp_now = {}
            if k == 1:
                for i in self.lis_t[k]:
                    out = torch_sparse_matmul(adjs[i], x)
                    temp_now[i] = out
                    res += self.W[coe_index] * out.view(
                        -1, self.num_heads, self.h_dim // self.num_heads
                    )
                    coe_index += 1
                temp_lst = temp_now
            else:
                for i, j in enumerate(self.lis_t[k]):
                    out = torch_sparse_matmul(adjs[j[:2]], temp_lst[j[2:]])
                    temp_now[j] = out
                    res += self.W[coe_index] * out.view(
                        -1, self.num_heads, self.h_dim // self.num_heads
                    )
                    coe_index += 1
                temp_lst = temp_now

        res = self.lin2(res.view(-1, self.h_dim))
        return res
