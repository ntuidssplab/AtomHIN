import os
from typing import Literal
import numpy as np
import torch
from torch_geometric import transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.data import Data
from ...misc import BaseConfig


class CoraConfig(BaseConfig):

    name: Literal['cora'] = 'cora'
    save_dir: str | None = None

    def load(self):
        transform = T.NormalizeFeatures()

        torch_dataset = Planetoid(
            root=os.path.join(os.path.expanduser(self.save_dir), 'Planetoid'),
            name='Cora',
            transform=transform,
        )
        data: Data = torch_dataset[0]
        data['feat'] = data.x
        data['label'] = data.y
        indices = rand_train_test_idx(data['label'])
        for split in ['train', 'val', 'test']:
            mask = torch.zeros_like(data[f'{split}_mask'])
            mask[indices[split]] = True
            data[f'{split}_mask'] = mask
        data.remove_tensor('x')
        data.remove_tensor('y')
        data = data.to_heterogeneous(
            node_type_names=['paper'],
            edge_type_names=[('paper', 'cite', 'paper')],
        )
        return data


def rand_train_test_idx(
    label, train_prop=.5, valid_prop=.25, ignore_negative=True
):
    """ randomly splits label into train/valid/test splits """
    if ignore_negative:
        labeled_nodes = torch.where(label != -1)[0]
    else:
        labeled_nodes = label

    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)

    perm = torch.as_tensor(np.random.permutation(n))

    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:]

    if not ignore_negative:
        return {
            'train': train_indices,
            'val': val_indices,
            'test': test_indices,
        }

    return {
        'train': labeled_nodes[train_indices],
        'val': labeled_nodes[val_indices],
        'test': labeled_nodes[test_indices],
    }
