from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Iterable, Literal, Tuple, TypedDict

import numpy as np
import scipy.sparse as sp

from ..shared.type import SplitDict
from .utils import find_data_file


class NodeParser:

    @staticmethod
    def load(raw_dir: str):
        node_file, ext = find_data_file('node', raw_dir)
        node_parser = NodeParser._csv_parser if ext == '.csv' else NodeParser._dat_parser
        return NodeParser._load_nodes(node_parser(node_file))

    @staticmethod
    def _dat_parser(node_dat: str):
        with open(node_dat, encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                if len(th) == 4:
                    # Then this line of node has attribute
                    node_id, node_name, node_type, node_attr = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    node_attr = list(node_attr.split(','))

                    assert len(node_attr) > 0

                    if len(node_attr) > 1:
                        # Then this line of node has attribute
                        node_attr = list(
                            map(
                                float,
                                [a.replace('\n', '') for a in node_attr]
                            )
                        )
                        yield node_id, node_type, node_attr
                    else:
                        yield node_id, node_type, None

                elif len(th) == 3:
                    # Then this line of node doesn't have attribute
                    node_id, node_name, node_type = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    yield node_id, node_type, None

                else:
                    raise Exception("Too few information to parse!")

    @staticmethod
    def _csv_parser(node_csv: str):
        with open(node_csv, encoding='utf-8') as f:
            next(f)
            for line in f:
                th = line.split(',')
                if len(th) == 3:
                    node_id, node_type, node_attr = th
                    node_id = int(node_id)
                    node_type = int(node_type)
                    node_attr = list(map(float, node_attr.split(' ')))

                    if len(node_attr) > 1:
                        # node has attribute
                        yield node_id, node_type, node_attr
                    else:
                        # node doesn't have attribute
                        yield node_id, node_type, None

                else:
                    raise Exception("Too few information to parse!")

    @staticmethod
    def _load_nodes(node_info_generator: Iterable[tuple]):
        """
        return nodes dict
            total: total number of nodes
            count: a dict of int, number of nodes for each type
            attr: a dict of np.array (or None), attribute matrices for each type of nodes
            shift: node_id shift for each type. You can get the id range of a type by
                        [ shift[node_type], shift[node_type]+count[node_type] )
        """
        nodes = {'total': 0, 'count': Counter(), 'attr': {}, 'shift': {}}
        for node_id, node_type, node_attr in node_info_generator:

            nodes['count'][node_type] += 1
            nodes['total'] += 1
            if node_attr is not None:
                nodes['attr'][node_id] = node_attr

        shift = 0
        attr = {}
        for i in range(len(nodes['count'])):
            nodes['shift'][i] = shift
            if shift in nodes['attr']:
                mat = []
                for j in range(shift, shift + nodes['count'][i]):
                    mat.append(nodes['attr'][j])
                attr[i] = np.array(mat)
            else:
                attr[i] = None
            shift += nodes['count'][i]
        # print(nodes['shift'])
        nodes['attr'] = attr
        return nodes


class LinkParser:

    @staticmethod
    def load(raw_dir: str, nodes):
        link_file, ext = find_data_file('link', raw_dir)
        parser = LinkParser._csv_parser if ext == '.csv' else LinkParser._dat_parser
        return LinkParser._load_links(parser(link_file), nodes)

    @staticmethod
    def _csv_parser(link_csv: str):
        with open(link_csv, encoding='utf-8') as f:
            next(f)
            for line in f:
                th = line.split(',')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(
                    th[2]
                ), float(th[3])
                yield (h_id, t_id, r_id, link_weight)

    @staticmethod
    def _dat_parser(link_dat: str):
        with open(link_dat, encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                h_id, t_id, r_id, link_weight = int(th[0]), int(th[1]), int(
                    th[2]
                ), float(th[3])
                yield (h_id, t_id, r_id, link_weight)

    @staticmethod
    def _load_links(link_info_generator: Iterable[tuple], nodes):
        """
        return links dict
            total: total number of links
            count: a dict of int, number of links for each type
            meta: a dict of tuple, explaining the link type is from what type of node to what type of node
            data: a dict of sparse matrices, each link type with one matrix. Shapes are all (nodes['total'], nodes['total'])
        """

        def get_node_type(node_id):
            for i in range(len(nodes['shift'])):
                if node_id < nodes['shift'][i] + nodes['count'][i]:
                    return i

        def list_to_sp_mat(li):
            data = [x[2] for x in li]
            i = [x[0] for x in li]
            j = [x[1] for x in li]
            return sp.coo_matrix(
                (data, (i, j)), shape=(nodes['total'], nodes['total'])
            ).tocsr()

        links = {
            'total': 0,
            'count': Counter(),
            'meta': {},
            'data': defaultdict(list)
        }

        for h_id, t_id, r_id, link_weight in link_info_generator:
            if r_id not in links['meta']:
                h_type = get_node_type(h_id)
                t_type = get_node_type(t_id)
                links['meta'][r_id] = (h_type, t_type)
            links['data'][r_id].append((h_id, t_id, link_weight))
            links['count'][r_id] += 1
            links['total'] += 1
        new_data = {}
        for r_id in links['data']:
            new_data[r_id] = list_to_sp_mat(links['data'][r_id])
        links['data'] = new_data
        return links


class LabelParser:

    @staticmethod
    def load(raw_dir: str, split: Literal['train', 'val', 'test'], nodes):
        label_file, ext = find_data_file(f'data_{split}', raw_dir)
        parser = LabelParser._csv_parser if ext == '.csv' else LabelParser._dat_parser
        return LabelParser._load_labels(parser(label_file), nodes)

    @staticmethod
    def _dat_parser(label_dat: str):
        with open(label_dat, encoding='utf-8') as f:
            for line in f:
                th = line.split('\t')
                node_id, node_name, node_type, node_label =\
                    int(th[0]), th[1], int(th[2]), list(map(int, th[3].split(',')))
                yield node_id, node_type, node_label

    @staticmethod
    def _csv_parser(label_csv: str):
        with open(label_csv, encoding='utf-8') as f:
            next(f)
            for line in f:
                th = line.split(',')
                node_id, node_type, node_label =\
                    int(th[0]), int(th[1]), list(map(int, th[2].split(' ')))
                yield node_id, node_type, node_label

    @staticmethod
    def _load_labels(label_info_generator: Iterable[tuple], nodes):
        """
        return labels dict
            num_classes: total number of labels
            total: total number of labeled data
            count: number of labeled data for each node type
            data: a numpy matrix with shape (self.nodes['total'], self.labels['num_classes'])
            mask: to indicate if that node is labeled, if False, that line of data is masked
        """
        labels = {
            'num_classes': 0,
            'total': 0,
            'count': Counter(),
            'data': None,
            'mask': None
        }
        nc = 0
        mask = np.zeros(nodes['total'], dtype=bool)
        data = [None for i in range(nodes['total'])]
        for node_id, node_type, node_label in label_info_generator:
            for label in node_label:
                nc = max(nc, label + 1)
            mask[node_id] = True
            data[node_id] = node_label
            labels['count'][node_type] += 1
            labels['total'] += 1
        labels['num_classes'] = nc
        new_data = np.zeros((nodes['total'], labels['num_classes']), dtype=int)
        for i, x in enumerate(data):
            if x is not None:
                for j in x:
                    new_data[i, j] = 1
        labels['data'] = new_data
        labels['mask'] = mask
        return labels


# Credit: https://github.com/THUDM/HGB/blob/ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe/NC/benchmark/scripts/data_loader.py#L192
def _load_labels_hgb_format(nodes, path, split=None):
    """
    return labels dict
        num_classes: total number of labels
        total: total number of labeled data
        count: number of labeled data for each node type
        data: a numpy matrix with shape (self.nodes['total'], self.labels['num_classes'])
        mask: to indicate if that node is labeled, if False, that line of data is masked
    """
    labels = {
        'num_classes': 0,
        'total': 0,
        'count': Counter(),
        'data': None,
        'mask': None
    }
    nc = 0
    mask = np.zeros(nodes['total'], dtype=bool)
    data = [None for i in range(nodes['total'])]
    with open(path, encoding='utf-8') as f:
        # Backward compatibility:
        # if split == train: use 2nd~5th fold
        # elif split == val: use 1st fold
        # This follows https://github.com/THUDM/HGB/blob/ca6fd5bb0c1ca32e63b132c8bfe8f11a4a6629fe/NC/benchmark/methods/RGCN/entity_classify.py#L145
        if split == 'train':
            lines = f.readlines()
            lines = lines[len(lines) // 5:]
        elif split == 'val':
            lines = f.readlines()
            lines = lines[:len(lines) // 5]
        else:
            lines = f
        for line in lines:
            th = line.split('\t')
            node_id, node_name, node_type, node_label = int(th[0]), th[1], int(
                th[2]
            ), list(map(int, th[3].split(',')))
            for label in node_label:
                nc = max(nc, label + 1)
            mask[node_id] = True
            data[node_id] = node_label
            labels['count'][node_type] += 1
            labels['total'] += 1
    labels['num_classes'] = nc
    new_data = np.zeros((nodes['total'], labels['num_classes']), dtype=int)
    for i, x in enumerate(data):
        if x is not None:
            for j in x:
                new_data[i, j] = 1
    labels['data'] = new_data
    labels['mask'] = mask
    return labels


def load_labels_hgb_format(nodes, raw_dir: str):
    labels_train = _load_labels_hgb_format(
        nodes, os.path.join(raw_dir, 'label.dat'), split='train'
    )
    labels_val = _load_labels_hgb_format(
        nodes, os.path.join(raw_dir, 'label.dat'), split='val'
    )
    labels_test = _load_labels_hgb_format(
        nodes, os.path.join(raw_dir, 'label.dat.test')
    )
    return labels_train, labels_val, labels_test


class GraphData(TypedDict):
    features: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    adj: np.ndarray
    ntype_idx_ptr: tuple[int, int, int, int]
    labels: np.ndarray
    label_masks: SplitDict[np.ndarray]


def process_raw_data(raw_dir: str) -> GraphData:
    """Load & process raw data from directory:

    Args:
        dat_dir (str): directory that consists of following data files
            [
                node.___
                link.___
                data_train.___
                data_val.___
                data_test.___
            ]
            in which ___ can be either "csv" or "dat".
            or the HGB format:
            [
                node.dat
                link.dat
                label.dat
                label.dat.test
            ]

    Returns:
        GraphData: (features, adj, ntype_idx_ptr, labels, label_masks)

        features (tuple[ndarray * 4]): Node features. Node features are stored in a list/tuple
            because each feature can have different dimension, making them not able to combined
            as a single ndarray.
        adj (ndarray): adjacency matrix for all nodes regardless node types.
        ntype_idx_ptr (tuple[int * 4]): Since indices of nodes are stored contiguous, one can
            access nodes w.r.t types using ntype_idx_ptr.
            E.g. [0, 10, 15, 20]
                [ 0:10] -> type0 nodes
                [10:15] -> type1 nodes
                [15:20] -> type2 nodes
                [20:  ] -> type3 nodes
        labels (ndarray): Labels of target nodes.
        label_masks (dict[str, ndarray]) Dict storing mask of train/val/test split.
    """

    nodes = NodeParser.load(raw_dir)
    links = LinkParser.load(raw_dir, nodes)
    if (
        os.path.isfile(os.path.join(raw_dir, 'label.dat'))
        and os.path.isfile(os.path.join(raw_dir, 'label.dat.test'))
    ):
        # HGB format label
        labels_train, labels_val, labels_test = load_labels_hgb_format(
            nodes, raw_dir
        )
    else:
        labels_train = LabelParser.load(raw_dir, 'train', nodes)
        labels_val = LabelParser.load(raw_dir, 'val', nodes)
        labels_test = LabelParser.load(raw_dir, 'test', nodes)

    def gen_features():
        for i in range(len(nodes['count'])):
            th = nodes['attr'][i]
            if th is None:
                # when i represents 'keyword'
                yield sp.eye(nodes['count'][i])
            else:
                yield th

    features = list(gen_features())
    adjM = sum(links['data'].values())

    num_targets = nodes['count'][0]
    labels = np.zeros((num_targets, labels_train['num_classes']), dtype=int)

    train_mask = labels_train['mask']
    val_mask = labels_val['mask']
    test_mask = labels_test['mask']

    labels[train_mask[:num_targets]] = labels_train['data'][train_mask]
    labels[val_mask[:num_targets]] = labels_val['data'][val_mask]
    labels[test_mask[:num_targets]] = labels_test['data'][test_mask]

    label_masks = {
        'train': train_mask[:num_targets],
        'val': val_mask[:num_targets],
        'test': test_mask[:num_targets],
    }

    return GraphData(
        features=features,
        adj=adjM,
        ntype_idx_ptr=nodes['shift'],
        labels=labels,
        label_masks=label_masks,
    )
