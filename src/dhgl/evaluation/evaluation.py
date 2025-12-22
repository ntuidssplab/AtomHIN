from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Literal, TypedDict

import numpy as np
import torch
from scipy import optimize
from sklearn.metrics import f1_score, roc_auc_score


def _handle_torch_tensor(y: torch.Tensor):
    if not isinstance(y, torch.Tensor):
        return y
    return y.detach().cpu().numpy()


def find_best_binary_threshold(
    y_true_labels: np.ndarray,
    y_pred_probas: np.ndarray,
    objective_fn: Callable[[np.ndarray, np.ndarray], Any],
    Ns: int = None,  # pylint: disable=invalid-name
    bounds: tuple[float, float] = None,
    **kwargs,
) -> np.ndarray:
    """Find the best thresholds for single-label binary classification task

    Args:
        y_pred_probas (np.ndarray): predicted distribution.  shape: (n_samples,)
        y_true_labels (np.ndarray): true labels.  shape: (n_samples,)
        objective_fn  (Callable[[y_true, y_pred], Value]) sklearn-format metrics function.
            Higher is better.
        bounds (tuple[float, float]): bounds for optimization. Default to (0, 1)
        Ns: number of grid points
        kwargs: additional args pass to optimizer function `scipy.optimize.brute`.
    """

    y_pred_probas = _handle_torch_tensor(y_pred_probas)
    y_true_labels = _handle_torch_tensor(y_true_labels)

    assert len(y_pred_probas.shape) == 1, (
        f'not binary single-label due to the shape = {y_pred_probas.shape}'
    )

    def loss_fn(thresholds: np.ndarray):
        pred_labels = (y_pred_probas > thresholds).astype(int)
        return -objective_fn(y_true_labels, pred_labels)

    if Ns is not None:
        kwargs['Ns'] = Ns
    bounds = bounds or (0., 1.)
    res = optimize.brute(loss_fn, (bounds, ), **kwargs)
    return res[0]


def find_best_thresholds(
    y_true_labels: np.ndarray,
    y_pred_probas: np.ndarray,
    objective_fn: (
        Callable[[np.ndarray, np.ndarray], Any]
        | Literal['macro_f1', 'micro_f1']
    ),
    **kwargs,
) -> np.ndarray:
    """Find the best thresholds for multi-label binary classification task. This may
        be very ineffective for single-label task. use `find_best_binary_threshold` for
        single-label task instead.

    Args:
        y_pred_probas   (np.ndarray): predicted distribution.
            shape (n_samples, n_classes)
        y_true_labels (np.ndarray): true labels
            shape (n_samples, n_classes)
        objective_fn  (Callable[[y_true, y_pred], Value]) sklearn-format metrics function.
            Higher is better.
    """

    y_pred_probas = _handle_torch_tensor(y_pred_probas)
    y_true_labels = _handle_torch_tensor(y_true_labels)

    if objective_fn in ('micro_f1', 'macro_f1'):
        fn_str = objective_fn

        def f1(y_true, y_pred):
            return f1_score(y_true, y_pred, average=fn_str[:5])

        objective_fn = f1

    def loss_fn(thresholds: np.ndarray):
        pred_labels = (y_pred_probas > thresholds).astype(int)
        return -objective_fn(y_true_labels, pred_labels)

    if 'x0' not in kwargs:
        kwargs['x0'] = [0.5] * y_true_labels.shape[-1]

    if 'disp' not in kwargs:
        kwargs['disp'] = 0

    return optimize.fmin(loss_fn, **kwargs)


class NodeClassificationMetrics(TypedDict):
    macro_f1: float
    micro_f1: float


LogitsLabelsTuple = tuple[torch.Tensor, torch.Tensor]


def node_classification_eval(
    train_data: LogitsLabelsTuple,
    *test_data: LogitsLabelsTuple,
    multi_label_thresholds: Literal['macro_f1', 'micro_f1']
    | float | np.ndarray = 'macro_f1',
):
    """Evaluation for both single-label and multi-label node classification task

    Args:
        data (LogitsLabelsTuple): (logits, labels). Note that the logits should NOT
            be the value after sigmoid
        multi_label_threshold (Literal[&#39;macro_f1&#39;, &#39;micro_f1&#39;]
            | float | np.ndarray):
            Thresholds used for multi-label classification task. if the thresholds is
            one of 'macro_f1', 'micro_f1'. The best thresholds will be found using
            `evaluation.find_best_thresholds`.
            When the task is single-label classification, this argument will be ignored.


    Returns:
        (train_metrics, test_metrics1, test_metrics2, ...)
    """
    if len(train_data[1].shape) == 1:  # single-label
        return tuple(
            single_label_eval(*data) for data in [train_data, *test_data]
        )

    ## multi-label
    if (
        isinstance(multi_label_thresholds, str)
        and multi_label_thresholds in ['macro_f1', 'micro_f1']
    ):
        logits, labels = train_data
        multi_label_thresholds = find_best_thresholds(
            labels, torch.sigmoid(logits), objective_fn=multi_label_thresholds
        )

    return tuple(
        multi_label_eval(*data, thresholds=multi_label_thresholds)
        for data in [train_data, *test_data]
    )


def multi_label_eval(
    logits: torch.Tensor,
    label: torch.Tensor,
    *,
    thresholds: np.ndarray | float = 0.5,
):

    assert len(label.shape) == 2  # multi-label

    probas = torch.sigmoid(logits).detach().cpu().numpy()
    label = label.cpu().numpy()

    pred = np.array(probas > thresholds, dtype=int)

    macro_f1 = f1_score(label, pred, average='macro')
    micro_f1 = f1_score(label, pred, average='micro')
    return NodeClassificationMetrics(macro_f1=macro_f1, micro_f1=micro_f1)


def single_label_eval(
    logits: torch.Tensor,
    label: torch.Tensor,
):
    assert len(label.shape) == 1  # single-label

    logits = logits.detach().cpu().numpy()
    label = label.cpu().numpy()

    pred = logits.argmax(1)

    macro_f1 = f1_score(label, pred, average='macro')
    micro_f1 = f1_score(label, pred, average='micro')
    return NodeClassificationMetrics(macro_f1=macro_f1, micro_f1=micro_f1)


import dgl

# def _mrr2(
#     positive_probas, negative_probas, positive_graph: dgl.DGLHeteroGraph,
#     negative_graph: dgl.DGLHeteroGraph
# ):
#     if len(positive_graph.canonical_etypes) > 1:
#         raise NotImplementedError
#     for etype in positive_graph.canonical_etypes:
#         positive_graph.edges[etype].data['y'] = torch.ones(
#             positive_graph.num_edges(etype=etype), device=positive_graph.device
#         )
#         positive_graph.edges[etype].data['p'] = positive_probas
#         negative_graph.edges[etype].data['y'] = torch.zeros(
#             negative_graph.num_edges(etype=etype), device=negative_graph.device
#         )
#         negative_graph.edges[etype].data['p'] = negative_probas
#     graph: dgl.DGLHeteroGraph = dgl.merge([positive_graph, negative_graph])

#     def copy_e(edges):
#         return {'p': edges.data['p'], 'y': edges.data['y']}

#     def mrr(nodes):
#         rank = torch.argsort(-nodes.mailbox['p'])
#         sorted_label_array = nodes.mailbox['y'][rank]
#         pos_index = (sorted_label_array == 1).nonzero()
#         if len(pos_index) == 0:
#             return {'mrr': torch.nan}
#         return {'mrr': 1 / (1 + pos_index[0][0])}

#     graph.update_all(copy_e, mrr)
# pos_edges = positive_graph.edges()
# neg_edges = negative_graph.edges()
# edge_list = (
#     np.concatenate(
#         list(map(_handle_torch_tensor, [pos_edges[0], neg_edges[0]]))
#     ),
#     np.concatenate(
#         list(map(_handle_torch_tensor, [pos_edges[1], neg_edges[1]]))
#     )
# )


def _ranking_metrics(
    probas, labels, positive_graph: dgl.DGLHeteroGraph,
    negative_graph: dgl.DGLHeteroGraph
):

    pos_edges = [
        list(map(_handle_torch_tensor, positive_graph.edges(etype=etype)))
        for etype in positive_graph.canonical_etypes
    ]
    neg_edges = [
        list(map(_handle_torch_tensor, negative_graph.edges(etype=etype)))
        for etype in negative_graph.canonical_etypes
    ]
    edge_list = list(map(np.concatenate, zip(*(pos_edges + neg_edges))))

    # mrr_list, cur_mrr = [], 0
    metrics = defaultdict(list)
    t_dict, labels_dict, conf_dict = defaultdict(list), defaultdict(
        list
    ), defaultdict(list)
    for i, h_id in enumerate(edge_list[0]):
        t_dict[h_id].append(edge_list[1][i])
        labels_dict[h_id].append(labels[i])
        conf_dict[h_id].append(probas[i])

    max_k = max(map(len, conf_dict.values()))
    dcg_weights = 1 / np.log2(2 + np.arange(max_k))
    mrr_weights = 1 / (np.arange(max_k) + 1)
    idcg_cumulative_weights = dcg_weights.cumsum()
    imrr_cumulative_weights = mrr_weights.cumsum()

    for h_id in t_dict.keys():
        conf_array = np.array(conf_dict[h_id])
        rank = np.argsort(-conf_array)
        sorted_label = np.array(labels_dict[h_id])[rank]
        n_pos = int(sorted_label.sum())
        if n_pos == 0:
            continue
        pos_index = np.where(sorted_label == 1)[0]
        mrr = (sorted_label * mrr_weights[:len(sorted_label)]).sum()
        dcg = (sorted_label * dcg_weights[:len(sorted_label)]).sum()
        metrics['amrr'].append(mrr / n_pos)
        metrics['nmrr'].append(mrr / imrr_cumulative_weights[n_pos - 1])
        metrics['ndcg'].append(dcg / idcg_cumulative_weights[n_pos - 1])
        metrics['mrr'].append(1 / (1 + np.min(pos_index)))

    return {met: np.mean(vals) for met, vals in metrics.items()}


# def _roc_auc(pos_proba, neg_proba):
#     from sklearn.metrics import roc_auc_score
#     pos_prob = _handle_torch_tensor(pos_proba)
#     neg_prob = _handle_torch_tensor(neg_proba)
#     labels = np.concatenate([np.ones_like(pos_prob), np.zeros_like(neg_prob)])
#     probas = np.concatenate([pos_prob, neg_prob])
#     return roc_auc_score(labels, probas)


def link_prediction_eval(
    pos_logits, neg_logits, positive_graph, negative_graph
):
    """
    :param edge_list: shape(2, edge_num)
    :param confidence: shape(edge_num,)
    :param labels: shape(edge_num,)
    :return: dict with all scores we need
    """
    pos_prob = _handle_torch_tensor(torch.sigmoid(pos_logits))
    neg_prob = _handle_torch_tensor(torch.sigmoid(neg_logits))
    labels = np.concatenate([np.ones_like(pos_prob), np.zeros_like(neg_prob)])
    probas = np.concatenate([pos_prob, neg_prob])
    return {
        **_ranking_metrics(probas, labels, positive_graph, negative_graph),
        'roc_auc':
        roc_auc_score(labels, probas),
    }
