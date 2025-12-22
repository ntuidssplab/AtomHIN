from __future__ import annotations

from functools import cached_property
from typing import ClassVar, Iterable, Literal

from pydantic import Field

from ...evaluation import (
    LogitsLabelsTuple,
    link_prediction_eval,
    node_classification_eval,
)
from ..misc import BaseConfig


class BaseEvaluatorConfig(BaseConfig):
    name: str

    early_stopping_objective: str
    metrics: ClassVar[list[str]]
    epochs_per_eval: int | float = 1
    """epochs per evaluation.
        This can be a fraction number in (0, 1) if using mini-batch training.
    """

    def eval(
        self, train_data: LogitsLabelsTuple, *test_data: LogitsLabelsTuple
    ) -> Iterable[dict[str, float]]:
        raise NotImplementedError


class HGBNodeClassificationEvaluatorConfig(BaseEvaluatorConfig):

    name: Literal['f1s', 'hgb-nc'] = 'f1s'
    metrics: ClassVar[list[str]] = ['macro_f1', 'micro_f1']
    multi_label_thresholds_objective: Literal['macro_f1', 'micro_f1'] = Field(
        'macro_f1', legacy='use hgb-mnc for multi-label tasks instead.',
        exclude=True
    )
    early_stopping_objective: Literal['macro_f1', 'micro_f1',
                                      'loss'] = 'macro_f1'

    def eval(
        self, train_data: LogitsLabelsTuple, *test_data: LogitsLabelsTuple
    ):

        return node_classification_eval(
            train_data,
            *test_data,
            multi_label_thresholds=self.
            multi_label_thresholds_objective,  # NOTE: no effect on single-label tasks
        )


class HGBMultiLabelNodeClassificationEvaluatorConfig(
    HGBNodeClassificationEvaluatorConfig
):

    name: Literal['hgb-mnc'] = 'hgb-mnc'
    multi_label_thresholds_objective: Literal['macro_f1',
                                              'micro_f1'] = 'macro_f1'

    def eval(
        self, train_data: LogitsLabelsTuple, *test_data: LogitsLabelsTuple
    ):

        return node_classification_eval(
            train_data,
            *test_data,
            multi_label_thresholds=self.multi_label_thresholds_objective,
        )


class OGBEvaluator(BaseEvaluatorConfig):
    name: Literal['acc'] = 'acc'
    metrics: ClassVar[list[str]] = ['acc']
    early_stopping_objective: Literal['acc'] = 'acc'

    def eval(
        self, train_data: LogitsLabelsTuple, *test_data: LogitsLabelsTuple
    ):

        for data in [train_data, *test_data]:
            logits, y_true = data
            yield self._evaluator.eval(
                {
                    'y_true': y_true.unsqueeze(dim=-1),
                    'y_pred': logits.argmax(1).unsqueeze(dim=-1),
                }
            )
        return

    @cached_property
    def _evaluator(self):
        from ogb.nodeproppred import Evaluator
        return Evaluator(name='ogbn-mag')


class HGBLinkPredEvaluator(BaseEvaluatorConfig):
    name: Literal['hgb-lp'] = 'hgb-lp'
    metrics: ClassVar[list[str]] = ['roc_auc', 'mrr', 'amrr', 'nmrr', 'ndcg']
    early_stopping_objective: Literal['loss', 'roc_auc', 'mrr'] = 'loss'
    calculate_train_metrics: bool | None = None

    def eval(self, train_data, *test_data):
        #  (
        #     pos_logits,
        #     neg_logits,
        #     data.positive_hg,
        #     data.negative_hg,
        # )

        if self.calculate_train_metrics:
            yield link_prediction_eval(train_data)
        else:
            yield {}
        for data in test_data:
            yield link_prediction_eval(*data)
        # for data in [train_data, *test_data]:
        #     yield link_prediction_eval(*data)
        return
