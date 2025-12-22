from __future__ import annotations

import json
from typing import Literal

from pydantic import field_validator

from ..misc import BaseConfig


class EarlyBreakingConfig(BaseConfig):
    """A trick which may be useful in large-scale hyperparameter searching.
    For each checkpoint, it checks whether current best metric better than the specified threshold.
    If not, the trainnig loop break immedidately.

    For example:
    checkpoints: [(20, 0.54), (40, 0.55), (80, 0.56)]

    Break if best/acc < 0.54 at epoch 20
    Break if best/acc < 0.55 at epoch 40
    Break if best/acc < 0.56 at epoch 80
    """
    metric: str
    mode: Literal['min', 'max']
    checkpoints: list[tuple[int, float]]
    # list of (epoch, metric-threshold)

    @field_validator('checkpoints', mode='before')
    @classmethod
    def load_list(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    def whether_break(
        self, cur_epoch: int, current_best_metrics: dict[str, float]
    ):
        assert self.metric in current_best_metrics
        for epoch, threshold in self.checkpoints:
            if epoch != cur_epoch:
                continue
            cur = current_best_metrics[self.metric]
            is_better = (
                cur > threshold if self.mode == 'max' else cur < threshold
            )
            if not is_better:
                return True
        return False
