from __future__ import annotations

from typing import Literal

from torch.optim import Optimizer
from torch.optim.lr_scheduler import OneCycleLR

from ..misc import BaseConfig


class SchedulerConfig(BaseConfig):

    name: Literal['one_cycle'] = 'one_cycle'
    max_lr_scale: float
    pct_start: float
    verbose: bool | None = None
    """Whether show learning rate in progress bar"""

    def init(self, optimizer: Optimizer, total_steps: int):
        scheduler = OneCycleLR(
            optimizer,
            total_steps=total_steps,
            max_lr=[
                pg['lr'] * self.max_lr_scale for pg in optimizer.param_groups
            ],
            pct_start=self.pct_start,
        )
        return scheduler
