import numpy as np
from tqdm import tqdm

from ..tracker_config import TrackerConfig
from .log import StdoutLogFile
from .metrics import MetricsLike

_ES_BAR_FORMAT = ("EarlyStoppingCounter: {n_fmt}/{total_fmt}, "
                  "{desc}")


class EarlyStoppingTqdm(tqdm):

    def __init__(self, early_stopping_rounds: int, **kwargs):
        super().__init__(
            range(early_stopping_rounds),
            bar_format=_ES_BAR_FORMAT if early_stopping_rounds else "{desc}",
            **kwargs,
        )
        return

    @classmethod
    def _get_free_pos(cls, instance=None):
        instances = [
            abs(inst.pos) for inst in cls._instances
            if inst is not instance and hasattr(inst, "pos")
        ]
        if instances:
            return max(instances) + 1
        return 0


class EarlyStoppingHandler:

    best_criterion: MetricsLike
    counter: int

    def __init__(
        self, log_file: StdoutLogFile, config: TrackerConfig, **tqdm_kwargs
    ):
        self.best_criterion = None
        self.counter = 0
        self.best_epoch = -1
        self.best_name = None
        self.early_stopping_rounds = config.early_stopping_rounds
        self.config = config
        self.log_file = log_file
        if config.progress == "tqdm":
            self.pbar = EarlyStoppingTqdm(
                config.early_stopping_rounds, **tqdm_kwargs
            )
        else:
            self.pbar = None
        return

    def __del__(self):
        if self.pbar is not None:
            self.pbar.close()

    def update(self, name: str, new_criterion: MetricsLike, epoch: int):

        if (
            self.best_criterion is not None
            and not np.isnan(self.best_criterion.value)
            and not new_criterion.better_than(self.best_criterion)
        ):
            self.counter += 1
            if self.pbar is not None:
                self.pbar.update()
            return

        self.best_criterion = new_criterion
        self.best_epoch = epoch
        self.best_name = name
        self.counter = 0
        if self.pbar is not None:
            self.pbar.reset()
        return

    def is_best_epoch(self, epoch: int) -> bool:
        return self.best_epoch >= 0 and self.best_epoch == epoch

    def should_stop(self, epoch: int) -> bool:

        if epoch == self.best_epoch:
            self._print_best_criterion("New")
            return False

        if self.early_stopping_rounds:
            print(
                "Early stopping counter:"
                f"{self.counter} / {self.early_stopping_rounds}",
                file=self.log_file
                if self.config.progress == "plain" else self.log_file.log
            )

        self._print_best_criterion("Current")
        return (
            self.early_stopping_rounds > 0
            and self.counter == self.early_stopping_rounds
        )

    def close(self):
        if self.pbar is not None:
            self.pbar.close()
        return

    def _print_best_criterion(self, new_or_current: str):
        if self.best_criterion is not None:
            msg = (
                f"{new_or_current} best {self.best_name}@epoch{self.best_epoch}: "
                f"{self.best_criterion}"
            )
            print(
                msg, file=self.log_file
                if self.config.progress == "plain" else self.log_file.log
            )
            if self.pbar is not None:
                self.pbar.set_description(msg)
        return
