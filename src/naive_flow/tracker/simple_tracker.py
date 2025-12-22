from typing import Union, Callable, overload, Literal

from torch import nn, optim
from .base_tracker import BaseTracker


class SimpleTracker(BaseTracker):

    model: nn.Module
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler._LRScheduler

    @overload
    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        *,
        log_root_dir: str = None,
        epochs_per_checkpoint: int = None,
        early_stopping_rounds: int = None,
        save_n_best: int = None,
        save_end: bool = None,
        comment: str = None,
        verbose: bool = None,
        progress: Literal['plain', 'tqdm', 'none'] = None,
        enable_logging: bool = True,
        from_checkpoint: Union[str, Callable] = None,
    ):
        """Tracker for classic setup: a model, an optimizer, and a scheduler (optional).

        Handle checkpoint saving, early stopping, and integrate with tensorboard.SummaryWriter

        Args:
            model (nn.Module): model would be saved automatically basen on the settings
            optimizer: 
            scheduler: Optional.
            log_root_dir (str, optional): `log_root_dir` for logs
                ```text
                LOG_ROOT_DIR
                ├── LOG_DIR_1
                └── LOG_DIR_2
                ```
                Here is an example of set `runs` as `log_root_dir`, Tracker will create
                `log_dir` following the naming rule (DATETIME_HOSTNAME_COMMENT) and those
                created `log_dir` will be put in `log_root_dir`. Note that here DATETIME
                and HOSTNAME will be generated in runtime automatically.
                While the COMMENT can be set by user in the `tracker_config__comment` field.

                ```text
                runs
                ├── Oct14_23-34-08_user_HGT_IMDB
                │   ├── events.out.tfevents.1697297648.user.1838590.0
                │   ├── log.log
                │   ├── Oct14_23-34-54_epoch_51_best
                │   └── Oct14_23-35-14_epoch_81_end
                ├── Oct15_00-17-18_user_HGT_acm
                │   ├── events.out.tfevents.1697300238.user.1861603.0
                │   ├── log.log
                │   ├── Oct15_00-19-14_epoch_74_best
                │   └── Oct15_00-20-01_epoch_104_end
                └── Oct15_00-21-21_user_HGT_dblp
                    ├── events.out.tfevents.1697300481.user.1863785.0
                    ├── log.log
                    ├── Oct15_00-21-45_epoch_48_best
                    └── Oct15_00-21-59_epoch_78_end
                ```

            epochs_per_checkpoint (int): Num of epochs per checkpoints. Default to 0
                By setting this field with positive integer, Tracker will save
                    the checkpoint regularly
                E.g.:
                    epochs_per_checkpoint == 1:
                        Tracker will save a checkpoint for every epoch

                    epochs_per_checkpoint == 0: Tracker will not save checkpoint regularly.
                        Note that Tracker will still save checkpoint at the end of training
                        and may save some checkpoints if `save_n_best` is set with 
                        poistive integer.

            early_stopping_rounds (int): Defaults to 0.
                Early stopping threshold. If early_stopping_rounds == 0, then early stopping would
                not be enable. I.e. model would train until the specified epoch

            save_n_best (int): Default to 1.
                only save n latest models with best validation scorea.

                If set to 0, no checkpoint would be saved due to its validation score. (still some
                checkpoints might be saved due to `epochs_per_checkpoint > 0`)

                If set to 1, every better model would be saved, and the previously best-saved
                models would get deleted.

            save_end (bool): Default to True.
                Whether to save the checkpoint of the last epoch.

                When True, the last epoch will always be saved.
                When False, The last checkpoint may still be saved if `save_n_best` > 0,
                and the last checkpoint has the best evaluation score.

            comment (str): Same as the argument of SummaryWriter. Defaults to None.

            enable_logging (bool): _description_. Defaults to True.
                whether log to file "log.log". It"s useful to turn this off when inference on kaggle

            from_checkpoint (str): Defaults to None. If omit, it will start a new training
                procedure, if a path to checkpoint is specified, the checkpoint will be load and
                start from which epoch the checkpoint saved.
                This argument can be set using
                    1. tracker.checkpoint.best(...) automatically get the best evaluated checkpoint
                    2. tracker.checkpoint.latest(...) automatically get the latest checkpoint
                    3. tracker.checkpoint.parse_args() use the command line arguments to decide
                        the behavior. Use "python ... --help" to get the manual
        """

    def __init__(
        self,
        model,
        optimizer,
        scheduler=None,
        from_checkpoint: Union[str, Callable] = None,
        **kwargs,
    ):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        super().__init__(
            **kwargs,
            from_checkpoint=from_checkpoint,
        )
        return

    def load(self, checkpoint_dict: dict):

        self.model.load_state_dict(checkpoint_dict['model'])
        if 'optimizer' in checkpoint_dict:
            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        if self.scheduler is not None and 'scheduler' in checkpoint_dict:
            self.scheduler.load_state_dict(checkpoint_dict['scheduler'])

        return

    def save(self) -> dict:

        scheduler_dict = self.scheduler.state_dict() if self.scheduler else {}

        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': scheduler_dict,
        }
