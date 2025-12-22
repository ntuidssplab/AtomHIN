from typing import Union, Callable, overload, Literal
from .base_tracker import BaseTracker, MetricsLike, BUILTIN_TYPES


class DummyTracker(BaseTracker):
    """Dummy tracker that takes neither models, optimizers, nor schedulers
    """

    @overload
    def __init__(
        self,
        *,
        log_root_dir: str = None,
        epochs_per_checkpoint: int = None,
        early_stopping_rounds: int = None,
        save_n_best: int = None,
        save_end: bool = None,
        comment: str = None,
        verbose: bool = None,
        progress: Literal['plain', 'tqdm', 'none'] = None,
        from_checkpoint: Union[str, Callable] = None,
    ):
        """Dummy tracker that takes neither models, optimizers, nor schedulers

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

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        from_checkpoint: Union[str, Callable] = None,
        **kwargs,
    ):
        super().__init__(
            from_checkpoint=from_checkpoint,
            **kwargs,
        )
        return

    def load(self, checkpoint_dict: dict):
        return

    def save(self) -> dict:
        return {}

    def range(self, to_epoch: int):
        raise RuntimeError('DummyTracker.range should never be accessed.')

    @overload
    def get_best_scalars(self):  # pylint: disable=arguments-differ
        ...

    def get_best_scalars(self, no_within_loop_warning: bool = False):
        assert no_within_loop_warning is False
        cached_scalar = self._scalar_cache.get_cached_scalars(0)
        return cached_scalar

    @overload
    def register_scalar( # pylint: disable=arguments-differ
        self,
        tag: str,
        scalar_type: Union[BUILTIN_TYPES, MetricsLike],
    ):
        ...

    def register_scalar(
        self, tag: str, scalar_type: BUILTIN_TYPES | MetricsLike,
        for_early_stopping: bool = False
    ):
        assert for_early_stopping is False
        return super().register_scalar(tag, scalar_type, for_early_stopping)
