from __future__ import annotations

import glob
import os
import shutil
import socket
import warnings
from datetime import datetime
from fnmatch import fnmatch
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    NamedTuple,
    Tuple,
    Type,
    TypedDict,
    Union,
    overload,
)

import termcolor
import torch

from .base.early_stopping_handler import EarlyStoppingHandler
from .base.history import CheckpointState, SaveReason
from .base.log import (
    Global,
    RangeTqdm,
    ScalarTqdms,
    StdoutLogFile,
    std_out_err_redirect_tqdm,
)
from .base.metrics import BUILTIN_METRICS, BUILTIN_TYPES, MetricsLike
from .checkpoint.utils import (
    TIME_FORMAT,
    list_checkpoints,
    list_checkpoints_later_than,
    parse_checkpoint_name,
)
from .tracker_config import TrackerConfig

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

__all__ = ["BaseTracker"]


class _CheckpointDict(TypedDict):

    user_data: dict
    epoch: int


class _Scalar(NamedTuple):

    tag: str
    metrics: MetricsLike

    @classmethod
    def from_arg(
        cls, tag: str, scalar_type: Union[BUILTIN_TYPES, MetricsLike]
    ):
        # ('a_loss/valid', 'loss')
        # ('b_loss/valid', 'loss')
        # ('a_metrics/valid', Metrics)
        # ('b_metrics/valid', Metrics)

        if isinstance(scalar_type, str):

            if BUILTIN_METRICS.get(scalar_type) is None:
                raise KeyError(
                    f"'{scalar_type}' is not a bulitin metrics. \n"
                    f"Currently available metrics are: {list(BUILTIN_METRICS.keys())}"
                )

            scalar_type = BUILTIN_METRICS.get(scalar_type)

        return cls(tag, scalar_type)


class _ScalarCache:

    def __init__(self):
        self.scalar_cache = [{"epoch": None}, {"epoch": None}]
        return

    def cache_scalar(self, tag: str, scalar, epoch: int):
        if self.scalar_cache[epoch & 1].get("epoch", None) != epoch:
            self.scalar_cache[epoch & 1] = {"epoch": epoch}

        self.scalar_cache[epoch & 1][tag] = scalar
        return

    def has_cached(self, epoch: int, tag: str = None):
        scalars = self.scalar_cache[epoch & 1]
        if scalars["epoch"] != epoch:
            return False
        if tag is None:
            return True
        return scalars.get(tag, False)

    def get_cached_scalars(self, epoch: int):
        scalars = self.scalar_cache[epoch & 1]
        assert scalars["epoch"] == epoch
        return scalars


# pylint: disable=invalid-name
MetricsArgT = Union[str, Tuple[str, Union[str, Type[MetricsLike]]]]


class BaseTracker:
    log_root_dir: str
    log_dir: str
    """_summary_
    \\RUNS
        ├─Oct12_19-00-17_DESKTOP-4A3P6E6nminst
        └─Oct12_19-01-16_DESKTOP-4A3P6E6nminst
    
        In the above case, \\RUNS is the log_root_dir, `Oct....` is the log_dir
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
        progress: Literal["plain", "tqdm", "none"] = None,
        enable_logging: bool = None,
        from_checkpoint: Union[str, Callable] = None,
    ):
        ...

    def __init__(self, from_checkpoint: Union[str, Callable] = None, **kwargs):

        kwargs.update(Global.tracker_params)
        if "config" in kwargs:
            warnings.warn(
                "The use of config as argument has been deprecated, "
                "use `**config.model_dump()` instead",
                # DeprecationWarning,
            )
            config = kwargs["config"]
        else:
            config = TrackerConfig.model_validate(kwargs)

        self.config = config

        self._es_metrics = None
        self._metrics: list[_Scalar] = []
        self._writer = None
        self._cur_epoch: int = None
        self._best_scalars: Dict[str, Any] = None
        self._scalar_cache = _ScalarCache()
        self._scalar_tqdms = None
        self._es_handler = None

        if from_checkpoint is None:
            # start a new training process
            self._log_dir = (
                new_time_formatted_log_dir(
                    config.comment, config.log_root_dir
                ) if config.log_root_dir else None
            )
            self._log_file = StdoutLogFile(self._log_dir)
            self.start_epoch = 0
            self._history: list[dict[str, float]] = []
        else:
            checkpoint_path = from_checkpoint if isinstance(from_checkpoint, str)\
                                                        else from_checkpoint(config.log_root_dir)
            self._log_dir, self._log_file, self.start_epoch, self._history =\
                    _load_checkpoint(checkpoint_path, config, user_load_hook=self.load)
            if self._history is None:
                self._history = []

        self._evaluated = False
        return

    @property
    def log_dir(self):
        assert self._log_dir is not None
        return self._log_dir

    @property
    def writer(self) -> SummaryWriter:
        assert self._writer is not None
        return self._writer

    @property
    def comment(self) -> str:
        return self.config.comment

    def _register_add_scalar_hook(self, writer: SummaryWriter):

        if getattr(writer.add_scalar, "__wrapper__", None) is self:
            # has been wrapped
            return writer

        self._writer = writer
        original_fn = writer.add_scalar

        @wraps(original_fn)
        def add_scalar_wrapper(
            tag,
            scalar_value,
            global_step=None,
            walltime=None,
            new_style=False,
            double_precision=False,
        ):
            original_fn(
                tag,
                scalar_value,
                global_step,
                walltime,
                new_style,
                double_precision,
            )
            return self._add_scalar_hook(tag, scalar_value, global_step)

        add_scalar_wrapper.__wrapper__ = self
        writer.add_scalar = add_scalar_wrapper
        return writer

    def create_summary_writer(
        self, max_queue: int = 10, flush_secs: int = 120,
        filename_suffix: str = ""
    ):
        """Create a registed summary writer. This is a shorthand of
        ```python
        writer = SummaryWriter(log_dir=tracker.log_dir, purge_step=tracker.start_epoch)
        writer = tracker.register_summary_writer(writer)
        ```

        Returns:
            SummaryWriter: return a summary writer which has been registed
        """
        assert self._writer is None, (
            "Try to create a new summary writer while there is one writer that has been registered"
        )
        writer = SummaryWriter( #pylint: disable=used-before-assignment
            log_dir=self.log_dir,
            purge_step=self.start_epoch,
            max_queue=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )
        return self._register_add_scalar_hook(writer)

    def register_summary_writer(self, writer: SummaryWriter):
        assert self._writer is None, (
            "Try to register a summary writer while another summary writer has been registerd."
        )
        return self._register_add_scalar_hook(writer)

    def register_scalar(
        self,
        tag: str,
        scalar_type: Union[BUILTIN_TYPES, MetricsLike],
        for_early_stopping: bool = False,
    ):

        if for_early_stopping:
            assert self._es_metrics is None, (
                "try to register a second scalar for early stopping, which is not allowed."
            )
            self._es_metrics = _Scalar.from_arg(tag, scalar_type)

        else:
            self._metrics.append(_Scalar.from_arg(tag, scalar_type))
        return

    def range(self, to_epoch: int, position: int | list[int] | None = None):
        """
        Args:
            to_epoch (int): Target epoch.
            position (int | list[int] | None, optional): position passed to tqdm. Defaults to None.
                The multiple positions specified, they will be spread to 
                Main bar, early-stopping-counter, scalars, respectively.
        """

        if to_epoch <= self.start_epoch:
            raise ValueError(
                f"expect to_epoch > {self.start_epoch}, got: to_epoch={to_epoch}"
            )

        positions = dict(
            enumerate(
                position if isinstance(position, Iterable) else [position]
            )
        )

        with std_out_err_redirect_tqdm() as orig_stdout:
            pbar = RangeTqdm(
                self.start_epoch,
                to_epoch,
                leave=True,
                position=positions.get(0, None),
                file=orig_stdout,
                dynamic_ncols=True,
                disable=(self.config.progress != "tqdm"),
            )
            self._es_handler = EarlyStoppingHandler(
                self._log_file,
                self.config,
                file=orig_stdout,
                position=positions.get(1, None),
            )
            if self.config.progress != "plain":
                self._scalar_tqdms = ScalarTqdms(
                    file=orig_stdout,
                    disable=self.config.progress != "tqdm",
                    position=positions.get(2, None),
                )
            for epoch in pbar:

                self._cur_epoch = epoch
                self._evaluated = False

                if self.config.progress == "plain":
                    print(f"Epoch {epoch} / {to_epoch}:", file=self._log_file)
                else:
                    print(
                        f"Epoch {epoch} / {to_epoch}:", file=self._log_file.log
                    )

                yield epoch

                if self._evaluated is False:
                    if self.config.early_stopping_rounds > 0:
                        pass
                        # warnings.warn(
                        #     "EarlyStoppingWarning:\n"
                        #     "Early stopping is enable while no criterion scalar has been added.\n"
                        #     "TODO: provide ways to suppress this warning\n"  #TODO
                        # )

                save_reason = SaveReason()

                if epoch == to_epoch - 1:
                    save_reason.end = True

                if (
                    self.config.epochs_per_checkpoint
                    and (epoch + 1 - self.start_epoch) %
                    self.config.epochs_per_checkpoint == 0
                ):
                    save_reason.regular = self.config.epochs_per_checkpoint

                if self._es_handler.should_stop(epoch):
                    if self.config.progress == "plain":
                        print("Early stopping!", file=self._log_file)
                    else:
                        print("Early stopping!", file=self._log_file.log)
                    save_reason.early_stopping = True
                    if self.config.save_end is True:
                        self._base_save(epoch, save_reason=save_reason)
                    break

                if self._es_handler.is_best_epoch(epoch):
                    save_reason.best = True
                    if self._es_metrics is not None:
                        self.get_best_scalars(True)

                if (
                    save_reason.end and self.config.save_end
                    or save_reason.regular
                    or save_reason.best and self.config.save_n_best
                ):
                    self._base_save(epoch, save_reason=save_reason)

        self._es_handler.close()
        self._scalar_tqdms.close()
        if self.config.progress == "plain":
            print(
                f"Training is finish for epochs: {to_epoch}",
                file=self._log_file
            )
        else:
            print(
                f"Training is finish for epochs: {to_epoch}",
                file=self._log_file.log
            )

        self.start_epoch = self._cur_epoch + 1
        return

    def get_best_scalars(self, no_within_loop_warning: bool = False):
        """Get the best scalars if scalar for earlystopping has been registered.

        Args:
            no_within_loop_warning (bool, optional): 
                If `get_best_scalars` is called within loop, the metric registered for earlystopping
                is evaluated as the best,
                the scalars having not been added will not be included.
                If the get_best_scalars is called after all of the scalars for 
                this epoched added, user can set `no_within_loop_warning=True` to suppress 
                warning.
                Defaults to False.

        Returns:
            dict[str, float]: best metrics
        """

        assert self._es_metrics is not None, (
            "You have to have registered summarywriter in tracker using "
            "register_scalar(..., for_early_stopping=True) so that tracker can know what are the "
            "best metrics for an epoch"
        )

        if self.start_epoch < self._cur_epoch + 1:
            # called within training loop
            if not self._evaluated:
                # called before  for-earlystopping scalar added
                # return self._best_scalars
                pass

            if no_within_loop_warning is False:
                warnings.warn(
                    "get_best_scalars called within loop. If this epoch is evaluated as the best"
                    ", and some of the scalars have not been added, they will not be included.\n"
                    "You can make sure the get_best_scalars is called after all of the scalars for "
                    "this epoched added, and pass no_within_loop_warning=True to suppress this "
                    "warning."
                )
            # update best_scalar
            if self._es_handler.is_best_epoch(self._cur_epoch):
                scalars = self._scalar_cache.get_cached_scalars(
                    self._cur_epoch
                )
                self._best_scalars = scalars
            return self._best_scalars

        assert self.start_epoch == self._cur_epoch + 1
        # call after training loop
        return self._best_scalars

    def is_regular_saving_epoch(self, epoch: int):
        """Return whether the epoch is a regular saving epoch determined by
            `epochs_per_checkpoint`

        """
        return (
            bool(self.config.epochs_per_checkpoint) and
            (epoch + 1 - self.start_epoch) % self.config.epochs_per_checkpoint
            == 0
        )

    def is_best_epoch(self, epoch: int):
        return self._es_handler.is_best_epoch(epoch)

    def add_scalar(
        self,
        tag: str,
        scalar_value,
        global_step=None,
    ):
        self._add_scalar_hook(tag, scalar_value, global_step)
        return

    def _add_scalar_hook(
        self,
        tag: str,
        scalar_value: float,
        global_step=None,
    ):
        if global_step is None:
            global_step = self._cur_epoch or 0
        else:
            if global_step > self._cur_epoch:
                warnings.warn(
                    f"A scalar {tag} is being added for epoch={global_step}, while the current "
                    f"epoch is {self._cur_epoch}"
                )

        def handle_tag():
            tem = tag.split("/", maxsplit=1)
            if len(tem) == 1:
                return tag, None
            return tem

        name, group = handle_tag()

        def find_registered_metric():
            if self._es_metrics is not None and tag == self._es_metrics.tag:
                if self._evaluated is True:
                    raise RuntimeError(
                        "More than one evaluation scalar were added during an epoch."
                    )

                metric_value = self._es_metrics.metrics(scalar_value)
                self._es_handler.update(tag, metric_value, global_step)
                self._evaluated = True
                return metric_value

            for pattern, met in self._metrics:
                if fnmatch(tag, pattern):
                    return met(scalar_value)

            if name in BUILTIN_METRICS:
                return BUILTIN_METRICS[name](scalar_value)
            return None

        cached_scalar = self._scalar_cache.has_cached(global_step, tag)
        if cached_scalar is not False:
            assert cached_scalar == scalar_value, (
                f"more than one unique scalar value was added for tag: {tag}, epoch: {global_step}"
            )
            return

        metric_value = find_registered_metric()
        if metric_value is None:
            return

        self._scalar_cache.cache_scalar(tag, scalar_value, global_step)

        group_str = f"/{group}" if group is not None else ""

        msg = f"{name:15}{group_str:6}: {metric_value}"
        if self._es_metrics is not None and tag == self._es_metrics.tag:
            attrs = ["underline"
                     ] if self._es_handler.is_best_epoch(global_step) else []
            msg = termcolor.colored(msg, "cyan", attrs=attrs)
        if self.config.progress == "plain":
            print(msg, file=self._log_file)
        else:
            print(msg, file=self._log_file.log)
            if self._scalar_tqdms is not None:
                self._scalar_tqdms.update(tag, msg)
        return

    def save_checkpoint(self, suffix: str = ""):
        """Save the current state as a checkpoint manually.

        Args:
            suffix (str, optional): Suffix append to the filename of checkpoint.
            Defaults to "".
        """

        return self._base_save(self._cur_epoch, suffix=suffix)

    def _base_save(
        self, cur_epoch: int, suffix: str = None,
        save_reason: SaveReason = None
    ) -> str:

        if self._scalar_cache.has_cached(cur_epoch):
            self._history.append(
                self._scalar_cache.get_cached_scalars(cur_epoch)
            )

        if suffix is None:
            suffix = "" if save_reason is None else _save_reason_suffix(
                save_reason
            )

        name = f"epoch_{cur_epoch}{suffix}"
        path = os.path.join(self.log_dir, name)
        os.makedirs(path, exist_ok=False)

        checkpoint_state = CheckpointState(
            best_metric_criterion=self._es_metrics and self._es_metrics.tag,
            best_metrics=self._es_metrics and self.get_best_scalars(True),
            save_reason=save_reason,
            epoch=cur_epoch,
            log_history=self._history,
        )
        with open(
            os.path.join(path, "checkpoint_state.json"), "w", encoding="utf8"
        ) as fout:
            fout.write(checkpoint_state.model_dump_json(indent=4))
        user_data = self.save()
        for key, item in user_data.items():
            torch.save(item, os.path.join(path, f"{key}.pth"))

        if self.config.verbose:
            print(f"Checkpoint: {name} has been saved.", file=self._log_file)
        else:
            print(
                f"Checkpoint: {name} has been saved.", file=self._log_file.log
            )

        if self.config.save_n_best:
            # purge redundant checkpoints
            best_checkpoints = list_checkpoints(self.log_dir, "_best")
            not_n_best = best_checkpoints[:-self.config.save_n_best]

            for filename in not_n_best:
                file = os.path.join(self.log_dir, filename)
                if os.path.isdir(file):
                    shutil.rmtree(file)
                else:
                    os.remove(file)  # for backward compatibility
                print(
                    f"Checkpoint: {filename} was purged "
                    f"due to save_n_best={self.config.save_n_best}.",
                    file=(
                        self._log_file
                        if self.config.verbose else self._log_file.log
                    ),
                )

        return name

    def __del__(self):
        if hasattr(self, "_log_file"):
            self._log_file.close()
        return

    def list_checkpoints(self):
        return list_checkpoints(self.log_dir)

    def load(self, checkpoint_dict: dict):
        raise NotImplementedError

    def save(self) -> dict:
        raise NotImplementedError


def formatted_now():
    return datetime.now().strftime(TIME_FORMAT)


def new_time_formatted_log_dir(
    comment: str,
    log_root_dir: str,
):
    current_time = formatted_now()
    comment = comment or ""
    log_dir = os.path.join(
        log_root_dir, current_time + "_" + socket.gethostname() + comment
    )

    os.makedirs(log_dir)

    return log_dir


def prompt_delete_later(checkpoint_name: str, to_delete: List[str]):

    to_delete_str = "\n".join(
        f"\t{i + 1}. {name}" for i, name in enumerate(to_delete)
    )

    print(
        f"You are trying to load checkpoint: {checkpoint_name}\n" +
        to_delete_str + "\n"
        f"While there are {len(to_delete)} checkpoints with larger epochs shown above, "
        "you can choose to delete[D] them or keep[K] them and create a new log_dir.\n"
        "Note that you can set the `tracker_config.delete_ok` to avoid this prompt."
    )

    while True:
        option = input("Delete [D], Keep [K]: ")
        if option.lower() == "d":
            return True

        if option.lower() == "k":
            return False


def _load_checkpoint(
    checkpoint_path: str,
    config: TrackerConfig,
    user_load_hook: Callable,
):
    """Load the specfiic checkpoint

    Args:
        checkpoint_path (str): path of saved checkpoint
            Notice that if the directory of the checkpoint does not in the
            `self.log_root_dir`, then it would be consideranother directory will be
            created in `self.log_root_dir`

    """

    def _load_data_legacy(checkpoint_path: str):
        checkpoint_dict: _CheckpointDict = torch.load(checkpoint_path)

        user_data = checkpoint_dict.get("user_data", checkpoint_dict)
        user_load_hook(user_data)
        start_epoch = checkpoint_dict.get("epoch", -1) + 1
        return start_epoch

    def _load_model_data_solely(checkpoint_path: str):
        user_data = {}
        name, _ = os.path.splitext(os.path.basename(checkpoint_path))
        user_data[name] = torch.load(checkpoint_path)
        user_load_hook(user_data)
        return

    def _load_data(checkpoint_path: str):

        with open(
            os.path.join(checkpoint_path, "checkpoint_state.json"), "r",
            encoding="utf8"
        ) as fin:
            checkpoint_state = CheckpointState.model_validate_json(fin.read())

        user_data = {}
        for path in glob.glob(os.path.join(checkpoint_path, "*.pth")):
            name, _ = os.path.splitext(os.path.basename(path))
            user_data[name] = torch.load(path)
        user_load_hook(user_data)
        start_epoch = checkpoint_state.epoch + 1
        return start_epoch, checkpoint_state.log_history

    if os.path.isdir(checkpoint_path):
        start_epoch, log_history = _load_data(checkpoint_path)
    elif parse_checkpoint_name(os.path.basename(checkpoint_path)) is not None:
        start_epoch = _load_data_legacy(checkpoint_path)
        log_history = None
    else:
        _load_model_data_solely(checkpoint_path)
        start_epoch = 0
        log_history = None

    checkpoint_log_dir = os.path.dirname(checkpoint_path)
    checkpoint_root_dir = os.path.dirname(checkpoint_log_dir)

    def need_create_new_dir(delete_ok: bool) -> bool:
        if os.path.abspath(checkpoint_root_dir) != os.path.abspath(
            config.log_root_dir
        ):
            return True

        to_delete = list_checkpoints_later_than(checkpoint_path)

        if len(to_delete) == 0:
            return False

        delete_ok = delete_ok if delete_ok is not None\
                        else prompt_delete_later(os.path.basename(checkpoint_path), to_delete)

        if delete_ok is False:
            return True

        for name in to_delete:
            path = os.path.join(checkpoint_log_dir, name)
            os.remove(path)
            if config.verbose:
                print(f"Checkpoint: {name} was purged.")

        return False

    if config.log_root_dir is not False:
        if need_create_new_dir(config.delete_ok):
            # create a new log_dir in the root_dir
            log_dir = new_time_formatted_log_dir(
                config.comment, config.log_root_dir
            )
        else:
            # use the existing log_dir
            log_dir = checkpoint_log_dir
    else:
        log_dir = None

    log_file = StdoutLogFile(log_dir)
    if config.verbose:
        print(
            f"Checkpoint {os.path.basename(checkpoint_path)} has been loaded.",
            file=log_file
        )
    else:
        print(
            f"Checkpoint {os.path.basename(checkpoint_path)} has been loaded.",
            file=log_file.log
        )
    return (log_dir, log_file, start_epoch, log_history)


def _save_reason_suffix(save_reason: SaveReason):
    if save_reason.best:
        return "_best"

    if save_reason.early_stopping or save_reason.end:
        return "_end"

    return ""


# def once_assign_property(attribute: str):

#     _attribute = f"_{attribute}"

#     def getter(self):
#         attr = getattr(self, _attribute, None)
#         if attr is None:
#             raise ValueError(f"'{attribute}' is referenced before assigned.")
#         return attr

#     def setter(self, value):
#         attr = getattr(self, _attribute, None)
#         if attr is not None:
#             raise ValueError(f"'{attribute}' is not mutable")

#         setattr(self, _attribute, value)
#         return

#     return property(getter, setter)
