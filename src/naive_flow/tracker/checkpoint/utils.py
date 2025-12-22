from typing import NamedTuple
import re
import os
from datetime import datetime

LOG_DIR_PATTERN = re.compile(r"(\d{6}_\d{2}-\d{2}-\d{2})_(.*)")
TIME_FORMAT = r"%y%m%d_%H-%M-%S"
CHECKPOINT_PATTERN = re.compile(r"(.*?)_?epoch[_\-](\d+)(.*)")


class LogDirParseResult(NamedTuple):
    time: datetime
    info: str


def parse_log_dir(log_dir_name: str):
    match = re.match(LOG_DIR_PATTERN, log_dir_name)
    if match is None:
        return None

    time_str = match.group(1)
    info = match.group(2)
    return LogDirParseResult(datetime.strptime(time_str, TIME_FORMAT), info)


class CheckpointParseResult(NamedTuple):
    time: str
    epoch: int
    suffix: str


def parse_checkpoint_name(checkpoint_name: str):
    match = re.match(CHECKPOINT_PATTERN, checkpoint_name)
    if match is None:
        return None
    return CheckpointParseResult(
        match.group(1), int(match.group(2)), match.group(3)
    )


def get_latest_log_dir(log_root_dir: str):
    """
    Structure of log_root_dir:
        └──log_root_dir
           ├─221012_16-08-48_DESKTOP-4A3P6E6
           |      20221017T01-44-38_epoch_10
           |      20221017T01-44-48_epoch_20
           │
           └─221013_17-09-52_DESKTOP-4A3P6E6
                   20220517T01-44-31_epoch_10
                   20220517T01-44-41_epoch_20
    """

    def get_dir_datetime(name: str) -> datetime:
        """check whether a name of dir is contains formatted time
        """
        result = parse_log_dir(name)
        if result is None:
            return None

        path = os.path.join(log_root_dir, name)
        if not os.path.isdir(path):
            return None

        return result.time

    arr = [
        (dir_name, get_dir_datetime(dir_name))
        for dir_name in os.listdir(log_root_dir)
    ]
    arr = list(filter(lambda tup: tup[1] is not None, arr))

    if len(arr) == 0:
        return None

    latest_dir, _ = max(arr, key=lambda tup: tup[1])
    latest_dir = os.path.join(log_root_dir, latest_dir)
    return latest_dir


def list_checkpoints(log_dir: str, comment_filter: str | None = None):
    """List the checkpoint within `log_dir`.

    Args:
        log_dir (str): log dir
        comment_filter (str, optional): filter to the comment (suffix) of checkpoints
    """

    def unorder_list():
        for name in os.listdir(log_dir):

            result = parse_checkpoint_name(name)
            if result is None:
                continue

            yield (result, name)

    checkpoints_list = list(unorder_list())
    checkpoints_list.sort(key=lambda t: t[0].epoch)

    if comment_filter is not None:
        return [
            name for parse_result, name in checkpoints_list
            if re.search(comment_filter, parse_result.suffix) is not None
        ]
    return [name for _, name in checkpoints_list]


def list_checkpoints_later_than(checkpoint_path: str):

    checkpoint_name = os.path.basename(checkpoint_path)
    log_dir = os.path.dirname(checkpoint_path)

    checkpoint_list = list_checkpoints(log_dir)
    idx = checkpoint_list.index(checkpoint_name)

    return checkpoint_list[idx + 1:]
