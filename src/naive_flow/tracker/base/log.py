import contextlib
import logging
import os
import sys
from typing import Dict, Literal

from tqdm import tqdm
from tqdm.contrib import DummyTqdmFile
from typing_extensions import deprecated


class Global:
    tracker_params = {}


def get_global():
    return Global.tracker_params


def set_global(
    progress: Literal["none", "tqdm", "plain"] = None,
    verbose: bool = None,
    log_root_dir: str = None,
):
    """Set global print options. This options will override aruguments for
        later initialized Tracker.
    """

    def gen():
        if progress is not None:
            yield ("progress", progress)
        if verbose is not None:
            yield ("verbose", verbose)
        if log_root_dir is not None:
            yield ("log_root_dir", log_root_dir)

    Global.tracker_params = dict(gen())
    return


@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    import naive_flow  # pylint: disable=import-outside-toplevel
    orig_out_err = sys.stdout, sys.stderr
    nf_out_err = naive_flow.stdout, naive_flow.stderr
    naive_flow.stdout, naive_flow.stderr = orig_out_err

    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err
        naive_flow.stdout, naive_flow.stderr = nf_out_err


class RangeTqdm(tqdm):

    def __init__(self, start: int, end: int, **kwargs):
        self.start = start
        self.end = end
        super().__init__(
            range(start, end),
            **kwargs,
            bar_format=(
                "Epoch: {cur}/{total_epochs}, {desc}"
                "|{bar}| "
                "[{elapsed}<{remaining}, {rate_fmt}{postfix}]"
            ),
        )
        return

    @property
    def format_dict(self):
        d = super().format_dict
        d.update({"total_epochs": self.end, "cur": self.start + self.n})
        return d


class _ScalarTqdm(tqdm):

    def __init__(self, **kwargs):
        super().__init__(bar_format="{desc}", **kwargs)
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


class ScalarTqdms:

    def __init__(self, **kwargs):
        self.position = kwargs.pop("position", None)
        self.kwargs = kwargs
        self.tqdms: Dict[str, _ScalarTqdm] = {}
        return

    def update(self, tag: str, msg: str):
        if tag not in self.tqdms:
            self.tqdms[tag] = _ScalarTqdm(
                position=self.position, **self.kwargs
            )
            if self.position is not None:
                self.position += 1

        self.tqdms[tag].set_description_str(msg)
        return

    def close(self):
        for scalar_tqdm in self.tqdms.values():
            scalar_tqdm.clear()
            scalar_tqdm.close()
        return

    def __del__(self):
        self.close()
        return


class _NullFile:
    """A dummy blackhole file-like object which acts like /dev/null"""

    def write(self, s: str):  # pylint: disable=unused-argument
        return

    def flush(self):
        return

    def close(self):
        return


class StdoutLogFile:

    def __init__(self, log_dir: str = None):
        """
        Args:
            log_file_root (str, optional): pass None or ignore to use stdout only.
        """
        if log_dir is not None:
            log_file_path = os.path.join(log_dir, "log.log")
            self.log = open(log_file_path, "a", encoding="utf-8")
        else:
            self.log = _NullFile()
        return

    def __del__(self):
        self.log.close()
        return

    def write(self, s: str):
        self.log.write(s)
        sys.stdout.write(s)
        return

    def flush(self):
        self.log.flush()
        sys.stdout.flush()
        return

    def close(self):
        self.log.close()
        return


FORMAT = ("%(asctime)s %(levelname)s\n"
          "%(message)s"
          "\n")


@deprecated("")
def del_file_handler(log_dir: str):
    name = __name__.rsplit(".", 3)[0]
    logger = logging.getLogger(name)
    for handler in logger.handlers:
        if handler.get_name() == log_dir:
            logger.removeHandler(handler)


@deprecated("")
def set_handlers(log_file_root: str = None):
    """
    Args:
        log_file_root (str, optional): pass None or ignore to use console handler only (without
            file handler).
    """

    name = __name__.rsplit(".", 3)[0]
    logger = logging.getLogger(name)
    if logger.level < logging.DEBUG:
        logger.setLevel(logging.DEBUG)
    if log_file_root is not None:
        log_file_path = os.path.join(log_file_root, "log.log")
        file_handler = logging.FileHandler(
            log_file_path, "a", encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(fmt=FORMAT))
        file_handler.setLevel(logging.DEBUG)
        file_handler.set_name(log_file_root)
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        logger.addHandler(file_handler)

    # stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(logging.INFO)
    # logger.addHandler(stream_handler)
    return


@deprecated("")
def get_logger(name: str = None, log_file_root: str = None):
    """
    Args:
        log_file_root (str, optional): pass None or ignore to use console handler only (without
            file handler).
    """
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    if log_file_root is not None:
        log_file_path = os.path.join(log_file_root, "log.log")
        file_handler = logging.FileHandler(
            log_file_path, "a", encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(fmt=FORMAT))
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    return logger
