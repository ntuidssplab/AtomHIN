import sys

from typing_extensions import deprecated

from . import tracker
from .config.utils import dump_config, load_env_file, strfconfig
from .tracker.base.log import get_global, set_global

stdout = sys.stdout
"""A way to get original stdout which may be redirected to tqdm."""

stderr = sys.stderr
"""A way to get original stderr which may be redirected to tqdm."""


@deprecated(
    "naive_flow.load_checkpoint has been deprecated for the new checkpoint format, "
    "as user can load the .pth in the checkpoint directory directly."
)
def load_checkpoint(checkpoint_path: str):
    # pylint: disable=import-outside-toplevel
    """Load the specific checkpoint save by Tracker"""
    import glob
    import os
    from typing import get_type_hints

    import torch

    from .tracker.base_tracker import _CheckpointDict

    if os.path.isfile(checkpoint_path):
        # backward compatibility
        checkpoint_dict: _CheckpointDict = torch.load(checkpoint_path)

        if set(get_type_hints(_CheckpointDict).keys()
               ) <= set(checkpoint_dict.keys()):
            return checkpoint_dict["user_data"]
        checkpoint_name = os.path.basename(checkpoint_path)
        raise ValueError(
            f"Checkpoint: {checkpoint_name} is not a checkpoint stored by Tracker."
        )
    else:
        user_data = {}
        for path in glob.glob(os.path.join(checkpoint_path, "*.pth")):
            name, _ = os.path.splitext(os.path.basename(path))
            user_data[name] = torch.load(path)
        return user_data
