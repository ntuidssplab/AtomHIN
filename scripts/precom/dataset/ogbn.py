from __future__ import annotations

from typing import Literal

from dhgl.script_utils.configs.dataset.hetero_dgl_dataset import MAGConfig as BaseMAG
from dhgl.script_utils.configs.dataset.hetero_dgl_dataset import NMAGConfig as BaseNMag

from .base import BaseDatasetConfig


class MAGConfig(BaseDatasetConfig, BaseMAG):

    name: Literal['mag', 'ogbn-mag'] = 'ogbn-mag'


class NMAGConfig(BaseDatasetConfig, BaseNMag):

    name: Literal['nmag', 'ogbn-nmag', 'atomic-mag'] = 'atomic-nmag'
