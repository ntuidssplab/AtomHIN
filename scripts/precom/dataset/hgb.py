from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from dhgl.script_utils.configs.dataset.hetero_dgl_dataset import ACMConfig as BaseACM
from dhgl.script_utils.configs.dataset.hetero_dgl_dataset import DBLPConfig as BaseDBLP
from dhgl.script_utils.configs.dataset.hetero_dgl_dataset import (
    FreebaseConfig as BaseFreebase,
)
from dhgl.script_utils.configs.dataset.hetero_dgl_dataset import IMDBConfig as BaseIMDB
from dhgl.script_utils.configs.dataset.hetero_dgl_dataset import (
    NIMDBConfig as BaseNIMDB,
)

from .base import BaseDatasetConfig


class ACMConfig(BaseDatasetConfig, BaseACM):

    name: Literal['acm']


class DBLPConfig(BaseDatasetConfig, BaseDBLP):

    name: Literal['dblp']


class IMDBConfig(BaseDatasetConfig, BaseIMDB):

    name: Literal['imdb']


class NIMDBConfig(BaseDatasetConfig, BaseNIMDB):

    name: Literal['atomic-imdb', 'nimdb']


class FreebaseConfig(BaseDatasetConfig, BaseFreebase):

    name: Literal['freebase']


HGBDatasetConfig = Annotated[ACMConfig | IMDBConfig | NIMDBConfig
                             | DBLPConfig | FreebaseConfig,
                             Field(discriminator='name')]
