from __future__ import annotations

from typing import Annotated

from pydantic import Field

from .hgb import HGBDatasetConfig
from .ogbn import MAGConfig, NMAGConfig

HeteroDatasetConfig = Annotated[HGBDatasetConfig | MAGConfig | NMAGConfig,
                                Field(discriminator='name')]
