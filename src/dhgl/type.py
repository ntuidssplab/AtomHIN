from __future__ import annotations

from typing import Literal

EType = str | tuple[str, str, str]
CEType = tuple[str, str, str]
NType = str
Split = Literal['train', 'val', 'test']
EWEIGHT = 'weight'
FEAT = 'feat'
