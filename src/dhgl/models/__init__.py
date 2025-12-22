from __future__ import annotations

from lazy_imports import try_import

from .adapted_gat import AdaptedGAT
from .HGT import HGT
from .SimpleHGN import SimpleHGN
from .TreeGNN import TreeGNN

with try_import() as node_former:
    from .NodeFormer import NodeFormer
