from __future__ import annotations

from .acm.acm_dataset import ACMDataset, HeteroACMDataset
from .dblp.dblp_dataset import AtomicDBLPDataset, DBLPDataset, HeteroDBLPDataset
from .freebase.freebase_dataset import FreebaseDataset
from .imdb.imdb_dataset import (
    AtomicIMDBDataset,
    HeteroIMDBDataset,
    IMDBDataset,
    NormalizedIMDBDataset,
)
from .link_prediction import SUPPORTED_DATASETS as LP_DATASETS
from .link_prediction import *
from .mag.mag_dataset import AtomicMAGDataset, MAGDataset, NormalizedMAGDataset

# from .synhin import SynHINDataset

__all__ = [
    'ACMDataset', 'DBLPDataset', 'IMDBDataset', 'FreebaseDataset',
    'MAGDataset', 'AtomicIMDBDataset', 'AtomicDBLPDataset', 'AtomicMAGDataset'
]

SUPPORTED_DATASETS = {
    dataset.__name__: dataset
    for dataset in (globals().get(key) for key in __all__)
} | LP_DATASETS

VARIANTS = {
    name: getattr(dataset, 'variants', [])
    for name, dataset in SUPPORTED_DATASETS.items()
}
