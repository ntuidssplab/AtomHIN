from __future__ import annotations

from .amazon.amazon_dataset import AmazonDataset, AtomicAmazonDataset
from .base import BaseLinkPredictionDataset, LinkPredDatasetLike
from .lastfm.lastfm_dataset import LastFMDataset
from .pubmed.pubmed_dataset import AtomicPubMedDataset, PubMedDataset

__all__ = [
    'AmazonDataset', 'AtomicAmazonDataset', 'LastFMDataset',
    'AtomicPubMedDataset', 'PubMedDataset'
]

SUPPORTED_DATASETS = {
    dataset.__name__: dataset
    for dataset in (globals().get(key) for key in __all__)
}

VARIANTS = {
    name: getattr(dataset, 'variants', [])
    for name, dataset in SUPPORTED_DATASETS.items()
}
