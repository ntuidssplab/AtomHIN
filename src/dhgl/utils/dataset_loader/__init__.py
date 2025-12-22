from __future__ import annotations

from typing import Literal, overload

import dgl

from ... import transforms
from ...data.base.base_schema import BaseHeteroGraphLike
from ...data.link_prediction.base import BaseLinkPredictionDataset
from .profiles import (
    PROFILES,
    AtomicLPDatasetLiteral,
    AtomicNCDatasetLiteral,
    LinkPredictionDatasetT,
    VanillaLPDatasetLiteral,
    VanillaNCDatasetLiteral,
)


@overload
def get_dataset(
    name: VanillaNCDatasetLiteral, profile: Literal['vanilla'] = 'vanilla',
    verbose: bool = None
) -> BaseHeteroGraphLike:
    ...


@overload
def get_dataset(
    name: AtomicNCDatasetLiteral, profile: Literal['atomic',
                                                   'srgcn'] = 'atomic',
    remove_unreachable: bool = True, verbose: bool = None, **kwargs
) -> BaseHeteroGraphLike:
    ...


@overload
def get_dataset(
    name: VanillaLPDatasetLiteral, profile: Literal['vanilla'] = 'vanilla',
    verbose: bool = None
) -> LinkPredictionDatasetT:
    ...


@overload
def get_dataset(
    name: AtomicLPDatasetLiteral, profile: Literal['atomic',
                                                   'srgcn'] = 'atomic',
    verbose: bool = None, **kwargs
) -> LinkPredictionDatasetT:
    ...


def get_dataset(
    name: str,
    profile: Literal['vanilla', 'atomic', 'srgcn'] | None = None,
    remove_unreachable: bool | None = None,
    verbose: bool = None,
    **kwargs,
) -> BaseHeteroGraphLike | LinkPredictionDatasetT:
    """Load a released dataset with optional schema refinement.

    This helper loads either a *vanilla* HIN benchmark (fixed schema) or an
    *atomic* HIN benchmark (maximally expressive schema), optionally applying
    task-specific schema refinement by selecting/unselecting node/edge types.

    The returned object depends on the dataset task:

    - For **node classification** datasets, returns a DGL heterograph.
    - For **link prediction** datasets, returns a dataset object containing
      training/validation/testing graphs and negative test edges.

    Profiles:
        - ``"vanilla"``: Load the standard benchmark schema. Schema refinement
          is **not supported**.
        - ``"atomic"``: Load the atomic HIN schema. Schema refinement is enabled
          via ``**kwargs``.
        - ``"srgcn"``: Load a predefined refined schema (e.g., searched/refined
          under sRGCN). Schema refinement is still possible via ``**kwargs``
          (overrides the profile config).

    Args:
        name (str):
            Dataset identifier. Examples: ``"imdb"``, ``"atomic-imdb"``,
            ``"atomic-amazon"``.
        profile ({"vanilla", "atomic", "srgcn"} or None):
            Which released configuration to load. If ``None``, the default
            profile for the dataset is used.
        remove_unreachable (bool or None):
            Whether to remove unreachable node types after loading/refinement.
            If ``None``, the default behavior is used (typically ``True`` for
            node classification atomic datasets and not applied to LP datasets).
        verbose (bool or None):
            Whether to print loading logs and dataset statistics.
        **kwargs:
            Schema refinement switches (atomic datasets only). Keys are type
            names and values are booleans indicating selection status.

    Returns:
        BaseHeteroGraphLike or LinkPredictionDatasetT:
            The loaded dataset object. For node classification datasets this is
            a DGL heterograph; for link prediction datasets this is a dataset
            object containing split graphs.

    Examples:
        >>> # Get the vanilla dataset (before atomization and schema refinement)
        >>> hg = get_dataset('imdb')
        >>> isinstance(hg, dgl.DGLHeteroGraph)
        True
        >>>
        >>> # Get the atomic dataset with the default profile
        >>> # (all node types unselected, all edge types selected)
        >>> hg = get_dataset('atomic-imdb')
        >>>
        >>> # Basic schema refinement: True selects, False unselects
        >>> hg = get_dataset('atomic-imdb', word=True, director=True, acts=False)
        >>>
        >>> # Basic schema refinement using a dictionary (for names with dashes)
        >>> hg = get_dataset(
        ...     'atomic-imdb', **{
        ...         'word': True,
        ...         'director': True,
        ...         'is-in': False
        ...     }
        ... )
        >>> # NOTE: select/unselect an etype also affects its inverse etype
        >>> ('keyword', 'is-in', 'movie') in hg.canonical_etypes
        False
        >>> ('movie', 'contains', 'keyword') in hg.canonical_etypes
        False
        >>>
        >>> # Use the schema refined on sRGCN
        >>> hg = get_dataset('atomic-imdb', profile='srgcn')
        >>>
        >>> # For link prediction dataset, a DGL dataset object will be returned
        >>> dataset = get_dataset('atomic-amazon')
        >>> dataset.graph, dataset.val_graph, dataset.test_graph, dataset.neg_test_graph
    """

    if name not in PROFILES:
        raise ValueError(
            f'Unknown dataset "{name}". Available: {list(PROFILES)}'
        )

    profile = list(PROFILES[name])[0] if profile is None else profile
    if profile not in PROFILES[name]:
        raise ValueError(
            f'Unsupported profile "{profile}" for dataset "{name}". '
            f'Available: {sorted(PROFILES[name].keys())}'
        )

    config = PROFILES[name][profile]
    if kwargs:
        if profile == 'vanilla':
            raise ValueError(
                'Schema refinement is not supported for vanilla datasets. '
                f'Use an atomic dataset name (e.g., "atomic-{name}") instead.'
            )
        config = config.update(**kwargs)
    hg = config.load(verbose=verbose)
    if isinstance(hg, dgl.DGLGraph):
        # default to True in NC datasets
        remove_unreachable = True if remove_unreachable is None else remove_unreachable

    if remove_unreachable:
        if isinstance(hg, BaseLinkPredictionDataset):
            raise ValueError(
                'remove_unreachable is only supported for node classification heterographs.'
            )
        hg = transforms.remove_unreachable(hg)
    return hg
