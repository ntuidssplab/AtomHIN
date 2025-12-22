from __future__ import annotations

from typing import ClassVar

from ....transforms import merge_etypes
from ....type import CEType, NType
from ..base import BaseHGBLinkPredictionDataset

VARIANTS = {
    'HGB':
    'https://www.dropbox.com/scl/fi/y2efm7uk2xwhardu14s9z/LastFM.zip?rlkey=muxbmh608hjw75189xkm5sqf7&st=v6iijuew&dl=0'
}


class LastFMDataset(BaseHGBLinkPredictionDataset):
    """

    Parameters
    ----------
    raw_path : str
        Specifying the directory that stores raw data
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """

    name: ClassVar[str] = 'lastfm'
    ntypes: ClassVar[list[NType]] = ['user', 'artist', 'tag']
    canonical_etypes: ClassVar[list[CEType]] = [
        ('user', 'user-artist', 'artist'),
        ('user', 'user-user', 'user'),
        ('artist', 'artist-tag', 'tag'),
        # inv etypes
        ('artist', 'artist-user', 'user'),
        ('user', 'user-user-inv', 'user'),
        ('tag', 'tag-artist', 'artist'),
    ]
    inverse_etypes: ClassVar[list[CEType]] = [
        ('artist', 'artist-user', 'user'),
        ('user', 'user-user-inv', 'user'),
        ('tag', 'tag-artist', 'artist'),
        #
        ('user', 'user-artist', 'artist'),
        ('user', 'user-user', 'user'),
        ('artist', 'artist-tag', 'tag'),
    ]
    target_etypes: ClassVar[list[CEType]] = [('user', 'user-artist', 'artist')]
    variants: ClassVar[list[str]] = list(VARIANTS)

    def __init__(
        self,
        raw_path: str = None,
        raw_dir: str = None,
        save_dir: str = None,
        force_reload: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            raw_path=VARIANTS.get(raw_path or list(VARIANTS)[0], raw_path),
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    @property
    def symmetric_(self):
        """Merge user-user-inv into user-user. This is an inplace operation"""

        self.graph = merge_etypes(
            self.graph,
            'user-user',
            etype_to_drop='user-user-inv',
        )
        if 'user-user' in self.val_graph.canonical_etypes:
            self.val_graph = merge_etypes(
                self.graph,
                'user-user',
                etype_to_drop='user-user-inv',
            )
        self.canonical_etypes = [
            e for e in self.canonical_etypes if e[1] != 'user-user-inv'
        ]
        return self
