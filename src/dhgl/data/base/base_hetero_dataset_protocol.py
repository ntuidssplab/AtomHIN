from typing import Protocol
from typing_extensions import runtime_checkable
from .base_schema import BaseGraphSchema


@runtime_checkable
class HeteroDGLDatasetLike(Protocol):
    """ Template for customizing heterogeneous graph datasets in DGL.

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

    def __init__(
        self,
        raw_path: str = None,
        raw_dir: str = None,
        save_dir: str = None,
        force_reload: bool = False,
        verbose: bool = False,
    ):
        ...

    def download(self):
        ...

    # download raw data to local disk

    def __getitem__(self, idx: int) -> BaseGraphSchema:
        ...

    def __len__(self) -> int:
        ...

    def save(self):
        ...

    # save processed data to directory `self.save_path`

    def load(self):
        ...

    # load processed data from directory `self.save_path`

    def has_cache(self):
        ...

    # check whether there are processed data in `self.save_path`

    def process(self):
        """process raw data to graphs, labels, splitting masks
        """
