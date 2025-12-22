from __future__ import annotations

import os
import zipfile
from typing import ClassVar
from urllib.parse import urlparse

import requests
from dgl import DGLHeteroGraph
from dgl.data import DGLDataset


def download_dataset(url: str, out_dir: str):
    os.makedirs(out_dir)
    file_name = os.path.basename(urlparse(url).path)
    file_path = os.path.join(out_dir, file_name)
    if not file_name.endswith('.zip'):
        raise NotImplementedError

    headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
    res = requests.get(url, headers=headers)
    assert res.status_code == 200
    with open(file_path, 'wb') as fout:
        fout.write(res.content)

    with zipfile.ZipFile(file_path, mode='r') as zipfin:
        zipfin.extractall(out_dir)
    return


class BaseHeteroDGLDataset(DGLDataset):
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

    name: ClassVar[str] = None

    def __init__(
        self,
        raw_path: str = None,
        raw_dir: str = None,
        save_dir: str = None,
        force_reload: bool = False,
        verbose: bool = False,
    ):

        assert bool(self.name), 'A name should be set for every datatset'

        def handle_path(path: str):
            if path is None:
                return path
            if path.startswith('http'):
                return path
            return os.path.abspath(os.path.expanduser(path))

        super().__init__(
            name=self.name,
            url=handle_path(raw_path),
            raw_dir=handle_path(raw_dir),
            save_dir=handle_path(save_dir),
            force_reload=force_reload,
            verbose=verbose,
        )

    def download(self):
        # download raw data to local disk
        if self.url.startswith('http'):
            if self.verbose:
                print(f'Downloading from {self.url} into {self.raw_path}')
            download_dataset(self.url, self.raw_path)
            return
        if not os.path.isdir(self.url):
            raise FileNotFoundError(
                f'Directory "{self.url}" for raw data does not exist. \n'
                'Either the specified path or variant is incorrect.\n' + (
                    f'Available variants: {list(self.variants)} for {self.name}.'
                    if hasattr(self, 'variants') else ''
                )
            )

        os.symlink(self.url, self.raw_path)
        return

    def __getitem__(self, idx: int) -> DGLHeteroGraph:
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def save(self):
        # save processed data to directory `self.save_path`
        pass

    def load(self):
        # load processed data from directory `self.save_path`
        pass

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        pass

    def process(self):
        """process raw data to graphs, labels, splitting masks
        """
        raise NotImplementedError
