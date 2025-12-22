from __future__ import annotations

import abc
import os
import warnings
from functools import reduce
from itertools import chain
from tempfile import TemporaryDirectory
from typing import Iterable, Literal, NamedTuple, TypedDict

import numpy as np
import torch
import zarr
import zarr.storage

# from line_profiler import profile
from psutil import Process
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import StackDataset as TorchStackDataset
from torch.utils.data import Subset as TorchSubset
from tqdm import tqdm
from typing_extensions import deprecated

from dhgl.utils.precomputation.feature_collector import _format_memory
from dhgl.utils.precomputation.memmap import (
    _fill_from_slices,
    dtype_convert,
    load_memmap_sparse_csr,
    save_csr_memmap,
)

from .sampler import SequentialChunkedSampler, SmartIndex


class _DenseMeta(TypedDict):
    indices: list[int]
    # dtype: str
    # shape_b_c_d: tuple[int, int, int]
    # filedir: str
    chunk_size: int


class _SparseMeta(TypedDict):
    indices: list[int]
    idx_dtype: str
    val_dtype: str
    nnz: int
    shape_b_c_d: tuple[int, int, int]
    filedir: str


class _Meta(TypedDict):
    sparse: _SparseMeta
    dense: _DenseMeta
    check_hash: str


DataType = list[torch.Tensor]


class _BaseDataset(Dataset):

    def __getitem__(self, item: Iterable | slice):
        if isinstance(item, (Iterable, slice)):
            return self.__getitems__(item)
        raise NotImplementedError('Use batch indices instead')

    @abc.abstractmethod
    def __getitems__(self, items):
        ...


class MixedStorage(zarr.storage.LocalStore):
    """A mixed storage that keep a given ratio of chunks in memory."""

    def __init__(self, root, memory_ratio, *, read_only=False):
        super().__init__(root, read_only=read_only)
        keys = list(self.sync_list())
        self._store_dict = {
            key: None
            for key in keys[:int(len(keys) * memory_ratio)]
        }
        return

    def sync_list(self):
        # docstring inherited
        to_strip = self.root.as_posix() + "/"
        for p in list(self.root.rglob("*")):
            if p.is_file():
                yield p.as_posix().replace(to_strip, "")

    async def _open(self) -> None:
        res = await super()._open()
        self._store_dict = {
            key: (await super(MixedStorage, self).get(key))
            for key in self._store_dict
        }
        return res

    async def clear(self) -> None:
        raise NotImplementedError

    async def get(
        self,
        key: str,
        prototype=None,
        byte_range=None,
    ):
        if key in self._store_dict:
            return await zarr.storage.MemoryStore.get(
                self, key, prototype, byte_range
            )
        return await super().get(key, prototype, byte_range)

    async def get_partial_values(
        self,
        prototype,
        key_ranges,
    ):
        raise NotImplementedError
        # print(f'fetching partial key={key_ranges}')
        # return await super().get_partial_values(prototype, key_ranges)


class TransposedDataset(_BaseDataset):

    dataset: Dataset[DataType]
    _meta: _Meta

    def __init__(
        self,
        dataset: Dataset[DataType],
        cache_dir: str,
        *,
        batch_size: int,
        chunk_size: int | None = None,
        cache_idx_dtype=None,
        cache_val_dtype=None,
        memory_ratio: float | None = None,
        verbose=None,
        **kwargs,
    ):
        if isinstance(dataset, TransposedDataset):
            raise ValueError('Dataset cannot be transposed twice.')
        self.dataset = dataset
        self.verbose = verbose

        cache_dir = os.path.join(cache_dir, self._dataset_hash(dataset))
        os.makedirs(cache_dir, exist_ok=True)

        self.cache_dir = cache_dir
        self._meta = self._try_load_transposed(
            cache_dir, chunk_size=chunk_size
        )
        if self._meta is None:
            self._meta = self._transpose(
                cache_dir,
                batch_size=batch_size,
                chunk_size=chunk_size,
                cache_idx_dtype=cache_idx_dtype,
                cache_val_dtype=cache_val_dtype,
                **kwargs,
            )
        if 'check_hash' in self._meta:
            assert self._meta['check_hash'] == self._dataset_hash(dataset), \
                f'Hash value of dataset does not match the value in cache: {cache_dir}. '

        if os.path.exists(os.path.join(self.cache_dir, 'data.zarr')):
            self.data_fp = zarr.open_array(
                # store=os.path.join(self.cache_dir, 'data.zarr'),
                store=MixedStorage(
                    root=os.path.join(self.cache_dir, 'data.zarr'),
                    memory_ratio=memory_ratio or 0.,
                    read_only=True,
                ),
                mode='r',
            )
        if verbose:
            print(f'Transposed dataset loaded from {cache_dir}')
        return

    def __len__(self):
        return len(self.dataset)

    def __getitems__(self, items):
        if isinstance(items, slice):
            items = SmartIndex.from_items(items)
        assert isinstance(items, SmartIndex), items
        # def _get_batch_from_cache(self, items: SmartIndex):

        if items.is_contiguous:
            items = items.to_slice()

        def _get_sparse(meta: _SparseMeta):

            ptr_fp = np.memmap(
                os.path.join(self.cache_dir, 'crow_indices.bin'),
                dtype=dtype_convert(np.long, 'str'),
                mode='r',
                shape=(meta['shape_b_c_d'][0], meta['shape_b_c_d'][1] + 1),
            )
            col_fp = np.memmap(
                os.path.join(self.cache_dir, 'col_indices.bin'),
                dtype=str(meta['idx_dtype']),
                mode='r',
                shape=(meta['nnz'], ),
            )
            val_fp = np.memmap(
                os.path.join(self.cache_dir, 'values.bin'),
                dtype=str(meta['val_dtype']),
                mode='r',
                shape=(meta['nnz'], ),
            )

            if isinstance(items, SmartIndex):
                # TODO
                raise NotImplementedError
            ptr, indices, data = _take_rows_from_2d_csr(
                ptr_fp, col_fp, val_fp, items
            )
            # b c d

            target_shape = ( # c b d
                meta['shape_b_c_d'][1], len(items),
                *meta['shape_b_c_d'][2:]
            )
            for csr in _rearrange_b_c_d_to_c_b_d(
                ptr, indices, data, target_shape
            ):
                csr = torch.sparse_csr_tensor(
                    *csr,
                    size=target_shape[1:],
                )
                yield csr

        def _get_dense(meta: _DenseMeta):
            nonlocal items
            assert os.path.exists(os.path.join(self.cache_dir, 'data.zarr'))
            data_fp = self.data_fp
            if 'ptr' not in data_fp.attrs:
                yield from torch.from_numpy(
                    np.array(items.take_from(data_fp))
                ).transpose(0, 1)
                return
            ptr = data_fp.attrs['ptr']
            arr = torch.from_numpy(np.array(items.take_from(data_fp)))
            for s, e in zip(ptr, ptr[1:]):
                yield arr[:, s:e]

        batch_feats = []
        if self._meta.get('sparse', None):
            batch_feats += [None] * len(self._meta['sparse']['indices'])
        if self._meta.get('dense', None):
            batch_feats += [None] * len(self._meta['dense']['indices'])

        if self._meta.get('sparse', None):
            for c, csr in zip(
                self._meta['sparse']['indices'],
                _get_sparse(self._meta['sparse'])
            ):
                batch_feats[c] = csr
        if self._meta.get('dense', None):
            for c, x in zip(
                self._meta['dense']['indices'],
                _get_dense(self._meta['dense'])
            ):
                batch_feats[c] = x

        return batch_feats

    # @property
    # def is_transposed(self) -> bool:
    #     return self._meta is not None

    # @property
    # def chunk_size(self) -> int:
    #     if not self.is_transposed:
    #         raise RuntimeError(
    #             'chunk_size only defined if the dataset has been transposed.'
    #         )
    #     return self._meta['dense'].get('chunk_size', None)

    @classmethod
    def _try_load_transposed(
        cls,
        cache_dir: str,
        chunk_size: int | None = None,
    ) -> bool:
        """If found and loaded, returns True. Returns False otherwise"""

        if not os.path.exists(os.path.join(cache_dir, 'meta.npz')):
            return None
        meta_fp = np.load(
            os.path.join(cache_dir, 'meta.npz'), allow_pickle=True
        )
        meta = {'sparse': {}, 'dense': {}}
        for k, v in meta_fp.items():
            k: str
            if 'dtype' in k:
                v = str(v)
            if 'shape' in k:
                v = tuple(v.tolist())
            if k.startswith('sparse__'):
                meta['sparse'][k.removeprefix('sparse__')] = v
            elif k.startswith('dense__'):
                meta['dense'][k.removeprefix('dense__')] = v
            else:
                meta[k] = v
        if chunk_size is not None:
            if chunk_size != meta['dense']['chunk_size']:
                return None
        return meta

    @classmethod
    def _dataset_hash(cls, dataset) -> str | None:
        if not hasattr(dataset, 'check_hash'):
            if hasattr(dataset, 'dataset'):
                # NOTE: in case the dataset is torch.utils.data.Subset
                return cls._dataset_hash(dataset.dataset)
            warnings.warn(
                'Dataset does not have attribute "check_hash", '
                'which is used in verification of loading transposed cache.'
            )
            raise NotImplementedError(
                'Dataset does not have attribute "check_hash", '
                'which is now required for automatic loading of tranposed cache'
            )
        return getattr(dataset, 'check_hash')

    @classmethod
    def _transpose_batch_sparse(
        cls,
        batch: DataType,
        out_filedir: str,
        idx_dtype=None,
        val_dtype=None,
    ):

        # batch: (c, b, d)
        indices = [
            i for i, feat in enumerate(batch) if feat.layout != torch.strided
        ]
        if len(indices) == 0:
            return None
        sparse_feats = [
            feat.to_sparse_csr() for feat in batch
            if feat.layout != torch.strided
        ]
        sparse_feat_shape_c_d = (len(sparse_feats), *batch[0].shape[1:])
        sparse_feats = _reshape_c_b_d_to_b_c_d(
            [f.crow_indices().numpy() for f in sparse_feats],
            [f.col_indices().numpy() for f in sparse_feats],
            [f.values().numpy() for f in sparse_feats],
            shape=(len(sparse_feats), *batch[0].shape)
        )
        idx_dtype = idx_dtype or sparse_feats.indices.dtype
        val_dtype = val_dtype or sparse_feats.data.dtype
        # XXX: no need to use memmap here
        save_csr_memmap(
            out_filedir,
            sparse_feats,
            idx_dtype=idx_dtype,
            val_dtype=val_dtype,
        )
        return _SparseMeta(
            indices=indices,
            idx_dtype=idx_dtype,
            val_dtype=val_dtype,
            nnz=sparse_feats._nnz(),
            shape_b_c_d=(len(sparse_feats), *sparse_feat_shape_c_d),
            filedir=out_filedir,
        )

    def _transposed_dense(
        self,
        out_dir: str,
        batch_size: int,
        chunk_size: int,
        dtype=None,
    ):

        def transpose(batch: DataType):
            # batch.features: (c, b, d)
            indices = [
                i for i, feat in enumerate(batch)
                if feat.layout == torch.strided
            ]
            if len(indices) == 0:
                return None
            dense_feats = [
                feat.to(dtype) for feat in batch
                if feat.layout == torch.strided
            ]
            dims = [feat.shape[-1] for feat in dense_feats]
            # dense_feats = dict(_group_by_dim(dense_feats))
            if len(set(dims)) == 1:
                # Backward compatibility
                dense_feats = torch.stack(dense_feats, dim=1)
                return indices, dense_feats
            ptr = np.cumsum([0, *dims]).tolist()
            dense_feats = torch.concatenate(dense_feats, dim=1)
            return indices, dense_feats, ptr

        def init_fp(ref_data: torch.Tensor, ptr=None):
            chunks = (chunk_size, *ref_data.shape[1:])
            shards = (batch_size, *ref_data.shape[1:])
            fp = zarr.create_array(
                os.path.join(out_dir, 'data.zarr'),
                shape=(len(self), *ref_data.shape[1:]),
                dtype=dtype_convert(ref_data.dtype, 'numpy'),
                chunks=chunks,
                shards=shards,
                compressors=None,
            )
            if ptr is not None:
                fp.attrs['ptr'] = ptr
            return fp

        class TemFp:
            n_samples = len(self)
            offset = 0
            indices = None
            fp = None

            def update(self, batch):

                res = transpose(batch)
                if res is None:
                    return
                self.indices, data, *ptr = res
                if self.offset == 0:
                    self.fp = init_fp(data, ptr=ptr[0] if ptr else None)
                self.fp: zarr.Array

                self.fp[self.offset:self.offset +
                        data.shape[0]] = data[:].numpy()

                self.offset += data.shape[0]

            def meta(self):
                if self.fp is None:
                    return {}
                return _DenseMeta(
                    indices=self.indices,
                    chunk_size=chunk_size,
                    # dtype=dtype_convert(self.fp.dtype, 'str'),
                    # shape_b_c_d=self.fp.shape,
                )

        return TemFp()

    def _export_transposed_sparse(
        self, out_dir: str, batch_metas: list[_SparseMeta]
    ):

        def reduce_fn(a, b):

            def equal(lval, rval):
                assert lval == rval, str((lval, rval))
                return lval

            return _SparseMeta(
                indices=equal(a['indices'], b['indices']),
                idx_dtype=equal(a['idx_dtype'], b['idx_dtype']),
                val_dtype=equal(a['val_dtype'], b['val_dtype']),
                nnz=a['nnz'] + b['nnz'],
                shape_b_c_d=(
                    a['shape_b_c_d'][0] + b['shape_b_c_d'][0],
                    *equal(a['shape_b_c_d'][1:], b['shape_b_c_d'][1:])
                ),
                filedir=None,
            )

        meta = reduce(reduce_fn, batch_metas)

        ptr_fp = np.memmap(
            os.path.join(out_dir, 'crow_indices.bin'),
            dtype=dtype_convert(np.long, 'str'),
            mode='w+',
            shape=(len(self), meta['shape_b_c_d'][1] + 1),
        )
        col_fp = np.memmap(
            os.path.join(out_dir, 'col_indices.bin'),
            dtype=dtype_convert(meta['idx_dtype'], 'str'),
            mode='w+',
            shape=(meta['nnz'], ),
        )
        val_fp = np.memmap(
            os.path.join(out_dir, 'values.bin'),
            dtype=dtype_convert(meta['val_dtype'], 'str'),
            mode='w+',
            shape=(meta['nnz'], ),
        )
        ptr_offset = 0
        offset = 0
        pbar = tqdm(
            [m['filedir'] for m in batch_metas], disable=not self.verbose
        )
        ps = Process()
        for b, filedir in enumerate(pbar):

            _meta, ptr, col, val = load_memmap_sparse_csr(filedir)

            assert ptr.shape[1] == meta['shape_b_c_d'][1] + 1
            batch_size = ptr.shape[0]

            ptr_fp[ptr_offset:ptr_offset + batch_size] = ptr[:] + offset
            col_fp[offset + ptr[0][0]:offset + ptr[-1][-1]] = col[:]
            val_fp[offset + ptr[0][0]:offset + ptr[-1][-1]] = val[:]

            ptr_fp.flush()
            col_fp.flush()
            val_fp.flush()

            mem = ps.memory_info().rss
            pbar.set_description(
                f'Exporting transposed sparse feat: ({b}) {offset = } Mem: {_format_memory(mem)}'
            )

            offset += ptr[-1][-1]
            ptr_offset = ptr_offset + batch_size
        assert ptr_offset == len(self)
        pbar.close()
        meta['idx_dtype'] = dtype_convert(meta['idx_dtype'], 'str')
        meta['val_dtype'] = dtype_convert(meta['val_dtype'], 'str')
        return meta

    def _transpose(
        self,
        out_dir: str,
        batch_size: int,
        chunk_size: int,
        cache_idx_dtype=None,
        cache_val_dtype=None,
        **batch_kwargs,
    ):
        """Save the transposed dataset in the out_dir"""

        _OUT_FILES = [
            'meta.npz',
            'crow_indices.bin',
            'col_indices.bin',
            'values.bin',
            'data.bin',
            'data.zarr'  # XXX
        ]
        assert all(
            not os.path.exists(os.path.join(out_dir, f)) for f in _OUT_FILES
        )
        assert self._meta is None, 'Cannot be transposed twice'
        data_loader = DataLoader(
            self.dataset,
            batch_sampler=SequentialChunkedSampler(
                range(len(self)),
                batch_size=batch_size,
                drop_last=False,
            ),
            collate_fn=lambda _: _,
            **batch_kwargs,
        )

        os.makedirs(out_dir, exist_ok=True)

        ps = Process()
        pbar = tqdm(
            data_loader, desc='Transposing dataset...',
            disable=not self.verbose
        )
        with TemporaryDirectory(dir=out_dir) as temp_dir:
            dense_fp = self._transposed_dense(
                temp_dir,
                batch_size=batch_size,
                chunk_size=chunk_size,
                dtype=cache_val_dtype,
            )
            sparse_metas = []

            for b, batch in enumerate(pbar):
                dense_fp.update(batch)
                sparse_metas.append(
                    self._transpose_batch_sparse(
                        batch,
                        os.path.join(temp_dir, f'batch-{b}-sparse'),
                        idx_dtype=cache_idx_dtype,
                        val_dtype=cache_val_dtype,
                    )
                )
                mem = ps.memory_info().rss
                pbar.set_description(
                    f'Transposing dataset ({b}): Mem: {_format_memory(mem)}'
                )
            pbar.close()

            meta = {}
            # Concatenate all batches into one.
            # if dense_metas[0] is not None:
            meta['dense'] = dense_fp.meta()
            if sparse_metas[0] is not None:
                meta['sparse'] = self._export_transposed_sparse(
                    temp_dir, sparse_metas
                )
            else:
                meta['sparse'] = {}

            check_hash = self._dataset_hash(self.dataset)
            check_hash = {} if check_hash is None else {
                'check_hash': check_hash
            }
            np.savez(
                os.path.join(temp_dir, 'meta.npz'),
                **{
                    f'sparse__{k}': v
                    for k, v in meta['sparse'].items()
                },
                **{
                    f'dense__{k}': v
                    for k, v in meta['dense'].items()
                },
                **check_hash,
            )

            for filename in _OUT_FILES:
                if os.path.exists(os.path.join(temp_dir, filename)):
                    os.replace(
                        os.path.join(temp_dir, filename),
                        os.path.join(out_dir, filename)
                    )

        return self._try_load_transposed(out_dir)


class MemoryDataset(Dataset):

    def __init__(
        self,
        dataset: Dataset[DataType],
        # feats: list[torch.Tensor],
        device=None,
        feat_fmt: Literal['strided', 'sparse_csr'] | None = None,
        verbose=None,
    ):

        def to_device(x: torch.Tensor):
            if feat_fmt is not None:
                x = x.to_sparse(layout=getattr(torch, feat_fmt))
            elif x.layout != torch.strided:
                # NOTE: use csr for row indexing
                x = x.to_sparse_csr()

            return x.to(device)

        if verbose:
            print(f'Moving dataset to {device}...')
        self.feats = list(map(to_device, dataset[:]))
        self.device = device
        return

    def __len__(self):
        return self.feats[0].shape[0]

    def __getitem__(self, items):
        if isinstance(items, SmartIndex):
            items = items.r_
        if isinstance(items, slice):
            # TODO
            raise NotImplementedError
        if not isinstance(items, torch.Tensor):
            items = torch.tensor(items)
        items = items.to(self.device)

        def get_rows(x: torch.Tensor):
            if x.is_sparse_csr:
                return torch.stack([x[i] for i in items])
            return x[items]

        return [get_rows(x) for x in self.feats]


class StackDataset(TorchStackDataset, _BaseDataset):

    def __getitems__(self, item):
        item = SmartIndex.from_items(item)

        def take_index(dataset):
            if isinstance(dataset, torch.Tensor):
                return dataset[item.r_]
            return dataset[item]

        if isinstance(self.datasets, dict):
            return {
                k: take_index(dataset)
                for k, dataset in self.datasets.items()
            }
        return tuple(take_index(dataset) for dataset in self.datasets)


# class SplitDataset(Dataset):

#     def __init__(
#         self,
#         dataset: Dataset[DataType],
#         indices: Sequence[int],
#         labels: torch.Tensor | None = None,
#         # verbose=True,
#     ):
#         self.dataset = dataset
#         self.indices = indices
#         self.labels = labels
#         assert len(dataset) == len(self.indices)
#         if self.labels is not None:
#             assert len(self.indices) == len(self.labels)
#         return

#     # def to(
#     #     self, device, feat_fmt: Literal['strided', 'sparse_csr'] | None = None
#     # ):
#     #     assert self._cache_dir == self._meta == None,\
#     #         f'Cannot move dataset to {device} while caching enabled'
#     #     if self.verbose:
#     #         print(f'Moving dataset to {device}...')
#     #     if isinstance(self.dataset, MemoryDataset):
#     #         self.dataset = MemoryDataset(self.dataset.feats, device, feat_fmt)
#     #         return self
#     #     self.dataset = MemoryDataset(
#     #         self.dataset[self.indices], device, feat_fmt
#     #     )
#     #     return self

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, item: Iterable | slice):
#         if isinstance(item, (Iterable, slice)):
#             return self.__getitems__(item)
#         raise NotImplementedError('Use batch indices instead')

#     def __getitems__(self, items):

#         items = SmartIndex.from_items(items)
#         # if isinstance(self.dataset, MemoryDataset):
#         #     batch = self.dataset[items]
#         # if self._meta is not None:
#         #     batch = self._get_batch_from_cache(items)
#         #     # batch_truth = self.dataset[self.indices[items]]
#         #     # for x, x_truth in zip(batch, batch_truth):
#         #     #     if not (x.to_dense() == x_truth.to_dense()).all():
#         #     #         breakpoint()
#         #     # print('Checked OK')
#         # else:
#         #     # batch = self.dataset[self.indices[items]]
#         batch = self.dataset[items]
#         if self.labels is not None:
#             return self.indices[items.r_], batch, self.labels[items.r_]
#         return self.indices[items.r_], batch


class Subset(TorchSubset):

    def __getitem__(self, items):
        return self.__getitems__(items)

    def __getitems__(self, items):
        if items == slice(None):
            return self.dataset[self.indices]
        return self.dataset[self.indices[SmartIndex.from_items(items).r_]]


def _rearrange_b_c_d_to_c_b_d(indptr, indices, data, target_shape):
    """Rearrange b c d -> c' b d """

    def expand_along_c(ptr, vec):
        return [vec[s:e] for s, e in zip(ptr, ptr[1:])]

    # indices_: (b' c' d)
    indices_ = [expand_along_c(ptr, indices) for ptr in indptr]

    # data_: (b' c' d)
    data_ = [expand_along_c(ptr, data) for ptr in indptr]

    # NOTE: both indices_ and data_ are (b' c' d)
    # zip(*indices_): c' b' d
    # zip(*data_): c' b' d
    for c_i, (i__, v__) in enumerate(zip(zip(*indices_), zip(*data_))):
        # i__, v__: (b', d)
        assert len(i__) == len(v__) == target_shape[1]

        yield _stack_as_csr(i__, v__)


@deprecated('Keep for now. Used in https://github.com/*')
def _rearrange_b_cd_to_c_b_d(indptr, indices, data, target_shape):
    """Rearrange b (cd) -> c' b d """


    indices_, data_ = zip(  # (b' c' d), (b' c' d)
        *(
        _reshape_cd_to_c_d(
            indices[s:e], data[s:e], (target_shape[0], target_shape[2])
        ) for s, e in zip(indptr, indptr[1:])
        )
    )

    # NOTE: both indices_ and data_ are (b' c' d)
    # zip(*indices_): c' b' d
    # zip(*data_): c' b' d
    for c_i, (i__, v__) in enumerate(zip(zip(*indices_), zip(*data_))):
        # i__, v__: (b', d)
        assert len(i__) == len(v__) == target_shape[1]

        yield _stack_as_csr(i__, v__)


@deprecated('Keep for now. Used in https://github.com/*')
def _reshape_cd_to_c_d(
    indices: np.ndarray, data: np.ndarray, shape: tuple[int, int]
):
    """reshape 2 (cd) -> 2 c' d"""
    assert len(shape) == 2
    row, col = np.divmod(indices, shape[-1])
    cols = [None] * shape[0]
    data_ = [None] * shape[0]
    uniques, starts, counts = np.unique(
        row, return_index=True, return_counts=True
    )
    ptr = 0
    cur = 0
    for i in range(shape[0]):
        if ptr >= len(uniques) or i < uniques[ptr]:
            # Empty row
            cols[i] = np.empty((0, ), dtype=indices.dtype)
            data_[i] = np.empty((0, ), dtype=data.dtype)
            continue
        assert uniques[ptr] == i
        s = starts[ptr]
        assert cur == s
        cur = starts[ptr] + counts[ptr]
        cols[i] = col[s:cur]
        data_[i] = data[s:cur]
        ptr += 1

    return cols, data_


def _stack_as_csr(indices: list[np.ndarray], data: list[np.ndarray]):
    """c' d -> c d """

    indptr = np.zeros(len(indices) + 1, dtype=indices[0].dtype)
    for i, col in enumerate(indices):
        indptr[i + 1] = indptr[i] + len(col)

    return indptr, np.concatenate(indices), np.concatenate(data)


class CSR3DReturnT(NamedTuple):
    indptr: np.ndarray
    """2D ptr: (b (c+1))"""

    indices: np.ndarray
    data: np.ndarray
    shape: tuple[int, int, int]

    def crow_indices(self):
        return self.indptr

    def col_indices(self):
        return self.indices

    def values(self):
        return self.data

    def _nnz(self):
        return len(self.data)


def _reshape_c_b_d_to_b_c_d(
    indptr: list[np.ndarray],
    indices: list[np.ndarray],
    data: list[np.ndarray],
    shape: tuple[int, int, int],
):

    # indptr: (c (b+1))
    # indices: (c nnz)
    C, B, D = shape
    assert all(len(ptr) - 1 == B for ptr in indptr)

    def _to_b_c_d():
        offset = 0
        for i in range(B):
            slices = [slice(ptr[i], ptr[i + 1]) for ptr in indptr]
            lengths = np.array([s.stop - s.start for s in slices])
            out_indptr = np.zeros(shape=len(slices) + 1, dtype=indptr[0].dtype)
            out_indptr[1:] = np.cumsum(lengths)
            out_indptr += offset
            offset = out_indptr[-1]
            yield (
                out_indptr,
                [indices[c][s] for c, s in enumerate(slices)],
                [data[c][s] for c, s in enumerate(slices)],
            )

    ptrs_, indices_, data_ = zip(*_to_b_c_d())

    ptrs_ = np.stack(ptrs_)
    indices_ = np.concatenate(sum(indices_, []))
    data_ = np.concatenate(sum(data_, []))
    return CSR3DReturnT(ptrs_, indices_, data_, (B, C, D))


def _take_rows_from_2d_csr(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    row_indices: np.ndarray | slice,
):
    if isinstance(row_indices, slice):
        return _get_compressed_vectors_with_slice(
            indptr, indices, data, row_indices
        )

    return _get_compressed_vectors(indptr, indices, data, row_indices)


def _get_compressed_vectors_with_slice(
    indptr_2d: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    row_slice: slice,
):
    slices = [
        slice(
            indptr_2d[row_slice.start][0], indptr_2d[row_slice.stop - 1][-1]
        )
    ]
    total_length = slices[0].stop - slices[0].start

    out_data = _fill_from_slices(data, slices, total_length)
    out_indices = _fill_from_slices(indices, slices, total_length)

    offset = indptr_2d[row_slice.start][0]
    out_indptr = indptr_2d[row_slice.start:row_slice.stop] - offset

    return out_indptr, out_indices, out_data


def _get_compressed_vectors(
    indptr_2d: np.ndarray, indices: np.ndarray, data: np.ndarray,
    row_idxs: np.ndarray
):
    ptr = indptr_2d[row_idxs].copy()
    slices = [slice(ptr_[0], ptr_[-1]) for ptr_ in ptr]
    lengths = np.array([s.stop - s.start for s in slices])
    total_length = lengths.sum()

    out_data = _fill_from_slices(data, slices, total_length)
    out_indices = _fill_from_slices(indices, slices, total_length)

    offsets = chain([0], np.cumsum(lengths)[:-1])
    for i, offset in enumerate(offsets):
        original_offset = ptr[i][0]
        ptr[i] += offset - original_offset

    return ptr, out_indices, out_data
