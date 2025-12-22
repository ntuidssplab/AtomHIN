from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from typing import Iterable, Literal, NamedTuple

import numpy as np
import torch
from numba import njit

_dtype_torch_to_numpy = {
    torch.bool: bool,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
}
_dtype_numpy_to_torch = {
    value: key
    for (key, value) in _dtype_torch_to_numpy.items()
} | {
    np.dtype(value): key
    for (key, value) in _dtype_torch_to_numpy.items()
}


# def dtype_torch_to_numpy(torch_dtype, numpy_dtype):
#     pass
def dtype_convert(
    src_dtype: str | torch.dtype | np.dtype,
    dst_lib: Literal['str', 'torch', 'numpy'],
):
    assert src_dtype
    assert dst_lib in ('str', 'torch', 'numpy')
    dtype = src_dtype
    if isinstance(src_dtype, torch.dtype):
        if dst_lib == 'torch':
            return src_dtype
        dtype = _dtype_torch_to_numpy[src_dtype]

    dtype = np.dtype(dtype)
    if dst_lib == 'str':
        return str(dtype)
    if dst_lib == 'torch':
        return _dtype_numpy_to_torch[dtype]
    assert dst_lib == 'numpy'
    return dtype


def save_stride_memmap(
    filedir: str,
    x: torch.Tensor,
    dtype: torch.dtype | None = None,
    metadata: dict | None = None,
):
    base_dir = os.path.dirname(filedir)
    assert os.path.isdir(base_dir), f'dir: {base_dir}'
    dtype = dtype_convert(dtype or x.dtype, 'numpy')
    with TemporaryDirectory(
        dir=base_dir, prefix=os.path.basename(filedir)
    ) as tmp_dir:
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        data = x.astype(dtype)
        filename = os.path.join(tmp_dir, 'data.bin')
        fp = np.memmap(filename, dtype=dtype, mode='w+', shape=data.shape)
        fp[:] = data[:]
        fp.flush()
        np.savez(
            os.path.join(tmp_dir, 'meta.npz'),
            _shape=np.array(data.shape),
            _dtype=str(np.dtype(dtype)),
            **(metadata or {}),
        )
        os.replace(tmp_dir, filedir)

    return


def save_csr_memmap(
    filedir: str,
    x_csr: torch.Tensor,
    idx_dtype: torch.dtype | None = None,
    val_dtype: torch.dtype | None = None,
    metadata: dict | None = None,
):
    """Saves a CSR-formatted sparse tensor into a memory-mapped directory.

    It is recommended to use `idx_dtype=torch.int32` to reduce disk space usage, especially
    for large sparse matrices with relatively small index ranges. If `idx_dtype` or
    `val_dtype` is not provided, the current tensor dtypes will be used.

    Args:
        path (str): Directory path where memory-mapped arrays and metadata will be saved.
        x_csr (torch.Tensor): A CSR-format sparse tensor (2D).
        idx_dtype (torch.dtype | None, optional): Desired dtype for index arrays.
            If None, the dtype from the tensor is used. Recommended: `torch.int32`.
        val_dtype (torch.dtype | None, optional): Desired dtype for values.
            If None, the current dtype of `.values()` is used.
        metadata (dict | None, optional): Additional metadata to store in the `.npz` file.
            Keys must not start with an underscore (`_`), which is reserved for internal metadata.

    Returns:
        None
    """

    def _save_memmap(filename, data: np.ndarray, dtype):
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        data = data.astype(dtype)
        if data.shape == (0, ):
            fp = np.memmap(filename, dtype=dtype, mode='w+', shape=(1, ))
            fp.flush()
            return

        fp = np.memmap(filename, dtype=dtype, mode='w+', shape=data.shape)
        fp[:] = data[:]
        fp.flush()
        return

    base_dir = os.path.dirname(filedir)
    assert os.path.isdir(base_dir), f'dir: {base_dir}'
    ptr_dtype = np.int64
    # NOTE: Scale of crow_indices depends on nnz.
    # Thus, it's safter to use large dtype.
    # Since #rows <<< nnz, the overhead is minimal
    idx_dtype = dtype_convert(idx_dtype or x_csr.crow_indices().dtype, 'numpy')
    val_dtype = dtype_convert(val_dtype or x_csr.values().dtype, 'numpy')
    metadata = metadata or {}
    for key in metadata:
        assert not key.startswith('_'), \
            f'Metadata key "{key}" should not start with an underscore.'

    with TemporaryDirectory(
        dir=base_dir, prefix=os.path.basename(filedir)
    ) as tmp_dir:
        _save_memmap(
            os.path.join(tmp_dir, 'crow_indices'), x_csr.crow_indices(),
            ptr_dtype
        )
        _save_memmap(
            os.path.join(tmp_dir, 'col_indices'), x_csr.col_indices(),
            idx_dtype
        )
        _save_memmap(
            os.path.join(tmp_dir, 'values'), x_csr.values(), val_dtype
        )
        np.savez(
            os.path.join(tmp_dir, 'meta.npz'),
            _ptr_shape=np.array(x_csr.crow_indices().shape
                                ),  # This allows 2d pointer
            _shape=np.array(x_csr.shape),
            _nnz=x_csr._nnz(),
            _ptr_dtype=str(np.dtype(ptr_dtype)),
            _idx_dtype=str(np.dtype(idx_dtype)),
            _val_dtype=str(np.dtype(val_dtype)),
            **metadata,
        )
        os.replace(tmp_dir, filedir)

    return


@njit
def _fill_from_slices(
    array: np.ndarray, slices: list[slice], total_length: int
) -> np.ndarray:
    out = np.empty(total_length, dtype=array.dtype)
    offset = 0
    for s in slices:
        values = array[s]
        out[offset:offset + len(values)] = values
        offset += len(values)
    return out


def _get_compressed_vectors_with_slice(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    row_slice: slice,
):
    slices = [slice(indptr[row_slice.start], indptr[row_slice.stop])]
    total_length = slices[0].stop - slices[0].start

    out_data = _fill_from_slices(data, slices, total_length)
    out_indices = _fill_from_slices(indices, slices, total_length)

    offset = indptr[row_slice.start]
    out_indptr = indptr[row_slice.start:row_slice.stop + 1] - offset

    return out_indptr, out_indices, out_data


@njit
def _get_compressed_vectors(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray,
    row_idxs: np.ndarray
):
    slices = [slice(indptr[i], indptr[i + 1]) for i in row_idxs]
    lengths = np.array([s.stop - s.start for s in slices])
    total_length = lengths.sum()

    out_data = _fill_from_slices(data, slices, total_length)
    out_indices = _fill_from_slices(indices, slices, total_length)

    out_indptr = np.zeros(shape=len(slices) + 1, dtype=indptr.dtype)
    out_indptr[1:] = np.cumsum(lengths)

    return out_indptr, out_indices, out_data


def take_rows_from_csr(
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


class MemmapReturnT(NamedTuple):
    meta: dict
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray

    def crow_indices(self):
        return self.indptr

    def col_indices(self):
        return self.indices

    def values(self):
        return self.data

    def _nnz(self):
        return len(self.data)


def load_memmap_sparse_csr(filedir: str) -> MemmapReturnT:
    """Load a CSR sparse matrix from a directory saved by `save_memmap`.
    Args:
        filedir (str): Directory previously saved using `save_memmap`.
    """
    _is_file = lambda name: os.path.isfile(os.path.join(filedir, name))
    assert all(map(_is_file, ['meta.npz', 'crow_indices', 'col_indices', 'values'])), \
        f'Not all required files are present in {filedir}.'

    meta = np.load(os.path.join(filedir, 'meta.npz'))
    ptr_shape = tuple(meta.get('_ptr_shape', (meta['_shape'][0] + 1, )))
    # If 2d ptr, require _ptr_shape to be set.
    crow_indices = np.memmap(
        os.path.join(filedir, 'crow_indices'),
        dtype=np.dtype(str(meta['_ptr_dtype'])),
        mode='r',
        shape=ptr_shape,
    )
    col_indices = np.memmap(
        os.path.join(filedir, 'col_indices'),
        dtype=np.dtype(str(meta['_idx_dtype'])),
        mode='r',
        shape=(meta['_nnz'], ),
    )
    values = np.memmap(
        os.path.join(filedir, 'values'),
        dtype=np.dtype(str(meta['_val_dtype'])),
        mode='r',
        shape=(meta['_nnz'], ),
    )
    return MemmapReturnT(meta, crow_indices, col_indices, values)


def load_memmap_stride(filedir: str):
    _is_file = lambda name: os.path.isfile(os.path.join(filedir, name))
    assert all(map(_is_file, ['meta.npz', 'data.bin'])), \
        f'Not all required files are present in {filedir}.'

    meta = np.load(os.path.join(filedir, 'meta.npz'))
    data = np.memmap(
        os.path.join(filedir, 'data.bin'),
        dtype=np.dtype(str(meta['_dtype'])),
        mode='r',
        shape=tuple(meta['_shape']),
    )
    return meta, data


def load_memmap(
    filedir: str,
    row_indices: Iterable[int] | None = None,
) -> tuple[dict, torch.Tensor]:
    if os.path.isfile(os.path.join(filedir, 'data.bin')):
        meta, data = load_memmap_stride(filedir)
        if row_indices is not None:
            return meta, torch.from_numpy(data[np.array(row_indices)])
        return meta, torch.from_numpy(np.array(data))
    meta, *loaded_data = load_memmap_sparse_csr(filedir)

    if row_indices is not None:
        if isinstance(row_indices, torch.Tensor):
            row_indices = row_indices.numpy()
        res = _get_compressed_vectors(*loaded_data, row_indices)
        return meta, torch.sparse_csr_tensor(
            *map(torch.from_numpy, res),
            size=(len(row_indices), *meta['_shape'][1:]),
        )

    _to_torch = lambda x: torch.from_numpy(np.array(x))
    return meta, torch.sparse_csr_tensor(
        *map(_to_torch, loaded_data), size=meta['_shape'].tolist()
    )


def load_memmap_meta(filedir: str) -> dict:
    assert os.path.isfile(os.path.join(filedir, 'meta.npz'))
    return np.load(os.path.join(filedir, 'meta.npz'))
