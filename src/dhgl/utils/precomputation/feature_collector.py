from __future__ import annotations

import contextlib
import hashlib
import json

# import multiprocessing as mulp
# mulp.set_start_method('spawn')
import os
import re
import tempfile
import warnings
from collections import defaultdict
from functools import lru_cache, reduce
from typing import Iterable, Mapping, NamedTuple, overload

import numpy as np
import psutil
import torch
from packaging import version
from tqdm import tqdm

from dhgl.type import CEType, EType, NType
from dhgl.utils.precomputation.memmap import (
    dtype_convert,
    load_memmap,
    save_csr_memmap,
    save_stride_memmap,
)

from .base import _generate_aliases
from .metagraph import MetaGraph

if version.parse(torch.__version__) >= version.parse('2.1.0'):
    from torch._tensor_str import printoptions
else:

    @contextlib.contextmanager
    def printoptions(*args, **kwargs):
        print_options = {
            'precision': torch._tensor_str.PRINT_OPTS.precision,
            'threshold': torch._tensor_str.PRINT_OPTS.threshold,
            'edgeitems': torch._tensor_str.PRINT_OPTS.edgeitems,
            'linewidth': torch._tensor_str.PRINT_OPTS.linewidth,
            'sci_mode': torch._tensor_str.PRINT_OPTS.sci_mode,
        }

        try:
            yield torch.set_printoptions(*args, **kwargs)
        finally:
            torch.set_printoptions(**print_options)


MetaPathT = tuple[CEType, ...]


def _format_memory(bytes_val: int) -> str:
    GB = 1024**3
    MB = 1024**2
    KB = 1024

    if bytes_val >= GB:
        return f"{bytes_val / GB:.1f}GB"
    elif bytes_val >= MB:
        return f"{bytes_val / MB:.1f}MB"
    else:
        return f"{bytes_val / KB:.0f}KB"


def _build_tree(mp_list: list[tuple[CEType]]):
    mp_tree = {}
    for mp in mp_list:
        sub_tree = mp_tree
        for eid in mp:
            if eid not in sub_tree:
                sub_tree[eid] = {}
            sub_tree = sub_tree[eid]
    return mp_tree


def sort_in_bfs_order(mps: list[tuple[CEType]]):

    def mp_bfs(mp_tree: dict):
        queue = []
        for eid, sub_tree in mp_tree.items():
            queue.append((eid, ))
        while len(queue):
            mp_ = queue.pop(0)
            if mp_ in mps:
                yield mp_
            sub_tree = mp_tree
            for eid in mp_:
                sub_tree = sub_tree[eid]
            for eid in sub_tree:
                queue.append((*mp_, eid))

    tree = _build_tree(mps)
    mps_ = list(mp_bfs(tree))
    assert set(mps_) == set(mps)
    return mps_


def sort_in_dfs_order(mps: list[tuple[CEType]]):

    def mp_dfs(mp_tree: dict, mp: tuple):
        for eid, sub_tree in mp_tree.items():
            mp_ = (*mp, eid)
            if mp_ in mps:
                yield mp_
            yield from mp_dfs(sub_tree, mp_)

    tree = _build_tree(mps)
    mps_ = list(mp_dfs(tree, tuple()))
    assert set(mps_) == set(mps)
    return mps_


from collections import OrderedDict


class LowRankMatrix(NamedTuple):
    """up @ down"""

    down: torch.Tensor
    up: torch.Tensor

    # def __init__(self, down, up):
    #     self.down = down
    #     self.up = up
    #     return

    @staticmethod
    def concat(xs: list[LowRankMatrix], dim: int):
        if dim != 1:
            raise NotImplementedError('Only support concatenating along dim 1')
        downs = torch.block_diag(*(x.down for x in xs))
        ups = torch.concat([x.up for x in xs], dim=1)
        rank = sum(x.up.shape[1] for x in xs)
        if rank >= 0.5 * min(downs.shape[1], ups.shape[0]):
            if ups.dtype != torch.float32 or downs.dtype != torch.float32:
                return (ups.float() @ downs.float()).to(ups.dtype)
            return ups @ downs
        # print(
        #     f'{(ups.numel() + downs.numel()) / (ups.shape[0] * downs.shape[1]):.2%}',
        #     end='\t'
        # )
        # return ups @ downs
        return LowRankMatrix(downs, ups)

    @property
    def dtype(self):
        which = np.argmin(
            torch.finfo(self.down.dtype).bits,
            torch.finfo(self.up.dtype).bits
        )
        return [self.down.dtype, self.up.dtype][which]

    @property
    def is_sparse_csr(self):
        return self.layout == torch.sparse_csr

    @property
    def shape(self):
        assert len(self.down.shape) == 2
        assert len(self.up.shape) == 2
        return (self.up.shape[0], self.down.shape[1])

    @property
    def layout(self):
        assert self.down.layout == self.up.layout
        return self.down.layout

    def __getitem__(self, item):
        return LowRankMatrix(self.down, self.up[item])

    def float(self):
        return self.up.float() @ self.down.float()

    def to_dense(self):
        dtype = self.up.dtype
        return (self.up.float() @ self.down.float()).to(dtype)

    def to(self, device):
        # return (self.up @ self.down).to(device)
        return LowRankMatrix(self.down.to(device), self.up.to(device))

    def __matmul__(self, rval: torch.Tensor):
        # res = LowRankMatrix(self.down @ rval, self.up)
        # return res
        return self.up @ (self.down @ rval)


class _SelfLoopMatrix:

    def __init__(self, mat: torch.Tensor):
        self.data = mat
        return

    def __matmul__(self, rval: torch.Tensor):
        return rval

    def to_sparse_csr(self):
        return self.data.to_sparse_csr()

    @property
    def layout(self):
        return self.data.layout

    @property
    def shape(self):
        return self.data.shape


class _LRUCache:

    def __init__(self, capacity: int | None):
        if capacity is not None and capacity < 0:
            raise ValueError("Capacity must be a positive integer")
        self.cache = OrderedDict()
        self.capacity = capacity
        return

    def __getitem__(self, key):
        if key not in self.cache:
            raise KeyError(f"{key} not found in cache")
        # Move key to the end to show it was recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def __setitem__(self, key, value):
        if key in self.cache:
            # Update and mark as recently used
            self.cache.move_to_end(key)
        self.cache[key] = value
        if self.capacity is not None and len(self.cache) > self.capacity:
            # Remove the first item (least recently used)
            self.cache.popitem(last=False)

    def set_capacity(self, new_capacity: int):
        while len(self.cache) > new_capacity:
            # Remove the first item (least recently used)
            self.cache.popitem(last=False)
        self.capacity = new_capacity
        return

    def __delitem__(self, key):
        del self.cache[key]

    def __contains__(self, key):
        return key in self.cache

    def __len__(self):
        return len(self.cache)

    def __iter__(self):
        return iter(self.cache)

    def items(self):
        return self.cache.items()

    def keys(self):
        return self.cache.keys()

    def values(self):
        return self.cache.values()

    def get(self, key, default=None):
        try:
            return self.__getitem__(key)
        except KeyError:
            return default

    def __repr__(self):
        return f"{self.__class__.__name__}({dict(self.cache)})"


class FeatureCollector:

    @overload
    def __init__(
        self,
        mg: MetaGraph,
        base_adjs: dict[CEType, torch.Tensor],
        *,
        custom_ntype_alias: Mapping[NType, str] | None = None,
        cache_dir: str | None = None,
        cache_idx_dtype: torch.dtype | None = None,
        cache_val_dtype: torch.dtype | None = None,
        readonly: bool = True,
        verbose: bool | int | None = None,
        legacy_mode: bool = False,
    ):
        ...

    @overload
    def __init__(
        self,
        ntypes: list[NType],
        cetypes: list[CEType],
        self_loop_cetypes: list[CEType],
        base_adjs: dict[CEType, torch.Tensor],
        *,
        custom_ntype_alias: Mapping[NType, str] | None = None,
        cache_dir: str | None = None,
        cache_idx_dtype: torch.dtype | None = None,
        cache_val_dtype: torch.dtype | None = None,
        readonly: bool = True,
        verbose: bool | int | None = None,
        legacy_mode: bool = False,
    ):
        ...

    def __init__(
        self,
        *args,
        custom_ntype_alias: Mapping[NType, str] | None = None,
        cache_dir: str | None = None,
        cache_idx_dtype: torch.dtype | None = None,
        cache_val_dtype: torch.dtype | None = None,
        readonly: bool = True,
        verbose: bool | int | None = None,
        legacy_mode: bool = False,
    ):
        if len(args) == 2:
            mg, base_adjs = args
        else:
            *mg_args, base_adjs = args
            mg = MetaGraph(*mg_args)
        mg: MetaGraph
        base_adjs: dict[CEType, torch.Tensor]
        assert all(isinstance(etype, tuple) for etype in base_adjs),\
            'require canonical etypes'

        self.mg = mg
        self._ntype_alias = custom_ntype_alias or _generate_aliases(mg.ntypes)
        self.cache_idx_dtype = cache_idx_dtype
        self.cache_val_dtype = cache_val_dtype and dtype_convert(
            cache_val_dtype, 'torch'
        )

        self.cache_dir = cache_dir
        self.readonly = readonly
        self.adjs: dict[CEType, torch.Tensor] = {}
        DENSITY = 0.01
        for etype, adj in base_adjs.items():
            if isinstance(adj, _SelfLoopMatrix):
                continue
            if adj.layout != torch.strided:
                density = adj._nnz() / adj.numel()
                if density > DENSITY:  # XXX: allow setting threshold
                    adj = adj.to_dense()
                    if verbose:
                        print(
                            f'{etype}: Density: {density:.2%} > {DENSITY:.2%}, converted to dense.'
                        )
                    assert not legacy_mode
            if mg.is_self_loop(etype):
                adj = _SelfLoopMatrix(adj)
            self.adjs[self.mg.to_canonical_etype(etype)] = adj

        self._temdir = None
        self._cache = _LRUCache(0)
        if not readonly and cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._temdir = tempfile.TemporaryDirectory(dir=self.cache_dir)
        self._pbar = None
        self.verbose = verbose or False
        self.legacy_mode = legacy_mode
        return

    @property
    def _cache_dir(self):
        if self.readonly:
            return None
        return self._temdir.name

    def __del__(self):
        if hasattr(self, '_temdir'):
            if self._temdir is not None:
                self._temdir.cleanup()
        return

    def fit_to_mps(self, mps: list[MetaPathT]):
        """Remove adjs that are not going to be used in specified metapaths.
        Can be used to save some memory.
        """
        raise NotImplementedError

    @staticmethod
    @lru_cache
    def _feat_hash(feat: torch.Tensor) -> str:
        if feat is None:
            return 'none'
        with printoptions(profile='default'):
            sha1 = hashlib.sha1(str(feat).encode())
            sha1.update(f'{feat.shape = }'.encode())
            if feat.dtype != torch.float:
                sha1.update(f'{feat.dtype = }'.encode())
            return sha1.hexdigest()

    def check_hash(
        self, mps: list[tuple[CEType]], feats: dict[str, torch.Tensor | None]
    ) -> str:
        """SHA1 hash of given metapaths and source features"""
        src_feats = {}
        for mp in mps:
            srctype = self._get_srctype(mp)
            src_feats[srctype] = feats[srctype]

        sha1 = hashlib.sha1(
            str(tuple(self._filter_self_loops(mp) for mp in mps)).encode()
        )
        for srctype, feat in src_feats.items():
            sha1.update(
                f'{srctype}: {self._feat_hash(feat).encode()}'.encode()
            )
        return sha1.hexdigest()

    def cache_name(
        self, mp: tuple[CEType], feat: torch.Tensor | None = None,
        max_length: int = 40
    ):
        if len(mp) > 1:
            mp = self._filter_self_loops(mp)
        HASH_LEN = 8
        assert max_length >= HASH_LEN
        sha1 = hashlib.sha1(str(mp).encode()).hexdigest()[:HASH_LEN]
        ntype_seq = [self.mg.to_canonical_etype(etype)[0] for etype in mp
                     ] + [self.mg.to_canonical_etype(mp[-1])[-1]]
        mpstr = ''.join(self._ntype_alias[ntype] for ntype in ntype_seq)
        if feat is not None and not self.legacy_mode:
            return f'{sha1}_{self._feat_hash(feat)[:4]}_{mpstr}'[:max_length]
        return f'{sha1}_{mpstr}'[:max_length]

    def _save_memmap(
        self, filedir: str, res: torch.Tensor, feat_hash: str, dtype=None
    ):
        assert not self.readonly
        os.makedirs(os.path.dirname(filedir), exist_ok=True)
        if res.layout == torch.strided:
            save_stride_memmap(
                filedir,
                res,
                dtype=dtype or self.cache_val_dtype,
                metadata={
                    'feat_hash': feat_hash,
                },
            )
            return
        save_csr_memmap(
            filedir,
            res.to_sparse_csr(),
            idx_dtype=self.cache_idx_dtype,
            val_dtype=dtype or self.cache_val_dtype,
            metadata={
                'feat_hash': feat_hash,
            },
        )
        return

    def _save_cache(
        self,
        export_dir: str,
        mp: tuple[CEType],
        src_feat: torch.Tensor,
        res: torch.Tensor,
        skip_exist: bool = False,
        next_mp_hint: CEType | None = None,
    ):

        def __save_in_memory(next_mp_hint: CEType):
            next_mp_hint = self._filter_self_loops(next_mp_hint)
            if next_mp_hint[:len(mp)] == mp:
                self._cache[mp] = (res, self._feat_hash(src_feat))
            return

        if next_mp_hint is not None:
            __save_in_memory(next_mp_hint)
        if export_dir is None:
            return
        cache_name = self.cache_name(mp, src_feat)
        feat_hash = self._feat_hash(src_feat)
        if os.path.exists(os.path.join(export_dir, cache_name)):
            if skip_exist:
                return
            raise RuntimeError(
                f'{os.path.join(export_dir, cache_name)} exists.'
            )
        if (
            self.cache_val_dtype is not None
            and torch.finfo(self.cache_val_dtype
                            ).bits < torch.finfo(res.dtype).bits
        ):
            # NOTE: if half precision used
            if not os.path.exists(os.path.join(self.cache_dir, cache_name)):
                # will save float16 in export_dir
                self._save_memmap( # save float32 in tmp
                    os.path.join(self._cache_dir, cache_name),
                    res,
                    feat_hash,
                    dtype=res.dtype,
                )
            if os.path.samefile(export_dir, self._cache_dir):
                # Cache has been saved in higher precision.
                return

        self._save_memmap(os.path.join(export_dir, cache_name), res, feat_hash)
        return

    def _has_cache(
        self,
        mp: tuple[CEType],
        src_feat: torch.Tensor | dict[NType, torch.Tensor],
        dirs: list[str],
    ) -> int | None:

        if isinstance(src_feat, Mapping):
            src_feat = src_feat[self._get_srctype(mp)]
        cache_name = self.cache_name(mp, src_feat)

        def __has_cache(cache_dir: str | None, ):
            if cache_dir is None:
                return None
            cache_path = os.path.join(cache_dir, cache_name)
            return os.path.exists(cache_path)

        for i, d in enumerate(dirs):
            if __has_cache(d):
                return i
        return None

    def has_cache(
        self,
        mp: tuple[CEType],
        src_feat: torch.Tensor | dict[NType, torch.Tensor],
    ) -> str | None:
        """
        Checks if a cached feature exists for the given metapath and source features.

        Looks for a cached hash file for `mp` in `self.cache_dir`, and
        verifies it matches the hash of `src_feat`.
        Returns the cached hash if found and valid, else None.

        Args:
            mp (tuple[ETYPE_ID]): Metapath (edge type IDs).
            src_feat (torch.Tensor): Source node features.

        Returns:
            hash or None: Cached feature hash if valid, else None.

        Raises:
            AssertionError: If a cache is found but the hash does not match `src_feat`.
        """

        return self._has_cache(mp, src_feat, [self.cache_dir]) is not None

    # def __load_cache_no_check(
    #     self,
    #     mp: tuple[CEType],
    #     cache_dir: str,
    #     row_indices: Iterable[int] | None = None,
    # ):
    #     cache_name = self.cache_name(mp)
    #     cache_path = os.path.join(cache_dir, cache_name)
    #     if not os.path.exists(cache_path):
    #         return None
    #     meta, cache = load_memmap(cache_path, row_indices)
    #     if self.verbose >= 2:
    #         print(
    #             f'cache loaded from {cache_path}',
    #             file=getattr(self, '_pbar', None)
    #         )
    #     return cache, meta['feat_hash']

    # def _find_cache_in_reference_dir(
    #     self,
    #     mp: tuple[CEType],
    #     src_feat: torch.Tensor,
    #     reference_dir: str,
    #     out_dir: str,
    # ):
    #     cache = self._find_cache(mp, src_feat, extra_in_dir=reference_dir)
    #     if cache is None:
    #         return None
    #     if out_dir is not None:
    #         name = self.cache_name(mp, src_feat)
    #         src_path = os.path.join(reference_dir, name)
    #         path = os.path.join(out_dir, name)
    #         if not os.path.exists(path):
    #             os.symlink(src_path, path)

    #     return cache

    def _find_cache(
        self,
        mp: tuple[CEType],
        src_feat: torch.Tensor,
        row_indices=None,
    ):

        assert len(mp)

        def _indexing(res):
            if res is None:
                return None
            cache, feat_hash = res
            if row_indices is not None:
                if cache.layout == torch.strided:
                    return cache[row_indices], feat_hash
                cache = cache.to_sparse_csr()
                return torch.stack([cache[i] for i in row_indices]), feat_hash
            return cache, feat_hash

        def _find_in_memory():
            if src_feat is None and len(mp) == 1:
                return self.adjs[self.mg.to_canonical_etype(mp[0])
                                 ], self._feat_hash(None)

            if len(mp) == 1 and self.mg.is_self_loop(mp[0]):
                return src_feat, self._feat_hash(src_feat)
            if mp not in self._cache:
                return None
            return self._cache[mp]

        def _find_in_disk(dir_: str | None):

            if dir_ is None:
                return None
            cache_name = self.cache_name(mp, src_feat)
            cache_path = os.path.join(dir_, cache_name)
            if not os.path.exists(cache_path):
                return None
            meta, cache = load_memmap(cache_path, row_indices)
            if self.verbose >= 2:
                print(
                    f'cache loaded from {cache_path}',
                    file=getattr(self, '_pbar', None)
                )
            return cache, meta['feat_hash']

        # NOTE: check order: memory -> cache_dir -> out_dir
        cache, feat_hash = (
            _indexing(_find_in_memory()) or _find_in_disk(self._cache_dir)
            or _find_in_disk(self.cache_dir) or (None, None)
        )
        if cache is None:
            return None

        assert feat_hash == self._feat_hash(src_feat), (
            f'Hash mismatch for {self.cache_name(mp, src_feat)}! '
            'Specify a different cache_dir to avoid collisions, '
            'required after node features have changed.'
        )
        return cache

    def _split_mp_with_bottleneck(
        self,
        mp: tuple[CEType],
        # src_feat: torch.Tesnor | None,
    ):

        def get_score_fn(mp: tuple[CEType]):

            def score_fn(args):
                pos, _, dim = args
                # return (dim, pos + 1 - (len(mp) // 2))
                return (dim, pos)  # NOTE: promote longer suffix

            return score_fn

        mp_ = self._filter_self_loops(mp)
        if len(mp_) == 0:
            return None
        mp = mp_
        bottlenecks = []
        for pos, etype in enumerate(mp):
            adj = self.adjs[etype]
            if adj.layout == torch.strided:
                bottlenecks.append((pos, etype, adj.shape[0]))

        if len(bottlenecks) == 0:
            return None
        bottleneck = min(bottlenecks, key=get_score_fn(mp))
        # if src_feat is not None:
        #     if bottleneck[-1] >= src_feat.shape[1]:
        #         return None
        if bottleneck[-1] >= self.adjs[mp[0]].shape[1]:
            return None
        assert bottleneck[0] + 1 < len(mp)
        assert self.adjs[mp[bottleneck[0] + 1]].layout == torch.strided
        prefix = mp[:bottleneck[0] + 1]
        suffix = mp[bottleneck[0] + 1:]

        # def strfmp(mp_):
        #     return f'{self.adjs[mp_[-1]].shape[0]} x {self.adjs[mp_[0]].shape[-1]}'

        # print(self.cache_name(mp), end=' = ')
        # print(self.cache_name(prefix), '+', self.cache_name(suffix), end='\t')
        # print(f'{strfmp(mp)} -> {strfmp(prefix)} + {strfmp(suffix)}')

        return prefix, suffix

    def _filter_self_loops(self, mp: tuple[CEType]):
        return tuple(e for e in mp if not self.mg.is_self_loop(e))

    def _collect_lora(
        self,
        prefix: tuple[CEType],
        suffix: tuple[CEType],
        src_feat: torch.Tensor | None,
        out_dir: str | None = None,
        next_mp_hint: tuple[CEType] | None = None,
    ):
        assert prefix
        assert suffix

        def _save(mp, src_feat, obj):
            path = os.path.join(out_dir, self.cache_name(mp, src_feat))
            if os.path.exists(path):
                return
            if len(self._filter_self_loops(mp)) > 1 or src_feat is not None:
                # save in float32 precision
                self._save_memmap(
                    path, obj, self._feat_hash(src_feat), dtype=obj.dtype
                )
            return

        prefix_res = self._collect(
            prefix, src_feat, out_dir=None, next_mp_hint=next_mp_hint
        )
        _save(prefix, src_feat, prefix_res)
        suffix_res = self._collect(
            suffix, None, out_dir=None, next_mp_hint=None
        )
        _save(suffix, None, suffix_res)
        # def keep_cache(mp, src_feat):
        #     cache_name = self.cache_name(mp, src_feat)
        #     shutil.move(
        #         os.path.join(self._cache_dir, cache_name),
        #         os.path.join(self.cache_dir, cache_name),
        #     )
        #     return

        # if self._has_cache(
        #     prefix, src_feat, [self._cache_dir, self.cache_dir]
        # ) is None:
        #     # if not self.has_cache(prefix, src_feat, out_dir=out_dir):
        #     prefix_res = self._collect(
        #         prefix, src_feat, out_dir=self.cache_dir, next_mp_hint=next_mp_hint
        #     )
        #     # cache should have saved in self._cache_dir
        #     if len(prefix) > 1 or src_feat is not None:
        #         if self._cache_dir is not None:
        #             keep_cache(prefix, src_feat)
        #         else:
        #             self._save_cache(
        #                 self.cache_dir, prefix, src_feat, prefix_res
        #             )
        # else:
        #     prefix_res = self._find_cache(prefix, src_feat)
        # # if not self.has_cache(suffix, None, out_dir=out_dir):
        # if self._has_cache(
        #     suffix, None, [self._cache_dir, self.cache_dir]
        # ) is None:
        #     suffix_res = self._collect(suffix, None, next_mp_hint=None)
        #     if len(suffix) > 1:
        #         if self._cache_dir is not None:
        #             keep_cache(suffix, None)
        #         else:
        #             self._save_cache(self.cache_dir, suffix, None, suffix_res)
        # else:
        #     suffix_res = self._find_cache(suffix, None)

        # if src_feat is not None:
        #     return suffix_res.T @ prefix_res

        return LowRankMatrix(prefix_res, suffix_res)

    def _collect(
        self,
        mp: tuple[CEType],
        src_feat: torch.Tensor | None,
        out_dir: str | None,
        next_mp_hint: tuple[CEType] | None = None,
    ):
        mp = self._filter_self_loops(mp)
        if len(mp) == 0:
            return src_feat

        cache = self._find_cache(mp, src_feat)
        if cache is not None:
            # NOTE: save in case out_dir is not where the cache found
            self._save_cache(
                out_dir, mp, src_feat, cache, skip_exist=True,
                next_mp_hint=next_mp_hint
            )
            return cache
        rval = self._collect(
            mp[:-1], src_feat, out_dir=self._cache_dir,
            next_mp_hint=next_mp_hint
        )
        if rval.layout not in (torch.strided, torch.sparse_coo):
            rval = rval.to_sparse_coo()

        if torch.finfo(self.adjs[mp[-1]].dtype
                       ).bits > torch.finfo(rval.dtype).bits:
            warnings.warn(
                f'The adj have dtype: {self.adjs[mp[-1]].dtype} but '
                f'either passed feature or saved cache is in {rval.dtype}.\n'
                'This may make computation slower. '
                'Restart with a clean cache_dir solves this issue.'
            )
            rval_t = reduce(
                torch.matmul, [self.adjs[eid].T for eid in mp[:-1]], src_feat.T
            )
            rval = rval_t.T
        feat = self.adjs[mp[-1]] @ rval
        self._save_cache(
            out_dir, mp, src_feat, feat, next_mp_hint=next_mp_hint
        )
        return feat

    def _get(
        self,
        mp: tuple[CEType],
        src_feat: torch.Tensor | None,
        row_indices: Iterable[int] | None,
    ) -> torch.Tensor:
        mp_ = mp
        mp = self._filter_self_loops(mp)
        assert len(mp_)
        if len(mp) == 0:
            mp = mp_

        tem = self._split_mp_with_bottleneck(mp)
        if tem is not None:
            prefix, suffix = tem
            prefix_res = self._find_cache(prefix, src_feat)
            suffix_res = self._find_cache(suffix, None, row_indices)
            assert prefix_res is not None, (
                f'Cache for {self.cache_name(prefix, src_feat)} not found in {self.cache_dir}. '
                'Ensure features were collected with the same cache_dir.'
            )
            assert suffix_res is not None, (
                f'Cache for {self.cache_name(suffix, None)} not found in {self.cache_dir}. '
                'Ensure features were collected with the same cache_dir.'
            )
            res = LowRankMatrix(prefix_res, suffix_res)
            if self.cache_val_dtype is not None and self.cache_val_dtype != res.dtype:
                # raise NotImplementedError()
                res = res.to(self.cache_val_dtype)
            return res
        res = self._find_cache(mp, src_feat, row_indices)
        assert res is not None, (
            f'Cache for {self.cache_name(mp, src_feat)} not found in {self.cache_dir}. '
            'Ensure features were collected with the same cache_dir.'
        )
        if self.cache_val_dtype is not None and self.cache_val_dtype != res.dtype:
            res = res.to(self.cache_val_dtype)
        return res

    def mp_to_canonical(self, mp: tuple[EType]):
        return tuple(self.mg.to_canonical_etype(e) for e in mp)

    def _get_srctype(self, mp: tuple[EType]) -> NType:
        return self.mg.to_canonical_etype(mp[0])[0]

    def _dump_meta(
        self,
        mps: list[tuple[CEType]],
        feats: dict[NType, torch.Tensor],
        out_dir: str,
    ):
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'meta.json')
        if os.path.exists(out_path):
            with open(out_path, encoding='utf8') as fin:
                meta = json.load(fin)
        else:
            meta = {}
        FEAT_HASH_LEN = 4
        MP_HASH_LEN = 8
        for ntype, feat in feats.items():
            feat_hash = self._feat_hash(feat)[:FEAT_HASH_LEN]
            meta[feat_hash] = ntype

        for mp in mps:
            if len(mp) > 1:
                mp = self._filter_self_loops(mp)
            sha1 = hashlib.sha1(str(mp).encode()).hexdigest()[:MP_HASH_LEN]
            meta[sha1] = {
                'etypes': [etype for (_, etype, __) in mp],
                'ntypes': [*[ntype for (ntype, _, __) in mp], mp[-1][-1]],
            }
        with open(out_path, 'w', encoding='utf8') as fout:
            json.dump(meta, fout, indent=4)
        return meta

    def precompute_features(
        self,
        mps: list[tuple[CEType]],
        feats: dict[NType, torch.Tensor],
        cache_size: int | None = None,
        sort_mps: bool = True,
    ):
        """
        Precompute features along given metapaths.

        Args:
            mps (list[tuple[ETYPE_ID]]): List of metapaths, each as a tuple of edge type IDs.
            feats (dict[NType, torch.Tensor]): Node features by node type.
            cache_size (int)

        Returns:
        Generator[tuple[int, torch.Tensor], None, None]:
            Yields tuples of metapath index (mp_id) and collected features (mps[mp_id]).
            Note that the order of output would not match the input `mps` order.

        """
        if not self.readonly:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._dump_meta(mps, feats, self.cache_dir)
        mps = list(map(self.mp_to_canonical, mps))
        id_mapping = dict(zip(mps, range(len(mps))))
        if sort_mps:
            mps = sort_in_dfs_order(mps)
        if cache_size is not None:
            self._cache.set_capacity(cache_size)

        for mp, next_mp in zip(mps, mps[1:] + [tuple()]):
            src_feat = feats[self._get_srctype(mp)]
            tem = self._split_mp_with_bottleneck(mp)
            if tem is not None:
                res = self._collect_lora(
                    *tem, src_feat, out_dir=self.cache_dir,
                    next_mp_hint=next_mp
                )
            else:
                res = self._collect(
                    mp,
                    src_feat,
                    out_dir=self.cache_dir,
                    next_mp_hint=next_mp,
                )
            if self.cache_val_dtype is not None and self.cache_val_dtype != res.dtype:
                res = res.to(self.cache_val_dtype)
            yield id_mapping[mp], res
        if cache_size is not None:
            self._cache.set_capacity(0)
        #self._temdir.cleanup()
        return

    def iget(
        self,
        mps: list[tuple[CEType]],
        feats: dict[NType, torch.Tensor] | None,
        row_indices: Iterable[int] | None = None,
    ):
        if not self.legacy_mode:
            # Backward compatibility
            pat = re.compile(r'\w{8}_\w{4}_\w+')
            if not any(
                [
                    pat.match(name) is not None
                    for name in os.listdir(self.cache_dir)
                ]
            ):
                self.legacy_mode = True

        with tqdm(
            mps, disable=not self.verbose, desc='Loading feats', leave=False
        ) as pbar:
            for mp_id, mp in enumerate(pbar):
                srctype = self._get_srctype(mp)
                srcfeat = None
                if feats is not None:
                    srcfeat = feats[srctype]
                res = self._get(mp, srcfeat, row_indices)
                yield res

    def get(
        self,
        mps: list[tuple[CEType]],
        feats: dict[NType, torch.Tensor] | None,
        row_indices: Iterable[int] | None = None,
    ) -> list[torch.Tensor]:
        results = np.empty((len(mps), ), dtype=object)
        for mp_id, res in enumerate(
            self.iget(mps, feats, row_indices=row_indices)
        ):
            results[mp_id] = res
        return results

    def ls(
        self, mps: list[tuple[CEType]], feats: dict[NType, torch.Tensor] | None
    ):
        paths = []
        for mp in mps:
            mp = self._filter_self_loops(mp)
            if len(mp) == 0:
                continue
            src_feat = feats[self._get_srctype(mp)]
            if len(mp) == 1 and src_feat is None:
                continue

            tem = self._split_mp_with_bottleneck(mp)
            if tem is not None:
                prefix, suffix = tem
                p1 = os.path.join(
                    self.cache_dir, self.cache_name(prefix, src_feat)
                )
                p2 = os.path.join(
                    self.cache_dir, self.cache_name(suffix, None)
                )
                if p1 not in paths:
                    paths.append(p1)
                if len(p2) > 1 and p2 not in paths:
                    paths.append(p2)
            else:
                paths.append(
                    os.path.join(
                        self.cache_dir, self.cache_name(mp, src_feat)
                    )
                )
        return paths

    # def precompute_features_parallel(
    #     self,
    #     mps: list[tuple[ETYPE_ID]],
    #     feats: dict[NType, torch.Tensor],
    #     num_workers: int = 0,
    #     out_dir: str | None = None,
    # ):
    #     """
    #     [WIP] NotImplementedError: Cannot access storage of SparseTensorImpl
    #     Customly destructing and resturcting the sparse tensors may solve the problem
    #     Precompute features along given metapaths.

    #     Args:
    #         mps (list[tuple[ETYPE_ID]]): List of metapaths, each as a tuple of edge type IDs.
    #         feats (dict[NType, torch.Tensor]): Node features by node type.
    #         out_dir (str | None, optional): Directory to save collected features.
    #             Could be the same as `cache_dir` or a different one.
    #             If `None`, features are not saved to disk.

    #     Returns:
    #     Generator[tuple[int, torch.Tensor], None, None]:
    #         Yields tuples of metapath index (mp_id) and collected features (mps[mp_id]).
    #         Note that the order of output would not match the input `mps` order.

    #     """
    #     from torch.multiprocessing import Process, Queue
    #     id_mapping = dict(zip(mps, range(len(mps))))
    #     mps = sort_in_bfs_order(mps)

    #     in_queue, out_queue = Queue(), Queue()

    #     mp_dict = defaultdict(list)
    #     for mp in mps:
    #         mp_dict[len(mp)].append(mp)
    #     print({k: len(mps_at_k) for k, mps_at_k in mp_dict.items()})

    #     assert min(mp_dict) >= 1
    #     for mp in mp_dict.pop(1, []):
    #         src_feat = feats[self._get_srctype(mp)]
    #         res = self.collect(
    #             mp,
    #             src_feat,
    #             out_dir=out_dir,
    #         )
    #         yield id_mapping[mp], res

    #     for k in range(min(mp_dict), max(mp_dict) + 1):

    #         batch_size = max(
    #             min(int(len(self.adjs)**0.5),
    #                 len(mp_dict[k]) // num_workers), 1
    #         )
    #         print(
    #             f'batch_size@{k} = {batch_size}, n_batches = {len(mp_dict[k]) / batch_size:.2f}'
    #         )
    #         for i in range(0, len(mp_dict[k]), batch_size):
    #             in_queue.put((i, mp_dict[k][i:i + batch_size]))


#
#     for _ in range(num_workers):
#         in_queue.put(None)

#     workers = [
#         Process(
#             target=FeatureCollector._mp_worker_fn,
#             args=(self, self.adjs, feats, out_dir, in_queue, out_queue)
#         ) for _ in range(num_workers)
#     ]
#     for w in workers:
#         w.start()
#     for _ in range(sum(len(mps) for mps in mp_dict.values())):
#         mp, res = out_queue.get()
#         yield id_mapping[mp], res

#     for w in workers:
#         w.join()
#     return

# @staticmethod
# def _mp_worker_fn(ref, adjs, feats, out_dir, in_queue, out_queue):
#     base_adjs = dict(zip(ref.cetypes, adjs))
#     collector = FeatureCollector(
#         cetypes=ref.cetypes,
#         base_adjs=base_adjs,
#         self_loop_etypes=[ref.cetypes[i] for i in ref.self_loop_etype_ids],
#         cache_dir=ref.cache_dir,
#         cache_idx_dtype=ref.cache_idx_dtype,
#         cache_val_dtype=ref.cache_val_dtype,
#         verbose=False,
#     )
#     while True:
#         task = in_queue.get()
#         if task is None:
#             break  # Sentinel

#         offset, mps = task

#         for i, res in collector.precompute_features(
#             mps, feats, out_dir=out_dir, sort_mps=False
#         ):
#             out_queue.put((mps[i], res))


class LabelFeatCollector(FeatureCollector):

    @classmethod
    def from_collector(cls, collector: FeatureCollector):
        return cls(
            collector.mg,
            collector.adjs,
            custom_ntype_alias=collector._ntype_alias,
            cache_dir=collector.cache_dir,
            cache_idx_dtype=collector.cache_idx_dtype,
            cache_val_dtype=collector.cache_val_dtype,
            verbose=collector.verbose,
            legacy_mode=False,
        )

    def diag_cache_name(self, mp):
        return os.path.join('diag', f'{self.cache_name(mp)}.pt')

    def lpa_diag(
        self,
        mps: list[tuple[CEType]],
        batch_size: int = 16,
        device: str = 'cuda',
    ):
        assert all(mp[0][0] == mp[-1][-1] for mp in mps)
        assert min(map(len, mps)) > 1
        # lmsps_dir = os.path.expanduser('~/data/lp-smag-vanilla/LMSPS/diag/')
        # def get_truth(mp: tuple[CEType]):
        #     ntypes = [*[s for s, _, __ in mp], mp[-1][-1]]
        #     name = ''.join(n[0].upper() for n in ntypes)
        #     path = os.path.join(lmsps_dir, f'{name}.pt')
        #     if not os.path.exists(path):
        #         print(f'{name} not exist')
        #         return None
        #     return torch.load(path)
        os.makedirs(self.cache_dir, exist_ok=True)

        mps = [mp for mp in mps if not self.has_diag(mp)]
        if not mps:
            return

        print(f'Diag will output to {self.cache_dir}')

        num_nodes = self.adjs[mps[0][-1]].shape[0]
        cur_device = self.adjs[mps[0][-1]].device

        self.adjs = {k: v.to(device) for k, v in self.adjs.items()}
        diags = defaultdict(list)
        with tqdm(torch.arange(num_nodes).split(batch_size)) as pbar:
            for batch in pbar:
                for mp, diag in self._lpa_diag(mps, batch, device=device):
                    diags[mp].append(diag)

        for mp, diag in diags.items():
            diag = torch.concat(diag).cpu()
            out_path = os.path.join(self.cache_dir, self.diag_cache_name(mp))
            # # diag_truth = get_truth(mp)
            # diag_truth = collector._collect(mp, None).to_dense().diag()
            # if diag_truth is not None:
            #     if not torch.allclose(
            #         diag_truth[:len(diag)].cpu(), diag, atol=1e-5
            #     ):
            #         breakpoint()
            #     else:
            #         print(f'MP: {collector.cache_name(mp, None)} checked OK')
            torch.save(diag, out_path)

        # put adj back to cpu
        self.adjs = {k: v.to(cur_device) for k, v in self.adjs.items()}
        return

    def _lpa_diag(
        self,
        mps: list[tuple[CEType]],
        batch: torch.Tensor,
        device='cuda',
    ):

        def _collect(
            mp: tuple[CEType],
            cache: dict,
        ):
            assert len(mp) > 0, mp
            if mp in cache:
                return cache[mp]

            lval = _collect(mp[1:], cache)

            rows = lval @ self.adjs[mp[0]]
            cache[mp] = rows
            return rows

        cache = {}
        for e0 in {mp[-1] for mp in mps}:
            a0 = self.adjs[e0]
            if a0.layout == torch.strided:
                rows = a0[batch]
            else:
                # rows = torch.stack([a0[i] for i in batch]).to_dense()
                rows = torch.stack([a0[i].to_dense() for i in batch])
            cache[(e0, )] = rows.to(device)
        batch = batch.to(device)

        ps = psutil.Process()
        with tqdm(mps, leave=False) as pbar:
            for mp in pbar:
                mp = self._filter_self_loops(mp)
                rows = _collect(mp, cache)
                if rows.layout == torch.strided:
                    diag = torch.take_along_dim(
                        rows, batch.view(-1, 1), dim=1
                    ).squeeze()
                else:
                    diag = torch.tensor(
                        [
                            row[i]
                            for i, row in zip(batch, rows.to_sparse_csr())
                        ]
                    )
                pbar.set_description(
                    f'MEM: {_format_memory(ps.memory_info().rss)}'
                )
                yield mp, diag

    def precompute_features(
        self,
        mps: list[tuple[CEType]],
        feats: torch.Tensor | dict[NType, torch.Tensor],
        cache_size: int | None = None,
        sort_mps: bool = True,
    ):
        """
        Precompute features along given metapaths.

        Args:
            mps (list[tuple[ETYPE_ID]]): List of metapaths, each as a tuple of edge type IDs.
            feats (dict[NType, torch.Tensor]): Node features by node type.
            cache_size (int)
            out_dir (str | None, optional): Directory to save collected features.
                Could be the same as `cache_dir` or a different one.
                If `None`, features are not saved to disk.

        Returns:
        Generator[tuple[int, torch.Tensor], None, None]:
            Yields tuples of metapath index (mp_id) and collected features (mps[mp_id]).
            Note that the order of output would not match the input `mps` order.

        """
        if isinstance(feats, torch.Tensor):
            tgt_ntype = self._get_srctype(mps[0])
            feats = {tgt_ntype: feats}
        src_feat = feats[self._get_srctype(mps[0])]
        num_nodes = self.adjs[mps[0][-1]].shape[0]
        # check if label only contains training label
        if src_feat.layout == torch.strided:
            __non_zero_entries = ((src_feat == 1).sum(1) > 0).sum().item()
        else:
            __non_zero_entries = len(
                torch.unique(src_feat.coalesce().indices()[0])
            )
        if __non_zero_entries >= num_nodes:
            raise ValueError(
                'Only training label should be used in label feats, '
                f'but found {__non_zero_entries} entries.'
            )
        gen = super().precompute_features(
            mps, feats, cache_size=cache_size, sort_mps=sort_mps
        )
        for mp_id, res in gen:
            mp = mps[mp_id]
            src_feat = feats[self._get_srctype(mp)]
            diag: torch.Tensor = self._get_diag(mp, None)
            if isinstance(res, LowRankMatrix):
                res = res.to_dense()
            res = res - diag.unsqueeze(-1) * src_feat
            yield mp_id, res
        return

    def has_diag(
        self,
        mp: tuple[CEType],
    ):
        mp = self._filter_self_loops(mp)
        assert len(mp)
        path = os.path.join(self.cache_dir, self.diag_cache_name(mp))
        return os.path.exists(path)

    def _get_diag(
        self,
        mp: tuple[CEType],
        row_indices: Iterable[int] | None,
    ) -> torch.Tensor:
        mp = self._filter_self_loops(mp)
        assert len(mp)
        path = os.path.join(self.cache_dir, self.diag_cache_name(mp))
        assert os.path.exists(path), (
            f'Cache for diag of {self.cache_name(mp)} not found in {self.cache_dir}. '
            'Ensure features were collected with the same cache_dir.'
        )
        diag = torch.load(path)
        if row_indices is not None:
            return diag[row_indices]
        return diag

    def _get(
        self,
        mp: tuple[CEType],
        src_feat: torch.Tensor | None,
        row_indices: Iterable[int] | None,
    ) -> torch.Tensor:
        mp = self._filter_self_loops(mp)
        assert len(mp)

        res = super()._get(mp, src_feat, row_indices)
        diag = self._get_diag(mp, row_indices)
        if row_indices is not None:
            src_feat = src_feat[row_indices]
        tem = diag.unsqueeze(-1) * src_feat
        if isinstance(res, LowRankMatrix):
            res = res.to_dense()
        res = res - tem.to(res.dtype)
        # if self.cache_val_dtype is not None and self.cache_val_dtype != res.dtype:
        #     res = res.to(self.cache_val_dtype)
        return res

    @classmethod
    def get_masksed_label(
        cls, label: torch.Tensor, train_indices: torch.Tensor,
        fmt: torch.layout = torch.strided
    ):

        def _check(label: torch.Tensor):
            if label.layout == torch.strided:
                __non_zero_entries = (label.sum(1) > 0).sum().item()
            else:
                __non_zero_entries = len(
                    torch.unique(label.coalesce().indices()[0])
                )
            if __non_zero_entries >= train_label.shape[0]:
                raise ValueError(
                    'Only training label should be used in label feats, '
                    f'but found {__non_zero_entries} entries.'
                )
            return True

        train_label = label[train_indices]
        if len(train_label.shape) == 2:
            # XXX: What is this?
            raise NotImplementedError
            _check(train_label)
            return train_label

        n_nodes = len(label)
        n_classes = label.max().item() + 1
        assert len(train_indices) < n_nodes
        train_label = torch.sparse_coo_tensor(
            torch.stack([train_indices, train_label]),
            values=torch.ones(len(train_label)), size=(n_nodes, n_classes)
        )
        return train_label.to_sparse(layout=fmt)
