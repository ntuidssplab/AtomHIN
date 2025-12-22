from __future__ import annotations

import abc
import hashlib
from typing import Iterable

import numpy as np
import torch

from dhgl.type import CEType, NType
from dhgl.utils.precomputation.feature_collector import FeatureCollector, LowRankMatrix


def _dict_tranpose(d: dict[NType, dict[NType, torch.Tensor]]):
    dt = {}
    for ntype, data in d.items():
        for sub_ntype, feat in data.items():
            if sub_ntype not in dt:
                dt[sub_ntype] = {}
            dt[sub_ntype][ntype] = feat
    return dt


class _GeneratorWithLen:

    def __init__(self, gen, n):
        self.gen = gen
        self.n = n

    def __iter__(self):
        yield from self.gen

    def __len__(self):
        return self.n


class BasePrecomDataset:

    def __init__(
        self,
        drop_unlabeled: bool = False,
        label_mask: torch.Tensor | None = None,
        allow_low_rank_feat: bool = False,
    ):
        self.drop_unlabeled = drop_unlabeled
        if drop_unlabeled:
            assert label_mask is not None,\
                'label_mask is required to drop unlabeled nodes but got None'
        self.inv_mapping = self._get_inv_mapping(
            label_mask
        ) if drop_unlabeled else None
        self.allow_low_rank = allow_low_rank_feat
        return

    def _get_inv_mapping(self, label_mask: torch.Tensor):
        assert self.drop_unlabeled is True
        assert label_mask is not None
        inv_mapping = torch.full((len(label_mask), ), -1)
        inv_mapping[label_mask] = torch.arange(label_mask.sum().item())
        return inv_mapping

    def __getitem__(self, item):
        if isinstance(item, (Iterable, slice)):
            return self.__getitems__(item)
        raise NotImplementedError('Use batch indices instead')

    def __getitems__(self, items):
        # items: b,c -> c x b
        # items: b -> C x b
        # items: :,c -> c x n
        if isinstance(items, tuple):
            bi, ci = items
        else:
            bi = items
            ci = None
        assert not isinstance(
            bi, int
        ), 'Indexing not by batch is extremely inefficient'

        if bi == slice(None):
            bi = None
        elif self.inv_mapping is not None:
            bi = self.inv_mapping[bi]

        def _handle_low_rank(feat: torch.Tensor | LowRankMatrix):
            if not self.allow_low_rank and isinstance(feat, LowRankMatrix):
                return feat.to_dense()
            return feat

        if isinstance(ci, int):
            return _handle_low_rank(self._get_feats(bi, [ci])[0])

        return list(map(_handle_low_rank, self._get_feats(bi, ci)))

    @property
    def T(self):
        return _GeneratorWithLen(
            (self[:, i] for i in range(self.shape[1])), self.shape[1]
        )

    def __len__(self):
        return self.shape[0]

    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, int]:
        ...

    @abc.abstractmethod
    def _get_feats(
        self, indices: Iterable[int] | None,
        c_indices: Iterable[int] | None = None
    ):
        ...

    @property
    @abc.abstractmethod
    def check_hash(self) -> str:
        """Used in verification of dataset caching"""


class PrecomDataset(BasePrecomDataset):
    """Standard precomputation-based dataset"""

    def __init__(
        self,
        collector: FeatureCollector,
        required_mps: list[tuple[CEType]],
        feats: dict[NType, torch.Tensor],
        drop_unlabeled: bool = False,
        label_mask: torch.Tensor | None = None,
        allow_low_rank_feat: bool = False,
    ):
        self.collector = collector
        self.required_mps = np.empty(len(required_mps), dtype=object)
        self.required_mps[:] = required_mps
        self.feats = feats
        if drop_unlabeled and label_mask is not None:
            self.n_samples = label_mask.sum().item()
        else:
            tgt_ntype = {mp[-1][-1] for mp in self.required_mps}
            if len(tgt_ntype) > 1:
                raise NotImplementedError
            tgt_ntype = list(tgt_ntype)[0]
            self.n_samples = len(feats[tgt_ntype])
        super().__init__(
            drop_unlabeled=drop_unlabeled, label_mask=label_mask,
            allow_low_rank_feat=allow_low_rank_feat
        )
        return

    @property
    def shape(self):
        """nxc"""
        return (self.n_samples, len(self.required_mps))

    @property
    def check_hash(self) -> str:
        sha1 = hashlib.sha1(self.__class__.__name__.encode())
        sha1.update(
            self.collector.check_hash(self.required_mps, self.feats).encode()
        )
        return sha1.hexdigest()

    def _get_feats(
        self,
        indices: Iterable[int] | None = None,
        c_indices: Iterable[int] | None = None,
    ):
        mps = self.required_mps
        if c_indices is not None:
            mps = mps[c_indices]

        feats = self.collector.get(
            mps,
            feats=self.feats,
            row_indices=indices,
        )

        # |MP| x bs x feat_dim
        return feats


class SlotDataset(BasePrecomDataset):
    """Use a collector per slot. Require more disk storage space but slightly faster"""

    def __init__(
        self,
        collector: FeatureCollector,
        required_mps: list[tuple[CEType]],
        feats: dict[NType, dict[NType, torch.Tensor]],
        drop_unlabeled: bool = False,
        label_mask: torch.Tensor | None = None,
        allow_low_rank_feat: bool = False,
    ):
        self.required_mps = np.empty(len(required_mps), dtype=object)
        self.required_mps[:] = required_mps
        self.feats_t = _dict_tranpose(feats)
        self.collectors = collector
        if isinstance(collector, FeatureCollector):
            # Backward Compatibility
            self.collectors = {ntype: collector for ntype in self.feats_t}
        if drop_unlabeled and label_mask is not None:
            self.n_samples = label_mask.sum().item()
        else:
            tgt_ntype = {mp[-1][-1] for mp in self.required_mps}
            if len(tgt_ntype) > 1:
                raise NotImplementedError
            tgt_ntype = list(tgt_ntype)[0]
            self.n_samples = len(list(feats[tgt_ntype].values())[0])
        super().__init__(
            drop_unlabeled=drop_unlabeled, label_mask=label_mask,
            allow_low_rank_feat=allow_low_rank_feat
        )
        return

    @property
    def shape(self):
        """nxc"""
        return (self.n_samples, len(self.required_mps))

    @property
    def check_hash(self) -> str:
        sha1 = hashlib.sha1(self.__class__.__name__.encode())
        for ntype, xs in self.feats_t.items():
            collector = self.collectors[ntype]
            mps = [mp for mp in self.required_mps if mp[0][0] in xs]
            sha1.update(collector.check_hash(mps, feats=xs).encode())
        return sha1.hexdigest()

    def _get_feats(
        self,
        indices: Iterable[int] | None = None,
        c_indices: Iterable[int] | None = None,
    ):
        mps = self.required_mps
        if c_indices is not None:
            mps = mps[c_indices]
        all_feats: dict[NType, dict[tuple[CEType, ...]:torch.Tensor]] = {}
        for ntype, xs in self.feats_t.items():
            collector = self.collectors[ntype]
            mps_ = [mp for mp in mps if mp[0][0] in xs]
            all_feats[ntype] = collector.get(
                mps_,
                feats=xs,
                row_indices=indices,
            )
            all_feats[ntype] = dict(zip(mps_, all_feats[ntype]))

        def concat(xs: list[torch.Tensor] | list[LowRankMatrix]):
            if any(isinstance(x, LowRankMatrix) for x in xs):
                return LowRankMatrix.concat(xs, dim=1)

            # NOTE: csr does not support concatenatation
            return torch.concat(
                [x.to_sparse_coo() if x.is_sparse_csr else x for x in xs],
                dim=1
            )

        # features: list[torch.Tensor] = [
        #     concat(xs) for xs in zip(*all_feats.values())
        # ]
        # XXX: This assume features in different slots have same format
        # If there were two formats, then could maintain two feature lists (one sparse one dense)
        # So let's keep this form for now
        features = [
            [xs[mp] for xs in all_feats.values() if mp in xs] for mp in mps
        ]
        del all_feats
        for i, xs in enumerate(features):
            # NOTE: inplace to avoid memory peak
            features[i] = concat(xs)
            del xs
        # |MP| x bs x feat_dim
        return features


class LazySlotDataset(BasePrecomDataset):

    def __init__(
        self,
        collector: FeatureCollector,
        required_mps: list[tuple[CEType]],
        feats: dict[NType, dict[NType, torch.Tensor]],
        drop_unlabeled: bool = False,
        label_mask: torch.Tensor | None = None,
        allow_low_rank_feat: bool = False,
    ):
        self.collector = collector
        self.required_mps = np.empty(len(required_mps), dtype=object)
        self.required_mps[:] = required_mps
        self.slots = list(_dict_tranpose(feats))
        self.feats = feats
        if drop_unlabeled and label_mask is not None:
            self.n_samples = label_mask.sum().item()
        else:
            tgt_ntype = {mp[-1][-1] for mp in self.required_mps}
            if len(tgt_ntype) > 1:
                raise NotImplementedError
            tgt_ntype = list(tgt_ntype)[0]
            self.n_samples = len(list(feats[tgt_ntype].values())[0])
        super().__init__(
            drop_unlabeled=drop_unlabeled, label_mask=label_mask,
            allow_low_rank_feat=allow_low_rank_feat
        )
        return

    @property
    def shape(self):
        """nxc"""
        return (self.n_samples, len(self.required_mps))

    @property
    def check_hash(self) -> str:
        sha1 = hashlib.sha1(self.__class__.__name__.encode())

        for slottype, xs in _dict_tranpose(self.feats).items():
            sha1.update(
                self.collector.check_hash(self.required_mps, feats=xs).encode()
            )
        return sha1.hexdigest()

    # from line_profiler import profile
    # @profile
    def _get_feats(
        self, indices: Iterable[int] | None,
        c_indices: Iterable[int] | None = None
    ):
        all_feats: list[torch.Tensor] = []

        collector: FeatureCollector = self.collector
        if (collector.cache_val_dtype or torch.float) != torch.float:
            # Ideally, precomputation should be done in higher precision
            # But the impact is probably minimal to change this exception to warning
            raise NotImplementedError
        required_mps = self.required_mps
        if c_indices is not None:
            required_mps = required_mps[c_indices]
        for mp, adj in zip(
            # pbar,
            required_mps,
            collector.iget(
                required_mps, None, collector.cache_dir, row_indices=indices
            )
        ):
            src_feats = self.feats[collector._get_srctype(mp)]
            # XXX: This assume features in different slots have same format
            # If there were two formats, then could maintain two feature lists (one sparse one dense)
            # So let's keep this form for now
            src_feat = torch.concat(
                [src_feats[slottype] for slottype in self.slots], dim=1
            )
            if adj.layout != torch.strided:
                adj = adj.to_sparse_coo()
                density = adj._nnz() / adj.numel()
                # print(
                #     f'Density({collector.cache_name(mp)}) = {density:.2%}',
                #     end='', file=pbar
                # )
                # if density > 0.1:  # XXX: allow custom threshold
                #     adj = adj.to_dense()
            else:
                # print(
                #     f'Density({collector.cache_name(mp)}) = dense',
                #     file=pbar,
                #     end='',
                # )
                pass
            # feats = [adj @ src_feats[slottype] for slottype in self.slots]
            # # XXX: This assume features in different slots have same format
            # # If there were two formats, then could maintain two feature lists (one sparse one dense)
            # # So let's keep this form for now
            # feat = torch.concat(feats, dim=1)
            feat = adj @ src_feat

            all_feats.append(feat)
        # |MP| x bs x feat_dim
        return all_feats

    def check(
        self,
        feats: list[torch.Tensor],
        items=None,
    ):
        from functools import reduce
        dense_adjs = {
            cetype: adj.to_dense()
            for cetype, adj in self.collector.adjs.items()
            if cetype not in self.collector._self_loops
        }

        def _collect_ground_truth(mp: tuple[CEType]):

            src_feat = torch.concatenate(
                [
                    self.feats[self.collector._get_srctype(mp)]
                    [ntype].to_dense() for ntype in self.slots
                ], dim=1
            )
            mp = self.collector._filter_self_loops(mp)
            seq = [dense_adjs[e].T for e in mp[::-1]] + [src_feat.T]
            res = reduce(torch.matmul, seq[::-1]).T
            if items is not None:
                res = res[items]
            return res

        from tqdm import tqdm
        for mp, feat in zip(tqdm(self.required_mps), feats):
            y = _collect_ground_truth(mp)
            if not torch.allclose(feat.to_dense(), y, atol=1e-4):
                raise RuntimeError()
        print('checked ok')
        return True
