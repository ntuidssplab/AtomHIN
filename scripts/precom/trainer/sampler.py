from __future__ import annotations

import warnings
from functools import cached_property
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    RandomSampler,
    Sampler,
    SequentialSampler,
)


class SmartIndex:

    def __init__(
        self,
        items: Sequence[int] | Sequence[slice] | slice,
        perm: Sequence[int] | None = None,
    ):
        if isinstance(items, SmartIndex):
            assert perm is None
            self.items = items.items
            self.perm = items.perm
            return
        if not isinstance(items, slice) and not isinstance(items, Iterable):
            raise NotImplementedError
        if items == slice(None):
            raise NotImplementedError
        self.items = items
        self.perm = perm
        if perm is not None:
            assert len(perm) == len(self)
        return

    @classmethod
    def from_items(cls, items):
        if isinstance(items, SmartIndex):
            return items
        return SmartIndex(items)

    @classmethod
    def post_shuffle(cls, items: Sequence[slice] | slice):
        perm = np.random.permutation(len(cls(items)))
        return SmartIndex(items, perm)

    @property
    def is_chunk(self):
        return isinstance(self.items, slice)

    @property
    def is_chunks(self):
        return isinstance(self.items,
                          Iterable) and isinstance(self.items[0], slice)

    @property
    def is_indices(self):
        return isinstance(self.items,
                          Iterable) and not isinstance(self.items[0], slice)

    @cached_property
    def is_contiguous(self):
        if self.is_indices:
            return (np.diff(self.items) == 1).all()
        if self.is_chunk:
            return True
        if self.is_chunks:
            return False
        raise NotImplementedError

    @cached_property
    def r_(self):
        if self.is_indices:
            indices = self.items
        elif self.is_chunk:
            indices = np.r_[self.items]
        elif self.is_chunks:
            indices = np.r_[tuple(self.items)]
        if self.perm is not None:
            return indices[self.perm]
        return indices

    def to_slice(self):
        # If items not shuffled
        assert self.is_contiguous
        if self.is_chunk:
            return self
        return SmartIndex(slice(self.items[0], self.items[-1] + 1))

    def __len__(self):
        if self.is_chunks:
            return sum(s.stop - s.start for s in self.items)
        if self.is_chunk:
            return self.items.stop - self.items.start
        return len(self.items)

    def __iter__(self):
        yield from self.r_

    def __take_without_permutation(self, x):
        if self.is_chunks:
            backend = torch if isinstance(x, torch.Tensor) else np
            return backend.concatenate([x[s] for s in self.items])
        return x[self.items]

    def take_from(self, x):
        res = self.__take_without_permutation(x)
        if self.perm is not None:
            return res[self.perm]
        return res


class SequentialChunkedSampler(Sampler[slice]):

    def __init__(self, data, batch_size: int, drop_last) -> None:
        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last
        return

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.data) // self.batch_size
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[slice]:
        for start in range(0, len(self.data), self.batch_size):
            end = start + self.batch_size
            if self.drop_last and end > len(self.data):
                break
            end = min(end, len(self.data))
            yield slice(start, end)


class RandomChunkedBatchSampler(Sampler[list[SmartIndex]]):

    def __init__(
        self, data, batch_size: int, chunk_size: int, drop_last: bool
    ) -> None:
        self.data = data
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.chunk_size = chunk_size
        self.remainder = 0
        if batch_size % chunk_size != 0:
            warnings.warn(
                f'chunk_size ({chunk_size}) is not multiple of batch_size ({batch_size})'
            )
            self.remainder = batch_size % chunk_size
        self.random_batch_sampler = BatchSampler(
            RandomSampler(range(self._get_length(chunk_size))),
            batch_size=(self.batch_size + self.chunk_size - 1) //
            self.chunk_size,
            drop_last=drop_last,
        )

    def _get_length(self, batch_size: int) -> int:
        if self.drop_last:
            return len(self.data) // batch_size
        return (len(self.data) + batch_size - 1) // batch_size

    def __len__(self) -> int:
        return self._get_length(self.batch_size)

    def __iter__(self) -> Iterator[SmartIndex]:
        assert self.drop_last

        def chunk_to_slice(chunk_i: int):
            s = chunk_i * self.chunk_size
            return slice(s, s + self.chunk_size)

        for batch_in_chunk in self.random_batch_sampler:
            chunks = [chunk_to_slice(i) for i in batch_in_chunk]
            if self.remainder > 0:
                chunks[-1] = slice(
                    chunks[-1].start, chunks[-1].start + self.remainder
                )

            yield SmartIndex.post_shuffle(chunks)
