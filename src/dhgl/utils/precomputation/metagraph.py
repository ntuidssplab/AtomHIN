from __future__ import annotations

import warnings
from functools import cached_property
from itertools import product
from typing import Callable, Mapping, NamedTuple, Self, overload

from dgl import DGLHeteroGraph

from ...type import CEType, EType, NType
from .base import _generate_aliases


class SelfLoop(NamedTuple):
    srctype: str
    etype: str
    dsttype: str

    @classmethod
    def from_ntype(cls, ntype: str):
        return cls(ntype, f'{ntype}-self', ntype)

    def __str__(self):
        return str(tuple(self))

    def __repr__(self):
        return f'<self-loop: {self.srctype}>'


def _get_self_loops_etypes(hg: DGLHeteroGraph):
    """Retrieve self-loop etypes from given hg"""
    self_loops = []
    for s, e, d in hg.canonical_etypes:
        if s != d:
            continue
        if hg.num_nodes(s) != hg.num_edges(etype=e):
            continue
        src, dst = hg.edges(etype=e)
        if (src == dst).all():
            self_loops.append((s, e, d))
    return self_loops


class MetaGraph:

    ntypes: list[NType]

    def __init__(
        self,
        ntypes: list[NType],
        canonical_etypes: list[CEType],
        # self_loop_etypes: list[CEType] | None = None,
    ):
        self.ntypes = ntypes.copy()
        self._canonical_etypes = canonical_etypes.copy()
        self._self_loops = list(map(SelfLoop.from_ntype, ntypes))
        self._self_looped = False
        return

    @classmethod
    def from_hg(cls, hg: DGLHeteroGraph) -> MetaGraph:
        """hg should not contain self-loops."""
        self_loop_etypes = _get_self_loops_etypes(hg)
        if self_loop_etypes:
            warnings.warn(
                'self-loop detected. It\'s recommended to add self loops using'
                ' MetaGraph.add_self_loops instead of adding real self loops to graphs.'
            )
            return cls(
                hg.ntypes,
                [e for e in hg.canonical_etypes if e not in self_loop_etypes],
            )
        return cls(hg.ntypes, hg.canonical_etypes)

    @property
    def canonical_etypes(self) -> list[CEType]:
        if not self._self_looped:
            return self._canonical_etypes
        return self._canonical_etypes + self._self_loops

    @cached_property
    def _to_canonical_etype(self) -> dict:
        return {c[1]: c for c in (self._canonical_etypes + self._self_loops)}

    @property
    def self_loop_etypes(self) -> list[CEType]:
        return self._self_loops

    def to_canonical_etype(self, etype: EType) -> CEType:
        if isinstance(etype, str):
            return self._to_canonical_etype[etype]
        return etype

    def is_self_loop(self, etype: EType):
        if isinstance(etype, SelfLoop):
            assert self.to_canonical_etype(etype) in self._self_loops
        return self.to_canonical_etype(etype) in self._self_loops

    def add_self_loops(self) -> Self:
        self._self_looped = True
        return self

    # @property
    # def self_loop_etypes(self) -> list[CEType]:
    #     return self._self_loops

    def get_ntype_id(self, ntype: NType):
        return self.ntypes.index(ntype)

    def remove_etypes(self, etypes: list[EType]) -> MetaGraph:
        canonical_etypes = self.canonical_etypes.copy()
        for etype in etypes:
            canonical_etypes.remove(self.to_canonical_etype(etype))
        return MetaGraph(self.ntypes, canonical_etypes)

    def metapaths(self, num_hops: int, dsttype: NType | None = None):
        """Get all metapaths within given num_hops

        Args:
            num_hops (int): num hops
            dsttype: (str): If specified, only return those metapaths ending with the given ntype.
        """

        mp_list = get_metapaths(
            self.ntypes,
            self._canonical_etypes,
            self._self_loops,
            num_hops,
        )
        if dsttype is not None:

            def end_with(mp: tuple[CEType, ...]):
                return mp[-1][-1] == dsttype

            mp_list = list(filter(end_with, mp_list))

        return mp_list


def get_metapaths(
    ntypes: list[NType],
    cetypes: list[CEType],
    self_loop_etypes: list[SelfLoop],
    num_layers: int,
):

    class MPDomain:

        def __init__(self, self_loops: list[int]):
            self.mp_domain = {
                (sl, ): mp_id
                for mp_id, sl in enumerate(self_loops)
            }
            self.ptr = len(self_loops)
            self.self_loops = set(self_loops)
            return

        def _is_valid(self, mp: tuple[int]):
            cur = cetypes[mp[0]][-1]
            for eid in mp[1:]:
                s, _, d = cetypes[eid]
                if s != cur:
                    return False
                cur = d
            return True

        def _rm_self_loops(self, mp: tuple[int]):
            return tuple(
                [
                    mp[0],
                    *[eid for eid in mp[1:] if eid not in self.self_loops]
                ]
            )

        def update(self, mp: tuple[int]):
            if not self._is_valid(mp):
                return -1
            mp = self._rm_self_loops(mp)
            mp_id = self.mp_domain.get(mp, None)
            if mp_id is not None:
                return mp_id
            self.mp_domain[mp] = self.ptr
            self.ptr += 1
            return self.ptr - 1

        def get(self, item: tuple[int]):
            return self.mp_domain[self._rm_self_loops(item)]

        @property
        def mp_list(self) -> list[tuple[int]]:
            mps = [None] * len(self)
            for mp, mpid in self.mp_domain.items():
                mps[mpid] = mp
            return mps

        def __len__(self):
            return self.ptr

    assert len(self_loop_etypes) == len(ntypes)
    cetypes = cetypes.copy()
    for self_loop_etype in self_loop_etypes:
        assert isinstance(self_loop_etype, tuple)
        if self_loop_etype not in cetypes:
            cetypes.append(self_loop_etype)
    self_loops = [
        cetypes.index(self_loop_etype) for self_loop_etype in self_loop_etypes
    ]
    mp_domain = MPDomain(self_loops)

    for l in range(num_layers + 1):
        if l == 0:
            continue

        for left_mp, r in product(mp_domain.mp_domain, range(len(cetypes))):
            mp = (*left_mp, r)

            r = mp[-1]
            mp_id = mp_domain.update(mp)
            if mp_id < 0:  # invalid metapath
                continue

    mp_list = mp_domain.mp_list
    mp_list = [tuple(cetypes[e] for e in mp) for mp in mp_list]
    return mp_list


class MPAdaptor:
    """An adaptor to switching metapath format"""

    @overload
    def __init__(
        self,
        ntypes: list[NType],
        canoncial_etypes: list[CEType],
        *,
        to_ntype_alias: Mapping | Callable | None = None,
    ):
        ...

    @overload
    def __init__(
        self,
        mg: MetaGraph,
        *,
        to_ntype_alias: Mapping | Callable | None = None,
    ):
        ...

    def __init__(
        self,
        ntypes: list[NType] | MetaGraph,
        canoncial_etypes: list[CEType] | None = None,
        *,
        to_ntype_alias: Mapping | Callable | None = None,
    ):
        self.to_ntype_alias = to_ntype_alias
        if to_ntype_alias is None:
            self.to_ntype_alias = _generate_aliases(ntypes)
        elif isinstance(to_ntype_alias, Callable):
            self.to_ntype_alias = {
                ntype: to_ntype_alias(ntype)
                for ntype in ntypes
            }
        self._inv_mapping = {
            self.to_ntype_alias[ntype]: ntype
            for ntype in ntypes
        }
        assert len(self._inv_mapping) == len(ntypes)
        if isinstance(ntypes, MetaGraph):  # overload 1
            self.mg = ntypes
        else:  # overload 2
            self.mg = MetaGraph(ntypes, canoncial_etypes)
        return

    @classmethod
    def from_metagraph(
        cls, mg: MetaGraph, to_ntype_alias: Mapping | Callable | None = None
    ):
        return cls(
            mg.ntypes, mg._canonical_etypes, to_ntype_alias=to_ntype_alias
        )

    @classmethod
    def from_hg(
        cls, hg: DGLHeteroGraph,
        to_ntype_alias: Mapping | Callable | None = None
    ):
        return MPAdaptor.from_metagraph(
            MetaGraph.from_hg(hg), to_ntype_alias=to_ntype_alias
        )

    @cached_property
    def _ntype_to_self_loop_etype(self):
        mapping = {}
        for cetype in self.mg._self_loops:
            s, _, d = cetype
            assert s == d
            mapping[s] = cetype
        assert len(mapping) == len(self.mg._self_loops)
        return mapping

    @cached_property
    def _short_to_canonical(self):
        ntype_mapping = self.to_ntype_alias
        to_canonical = {}
        for s, e, d in self.mg.canonical_etypes:
            to_canonical[(ntype_mapping[s], ntype_mapping[d])] = (s, e, d)
        if len(to_canonical) != len(self.mg.canonical_etypes):
            raise NotImplementedError

        assert len(to_canonical) == len(self.mg.canonical_etypes)
        return to_canonical

    def remove_self_loops(self, mp: tuple[CEType, ...]):
        if len(mp) == 1 and self.mg.is_self_loop(mp[0]):
            return mp
        return tuple(e for e in mp if not self.mg.is_self_loop(e))

    def to_canonical_metapath(self, mp: tuple[EType, ...]):
        """
        transform the given metapath into canonical form.
        A canonical form of metapath satisfy:
        1. All etypes are canonical
        2. Starting with a self-loop etype
        3. No self-loop in the middle of metapaths
        """
        mp = list(self.mg.to_canonical_etype, mp)
        mp = self.remove_self_loops(mp)
        srctype = mp[0][0]
        return (SelfLoop.from_ntype(srctype), *mp)

    def short_to_canonical(self, src_mp: str) -> tuple[CEType, ...]:
        assert isinstance(src_mp, str)
        mp = src_mp[::-1]
        ntype = self._inv_mapping[mp[0]]
        self_loop_prefix = (self._ntype_to_self_loop_etype[ntype], )
        if len(mp) == 1:  #self-loop
            return self_loop_prefix
        return self_loop_prefix + tuple(
            self._short_to_canonical[n1, n2] for n1, n2 in zip(mp, mp[1:])
        )

    def canonical_to_short(self, src_mp: tuple[CEType, ...]) -> str:
        if len(src_mp) == 1 and self.mg.is_self_loop(src_mp[0]):
            return self.to_ntype_alias[src_mp[0][0]]
        src_mp = self.remove_self_loops(src_mp)
        ntypes = [*[s for s, _, __ in src_mp], src_mp[-1][-1]]
        return ''.join(self.to_ntype_alias[n] for n in ntypes[::-1])
