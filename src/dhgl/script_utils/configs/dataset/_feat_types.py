from __future__ import annotations

import abc
from typing import ClassVar, Literal, Mapping, TypeVar, Union, get_args, get_origin

from pydantic import (
    RootModel,
    SerializeAsAny,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)

from ....data.schema import BaseHeteroGraphLike
from ....transforms import remove_node_features
from ....type import FEAT, NType
from ...misc import BaseConfig


class BaseFeatType(BaseConfig):

    @abc.abstractmethod
    def apply_(
        self, hg: BaseHeteroGraphLike, ntype: str
    ) -> BaseHeteroGraphLike:
        ...


FeatTypeT = TypeVar('FeatTypeT', bound=BaseFeatType)


class FeatTypes(RootModel[FeatTypeT]):
    """Almost equivalent to dict[str, FeatType] with auto adapting.

    For example FeatTypes[SubclassFeatType | BasicFeatType]
    will cause every value try parsing as SubclassFeatType first, then BasicFeatType, etc.
    If none of types is matched, a ValidationError will be raised.

    Example Annotation:
    >>> feat_types: FeatTypes[BasicFeatType | SomeOtherFeatValueType]

    Example Usage:
    >>> hg = model.feat_types[ntype].apply_(hg, ntype)
    >>> assert hg is hg # the apply_ should be assumed inplace
    """

    root: dict[NType, SerializeAsAny[FeatTypeT]]

    @field_validator('root', mode='before')
    @classmethod
    def parse(cls, data):
        if not isinstance(data, Mapping):
            return data
        _str, annotated_type = get_args(cls.model_fields['root'].annotation)
        all_feat_types, _serialize_as_any = get_args(annotated_type)
        if get_origin(all_feat_types) in (Union, type(int | str)):
            adapters = [
                TypeAdapter(feat_type)
                for feat_type in get_args(all_feat_types)
            ]
        else:
            adapters = [TypeAdapter(all_feat_types)]
        errors = []
        for ntype, feat_type in data.items():
            for adapter in adapters:
                try:
                    data[ntype] = adapter.validate_python(feat_type)
                    break
                except ValidationError as e:
                    errors.append(e)
            else:
                raise ExceptionGroup(
                    f'"{ntype}": "{feat_type}" is not a valid feat_type', [
                        *errors,
                    ]
                )

        return data

    def __getitem__(self, item):
        return self.root[item]

    def __iter__(self):
        yield from self.root

    def __getattr__(self, attr_):
        return getattr(self.root, attr_)

    def items(self):
        yield from self.root.items()

    def apply_(self, hg: BaseHeteroGraphLike):
        for ntype, feat_type in self.items():
            hg = feat_type.apply_(hg, ntype)
        return hg


class BasicFeatType(BaseFeatType):
    """
    Format: {mode}_{fmt}
    E.g. nid_coo
    E.g. ntype_id
    """
    mode: Literal['original', 'none', 'ntype', 'nid', 'zero']
    fmt: Literal['id', 'coo'] | None = None

    def apply_(self, hg: BaseHeteroGraphLike, ntype: str):
        if self.mode == 'original':
            return hg
        if self.mode == 'none':
            if 'feat' in hg.nodes[ntype].data:
                hg.nodes[ntype].data.pop('feat')
            return hg
        return remove_node_features(
            hg, self.mode, ntype, fmt=self.fmt, in_place=True
        )

    @model_serializer()
    @property
    def to_str(self):
        if self.fmt is None:
            return self.mode
        return f'{self.mode}_{self.fmt}'

    @model_validator(mode='before')
    @classmethod
    def parse(cls, data):
        if isinstance(data, str):
            if data.endswith('_id'):
                return {'mode': data.removesuffix('_id'), 'fmt': 'id'}
            if data.endswith('_emb'):
                return {'mode': data.removesuffix('_emb'), 'fmt': 'id'}
            if data.endswith('_coo'):
                return {'mode': data.removesuffix('_coo'), 'fmt': 'coo'}
            return {'mode': data}
        return data


class RandomFeatType(BaseFeatType):
    """
    Format:
    u(dim, a, b, seed)
    u(dim, a, b)
    """
    _args: ClassVar[tuple[str]] = ('dim', 'a', 'b', 'seed')
    mode: Literal['randn'] = 'randn'
    seed: int | None = None
    dim: int
    a: float
    b: float

    def apply_(self, hg: BaseHeteroGraphLike, ntype):
        import torch
        gen = None
        if self.seed is not None:
            gen = torch.Generator().manual_seed(self.seed)
        u = torch.rand((hg.num_nodes(ntype), self.dim), generator=gen)
        assert self.b > self.a
        hg.nodes[ntype].data[FEAT] = u * (self.b - self.a) + self.a
        return hg

    @field_validator('seed', mode='before')
    @classmethod
    def parse_hex(cls, v):
        if not isinstance(v, str):
            return v
        if v.startswith('0x'):
            return int(v, base=16)
        return v

    @model_validator(mode='before')
    @classmethod
    def parse(cls, data):
        if not isinstance(data, str):
            return data
        if data.lower().startswith('u'):
            import ast
            args = ast.literal_eval(data.lower().removeprefix('u'))
            return dict(zip(cls._args, args))

        return data

    @model_serializer
    @property
    def to_str(self):
        args = (self.dim, self.a, self.b)
        if self.seed is not None:
            args += (self.seed, )
        return f'u{args}'
