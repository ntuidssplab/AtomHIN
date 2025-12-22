from __future__ import annotations

from typing import Generic, Mapping, NamedTuple, Protocol, TypedDict, TypeVar

from dgl import DGLHeteroGraph

_DataKey = str
_DataType = object
_NType = str
_EType = str | tuple[str, str, str]
_SplitDictValueT = TypeVar('_SplitDictValueT')


class SplitDict(Generic[_SplitDictValueT], TypedDict):
    train: _SplitDictValueT
    val: _SplitDictValueT
    test: _SplitDictValueT


_NodeDataDictViewT = TypeVar(
    '_NodeDataDictViewT', bound=Mapping[_DataKey, _DataType]
)
_EdgeDataDictViewT = TypeVar(
    '_EdgeDataDictViewT', bound=Mapping[_DataKey, _DataType]
)


class NodeSpace(NamedTuple, Generic[_NodeDataDictViewT]):
    data: _NodeDataDictViewT


class EdgeSpace(NamedTuple, Generic[_EdgeDataDictViewT]):
    data: _EdgeDataDictViewT


class NodeSpaceProtocol(NamedTuple):
    data: dict


class HeteroNodeProtocol(Protocol):

    def __getitem__(self, key) -> NodeSpaceProtocol:
        ...

    def __call__(self, ntype=None):
        """Return the nodes."""


class EdgeSpaceProtocol(NamedTuple):
    data: dict


class HeteroEdgeProtocol(Protocol):

    def __getitem__(self, key) -> EdgeSpaceProtocol:
        ...

    def __call__(self, etype=None):
        """Return the edges."""


_NodeDictViewT = TypeVar(
    '_NodeDictViewT', bound=Mapping[_NType, NodeSpaceProtocol]
)
_EdgeDictViewT = TypeVar(
    '_EdgeDictViewT', bound=Mapping[_EType, EdgeSpaceProtocol]
)
_NdataDictViewT = TypeVar(
    '_NdataDictViewT', bound=Mapping[_DataKey, Mapping[_NType, _DataType]]
)
_EdataDictViewT = TypeVar(
    '_EdataDictViewT', bound=Mapping[_DataKey, Mapping[_NType, _DataType]]
)


class HeteroGraphSchema(
    DGLHeteroGraph, Generic[_NodeDictViewT, _NdataDictViewT, _EdgeDictViewT,
                            _EdataDictViewT]
):
    """Provide `NodeDictView` and `NodeDataDictView` to create a heterogeneous graph schema

    >>> MyHeterGraph = HeteroGraphSchema[MyNodeDictView, MyNodeDataDictView]

    """
    nodes: _NodeDictViewT | HeteroNodeProtocol
    ndata: _NdataDictViewT

    edges: _EdgeDictViewT | HeteroEdgeProtocol
    edata: _EdataDictViewT
