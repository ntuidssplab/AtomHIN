"""
There is no actual functionality of this schema code. The purpose of this schema
is to provide type hints for python language service of your IDE.

For example, in VS Code, you should install `Pylance` for python language service,
featuring **autocompletion**, **error checks**, and so on.
"""
from __future__ import annotations

from typing import Generic, TypedDict, TypeVar

import torch

from ..shared.type import (
    EdgeSpace,
    HeteroGraphSchema,
    NodeSpace,
    _EdgeDictViewT,
    _NodeDictViewT,
)


class _NodeData(TypedDict):
    feat: torch.Tensor


class TargetNodeData(_NodeData):
    label: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor


class _WeightedEdgeData(TypedDict):
    weight: torch.Tensor


class _EdgeData(TypedDict):
    pass


BaseNodeSpace = NodeSpace[_NodeData]
BaseTargetNodeSpace = NodeSpace[TargetNodeData]
BaseEdgeSpace = EdgeSpace[_EdgeData]
BaseWeightedEdgeSpace = EdgeSpace[_WeightedEdgeData]

# NodeDictView = Dict[str, Union[NodeSpace[NodeData], NodeSpace[TargetNodeData]]]

_NodesLiteralT = TypeVar('_NodesLiteralT', bound=str)
_EdgesLiteralT = TypeVar('_NodesLiteralT', bound=tuple[str, str, str])
_TargetNodeLiteralT = TypeVar('_TargetNodeLiteralT', bound=str)


class NdataDictView(TypedDict, Generic[_NodesLiteralT, _TargetNodeLiteralT]):
    feat: dict[_NodesLiteralT, torch.Tensor]
    label: dict[_TargetNodeLiteralT, torch.Tensor]
    train_mask: dict[_TargetNodeLiteralT, torch.Tensor]
    val_mask: dict[_TargetNodeLiteralT, torch.Tensor]
    test_mask: dict[_TargetNodeLiteralT, torch.Tensor]


class EdataDictView(TypedDict, Generic[_EdgesLiteralT]):
    weight: dict[_EdgesLiteralT, torch.Tensor]


class BaseHeteroGraphLike(
    HeteroGraphSchema[_NodeDictViewT, NdataDictView[_NodesLiteralT,
                                                    _TargetNodeLiteralT],
                      _EdgeDictViewT, EdataDictView[_EdgesLiteralT]],
    Generic[_NodeDictViewT, _NodesLiteralT, _TargetNodeLiteralT,
            _EdgeDictViewT, _EdgesLiteralT],
):
    """Provide `NodeDictView` and `NodeDataDictView` to create a graph schema

    >>> MyHeterGraph = BaseHeteroGraphLike[MyNodeDictView, NodesLiteral, TargetNodeLiteral]

    """


BaseGraphSchema = BaseHeteroGraphLike
# BaseHeteroGraphLike = TypeVar('BaseHeteroGraphLike', bound=BaseGraphSchema)
