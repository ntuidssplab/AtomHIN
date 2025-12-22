"""
There is no actual functionality of this schema code. The purpose of this schema
is to provide type hints for python language service of your IDE.

For example, in VS Code, you should install `Pylance` for python language service,
featuring **autocompletion**, **error checks**, and so on.
"""
from __future__ import annotations

from typing import Literal, Never, TypedDict

from ..base.base_schema import (
    BaseEdgeSpace,
    BaseGraphSchema,
    BaseNodeSpace,
    BaseTargetNodeSpace,
)


class NodeDictView(TypedDict):
    author: BaseTargetNodeSpace
    paper: BaseNodeSpace
    conference: BaseNodeSpace
    term: BaseNodeSpace


NodesLiteral = Literal['author', 'paper', 'conference', 'term']
TargetNodeLiteral = Literal['author']

EdgeDictView = TypedDict(
    'EdgeDictView', {
        'writing': BaseEdgeSpace,
        'has': BaseEdgeSpace,
        'contains': BaseEdgeSpace,
        'pubs-in': BaseEdgeSpace,
        'written-by': BaseEdgeSpace,
        'is-in': BaseEdgeSpace,
    }
)
EdgesLiteral = Never

DBLPGraphSchema = BaseGraphSchema[NodeDictView, NodesLiteral,
                                  TargetNodeLiteral, EdgeDictView,
                                  EdgesLiteral]
