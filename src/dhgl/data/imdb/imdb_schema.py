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
    movie: BaseTargetNodeSpace
    director: BaseNodeSpace
    actor: BaseNodeSpace
    keyword: BaseNodeSpace


NodesLiteral = Literal['movie', 'director', 'actor', 'keyword']
TargetNodeLiteral = Literal['movie']

EdgeDictView = TypedDict(
    'EdgeDictView', {
        'acts': BaseEdgeSpace,
        'directed': BaseEdgeSpace,
        'is-in': BaseEdgeSpace,
        'contains': BaseEdgeSpace,
        'directed-by': BaseEdgeSpace,
        'stars': BaseEdgeSpace,
    }
)
EdgesLiteral = Never

IMDBGraphSchema = BaseGraphSchema[NodeDictView, NodesLiteral,
                                  TargetNodeLiteral, EdgeDictView,
                                  EdgesLiteral]
