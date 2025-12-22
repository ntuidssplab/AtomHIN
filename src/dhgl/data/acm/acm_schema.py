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
    author: BaseNodeSpace
    paper: BaseTargetNodeSpace
    subject: BaseNodeSpace
    term: BaseNodeSpace


NodesLiteral = Literal['author', 'paper', 'subject', 'term']
TargetNodeLiteral = Literal['paper']
EdgeDictView = TypedDict(
    'EdgeDictView', {
        'writing': BaseEdgeSpace,
        'cited': BaseEdgeSpace,
        'citing': BaseEdgeSpace,
        'contains': BaseEdgeSpace,
        'is-about': BaseEdgeSpace,
        'written-by': BaseEdgeSpace,
        'has': BaseEdgeSpace,
        'is-in': BaseEdgeSpace,
    }
)
# ('author', 'writing', 'paper'): BaseEdgeSpace,
# ('paper', 'cited', 'paper'): BaseEdgeSpace,
# ('paper', 'citing', 'paper'): BaseEdgeSpace,
# ('paper', 'contains', 'term'): BaseEdgeSpace,
# ('paper', 'is-about', 'subject'): BaseEdgeSpace,
# ('paper', 'written-by', 'author'): BaseEdgeSpace,
# ('subject', 'has', 'paper'): BaseEdgeSpace,
# ('term', 'is-in', 'paper'): BaseEdgeSpace,
EdgesLiteral = Never

ACMGraphSchema = BaseGraphSchema[NodeDictView, NodesLiteral, TargetNodeLiteral,
                                 EdgeDictView, EdgesLiteral]
