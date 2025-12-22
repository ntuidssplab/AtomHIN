"""
There is no actual functionality of this schema code. The purpose of this schema
is to provide type hints for python language service of your IDE.

For example, in VS Code, you should install `Pylance` for python language service,
featuring **autocompletion**, **error checks**, and so on.
"""
from __future__ import annotations

from typing import Literal, Mapping, TypedDict

from ..base.base_schema import (
    BaseEdgeSpace,
    BaseGraphSchema,
    BaseNodeSpace,
    BaseTargetNodeSpace,
)


class NodeDictView(TypedDict):
    paper: BaseTargetNodeSpace
    field_of_study: BaseNodeSpace
    institution: BaseNodeSpace
    author: BaseNodeSpace


NodesLiteral = Literal['paper', 'field_of_study', 'institution', 'author']
TargetNodeLiteral = Literal['paper']

EdgeDictView = Mapping[Literal['abc'], BaseEdgeSpace]
EdgesLiteral = Literal['abc']

MAGGraphSchema = BaseGraphSchema[NodeDictView, NodesLiteral, TargetNodeLiteral,
                                 EdgeDictView, EdgesLiteral]
