"""
There is no actual functionality of this schema code. The purpose of this schema
is to provide type hints for python language service of your IDE.

For example, in VS Code, you should install `Pylance` for python language service,
featuring **autocompletion**, **error checks**, and so on.
"""
from __future__ import annotations

from typing import Literal, TypedDict

from ..base.base_schema import (
    BaseEdgeSpace,
    BaseGraphSchema,
    BaseNodeSpace,
    BaseTargetNodeSpace,
    BaseWeightedEdgeSpace,
)


class NodeDictView(TypedDict):
    actor: BaseNodeSpace
    color: BaseNodeSpace
    content_rating: BaseNodeSpace
    country: BaseNodeSpace
    director: BaseNodeSpace
    keyword: BaseNodeSpace
    language: BaseNodeSpace
    movie: BaseTargetNodeSpace
    numerical: BaseNodeSpace
    word: BaseNodeSpace


NodesLiteral = Literal['actor', 'color', 'content_rating', 'country',
                       'director', 'keyword', 'language', 'movie', 'numerical',
                       'word']
TargetNodeLiteral = Literal['movie']

EdgeDictView = TypedDict(
    'EdgeDictView', {
        'acts': BaseEdgeSpace,
        'is-type-of': BaseEdgeSpace,
        'is-rating-for': BaseEdgeSpace,
        'is-country-of': BaseEdgeSpace,
        'directed': BaseEdgeSpace,
        'is-in': BaseEdgeSpace,
        'is-language-of': BaseEdgeSpace,
        'contains': BaseEdgeSpace,
        'contains-word': BaseWeightedEdgeSpace,
        'directed-by': BaseEdgeSpace,
        'has-color': BaseEdgeSpace,
        'has-numerical': BaseWeightedEdgeSpace,
        'has-rating': BaseEdgeSpace,
        'is-from-country': BaseEdgeSpace,
        'is-in-language': BaseEdgeSpace,
        'stars': BaseEdgeSpace,
        'is-numerical-of': BaseWeightedEdgeSpace,
        'is-word-of': BaseWeightedEdgeSpace,
    }
)
EdgesLiteral = Literal['contains-word', 'has-numerical', 'is-numerical-of',
                       'is-word-of']

NIMDBGraphSchema = BaseGraphSchema[NodeDictView, NodesLiteral,
                                   TargetNodeLiteral, EdgeDictView,
                                   EdgesLiteral]
