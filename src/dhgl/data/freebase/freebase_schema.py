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
    book: BaseTargetNodeSpace
    business: BaseNodeSpace
    film: BaseNodeSpace
    location: BaseNodeSpace
    music: BaseNodeSpace
    organization: BaseNodeSpace
    people: BaseNodeSpace
    sports: BaseNodeSpace


NodesLiteral = Literal['book', 'business', 'film', 'location', 'music',
                       'organization', 'people', 'sports']
TargetNodeLiteral = Literal['book']

EdgeDictView = TypedDict(
    'EdgeDictView', {
        'book-about-organization': BaseEdgeSpace,
        'book-and-book': BaseEdgeSpace,
        'book-and-book-inv': BaseEdgeSpace,
        'book-on-location': BaseEdgeSpace,
        'book-on-sports': BaseEdgeSpace,
        'book-to-film': BaseEdgeSpace,
        'business-about-book-inv': BaseEdgeSpace,
        'music-in-book-inv': BaseEdgeSpace,
        'people-to-book-inv': BaseEdgeSpace,
        'business-about-book': BaseEdgeSpace,
        'business-about-film': BaseEdgeSpace,
        'business-about-music': BaseEdgeSpace,
        'business-about-sports': BaseEdgeSpace,
        'business-and-business': BaseEdgeSpace,
        'business-and-business-inv': BaseEdgeSpace,
        'business-on-location': BaseEdgeSpace,
        'organization-for-business-inv': BaseEdgeSpace,
        'people-in-business-inv': BaseEdgeSpace,
        'book-to-film-inv': BaseEdgeSpace,
        'business-about-film-inv': BaseEdgeSpace,
        'film-and-film': BaseEdgeSpace,
        'film-and-film-inv': BaseEdgeSpace,
        'location-in-film-inv': BaseEdgeSpace,
        'music-in-film-inv': BaseEdgeSpace,
        'organization-in-film-inv': BaseEdgeSpace,
        'people-to-film-inv': BaseEdgeSpace,
        'sports-in-film-inv': BaseEdgeSpace,
        'book-on-location-inv': BaseEdgeSpace,
        'business-on-location-inv': BaseEdgeSpace,
        'location-and-location': BaseEdgeSpace,
        'location-and-location-inv': BaseEdgeSpace,
        'location-in-film': BaseEdgeSpace,
        'music-on-location-inv': BaseEdgeSpace,
        'organization-on-location-inv': BaseEdgeSpace,
        'people-on-location-inv': BaseEdgeSpace,
        'sports-on-location-inv': BaseEdgeSpace,
        'business-about-music-inv': BaseEdgeSpace,
        'music-and-music': BaseEdgeSpace,
        'music-and-music-inv': BaseEdgeSpace,
        'music-for-sports': BaseEdgeSpace,
        'music-in-book': BaseEdgeSpace,
        'music-in-film': BaseEdgeSpace,
        'music-on-location': BaseEdgeSpace,
        'organization-to-music-inv': BaseEdgeSpace,
        'people-to-music-inv': BaseEdgeSpace,
        'book-about-organization-inv': BaseEdgeSpace,
        'organization-and-organization': BaseEdgeSpace,
        'organization-and-organization-inv': BaseEdgeSpace,
        'organization-for-business': BaseEdgeSpace,
        'organization-in-film': BaseEdgeSpace,
        'organization-on-location': BaseEdgeSpace,
        'organization-to-music': BaseEdgeSpace,
        'organization-to-sports': BaseEdgeSpace,
        'people-in-organization-inv': BaseEdgeSpace,
        'people-and-people': BaseEdgeSpace,
        'people-and-people-inv': BaseEdgeSpace,
        'people-in-business': BaseEdgeSpace,
        'people-in-organization': BaseEdgeSpace,
        'people-on-location': BaseEdgeSpace,
        'people-to-book': BaseEdgeSpace,
        'people-to-film': BaseEdgeSpace,
        'people-to-music': BaseEdgeSpace,
        'people-to-sports': BaseEdgeSpace,
        'book-on-sports-inv': BaseEdgeSpace,
        'business-about-sports-inv': BaseEdgeSpace,
        'music-for-sports-inv': BaseEdgeSpace,
        'organization-to-sports-inv': BaseEdgeSpace,
        'people-to-sports-inv': BaseEdgeSpace,
        'sports-and-sports': BaseEdgeSpace,
        'sports-and-sports-inv': BaseEdgeSpace,
        'sports-in-film': BaseEdgeSpace,
        'sports-on-location': BaseEdgeSpace,
    }
)
EdgesLiteral = Never

FreebaseSchema = BaseGraphSchema[NodeDictView, NodesLiteral, TargetNodeLiteral,
                                 EdgeDictView, EdgesLiteral]
