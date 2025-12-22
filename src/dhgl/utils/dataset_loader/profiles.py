from __future__ import annotations

from typing import Literal

from dhgl.data.link_prediction.base import BaseLinkPredictionDataset

from .adapter import *

VanillaNCDatasetLiteral = Literal['imdb', 'freebase', 'dblp', 'acm', 'mag']
AtomicNCDatasetLiteral = Literal['atomic-imdb', 'atomic-freebase',
                                 'atomic-dblp', 'atomic-acm', 'atomic-mag']
VanillaLPDatasetLiteral = Literal['amazon', 'lastfm', 'pubmed']
AtomicLPDatasetLiteral = Literal['atomic-amazon', 'atomic-lastfm',
                                 'atomic-pubmed']
LinkPredictionDatasetT = BaseLinkPredictionDataset

PROFILES = {
    'imdb': {
        'vanilla':
        ADAPTER.validate_python(
            dict(name='imdb', raw_path='HGB', non_tgt_feat='original')
        )
    },
    'freebase': {
        'vanilla':
        ADAPTER.validate_python(
            dict(name='freebase', raw_path='HGB', feat_types='nid_coo')
        )
    },
    'dblp': {
        'vanilla':
        ADAPTER.validate_python(
            dict(name='dblp', raw_path='HGB', non_tgt_feat='original')
        )
    },
    'acm': {
        'vanilla':
        ADAPTER.validate_python(
            dict(name='acm', raw_path='HGB', non_tgt_feat='original')
        )
    },
    'mag': {
        'vanilla':
        ADAPTER.validate_python(dict(name='mag', non_tgt_feat='original'))
    },
    'lastfm': {
        'vanilla':
        ADAPTER.validate_python(
            dict(
                name='lastfm',
                raw_path='HGB',
                feat_types='nid_coo',
                use_symmetric_user_user=True,
                fix_valid=True,
            )
        )
    },
    'amazon': {
        'vanilla':
        ADAPTER.validate_python(
            dict(name='amazon', raw_path='HGB', feat_types='original')
        )
    },
    'pubmed': {
        'vanilla':
        ADAPTER.validate_python(
            dict(
                name='pubmed',
                raw_path='HGBRe',
                feat_types='original',
            )
        )
    },
    'atomic-imdb': {
        'atomic':
        AtomicIMDBConfig(
            name='atomic-imdb', raw_path='HGB', prepropagation=True
        )
    },
    'atomic-freebase': {
        'atomic':
        AtomicFreebaseConfig(
            name='freebase',
            raw_path='HGB',
            prepropagation=True,
        ),
    },
    'atomic-dblp': {
        'atomic':
        AtomicDBLPConfig(
            name='atomic-dblp',
            raw_path='HGB',
            prepropagation=True,
        )
    },
    'atomic-acm': {
        'atomic':
        AtomicACMConfig(
            name='acm',
            raw_path='HGB',
            prepropagation=True,
        )
    },
    'atomic-mag': {
        'atomic':
        AtomicMAGConfig(
            name='atomic-mag',
            prepropagation=True,
            symmetric_citing=True,
        )
    },
    'atomic-lastfm': {
        'atomic':
        LastFMConfig(
            name='lastfm',
            raw_path='HGB',
            use_symmetric_user_user=True,
            fix_valid=True,
            prepropagation=True,
        )
    },
    'atomic-amazon': {
        'atomic':
        AtomicAmazonConfig(
            name='atomic-amazon',
            raw_path='HGB',
            prepropagation=True,
        )
    },
    'atomic-pubmed': {
        'atomic':
        AtomicPubMedConfig(
            name='atomic-pubmed',
            raw_path='HGBRe',
            prepropagation=True,
        )
    },
}
PROFILES['atomic-imdb']['srgcn'] = PROFILES['atomic-imdb']['atomic'].update(
    **{
        'keyword': False,
        'director': True,
        'actor': False,
        'color': False,
        'language': True,
        'country': True,
        'content_rating': True,
        'numerical': True,
        'word': True,
        'movie': False,
        'is-type-of': False,
        'has-color': False,
        'is-country-of': False,
        'is-from-country': False,
        'is-language-of': False,
        'is-in-language': False,
    },
)
PROFILES['atomic-freebase']['srgcn'] = PROFILES['atomic-freebase'][
    'atomic'].update(
        **{
            "book": True,
            "business": True,
            "film": False,
            "location": False,
            "music": True,
            "organization": True,
            "people": False,
            "sports": True,
            'book-to-film': False,
            'book-to-film-inv': False,
            'business-on-location': False,
            'business-on-location-inv': False,
            'music-in-book': False,
            'music-in-book-inv': False,
            'music-in-film': False,
            'music-in-film-inv': False,
            'organization-and-organization': False,
            'organization-and-organization-inv': False,
            'organization-for-business': False,
            'organization-for-business-inv': False,
            'organization-in-film': False,
            'organization-in-film-inv': False,
            'organization-to-sports': False,
            'organization-to-sports-inv': False,
            'people-in-business': False,
            'people-in-business-inv': False,
            'people-in-organization': False,
            'people-in-organization-inv': False,
            'people-to-music': False,
            'people-to-music-inv': False,
            'people-to-sports': False,
            'people-to-sports-inv': False,
        },
    )
PROFILES['freebase'] |= PROFILES['atomic-freebase']
PROFILES['atomic-dblp']['srgcn'] = PROFILES['atomic-dblp']['atomic'].update(
    **{
        "authorfeat": False,
        "conference": True,
        "numerical": False,
        "paper": False,
        "paperfeat": True,
        "term": False,
        "author": False,
        "has-authorfeat": False,
        "is-authorfeat-of": False,
        "has-numerical": False,
        "is-numerical-of": False,
    },
)
PROFILES['atomic-acm']['srgcn'] = PROFILES['atomic-acm']['atomic'].update(
    prepropagation=True,
    term=True,
)
PROFILES['acm'] |= PROFILES['atomic-acm']
PROFILES['atomic-mag']['srgcn'] = PROFILES['atomic-mag']['atomic'].update(
    **{
        "paper": False,
        "institution": False,
        "field_of_study": True,
        "author": False,
        "numerical": True,
        "year": True,
        "year-of-publication": False,
        "published-in-year": False,
    },
)
PROFILES['atomic-lastfm']['srgcn'] = PROFILES['atomic-lastfm'][
    'atomic'].update(
        **{
            "user": False,
            "artist": False,
            "tag": True,
            "user-artist": False,
            "artist-user": False,
            "user-user": False,
        },
    )
PROFILES['lastfm'] |= PROFILES['atomic-lastfm']
PROFILES['atomic-amazon']['srgcn'] = PROFILES['atomic-amazon'][
    'atomic'].update(
        **{
            "product": True,
            "price": False,
            "sales_rank": False,
            "brand": False,
            "category": False,
            "product-brand": False,
            "brand-product": False,
            "product-category": False,
            "category-product": False,
        },
    )
PROFILES['atomic-pubmed']['srgcn'] = PROFILES['atomic-pubmed'][
    'atomic'].update(
        **{
            "species": True,
            "disease": True,
            "chemical": True,
            "gene": True,
            "species_feat": False,
            "disease_feat": True,
            "chemical_feat": False,
            "gene_feat": False,
            'species-species': False,
            'species-species-inv': False,
            'species-disease': False,
            'disease-species': False,
            'disease-disease': False,
            'disease-disease-inv': False,
            'chemical-species': False,
            'species-chemical': False,
            'chemical-disease': False,
            'disease-chemical': False,
            'gene-gene': False,
            'gene-gene-inv': False,
            'species-has-feat': False,
            'feat-of-species': False,
            'gene-has-feat': False,
            'feat-of-gene': False,
        },
    )
