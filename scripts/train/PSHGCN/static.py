from __future__ import annotations

from itertools import product


class AmazonEtype:
    """An ugly workaround for the problem of PSHGCN code that only accept character as a node type"""

    def __init__(self, etypes: list[tuple[str, str, str]]):
        for etype in etypes:
            if isinstance(etype, str):
                assert len(etype) == 2
        self.etypes = etypes
        return

    def __getitem__(self, idx: int):
        seq = [
            (e[0] + e[-1] if isinstance(e, tuple) else e) for e in self.etypes
        ]
        seq = ''.join(seq)
        assert len(seq) % 2 == 0
        if isinstance(idx, slice):
            res = seq[idx]
            if len(res) == 0:
                return ''
            if len(res) % 2 != 0:
                raise NotImplementedError

            def half(i):
                if i is None:
                    return None
                assert i % 2 == 0
                return i // 2

            if idx.step not in (None, -1):
                breakpoint()
            if idx.step == -1:
                assert idx.start == idx.stop == None
                return AmazonEtype([e[::-1] for e in self.etypes[::-1]])
            if idx.step is not None:
                breakpoint()
            idx_ = slice(half(idx.start), half(idx.stop), idx.step)
            return AmazonEtype(self.etypes[idx_])

        return seq[idx]

    def __len__(self):
        return 2 * len(self.etypes)

    def __eq__(self, other):
        if isinstance(other, str):
            return repr(self) == other
        return repr(self) == repr(other)

    def __add__(self, other):
        if isinstance(other, AmazonEtype):
            return AmazonEtype(self.etypes + other.etypes)
        return AmazonEtype(
            self.etypes + [other[i:i + 2] for i in range(0, len(other), 2)]
        )

    def __radd__(self, other):
        if isinstance(other, AmazonEtype):
            return AmazonEtype(other.etypes + self.etypes)
        return AmazonEtype(
            [other[i:i + 2] for i in range(0, len(other), 2)] + self.etypes
        )

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        seq = [
            (e[0] + e[1] + e[-1] if isinstance(e, tuple) else e)
            for e in self.etypes
        ]
        return ''.join(seq)

    # def __str__(self):
    #     seq = [
    #         (e[0] + e[1] + e[-1] if isinstance(e, tuple) else e)
    #         for e in self.etypes
    #     ]
    #     return ''.join(seq)


ETYPE_MAPPERS = {
    'acm': {
        'citing': 'PP',
        'written-by': 'PA',
        'is-about': 'PC',
        'contains': 'PK',
        'writing': 'AP',
        'has': 'CP',
        'is-in': 'KP',
    },
    'dblp': {
        'writing': 'AP',
        'written-by': 'PA',
        'has': 'VP',
        'pubs-in': 'PV',
        'contains': 'PT',
        'is-in': 'TP',
    },
    'imdb': {
        'acts': 'AM',
        'directed': 'DM',
        'is-in': 'KM',
        'contains': 'MK',
        'directed-by': 'MD',
        'stars': 'MA',
    },
    'nimdb': {
        'acts': 'AM',
        'is-type-of': 'CM',
        'is-rating-for': 'RM',
        'is-country-of': 'SM',
        'directed': 'DM',
        'is-in': 'KM',
        'is-language-of': 'LM',
        'contains': 'MK',
        'contains-word': 'MW',
        'directed-by': 'MD',
        'has-color': 'MC',
        'has-numerical': 'MN',
        'has-rating': 'MR',
        'is-from-country': 'MS',
        'is-in-language': 'ML',
        'stars': 'MA',
        'is-numerical-of': 'NM',
        'is-word-of': 'WM',
    },
    'lastfm': {
        'user-artist': 'UA',
        'artist-user': 'AU',
        'user-user': 'UU',
        'artist-tag': 'AT',
        'tag-artist': 'TA',
    },
    'amazon': {
        'co-view': 'PP',
        'co-purchase': 'UU',
    },
    'freebase': {
        'book-about-organization': 'BO',
        'book-about-organization-inv': 'OB',
        'book-and-book': 'BB',
        'book-and-book-inv': 'BB',
        'book-on-location': 'BL',
        'book-on-location-inv': 'LB',
        'book-on-sports': 'BS',
        'book-on-sports-inv': 'SB',
        'book-to-film': 'BF',
        'book-to-film-inv': 'FB',
        'business-about-book': 'EB',
        'business-about-book-inv': 'BE',
        'business-about-film': 'EF',
        'business-about-film-inv': 'FE',
        'business-about-music': 'EM',
        'business-about-music-inv': 'ME',
        'business-about-sports': 'ES',
        'business-about-sports-inv': 'SE',
        'business-and-business': 'EE',
        'business-and-business-inv': 'EE',
        'business-on-location': 'EL',
        'business-on-location-inv': 'LE',
        'film-and-film': 'FF',
        'film-and-film-inv': 'FF',
        'location-and-location': 'LL',
        'location-and-location-inv': 'LL',
        'location-in-film': 'LF',
        'location-in-film-inv': 'FL',
        'music-and-music': 'MM',
        'music-and-music-inv': 'MM',
        'music-for-sports': 'MS',
        'music-for-sports-inv': 'SM',
        'music-in-book': 'MB',
        'music-in-book-inv': 'BM',
        'music-in-film': 'MF',
        'music-in-film-inv': 'FM',
        'music-on-location': 'ML',
        'music-on-location-inv': 'LM',
        'organization-and-organization': 'OO',
        'organization-and-organization-inv': 'OO',
        'organization-for-business': 'OE',
        'organization-for-business-inv': 'EO',
        'organization-in-film': 'OF',
        'organization-in-film-inv': 'FO',
        'organization-on-location': 'OL',
        'organization-on-location-inv': 'LO',
        'organization-to-music': 'OM',
        'organization-to-music-inv': 'MO',
        'organization-to-sports': 'OS',
        'organization-to-sports-inv': 'SO',
        'people-and-people': 'PP',
        'people-and-people-inv': 'PP',
        'people-in-business': 'PE',
        'people-in-business-inv': 'EP',
        'people-in-organization': 'PO',
        'people-in-organization-inv': 'OP',
        'people-on-location': 'PL',
        'people-on-location-inv': 'LP',
        'people-to-book': 'PB',
        'people-to-book-inv': 'BP',
        'people-to-film': 'PF',
        'people-to-film-inv': 'FP',
        'people-to-music': 'PM',
        'people-to-music-inv': 'MP',
        'people-to-sports': 'PS',
        'people-to-sports-inv': 'SP',
        'sports-and-sports': 'SS',
        'sports-and-sports-inv': 'SS',
        'sports-in-film': 'SF',
        'sports-in-film-inv': 'FS',
        'sports-on-location': 'SL',
        'sports-on-location-inv': 'LS'
    },
    'namazon': {
        'co-view': AmazonEtype([('I', 'v', 'I')]),
        'co-purchase': AmazonEtype([('I', 'p', 'I')]),
        'product-price': 'IP',
        'price-product': 'PI',
        'product-sales_rank': 'IS',
        'sales_rank-product': 'SI',
        'product-brand': 'IB',
        'brand-product': 'BI',
        'product-category': 'IC',
        'category-product': 'CI',
    },
    'pubmed': {
        f'{n1}-{n2}': f'{n1[0].upper()}{n2[0].upper()}'
        for n1, n2 in product(
            ['chemical', 'disease', 'gene', 'species'],
            ['chemical', 'disease', 'gene', 'species']
        )
    }
    #     'product': 2,
    #     'price': 1,
    #     'sales_rank': 1,
    #     'brand': 1,
    #     'category': 1,
    #     'num_layers': 3,
    #     'product-price|price-product': 0,
    #     'product-sales_rank|sales_rank-product': 0,
    #     'product-brand|brand-product': 0,
    #     'product-category|category-product': 0,
}
