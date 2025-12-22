import torch
from dgl import function as fn
from ..data.schema import DBLPGraphSchema, BaseHeteroGraphLike, ACMGraphSchema, IMDBGraphSchema
from .. import transforms

Ntype = str
EType = list[str] | list[tuple[str, str, str]]


def feat_propagate(
    hg: BaseHeteroGraphLike,
    meta_path: list[EType],
    reduce_fn=fn.sum,
):
    """Propagate node features along the given metapath

    Args:
        hg (BaseHeteroGraphLike)
        meta_path (list[EType]): the metapath
        reduce_fn (_type_, optional): reduce function to reduce features from multiple nodes.
            Defaults to fn.sum.
    
    Example:
    >>> terms_for_authors = dhgl.utils.feat_propagate(dblp_hg, [('term', 'is-in', 'paper'), ('paper', 'written-by', 'author')])
    """

    def get_src_dst_ntype_of_metapath(meta_path: list[EType]):
        assert meta_path, 'got metapath with zero length'
        mid_ntype = None
        for etype in meta_path:
            src_ntype, _, dst_ntype = hg.to_canonical_etype(etype)
            assert mid_ntype is None or mid_ntype == src_ntype, (
                f'The provided metapath not connected: {meta_path = }'
            )
            mid_ntype = dst_ntype
        return hg.to_canonical_etype(meta_path[0])[0], mid_ntype

    _src_ntype, dst_ntype = get_src_dst_ntype_of_metapath(meta_path)

    with hg.local_scope():
        feat_name = 'feat'
        for etype in meta_path:
            hg.apply_edges(fn.copy_u(feat_name, 'm'), etype=etype)
            hg.update_all(
                fn.copy_e('m', 'm'), reduce_fn('m', 'h'), etype=etype
            )
            feat_name = 'h'
        return hg.nodes[dst_ntype].data['h']


def _get_populars_ids(
    agged_ids: torch.Tensor,
    target_dim: int = None,
    threshold: int = None,
    verbose: str = None,
):
    assert bool(target_dim is None) ^ bool(threshold is None)

    id_used_counts = agged_ids.bool().sum(dim=0)
    if threshold is None:
        for t in range(1, agged_ids.size(0)):
            indices = id_used_counts > t
            dim = indices.sum().item()
            if target_dim == -1 or dim <= target_dim:
                if verbose is not None:
                    print(
                        f'Got {target_dim = } for {verbose}, resulting {dim = } '
                        f'with threshold = {t}'
                    )
                return indices.nonzero().flatten()
        raise ValueError(
            f'Cannot find threshold statisfying target_dim={target_dim}'
        )

    res = (id_used_counts > threshold).nonzero().flatten()
    if verbose:
        print(f'Got {threshold = } for {verbose}, resulting dim = {len(res)}.')
    return res


def dblp_to_tabular(
    dblp_hg: DBLPGraphSchema,
    term_tgt_dim: int = 1000,
):
    """Transform DBLP graph to tabular form

    Args:
        dblp_hg (DBLPGraphSchema)
        term_tgt_dim (int, optional): Target dimension of the features for terms. The
            result feature dimension for term wil be less than or equal to the term_tgt_dim.
            Smaller term_tgt_dim would cause more term nodes(columns) to be dropped, and
            the terms owned by fewer authors would have higher priority to be dropped.

    Returns:
        Tensor: 2d tensor with shape (#tgt_nodes, dim_conference + dim_term). 
            Where dim_conference = 20, and dim_term <= term_tgt_dim.
    """
    dblp_hg = transforms.to_dense(dblp_hg)

    TPA = [('term', 'is-in', 'paper'), ('paper', 'written-by', 'author')]
    CPA = [('conference', 'has', 'paper'), ('paper', 'written-by', 'author')]

    terms = feat_propagate(dblp_hg, TPA)

    def get_popular_terms(target_dim: int):
        terms_used_counts = terms.bool().long().sum(dim=0)
        for threshold in range(dblp_hg.num_nodes('author')):
            indices = terms_used_counts > threshold
            dim = indices.sum()
            if dim <= target_dim:
                return indices
        raise ValueError(
            f'Cannot find threshold statisfying target_dim={target_dim}'
        )

    terms = terms[:, get_popular_terms(term_tgt_dim)]
    conferences = feat_propagate(dblp_hg, CPA)

    return torch.concat([terms, conferences], dim=1)


def acm_to_tabular(
    acm_hg: ACMGraphSchema,
    author_threshold: int = 5,
    paper_threshold: int = 5,
    term_tgt_dim: int = 1000,
    verbose: bool = False,
):

    acm_hg = transforms.remove_non_target_feature(acm_hg, 'id')
    acm_hg = transforms.to_dense(acm_hg)

    AP = [('author', 'writing', 'paper')]
    SP = [('subject', 'has', 'paper')]
    TP = [('term', 'is-in', 'paper')]
    PP = [('paper', 'citing', 'paper')]

    authors = feat_propagate(acm_hg, AP)
    papers = feat_propagate(acm_hg, PP)
    terms = feat_propagate(acm_hg, TP)
    subjects = feat_propagate(acm_hg, SP)

    get_verbose = (lambda x: x) if verbose else (lambda x: None)
    return torch.concat(
        [
            authors[:,
                    _get_populars_ids(
                        authors, threshold=author_threshold,
                        verbose=get_verbose('author')
                    )],
            papers[:,
                   _get_populars_ids(
                       papers, threshold=paper_threshold,
                       verbose=get_verbose('paper')
                   )],
            terms[:,
                  _get_populars_ids(
                      terms, target_dim=term_tgt_dim,
                      verbose=get_verbose('term')
                  )],
            subjects[:,
                     _get_populars_ids(
                         subjects, threshold=1.5,
                         verbose=get_verbose('subject')
                     )],
        ], dim=1
    )


def imdb_to_tabular(
    imdb_hg: IMDBGraphSchema,
    actor_threshold: int = 3,
    director_threshold: int = 1,
    keyword_threshold: int = 3,
    verbose: bool = False,
):
    imdb_hg = transforms.remove_non_target_feature(imdb_hg, 'id')
    imdb_hg = transforms.to_dense(imdb_hg)

    # rest cols (> 16) are aggregated ids
    moive_feat = imdb_hg.nodes['movie'].data['feat'][:, :16]

    AM = [('actor', 'acts', 'movie')]
    DM = [('director', 'directed', 'movie')]
    KM = [('keyword', 'is-in', 'movie')]

    actors = feat_propagate(imdb_hg, AM)
    directors = feat_propagate(imdb_hg, DM)
    keywords = feat_propagate(imdb_hg, KM)

    get_verbose = (lambda x: x) if verbose else (lambda x: None)
    actors = actors[:,
                    _get_populars_ids(
                        actors, threshold=actor_threshold,
                        verbose=get_verbose('actor')
                    )]
    directors = directors[:,
                          _get_populars_ids(
                              directors, threshold=director_threshold,
                              verbose=get_verbose('director')
                          )]
    keywords = keywords[:,
                        _get_populars_ids(
                            keywords, threshold=keyword_threshold,
                            verbose=get_verbose('keyword')
                        )]

    return torch.concat([moive_feat, actors, directors, keywords], dim=1)
