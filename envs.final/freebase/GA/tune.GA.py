from __future__ import annotations

import os

import naive_flow as nf
from optuna.samplers import NSGAIISampler
from ray import tune

from dhgl.script_utils.tuner import BaseTuneConfig, ParametersInvalidError

BASE_ENV_FILE = os.path.abspath(
    os.path.join(__file__, '..', '_freebase.sRGCN.env')
)
assert os.path.isfile(BASE_ENV_FILE)
CACHE_DIR = os.path.abspath(
    os.path.join(__file__, '..', '_freebase.sRGCN.tune-cache')
)
TARGET_NTYPE = 'book'
NT_NTYPES = [
    'business', 'film', 'location', 'music', 'organization', 'people', 'sports'
]
NTYPES = [TARGET_NTYPE, *NT_NTYPES]
NTYPE_FEAT = ['none', 'nid_coo']

# yapf: disable
UDCETYPES = [
    [('book', 'book-about-organization', 'organization'), ('organization', 'book-about-organization-inv', 'book')],
    [('book', 'book-and-book', 'book'), ('book', 'book-and-book-inv', 'book')],
    [('book', 'book-on-location', 'location'), ('location', 'book-on-location-inv', 'book')],
    [('book', 'book-on-sports', 'sports'), ('sports', 'book-on-sports-inv', 'book')],
    [('book', 'book-to-film', 'film'), ('film', 'book-to-film-inv', 'book')],
    [('business', 'business-about-book', 'book'), ('book', 'business-about-book-inv', 'business')],
    [('business', 'business-about-film', 'film'), ('film', 'business-about-film-inv', 'business')],
    [('business', 'business-about-music', 'music'), ('music', 'business-about-music-inv', 'business')],
    [('business', 'business-about-sports', 'sports'), ('sports', 'business-about-sports-inv', 'business')],
    [('business', 'business-and-business', 'business'), ('business', 'business-and-business-inv', 'business')],
    [('business', 'business-on-location', 'location'), ('location', 'business-on-location-inv', 'business')],
    [('film', 'film-and-film', 'film'), ('film', 'film-and-film-inv', 'film')],
    [('location', 'location-and-location', 'location'), ('location', 'location-and-location-inv', 'location')],
    [('location', 'location-in-film', 'film'), ('film', 'location-in-film-inv', 'location')],
    [('music', 'music-and-music', 'music'), ('music', 'music-and-music-inv', 'music')],
    [('music', 'music-for-sports', 'sports'), ('sports', 'music-for-sports-inv', 'music')],
    [('music', 'music-in-book', 'book'), ('book', 'music-in-book-inv', 'music')],
    [('music', 'music-in-film', 'film'), ('film', 'music-in-film-inv', 'music')],
    [('music', 'music-on-location', 'location'), ('location', 'music-on-location-inv', 'music')],
    [('organization', 'organization-and-organization', 'organization'), ('organization', 'organization-and-organization-inv', 'organization')],
    [('organization', 'organization-for-business', 'business'), ('business', 'organization-for-business-inv', 'organization')],
    [('organization', 'organization-in-film', 'film'), ('film', 'organization-in-film-inv', 'organization')],
    [('organization', 'organization-on-location', 'location'), ('location', 'organization-on-location-inv', 'organization')],
    [('organization', 'organization-to-music', 'music'), ('music', 'organization-to-music-inv', 'organization')],
    [('organization', 'organization-to-sports', 'sports'), ('sports', 'organization-to-sports-inv', 'organization')],
    [('people', 'people-and-people', 'people'), ('people', 'people-and-people-inv', 'people')],
    [('people', 'people-in-business', 'business'), ('business', 'people-in-business-inv', 'people')],
    [('people', 'people-in-organization', 'organization'), ('organization', 'people-in-organization-inv', 'people')],
    [('people', 'people-on-location', 'location'), ('location', 'people-on-location-inv', 'people')],
    [('people', 'people-to-book', 'book'), ('book', 'people-to-book-inv', 'people')],
    [('people', 'people-to-film', 'film'), ('film', 'people-to-film-inv', 'people')],
    [('people', 'people-to-music', 'music'), ('music', 'people-to-music-inv', 'people')],
    [('people', 'people-to-sports', 'sports'), ('sports', 'people-to-sports-inv', 'people')],
    [('sports', 'sports-and-sports', 'sports'), ('sports', 'sports-and-sports-inv', 'sports')],
    [('sports', 'sports-in-film', 'film'), ('film', 'sports-in-film-inv', 'sports')],
    [('sports', 'sports-on-location', 'location'), ('location', 'sports-on-location-inv', 'sports')],
]
# yapf: enable


def cetype_to_name(cetypes: list):
    return '|'.join(cetype[1] for cetype in cetypes)


UDCETYPES_NAME = list(map(cetype_to_name, UDCETYPES))
space = {}
space = {t: tune.choice([0, 1]) for t in [*NTYPES, *UDCETYPES_NAME]}
space['num_layers'] = tune.randint(1, 12)
space['book'] = 1
# space['num_heads'] = 8

POINTS_TO_EVALUATE = [
    { # Vanilla l6
        # 'book': 1,
        'business': 1,
        'film': 1,
        'location': 1,
        'music': 1,
        'organization': 1,
        'people': 1,
        'sports': 1,
        'num_layers': 6,
        'book-about-organization|book-about-organization-inv': 0,
        'book-and-book|book-and-book-inv': 0,
        'book-on-location|book-on-location-inv': 0,
        'book-on-sports|book-on-sports-inv': 0,
        'book-to-film|book-to-film-inv': 0,
        'business-about-book|business-about-book-inv': 0,
        'business-about-film|business-about-film-inv': 0,
        'business-about-music|business-about-music-inv': 0,
        'business-about-sports|business-about-sports-inv': 0,
        'business-and-business|business-and-business-inv': 0,
        'business-on-location|business-on-location-inv': 0,
        'film-and-film|film-and-film-inv': 0,
        'location-and-location|location-and-location-inv': 0,
        'location-in-film|location-in-film-inv': 0,
        'music-and-music|music-and-music-inv': 0,
        'music-for-sports|music-for-sports-inv': 0,
        'music-in-book|music-in-book-inv': 0,
        'music-in-film|music-in-film-inv': 0,
        'music-on-location|music-on-location-inv': 0,
        'organization-and-organization|organization-and-organization-inv': 0,
        'organization-for-business|organization-for-business-inv': 0,
        'organization-in-film|organization-in-film-inv': 0,
        'organization-on-location|organization-on-location-inv': 0,
        'organization-to-music|organization-to-music-inv': 0,
        'organization-to-sports|organization-to-sports-inv': 0,
        'people-and-people|people-and-people-inv': 0,
        'people-in-business|people-in-business-inv': 0,
        'people-in-organization|people-in-organization-inv': 0,
        'people-on-location|people-on-location-inv': 0,
        'people-to-book|people-to-book-inv': 0,
        'people-to-film|people-to-film-inv': 0,
        'people-to-music|people-to-music-inv': 0,
        'people-to-sports|people-to-sports-inv': 0,
        'sports-and-sports|sports-and-sports-inv': 0,
        'sports-in-film|sports-in-film-inv': 0,
        'sports-on-location|sports-on-location-inv': 0
    },
    { # Vanilla l7
        # 'book': 1,
        'business': 1,
        'film': 1,
        'location': 1,
        'music': 1,
        'organization': 1,
        'people': 1,
        'sports': 1,
        'num_layers': 7,
        'book-about-organization|book-about-organization-inv': 0,
        'book-and-book|book-and-book-inv': 0,
        'book-on-location|book-on-location-inv': 0,
        'book-on-sports|book-on-sports-inv': 0,
        'book-to-film|book-to-film-inv': 0,
        'business-about-book|business-about-book-inv': 0,
        'business-about-film|business-about-film-inv': 0,
        'business-about-music|business-about-music-inv': 0,
        'business-about-sports|business-about-sports-inv': 0,
        'business-and-business|business-and-business-inv': 0,
        'business-on-location|business-on-location-inv': 0,
        'film-and-film|film-and-film-inv': 0,
        'location-and-location|location-and-location-inv': 0,
        'location-in-film|location-in-film-inv': 0,
        'music-and-music|music-and-music-inv': 0,
        'music-for-sports|music-for-sports-inv': 0,
        'music-in-book|music-in-book-inv': 0,
        'music-in-film|music-in-film-inv': 0,
        'music-on-location|music-on-location-inv': 0,
        'organization-and-organization|organization-and-organization-inv': 0,
        'organization-for-business|organization-for-business-inv': 0,
        'organization-in-film|organization-in-film-inv': 0,
        'organization-on-location|organization-on-location-inv': 0,
        'organization-to-music|organization-to-music-inv': 0,
        'organization-to-sports|organization-to-sports-inv': 0,
        'people-and-people|people-and-people-inv': 0,
        'people-in-business|people-in-business-inv': 0,
        'people-in-organization|people-in-organization-inv': 0,
        'people-on-location|people-on-location-inv': 0,
        'people-to-book|people-to-book-inv': 0,
        'people-to-film|people-to-film-inv': 0,
        'people-to-music|people-to-music-inv': 0,
        'people-to-sports|people-to-sports-inv': 0,
        'sports-and-sports|sports-and-sports-inv': 0,
        'sports-in-film|sports-in-film-inv': 0,
        'sports-on-location|sports-on-location-inv': 0
    },
]
# assert all(set(point) == set(space) for point in POINTS_TO_EVALUATE)


def param_to_config(param):

    if 'book' not in param:
        # NOTE: this is probably a bug of raytune that while resuming with searcher-log-dir
        # the constant params would not pass
        param['book'] = 1
    feat_types = {ntype: NTYPE_FEAT[param[ntype]] for ntype in NTYPES}
    # feat_types[TARGET_NTYPE] = NTYPE_FEAT[0]

    exclude_edge_types = [
        cetype for cetype in UDCETYPES if param[cetype_to_name(cetype)]
    ]
    exclude_edge_types = [cetype[1] for cetype in sum(exclude_edge_types, [])]

    env_data = nf.load_env_file(BASE_ENV_FILE)
    env_data['dataset_config'].update(
        {
            'feat_types': feat_types,
            'exclude_edge_types': exclude_edge_types,
        }
    )
    env_data['hgnn_config'].update(
        num_layers=param['num_layers'],
        # num_heads=param['num_heads'],
    )
    return env_data


tune_config = BaseTuneConfig(
    metric=['loss/val', 'micro_f1/val', 'macro_f1/val'],
    mode=['min', 'max', 'max'],
    space=space,
    oom_report_value=[2.0, 0.5, 0.3],
    param_to_config=param_to_config,
    repeat=5,
    points_to_evaluate=POINTS_TO_EVALUATE,
    sampler=NSGAIISampler(mutation_prob=0.125),
    reduction_factor=4,
    cache_dir=CACHE_DIR,
)
