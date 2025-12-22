import os

import naive_flow as nf
from optuna.samplers import NSGAIISampler
from ray import tune

from dhgl.script_utils.tuner import BaseTuneConfig

BASE_ENV_FILE = os.path.abspath(
    os.path.join(__file__, '..', '_nimdb.sRGCN.env')
)
CACHE_DIR = os.path.abspath(
    os.path.join(__file__, '..', '_nimdb.sRGCN.tune-cache')
)
assert os.path.isfile(BASE_ENV_FILE)
TARGET_NTYPE = 'movie'
NT_NTYPES = [
    'keyword',
    'director',
    'actor',
    'color',
    'language',
    'country',
    'content_rating',
    'numerical',
    'word',
]
NTYPES = [TARGET_NTYPE, *NT_NTYPES]
NTYPE_FEAT = ['none', 'nid']
# yapf: disable
UDCETYPES = [
    [('actor', 'acts', 'movie'), ('movie', 'stars', 'actor')],
    [('color', 'is-type-of', 'movie'), ('movie', 'has-color', 'color')],
    [('content_rating', 'is-rating-for', 'movie'), ('movie', 'has-rating', 'content_rating')],
    [('country', 'is-country-of', 'movie'), ('movie', 'is-from-country', 'country')],
    [('director', 'directed', 'movie'), ('movie', 'directed-by', 'director')],
    [('keyword', 'is-in', 'movie'), ('movie', 'contains', 'keyword')],
    [('language', 'is-language-of', 'movie'), ('movie', 'is-in-language', 'language')],
    [('movie', 'contains-word', 'word'), ('word', 'is-word-of', 'movie')],
    [('movie', 'has-numerical', 'numerical'), ('numerical', 'is-numerical-of', 'movie')],
]
# yapf: enable


def cetype_to_name(cetypes: list):
    return '|'.join(cetype[1] for cetype in cetypes)


UDCETYPES_NAME = list(map(cetype_to_name, UDCETYPES))
space = {}
space = {t: tune.choice([0, 1]) for t in [*NT_NTYPES, *UDCETYPES_NAME]}
space['num_layers'] = tune.randint(2, 12)
# space['num_heads'] = 8

POINTS_TO_EVALUATE = [
    { # IMDB
        'keyword': 0,
        'director': 0,
        'actor': 0,
        'color': 1,
        'language': 1,
        'country': 1,
        'content_rating': 1,
        'numerical': 1,
        'word': 1,
        'num_layers': 8,
        'acts|stars': 0,
        'is-type-of|has-color': 1,
        'is-rating-for|has-rating': 1,
        'is-country-of|is-from-country': 1,
        'directed|directed-by': 0,
        'is-in|contains': 0,
        'is-language-of|is-in-language': 1,
        'contains-word|is-word-of': 1,
        'has-numerical|is-numerical-of': 1,
    },
]
# assert all(set(point) == set(space) for point in POINTS_TO_EVALUATE)


class ParametersInvalidError(ValueError):
    pass


def param_to_config(param):

    if all(param[ntype] == 0 for ntype in NT_NTYPES):
        raise ParametersInvalidError('Invalid parameters combinition')
    feat_types = {ntype: NTYPE_FEAT[param[ntype]] for ntype in NT_NTYPES}
    feat_types[TARGET_NTYPE] = NTYPE_FEAT[0]

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
    metric='loss/val',
    mode='min',
    space=space,
    param_to_config=param_to_config,
    repeat=10,
    reduction_factor=4,
    points_to_evaluate=POINTS_TO_EVALUATE,
    sampler=NSGAIISampler(),
    cache_dir=CACHE_DIR,
)
