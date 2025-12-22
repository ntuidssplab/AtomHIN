from __future__ import annotations

import os
from itertools import product

import naive_flow as nf
from optuna.samplers import NSGAIISampler
from ray import tune

from dhgl.script_utils.tuner import BaseTuneConfig
from scripts.train.trainer import TrainerConfig

BASE_ENV_FILE = os.path.abspath(
    os.path.join(__file__, '..', '_ndblp.sRGCN.env')
)
assert os.path.isfile(BASE_ENV_FILE)
CACHE_DIR = os.path.abspath(
    os.path.join(__file__, '..', '_ndblp.sRGCN.tune-cache')
)

TARGET_NTYPE = 'author'
NT_NTYPES = [
    'authorfeat',
    'conference',
    'numerical',
    'paper',
    'paperfeat',
    'term',
]
NTYPES = [TARGET_NTYPE, *NT_NTYPES]
NTYPE_FEAT = ['none', 'nid']
# yapf: disable
UDCETYPES = [
    [('author', 'has-authorfeat', 'authorfeat'), ('authorfeat', 'is-authorfeat-of', 'author')],
    [('conference', 'has', 'paper'), ('paper', 'pubs-in', 'conference')],
    [('paper', 'contains', 'term'), ('term', 'is-in', 'paper')],
    [('paper', 'has-paperfeat', 'paperfeat'), ('paperfeat', 'is-paperfeat-of', 'paper')],
    [('author', 'writing', 'paper'), ('paper', 'written-by', 'author')],
    [('term', 'has-numerical', 'numerical'), ('numerical', 'is-numerical-of', 'term')],
]
# yapf: enable


def cetype_to_name(cetypes: list):
    return '|'.join(cetype[1] for cetype in cetypes)


UDCETYPES_NAME = list(map(cetype_to_name, UDCETYPES))
space = {}
space = {t: tune.choice([0, 1]) for t in [*NT_NTYPES, *UDCETYPES_NAME]}
space['num_layers'] = tune.randint(4, 14)

POINTS_TO_EVALUATE = [
    { #
        # 'author': 0,
        'authorfeat': 0,
        'conference': 1,
        'numerical': 0,
        'paper': 0,
        'paperfeat': 0,
        'term': 1,
        'num_layers': 8,
        'has-authorfeat|is-authorfeat-of': 1,
        'has|pubs-in': 0,
        'contains|is-in': 0,
        'has-paperfeat|is-paperfeat-of': 1,
        'writing|written-by': 0,
        'has-numerical|is-numerical-of': 1,
    }
]


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
    env_data['hgnn_config'].update(num_layers=param['num_layers'], )
    return env_data


tune_config = BaseTuneConfig(
    metric=['loss/val', 'micro_f1/val', 'macro_f1/val'],
    mode=['min', 'max', 'max'],
    oom_report_value=[1.0, 0.9, 0.9],
    space=space,
    param_to_config=param_to_config,
    repeat=10,
    points_to_evaluate=POINTS_TO_EVALUATE,
    sampler=NSGAIISampler(),
    reduction_factor=2,
    cache_dir=CACHE_DIR,
)
