import os
from itertools import product

import naive_flow as nf
from optuna.samplers import NSGAIISampler
from ray import tune

from dhgl.data.link_prediction.amazon import NormalizedAmazonDataset
from dhgl.script_utils.tuner import BaseTuneConfig
from scripts.train.linkpred import TrainerConfig

BASE_ENV_FILE = os.path.abspath(
    os.path.join(__file__, '..', '_namazon.sRGCN.env')
)
assert os.path.isfile(BASE_ENV_FILE)
CACHE_DIR = os.path.abspath(
    os.path.join(__file__, '..', '_namazon.sRGCN.tune-cache')
)

NTYPES = NormalizedAmazonDataset.ntypes
NTYPE_FEAT = ['none', 'nid', 'nid_coo']
UDCETYPES = list(
    zip(
        NormalizedAmazonDataset.canonical_etypes[:6],
        NormalizedAmazonDataset.canonical_etypes[6:]
    )
)


def cetype_to_name(cetypes: list):
    return '|'.join(cetype[1] for cetype in cetypes)


UDCETYPES_NAME = list(map(cetype_to_name, UDCETYPES))
space = {}
space = {t: tune.choice([0, 1]) for t in [*NTYPES, *UDCETYPES_NAME]}
space['product'] = tune.choice([0, 2])
space['num_layers'] = tune.randint(1, 13)

POINTS_TO_EVALUATE = [
    { # Vanilla
        'product': 0,
        'price': 1,
        'sales_rank': 1,
        'brand': 1,
        'category': 1,
        'num_layers': 3,
        'co-view|co-view-inv': 0,
        'co-purchase|co-purchase-inv': 0,
        'product-price|price-product': 0,
        'product-sales_rank|sales_rank-product': 0,
        'product-brand|brand-product': 0,
        'product-category|category-product': 0,
    },
    { # ALL
        'product': 2,
        'price': 1,
        'sales_rank': 1,
        'brand': 1,
        'category': 1,
        'num_layers': 3,
        'co-view|co-view-inv': 0,
        'co-purchase|co-purchase-inv': 0,
        'product-price|price-product': 0,
        'product-sales_rank|sales_rank-product': 0,
        'product-brand|brand-product': 0,
        'product-category|category-product': 0,
    }
]


class ParametersInvalidError(ValueError):
    pass


def param_to_config(param):

    if all(param[ntype] == 0 for ntype in NTYPES):
        raise ParametersInvalidError('Invalid parameters combinition')
    feat_types = {ntype: NTYPE_FEAT[param[ntype]] for ntype in NTYPES}

    exclude_edge_types = [
        cetype for cetype in UDCETYPES if param[cetype_to_name(cetype)]
    ]
    exclude_edge_types = [
        cetype[1] for cetype in sum(map(list, exclude_edge_types), [])
    ]

    env_data = nf.load_env_file(BASE_ENV_FILE)
    env_data['dataset_config'].update(
        {
            'feat_types': feat_types,
            'exclude_edge_types': exclude_edge_types,
        }
    )
    env_data['hgnn_config'].update(num_layers=param['num_layers'], )
    return TrainerConfig.model_validate_strings(env_data)


[param_to_config(p) for p in POINTS_TO_EVALUATE]

tune_config = BaseTuneConfig(
    metric='roc_auc/val',
    mode='max',
    oom_report_value=0.6,
    space=space,
    param_to_config=param_to_config,
    repeat=5,
    points_to_evaluate=POINTS_TO_EVALUATE,
    sampler=NSGAIISampler(mutation_prob=0.1),
    reduction_factor=2,
    cache_dir=CACHE_DIR,
)
