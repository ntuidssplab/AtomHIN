import os
from itertools import product

import naive_flow as nf
from optuna.samplers import NSGAIISampler
from ray import tune

from dhgl.data.link_prediction.pubmed import NormalizedPubMedDataset
from dhgl.script_utils.tuner import BaseTuneConfig
from scripts.train.linkpred import TrainerConfig

BASE_ENV_FILE = os.path.abspath(
    os.path.join(__file__, '..', '_npubmed.sRGCN.env')
)
assert os.path.isfile(BASE_ENV_FILE)
CACHE_DIR = os.path.abspath(
    os.path.join(__file__, '..', '_npubmed.sRGCN.tune-cache')
)

NTYPES = NormalizedPubMedDataset.ntypes

NormalizedPubMedDataset.canonical_etypes
NTYPE_FEAT = ['none', 'nid']
# yapf: disable
UDCETYPES = list(
    zip(
        NormalizedPubMedDataset.canonical_etypes[:14],
        NormalizedPubMedDataset.canonical_etypes[14:]
    )
)
# yapf: enable


def cetype_to_name(cetypes: list):
    return '|'.join(cetype[1] for cetype in cetypes)


UDCETYPES_NAME = list(map(cetype_to_name, UDCETYPES))
space = {}
space = {t: tune.choice([0, 1]) for t in [*NTYPES, *UDCETYPES_NAME]}
space['num_layers'] = tune.randint(0, 10)

POINTS_TO_EVALUATE = [
    { # Vanilla
        'species': 0,
        'disease': 0,
        'chemical': 0,
        'gene': 0,
        'species_feat': 1,
        'disease_feat': 1,
        'chemical_feat': 1,
        'gene_feat': 1,
        'num_layers': 1,
        'species-species|species-species-inv': 0,
        'species-disease|disease-species': 0,
        'disease-disease|disease-disease-inv': 0,
        'chemical-species|species-chemical': 0,
        'chemical-disease|disease-chemical': 0,
        'chemical-chemical|chemical-chemical-inv': 0,
        'chemical-gene|gene-chemical': 0,
        'gene-species|species-gene': 0,
        'gene-disease|disease-gene': 0,
        'gene-gene|gene-gene-inv': 0,
        'species-has-feat|feat-of-species': 1,
        'disease-has-feat|feat-of-disease': 1,
        'chemical-has-feat|feat-of-chemical': 1,
        'gene-has-feat|feat-of-gene': 1,
    },
    { # nid
        'species': 1,
        'disease': 1,
        'chemical': 1,
        'gene': 1,
        'species_feat': 0,
        'disease_feat': 0,
        'chemical_feat': 0,
        'gene_feat': 0,
        'num_layers': 1,
        'species-species|species-species-inv': 0,
        'species-disease|disease-species': 0,
        'disease-disease|disease-disease-inv': 0,
        'chemical-species|species-chemical': 0,
        'chemical-disease|disease-chemical': 0,
        'chemical-chemical|chemical-chemical-inv': 0,
        'chemical-gene|gene-chemical': 0,
        'gene-species|species-gene': 0,
        'gene-disease|disease-gene': 0,
        'gene-gene|gene-gene-inv': 0,
        'species-has-feat|feat-of-species': 1,
        'disease-has-feat|feat-of-disease': 1,
        'chemical-has-feat|feat-of-chemical': 1,
        'gene-has-feat|feat-of-gene': 1,
    },
    { # nid + all etypes
        'species': 1,
        'disease': 1,
        'chemical': 1,
        'gene': 1,
        'species_feat': 0,
        'disease_feat': 0,
        'chemical_feat': 0,
        'gene_feat': 0,
        'num_layers': 1,
        'species-species|species-species-inv': 0,
        'species-disease|disease-species': 0,
        'disease-disease|disease-disease-inv': 0,
        'chemical-species|species-chemical': 0,
        'chemical-disease|disease-chemical': 0,
        'chemical-chemical|chemical-chemical-inv': 0,
        'chemical-gene|gene-chemical': 0,
        'gene-species|species-gene': 0,
        'gene-disease|disease-gene': 0,
        'gene-gene|gene-gene-inv': 0,
        'species-has-feat|feat-of-species': 0,
        'disease-has-feat|feat-of-disease': 0,
        'chemical-has-feat|feat-of-chemical': 0,
        'gene-has-feat|feat-of-gene': 0,
    },
    { # all ntypes
        'species': 1,
        'disease': 1,
        'chemical': 1,
        'gene': 1,
        'species_feat': 1,
        'disease_feat': 1,
        'chemical_feat': 1,
        'gene_feat': 1,
        'num_layers': 1,
        'species-species|species-species-inv': 0,
        'species-disease|disease-species': 0,
        'disease-disease|disease-disease-inv': 0,
        'chemical-species|species-chemical': 0,
        'chemical-disease|disease-chemical': 0,
        'chemical-chemical|chemical-chemical-inv': 0,
        'chemical-gene|gene-chemical': 0,
        'gene-species|species-gene': 0,
        'gene-disease|disease-gene': 0,
        'gene-gene|gene-gene-inv': 0,
        'species-has-feat|feat-of-species': 1,
        'disease-has-feat|feat-of-disease': 1,
        'chemical-has-feat|feat-of-chemical': 1,
        'gene-has-feat|feat-of-gene': 1,
    },
    { # zero layers + nid
        'species': 1,
        'disease': 1,
        'chemical': 1,
        'gene': 1,
        'species_feat': 0,
        'disease_feat': 0,
        'chemical_feat': 0,
        'gene_feat': 0,
        'num_layers': 0,
        'species-species|species-species-inv': 0,
        'species-disease|disease-species': 0,
        'disease-disease|disease-disease-inv': 0,
        'chemical-species|species-chemical': 0,
        'chemical-disease|disease-chemical': 0,
        'chemical-chemical|chemical-chemical-inv': 0,
        'chemical-gene|gene-chemical': 0,
        'gene-species|species-gene': 0,
        'gene-disease|disease-gene': 0,
        'gene-gene|gene-gene-inv': 0,
        'species-has-feat|feat-of-species': 1,
        'disease-has-feat|feat-of-disease': 1,
        'chemical-has-feat|feat-of-chemical': 1,
        'gene-has-feat|feat-of-gene': 1,
    },
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
