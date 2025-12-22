import os

import naive_flow as nf
from optuna.samplers import NSGAIISampler
from ray import tune

from dhgl.script_utils.tuner import BaseTuneConfig, ParametersInvalidError
from scripts.precom.trainer.config import TrainerConfig

BASE_ENV_FILE = {
    'cpu': os.path.abspath(os.path.join(__file__, '..', '_nmag.SRGCN.env')),
}
CACHE_DIR = os.path.abspath(
    os.path.join(__file__, '..', '_nmag.SRGCN.tune-cache')
)
assert all(os.path.isfile(f) for f in BASE_ENV_FILE.values())

# yapf: disable
space = {}
BASES = {}
space['_dataset_config__feat_types__paper'] = tune.choice([0, 1])
space['_dataset_config__feat_types__institution'] = tune.choice([0, 1])
space['_dataset_config__feat_types__field_of_study'] = tune.choice([0, 1])
space['_dataset_config__feat_types__author'] = tune.choice([0, 1])
space['_dataset_config__feat_types__numerical'] = tune.choice([0, 1])
space['_dataset_config__feat_types__year'] = tune.choice([0, 1])

space['_dataset_config__exclude_edge_types__PY'] = tune.choice([0, 1])
space['_dataset_config__exclude_edge_types__PP'] = tune.choice([0, 1])
space['_dataset_config__exclude_edge_types__PA'] = tune.choice([0, 1])
space['_dataset_config__exclude_edge_types__PN'] = tune.choice([0, 1])
space['_dataset_config__exclude_edge_types__PF'] = tune.choice([0, 1])
space['_dataset_config__exclude_edge_types__AI'] = tune.choice([0, 1])

DIMS = [256]
CHOICES = {
    '_dataset_config__feat_types__paper': ['none', *[f'u({d}, -0.5, 0.5, 0xAA9)' for d in DIMS]],
    '_dataset_config__feat_types__institution': ['none', *[f'u({d}, -0.5, 0.5, 0xAAA)' for d in DIMS]],
    '_dataset_config__feat_types__field_of_study': ['none', *[f'u({d}, -0.5, 0.5, 0xAAB)' for d in DIMS]],
    '_dataset_config__feat_types__author': ['none', *[f'u({d}, -0.5, 0.5, 0xAAC)' for d in DIMS]],
    '_dataset_config__feat_types__numerical': ['none', 'nid'],
    '_dataset_config__feat_types__year': ['none', 'nid'],
}
ETYPE_MAPPING = {
    '_dataset_config__exclude_edge_types__PY': ['year-of-publication', 'published-in-year'],
    '_dataset_config__exclude_edge_types__PP': ['cites'],
    '_dataset_config__exclude_edge_types__PA': ['writes', 'written_by'],
    '_dataset_config__exclude_edge_types__PN': ["has-numerical", "is-numerical-of"],
    '_dataset_config__exclude_edge_types__PF': ['contains', 'has_topic'],
    '_dataset_config__exclude_edge_types__AI': ['affiliated_with', 'affiliates'],
}
# yapf: enable
point = {}  #YNFPI
point['_dataset_config__feat_types__paper'] = 1
point['_dataset_config__feat_types__institution'] = 1
point['_dataset_config__feat_types__field_of_study'] = 1
point['_dataset_config__feat_types__author'] = 0
point['_dataset_config__feat_types__numerical'] = 1
point['_dataset_config__feat_types__year'] = 1
point['_dataset_config__exclude_edge_types__PY'] = 1
point['_dataset_config__exclude_edge_types__PN'] = 0
point['_dataset_config__exclude_edge_types__PF'] = 0
point['_dataset_config__exclude_edge_types__AI'] = 0
point['_dataset_config__exclude_edge_types__PP'] = 0
point['_dataset_config__exclude_edge_types__PA'] = 0

point2 = {}  #YNFPI
point2['_dataset_config__feat_types__paper'] = 0
point2['_dataset_config__feat_types__institution'] = 0
point2['_dataset_config__feat_types__field_of_study'] = 1
point2['_dataset_config__feat_types__author'] = 0
point2['_dataset_config__feat_types__numerical'] = 1
point2['_dataset_config__feat_types__year'] = 1
point2['_dataset_config__exclude_edge_types__PY'] = 1
point2['_dataset_config__exclude_edge_types__PN'] = 0
point2['_dataset_config__exclude_edge_types__PF'] = 0
point2['_dataset_config__exclude_edge_types__AI'] = 0
point2['_dataset_config__exclude_edge_types__PP'] = 0
point2['_dataset_config__exclude_edge_types__PA'] = 0

point3 = {}  #AYNF
point3['_dataset_config__feat_types__paper'] = 0
point3['_dataset_config__feat_types__institution'] = 0
point3['_dataset_config__feat_types__field_of_study'] = 1
point3['_dataset_config__feat_types__author'] = 1
point3['_dataset_config__feat_types__numerical'] = 1
point3['_dataset_config__feat_types__year'] = 1
point3['_dataset_config__exclude_edge_types__PY'] = 1
point3['_dataset_config__exclude_edge_types__PN'] = 0
point3['_dataset_config__exclude_edge_types__PF'] = 0
point3['_dataset_config__exclude_edge_types__AI'] = 0
point3['_dataset_config__exclude_edge_types__PP'] = 0
point3['_dataset_config__exclude_edge_types__PA'] = 0

POINTS_TO_EVALUATE = [point, point2, point3]


def check_valid(param):
    feat_args = [
        '_dataset_config__feat_types__paper',
        '_dataset_config__feat_types__institution',
        '_dataset_config__feat_types__field_of_study',
        '_dataset_config__feat_types__author',
        '_dataset_config__feat_types__numerical',
        '_dataset_config__feat_types__year',
    ]
    v = sum(param[a] for a in feat_args)
    return v > 0


def param_to_config(param):

    if not check_valid(param):
        raise ParametersInvalidError()
    env_file = BASE_ENV_FILE['cpu']
    env_data = nf.load_env_file(
        env_file, preset_env_vars={
            '__file__': env_file,
            '__dir__': os.path.dirname(env_file),
            **os.environ,
        }
    )

    for key in space:
        key: str
        if key.startswith('_'):
            if key in BASES:
                val = BASES[key]**param[key]
                key = key.removeprefix('_')
            elif key in CHOICES:
                val = CHOICES[key][param[key]]
                key = key.removeprefix('_')
            else:
                continue
        else:
            val = param[key]
        data = env_data
        scopes = key.split('__')
        for scope in scopes[:-1]:
            if scope not in data:
                data[scope] = {}
            data = data[scope]
        data.update({scopes[-1]: val})

    exclude_edge_types = []
    for key, vals in ETYPE_MAPPING.items():
        if param[key]:
            exclude_edge_types += vals
    env_data['hgnn_config']['mp_config']['exclude_edge_types'
                                         ] = exclude_edge_types

    config = TrainerConfig.model_validate_strings(env_data)
    return config


# Validate points
[param_to_config(p) for p in POINTS_TO_EVALUATE]

tune_config = BaseTuneConfig(
    metric='acc/val',
    mode='max',
    oom_report_value=0.5,
    space=space,
    param_to_config=param_to_config,
    repeat=1,
    points_to_evaluate=POINTS_TO_EVALUATE,
    sampler=NSGAIISampler(mutation_prob=0.2),
    reduction_factor=4,
    cache_dir=CACHE_DIR,
)
