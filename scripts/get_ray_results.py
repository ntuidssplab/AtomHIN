import argparse
import os

from ray import tune

from dhgl.script_utils.tuner import BaseTuneConfig, import_from_path
from scripts.raytune.tuner import get_trainable

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'config',
        type=str,
        help='path to the .py config file',
    )
    arg_parser.add_argument(
        '-n',
        '--num_samples',
        type=int,
        required=True,
    )
    args = arg_parser.parse_args()

    RESULT_DIR = 'ray_results'
    CONFIG_PY, NUM_SAMPLES = args.config, args.num_samples
    tune_config: BaseTuneConfig = import_from_path(
        'tune_config', CONFIG_PY
    ).tune_config

    res_path = os.path.abspath(
        os.path.join(
            RESULT_DIR, (
                os.path.normpath(CONFIG_PY).replace(os.path.sep, '.') +
                f'_{tune_config.hash_str[:8]}' + f'-n{NUM_SAMPLES}'
            )
        )
    )
    restored_tuner = tune.Tuner.restore(
        res_path, trainable=get_trainable(os.path.abspath(CONFIG_PY))
    )
    results = restored_tuner.get_results()

    objective = tune_config.mean_metric[
        0] if tune_config.multi_objective else tune_config.mean_metric
    mode = tune_config.mode[
        0] if tune_config.multi_objective else tune_config.mode
    df = results.get_dataframe()

    metric_cols = ['macro_f1/val/mean', 'micro_f1/val/mean']
    if 'acc/val/mean' in df.columns:
        metric_cols = ['acc/val/mean']
    elif 'roc_auc/val/mean' in df.columns:
        metric_cols = ['roc_auc/val/mean', 'mrr/val/mean']

    query = ['logdir', objective] + [
        met for met in metric_cols if met != objective
    ] + [met.replace('val', 'test') for met in metric_cols]

    if mode == 'min':
        print(df.iloc[df[objective].argsort()[:10]][query])
    else:
        print(df.iloc[(-df[objective]).argsort()[:10]][query])
