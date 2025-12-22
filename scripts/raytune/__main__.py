from __future__ import annotations

import argparse
import os
import shutil

import ray
import rich
from optuna.samplers import NSGAIISampler
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.optuna import OptunaSearch

from dhgl.script_utils.tuner import BaseTuneConfig, import_from_path

from .tuner import get_trainable


def get_num_samples(log_dir: str):
    import pickle
    with open(os.path.join(log_dir, 'tuner.pkl'), 'rb') as fin:
        state = pickle.load(fin)
        return state['_tune_config'].num_samples


def main():
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
    arg_parser.add_argument(
        '--rounds',
        type=int,
        required=False,
        help='Number of rounds, overriding "repeat" set in the config.py file',
    )
    arg_parser.add_argument(
        '--restart-errored',
        action='store_true',
        help=
        'When resuming from interrupted experiment, whether to restart errored trials',
    )
    arg_parser.add_argument(
        '--log-dir',
        type=str,
        required=False,
        help='log dir used to store/restore the experiment',
    )
    arg_parser.add_argument(
        '--searcher-log-dir',
        type=str,
        required=False,
        help=
        'Specify the log dir only for restoring searcher. Can be used to increase the num_samples',
    )
    arg_parser.add_argument(
        '--gpu',
        type=float,
        default=1,
        help='number of gpu to allocate per trial',
    )
    arg_parser.add_argument(
        '--cpu',
        type=float,
        default=1,
        help='number of cpu to allocate per trial',
    )
    arg_parser.add_argument(
        '--gpu-util',
        '--gpu_util',
        type=float,
        required=False,
        help='ray.tune.utils.wait_for_gpu(target_util)',
    )
    arg_parser.add_argument(
        '--gpu-delay',
        '--gpu_delay',
        type=int,
        default=100,
        help='ray.tune.utils.wait_for_gpu(delay sec.)',
    )
    arg_parser.add_argument(
        '--gpu-retry',
        '--gpu_retry',
        type=int,
        default=100000,
        help='ray.tune.utils.wait_for_gpu(retry)',
    )
    arg_parser.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help='Suppress confirmation prompts',
    )
    arg_parser.add_argument(
        '--temp_dir',
        '--temp-dir',
        type=str,
        required=False,
        help='ray.init(_temp_dir=<temp-dir>)',
    )
    args = arg_parser.parse_args()

    def _get_wait_for_gpu_args():
        gpu_kwargs = {
            'target_util': args.gpu_util,
            'retry': args.gpu_retry,
            'delay_s': args.gpu_delay,
        }
        return {k: v for k, v in gpu_kwargs.items() if v is not None}

    config_py_path = args.config
    assert os.path.isfile(
        config_py_path
    ), f'The provided path: {config_py_path} not found.'

    tune_config: BaseTuneConfig = import_from_path(
        'tune_config', config_py_path
    ).tune_config

    storage_path = os.path.abspath('ray_results')
    run_name = (
        os.path.normpath(config_py_path).replace(os.path.sep, '.') +
        f'_{tune_config.hash_str[:8]}' + f'-n{args.num_samples}'
    )

    log_dir = (args.log_dir
               and os.path.abspath(args.log_dir
                                   )) or os.path.join(storage_path, run_name)

    if args.searcher_log_dir is not None:
        assert tune.Tuner.can_restore(os.path.abspath(args.searcher_log_dir))
        # # XXX
        # assert args.num_samples > get_num_samples(args.searcher_log_dir), (
        #     'haven\'t find a reason to use searcher checkpoint not for increasing num_samples'
        # )
        assert not os.path.exists(log_dir), (
            'Using a search checkpoint requires starting a new experiment'
        )

    REPEAT = args.rounds or tune_config.repeat
    rich.print(
        BaseTuneConfig.model_validate(
            {
                **tune_config.model_dump(mode='python'), 'repeat': REPEAT
            }
        )
    )
    if not args.quiet:
        input('Check the config above, press Enter to continue...')
    if ray.is_initialized():
        ray.shutdown()
    ray.init(_temp_dir=args.temp_dir)

    train_with_gpu = tune.with_resources(
        get_trainable(
            os.path.abspath(config_py_path),
            gpu_waiting_kwargs=_get_wait_for_gpu_args(),
            repeat=REPEAT,
        ), {
            'gpu': args.gpu,
            'cpu': args.cpu,
        }
    )
    if tune.Tuner.can_restore(log_dir):
        tuner = tune.Tuner.restore(
            log_dir,
            train_with_gpu,
            restart_errored=args.restart_errored,
        )
    else:
        os.makedirs(log_dir, exist_ok=True)
        shutil.copy(config_py_path, os.path.join(log_dir, 'tune_config.py'))
        search_alg = OptunaSearch(
            metric=tune_config.mean_metric,
            mode=tune_config.mode,
            sampler=tune_config.sampler or NSGAIISampler(),
            points_to_evaluate=tune_config.points_to_evaluate,
        )
        if args.searcher_log_dir is not None:
            search_alg.restore_from_dir(os.path.abspath(args.searcher_log_dir))
        scheduler = AsyncHyperBandScheduler(
            metric=tune_config.mean_metric[0]
            if tune_config.multi_objective else tune_config.mean_metric,
            mode=tune_config.mode[0]
            if tune_config.multi_objective else tune_config.mode,
            grace_period=1,
            max_t=REPEAT,
            reduction_factor=tune_config.reduction_factor or 2,
        )
        tuner = tune.Tuner(
            train_with_gpu,
            tune_config=tune.TuneConfig(
                search_alg=search_alg,
                scheduler=scheduler,
                num_samples=args.num_samples,
                # max_concurrent_trials=1,
                reuse_actors=False,
            ),
            run_config=tune.RunConfig(
                storage_path=storage_path,
                name=run_name,
                log_to_file=True,
                failure_config=tune.FailureConfig(max_failures=1),
                sync_config=tune.SyncConfig(sync_artifacts=True),
            ),
            param_space=None if args.searcher_log_dir else tune_config.space,
        )
    tuner.fit()


if __name__ == '__main__':
    main()
