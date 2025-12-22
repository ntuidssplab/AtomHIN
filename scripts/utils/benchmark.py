from __future__ import annotations

import argparse
import gc
import os
import pprint
import re
import time
from glob import glob

import naive_flow as nf
import torch
from pydantic import TypeAdapter, ValidationError
from tqdm import tqdm

from dhgl.script_utils.benchmark_recorder import BenchmarkRecorder
from scripts.precom.trainer import TrainerConfig as PrecomConfig
from scripts.train.linkpred import TrainerConfig as LinkPredTrainerConfig
from scripts.train.trainer import TrainerConfig

Config = TypeAdapter(TrainerConfig | PrecomConfig | LinkPredTrainerConfig)


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'envs_dir',
        type=str,
        nargs='+',
        help='path(s) to the .env files',
    )
    arg_parser.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help='skip confirmation',
    )
    arg_parser.add_argument(
        '--check',
        action='store_true',
        help='check envs',
    )
    arg_parser.add_argument(
        '-r',
        '--rounds',
        type=int,
        default=5,
    )
    arg_parser.add_argument(
        '--round-robin',
        '--rr',
        action='store_true',
        help='Run experiments in round-robin mode.',
    )
    arg_parser.add_argument(
        '--progress',
        choices=['none', 'tqdm'],
        default='none',
        help='Arg passed to nf.set_global',
    )
    arg_parser.add_argument(
        '--tqdm',
        action='store_true',
        help='shortcut to --progress=tqdm',
    )
    arg_parser.add_argument(
        '--verbose',
        action='store_true',
        default=False,
        help='Arg passed to nf.set_global',
    )
    arg_parser.add_argument(
        '--log-root-dir',
        '--log_root_dir',
        required=False,
        help='Arg passed to nf.set_global',
    )
    arg_parser.add_argument(
        '-f', '--filter', type=str, help='regex to filename to keep'
    )
    arg_parser.add_argument(
        '-i', '--ignore', type=str, default=r'^_',
        help='regex to filename to ignore'
    )
    args = arg_parser.parse_args()
    envs_dirs = args.envs_dir
    single_file_mode = len(envs_dirs) == 1 and os.path.isfile(envs_dirs[0])

    def filiter_dir(envs_dirs: list[str]):

        def _filtier_dir(envs_dir: str):
            if os.path.isfile(envs_dir):
                yield os.path.dirname(envs_dir), os.path.basename(envs_dir)
                return
            for file in glob('**/*.env', root_dir=envs_dir, recursive=True):
                if re.search(args.ignore, os.path.basename(file)) is not None:
                    continue
                if args.filter is not None:
                    if re.search(args.filter, file):
                        yield envs_dir, file
                else:
                    yield envs_dir, file

        all_envs = []
        print('[')
        for envs_dir in envs_dirs:
            for d, f in _filtier_dir(envs_dir):
                print(f'    "{os.path.join(d, f)}",')
                all_envs.append((d, f))
            # env_files = list(_filtier_dir(envs_dir))
            # print(envs_dir)
            # pprint(env_files)
            # all_envs.extend([(envs_dir, env_file) for env_file in env_files])
        print(']')
        if not args.quiet and not args.check and not single_file_mode:
            input('check if the above profiles correct...')
        return all_envs

    all_envs = filiter_dir(envs_dirs)

    for envs_dir, env_file in all_envs:
        check_env(envs_dir, env_file)
    if args.check:
        return

    log_root_dir = args.log_root_dir
    if args.log_root_dir is None:
        log_root_dir = nf.tracker.new_time_formatted_log_dir('', 'runs_b')
    nf.set_global(
        progress='tqdm' if args.tqdm else args.progress, verbose=args.verbose,
        log_root_dir=log_root_dir
    )
    if single_file_mode:
        assert len(all_envs) == 1
        results = run(*all_envs[0], n_rounds=args.rounds).get_df()
        pprint.pprint(
            results.drop('profile', axis=1).aggregate(['mean',
                                                       'std']).to_dict()
        )
    elif args.round_robin:
        for i in tqdm(range(args.rounds)):
            for envs_dir, env_file in tqdm(all_envs):
                run(envs_dir, env_file, n_rounds=i + 1)
                gc.collect()
    else:
        for envs_dir, env_file in tqdm(all_envs):
            run(envs_dir, env_file, n_rounds=args.rounds)
            gc.collect()
    return


def check_env(root: str, env_file: str):
    try:
        env_path = os.path.join(root, env_file)
        data = nf.load_env_file(
            env_path, preset_env_vars={
                '__file__': env_path,
                '__dir__': os.path.dirname(env_path)
            }
        )
        Config.validate_strings(data)
    except ValidationError as err:
        print(f'Validation error occurred for {os.path.join(root, env_file)}')
        raise err
    return


def run(root: str, env_file: str, n_rounds: int):
    profile = '.'.join(env_file.split(os.path.sep))
    if profile.endswith('.env'):
        profile = profile[:-4]
    res_path = os.path.join(root, env_file[:-4] + '.csv')
    oom_path = os.path.join(
        os.path.dirname(res_path),
        '_OOM.' + os.path.basename(env_file)[:-4] + '.csv'
    )
    if os.path.exists(oom_path):
        return
    benchmark = BenchmarkRecorder(res_path)

    with tqdm(
        range(len(benchmark), n_rounds), leave=False,
        disable=(n_rounds - len(benchmark)) <= 1
    ) as pbar:
        for i in pbar:
            env_path = os.path.join(root, env_file)
            config = Config.validate_strings(
                nf.load_env_file(
                    env_path, preset_env_vars={
                        '__file__': env_path,
                        '__dir__': os.path.dirname(env_path)
                    }
                )
            )
            gc.collect()
            torch.cuda.empty_cache()
            pbar.set_description(profile)
            try:
                metrics = config.run()
                if hasattr(metrics, 'get_best_scalars'):
                    metrics = metrics.get_best_scalars()
                metrics['profile'] = profile
                benchmark.add_row(metrics)
                del config
            except torch.cuda.OutOfMemoryError as e:
                if os.path.exists(res_path):
                    raise e
                with open(oom_path, 'w', encoding='utf8') as fout:
                    pass
                pbar.write(f'OOM with profile: {profile}. skipped.')
                time.sleep(1.)

            if len(benchmark) - i - 1 > 0:
                pbar.update(len(benchmark) - i - 1)
            if n_rounds - len(benchmark) <= 0:
                # NOTE: check if results from other processes
                break
            # yield profile, tracker.get_best_scalars()
    return benchmark


if __name__ == '__main__':
    main()
