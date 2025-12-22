from __future__ import annotations

import json
import os
from argparse import ArgumentParser

import naive_flow as nf

from dhgl.script_utils import trainer

try:
    from .trainer.config import TrainerConfig
except ImportError as e:
    if 'attempted relateive import' not in str(e.msg):
        raise e
    from dhgl.script_utils import correct_module_path
    e.add_note(
        'You are trying to run the py file directly, please you '
        f'"python -m {correct_module_path(__file__)}" instead'
    )
    raise e


def main():
    arg_parser = ArgumentParser('train-node-classification')
    arg_parser.add_argument(
        'env',
        type=str,
        help='path to the .env file to use as config',
    )
    arg_parser.add_argument(
        '--check',
        nargs='?',
        const=True,
    )
    args = arg_parser.parse_args()
    env_path = args.env
    assert os.path.isfile(env_path), 'No env file found'

    data = nf.load_env_file(
        env_path, preset_env_vars={
            '__file__': env_path,
            '__dir__': os.path.dirname(env_path),
            **os.environ,
        }
    )
    config = TrainerConfig.model_validate_strings(data)
    print(nf.strfconfig(config, description=args.check))
    if args.check:
        return
    tracker = trainer.train(config)
    print('best:')
    best_scalars = tracker.get_best_scalars()
    print(best_scalars)
    if config.tracker_config.save_end:
        print('last:')
        last_scalars = tracker._history[-1]
        print(last_scalars)
        with open(
            os.path.join(tracker.log_dir, 'results_last.json'), 'w',
            encoding='utf8'
        ) as fout:
            json.dump(last_scalars, fout, indent=4)
    with open(
        os.path.join(tracker.log_dir, 'results.json'), 'w', encoding='utf8'
    ) as fout:
        json.dump(best_scalars, fout, indent=4)


if __name__ == '__main__':
    main()
