from __future__ import annotations

import gc
import hashlib
import json
import os
import warnings
from collections import defaultdict

import numpy as np
import ray
import torch
from ray import tune

import naive_flow as nf
from dhgl.script_utils.configs.cv import CVConfig
from dhgl.script_utils.tuner import BaseTuneConfig, import_from_path

# Legacy
from scripts.train.trainer.config import TrainerConfig


def get_trainable(path: str, gpu_waiting_kwargs: dict = None, **kwargs):

    if gpu_waiting_kwargs is None:
        gpu_waiting_kwargs = {}

    def train(param):
        tune_config = import_from_path('tune_config', path).tune_config
        param['tune_config'] = tune_config
        for k, v in kwargs.items():
            param[k] = v
        return _train(tune_config, param, **gpu_waiting_kwargs)

    return train


def _fetch_cache(config: TrainerConfig, cache_dir: str):
    if cache_dir is None:
        return None

    if not os.path.isabs(cache_dir):
        warnings.warn(f'Not an abs path: {cache_dir}')
        return None

    configstr = nf.strfconfig(config)
    sha1 = hashlib.sha1(configstr.encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_dir, f'{sha1}.json')

    if not os.path.isfile(cache_path):
        return None  # No cache

    with open(cache_path, encoding='utf-8') as f:
        obj = json.load(f)
        obj.pop('config', None)
        return obj

    return None


def _fetch_cache_cv(config: TrainerConfig, tune_config: BaseTuneConfig):
    if tune_config.cv_config is None:
        # TODO
        pass

    def get_config(i):
        assert config.cv_config is None
        data = config.model_dump()
        data['cv_config'] = CVConfig(
            seed=tune_config.cv_config.seed,
            num_folds=tune_config.cv_config.num_folds,
            ith_fold=i,
        )
        return config.model_validate(data)

    configs = list(map(get_config, range(tune_config.cv_config.num_folds)))
    caches = [_fetch_cache(c, tune_config.cache_dir) for c in configs]

    def pop_results(cache: dict):
        try:
            return {k: v.pop(0) for k, v in cache.items()}
        except IndexError:
            return None

    results = defaultdict(list)
    while True:
        if not caches:
            break
        cache = caches.pop(0)
        if cache is None:
            break
        tem = pop_results(cache)
        if tem is None:
            break
        for k, v in tem.items():
            results[k].append(v)
        caches.append(cache)
    if results:
        return results
    return None


def _save_cache(config: TrainerConfig, results: dict, cache_dir: str):
    """Save value to cache with SHA-1 filename and SHA-256 validation."""
    if cache_dir is None:
        return None

    if not os.path.isabs(cache_dir):
        warnings.warn(f'Not an abs path: {cache_dir}')
        return None
    os.makedirs(cache_dir, exist_ok=True)
    configstr = nf.strfconfig(config)
    sha1 = hashlib.sha1(configstr.encode('utf-8')).hexdigest()
    cache_path = os.path.join(cache_dir, f'{sha1}.json')

    results = results.copy()
    results['config'] = config.model_dump_json()
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    return


def _save_cache_cv(config: TrainerConfig, results: dict, cache_dir: str):

    def _filter(vs: list):
        return [
            v for i, v in enumerate(vs)
            if (i % config.cv_config.num_folds) == config.cv_config.ith_fold
        ]

    results = {k: _filter(vs) for k, vs in results.items()}
    return _save_cache(config, results, cache_dir)


def _train(tune_config: BaseTuneConfig, param, **gpu_waiting_kwargs):

    def _get_config(i: int | None = None):
        config = tune_config.param_to_config(param)
        if isinstance(config, dict):  # Legacy
            config = TrainerConfig.model_validate_strings(config)
        if i is None or tune_config.cv_config is None:
            return config
        assert config.cv_config is None
        data = config.model_dump()
        data['cv_config'] = CVConfig(
            seed=tune_config.cv_config.seed,
            num_folds=tune_config.cv_config.num_folds,
            ith_fold=i % tune_config.cv_config.num_folds,
        )
        return config.model_validate(data)

    tune_config: BaseTuneConfig = param['tune_config']
    nf.set_global(progress='none', verbose=False, log_root_dir='runs')
    config = _get_config()
    nf.dump_config(config, 'config.env')

    metrics_history = defaultdict(list)
    cache_iters = 0
    if tune_config.cache_dir is not None:
        if tune_config.cv_config is not None:
            tem = _fetch_cache_cv(config, tune_config)
        else:
            tem = _fetch_cache(config, tune_config.cache_dir)
        if tem is not None:
            for k, v in tem.items():
                metrics_history[k] = v
                cache_iters = len(v)
            for i in range(min(param['repeat'], cache_iters)):
                metrics = {}
                for k, v in metrics_history.items():
                    metrics[k] = v[i]
                    metrics[f'{k}/mean'] = np.mean(metrics_history[k][:i + 1])
                    metrics[f'{k}/std'] = np.std(metrics_history[k][:i + 1])
                metrics['iters'] = i + 1
                tune.report(metrics)

    if param['repeat'] <= cache_iters:
        return

    if ray.get_gpu_ids():
        # if tune.get_context().get_trial_resources().required_resources['GPU']
        tune.utils.wait_for_gpu(**gpu_waiting_kwargs)

    try:
        for i in range(cache_iters, param['repeat']):
            gc.collect()
            torch.cuda.empty_cache()
            config = _get_config(i)
            if tune_config.trainer_fn is not None:  # Legacy
                tracker = tune_config.trainer_fn(config)
            else:
                tracker = config.run()

            if isinstance(tracker, nf.tracker.BaseTracker):
                metrics = tracker.get_best_scalars()
            else:
                # if trainer_fn return metrics dict
                metrics = tracker

            # Calculate accumulated metrics
            for k, v in metrics.copy().items():
                metrics_history[k].append(v)
                metrics[f'{k}/mean'] = np.mean(metrics_history[k])
                metrics[f'{k}/std'] = np.std(metrics_history[k])

            metrics['iters'] = i + 1
            if tune_config.cache_dir is not None:
                if tune_config.cv_config is not None:
                    _save_cache_cv(
                        config, metrics_history, tune_config.cache_dir
                    )
                else:
                    _save_cache(config, metrics_history, tune_config.cache_dir)
            tune.report(metrics)
    except torch.cuda.OutOfMemoryError as oom:
        if tune_config.oom_report_value is not None:
            if tune_config.multi_objective:
                mets = dict(
                    zip(tune_config.metric, tune_config.oom_report_value)
                )
                mets.update(
                    dict(
                        zip(
                            tune_config.mean_metric,
                            tune_config.oom_report_value
                        )
                    )
                )
            else:
                mets = {tune_config.metric: tune_config.oom_report_value}
                if not tune_config.metric.endswith('/mean'):
                    mets[f'{tune_config.metric}/mean'
                         ] = tune_config.oom_report_value
            tune.report(mets)
        raise oom
