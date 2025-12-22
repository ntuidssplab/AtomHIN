from __future__ import annotations

import math
import os
from typing import Callable, Mapping

import naive_flow as nf
import torch
from torch import Tensor, nn
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from dhgl import hgget as H
from dhgl.script_utils.trainer.base import PredDataT
from dhgl.type import Split

from .base import BatchData, HGNNReturnT
from .config import DataLoadingConfig, TrainerConfig
from .data_utils import MemoryDataset, StackDataset, Subset, TransposedDataset
from .sampler import RandomChunkedBatchSampler, SequentialChunkedSampler


def _train_step(
    model: nn.Module,
    device,
    data_loader: DataLoader,
    forward_fn: Callable[..., Tensor | PredDataT],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer,
    scheduler=None,
    scaler: GradScaler | None = None,
    amp: torch.dtype | None = None,
    max_iters: int | None = None,
    grad_max_norm: float | None = None,
    grad_max_value: float | None = None,
    verbose: bool = False,
):

    model.train()

    assert not ((amp is None) ^ (scaler is None))

    def _batch_train_step():
        for i, data in enumerate(
            tqdm(
                data_loader, leave=False, disable=not verbose, desc='Training',
                position=1, file=nf.stdout, total=max_iters
            )
        ):
            data: BatchData
            labels = data.labels.to(device)

            with torch.autocast(
                torch.device(device).type, dtype=amp, enabled=scaler
                is not None
            ):
                preds = forward_fn(data.batch_indices, data.features)

                train_loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(train_loss).backward()
            else:
                train_loss.backward()
            if grad_max_value is not None:
                nn.utils.clip_grad_value_(model.parameters(), grad_max_value)
            if grad_max_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)

            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            yield (
                preds.detach().cpu(),
                labels.cpu(),
                train_loss.detach().item(),
            )
            if max_iters is not None and i + 1 >= max_iters:
                return

    logits, labels, losses = zip(*_batch_train_step())
    logits = torch.concat(logits)
    labels = torch.concat(labels)
    return (logits, labels, sum(losses) / len(losses))


@torch.no_grad()
def _eval_step(
    model: nn.Module,
    device,
    data_loader: DataLoader,
    forward_fn: Callable[..., Tensor | PredDataT],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    amp: torch.dtype | None = None,
    verbose: bool = False,
):

    model.eval()

    def _batch_eval_step():
        for data in tqdm(
            data_loader, leave=False, disable=not verbose, desc='Evaluating',
            position=1, file=nf.stdout
        ):
            data: BatchData
            labels = data.labels.to(device)
            batch_size = len(data.labels)
            with torch.autocast(
                torch.device(device).type, dtype=amp, enabled=amp is not None
            ):
                preds = forward_fn(data.batch_indices, data.features)
                loss = loss_fn(preds, labels)

            yield (
                preds.detach().cpu(),
                labels.cpu(),
                loss.detach().cpu(),
                batch_size,
            )

    logits, labels, losses, batch_sizes = zip(*_batch_eval_step())
    losses = Tensor(losses)
    batch_sizes = Tensor(batch_sizes)
    return (
        torch.concat(logits),
        torch.concat(labels),
        ((losses * batch_sizes).sum() / batch_sizes.sum()).item(),
    )


def _split_dataset(
    config: DataLoadingConfig,
    dataset,
    indices: dict[Split, torch.Tensor],
    labels: dict[Split, torch.Tensor],
):
    datasets = {}

    def split_dataset(split, dataset_):
        subset = Subset(dataset_, indices[split])
        if config[split].name == 'memory':
            return MemoryDataset(
                subset,
                device=config[split].device,
                feat_fmt=config[split].feat_fmt,
                verbose=config[split].verbose,
            )
        assert config[split].name == 'disk'
        if config[split].cache_dir is None:
            # Data will be loaded on the fly
            return subset
        cache_dir = config[split].mk_cache_dir()
        sub_cache_dir = os.path.expanduser(os.path.join(cache_dir, split))
        subset = TransposedDataset(
            subset,
            cache_dir=sub_cache_dir,
            batch_size=config[split].batch_size,
            chunk_size=config[split].chunk_size,
            cache_idx_dtype=config[split].cache_idx_dtype,
            cache_val_dtype=config[split].cache_val_dtype,
            memory_ratio=(
                config[split].memory_ratio if
                (config[split].memory_ratio or 0) < 1. else 0
            ),
            verbose=config[split].verbose,
        )
        if config[split].memory_ratio == 1.:
            # XXX: This could cause problem if there are sparse feat.
            # Only support for dense feat for now
            subset = MemoryDataset(
                subset,
                device='cpu',
                verbose=config[split].verbose,
            )

        return subset

    for split in ('train', 'val', 'test'):
        if isinstance(dataset, Mapping):
            subset = StackDataset(
                **{
                    k: split_dataset(split, v)
                    for k, v in dataset.items()
                }
            )
        else:
            subset = split_dataset(split, dataset)
        datasets[split] = StackDataset(indices[split], subset, labels[split])
    return datasets


def train(config: TrainerConfig):
    """Trainer for the MODEL"""

    hg, feats = config.dataset_config.load()
    hgnn_data: HGNNReturnT = config.hgnn_config.init(hg, feats, config)

    indices = {split: H.index(hg, split) for split in ('train', 'val', 'test')}
    labels = {split: H.label(hg, split) for split in ('train', 'val', 'test')}
    del hg  # save memory
    datasets = _split_dataset(
        config.data_loading_config, hgnn_data['dataset'], indices, labels
    )

    data_loaders = {
        split:
        DataLoader(
            datasets[split],
            batch_sampler=SequentialChunkedSampler(
                range(len(indices[split])),
                batch_size=(
                    config.batch_config[split].batch_size
                    or len(indices[split])
                ),
                drop_last=split == 'train',
            ),
            num_workers=config.batch_config[split].num_workers,
            pin_memory=config.batch_config[split].pin_memory,
            persistent_workers=config.batch_config[split].persistent_workers,
            collate_fn=BatchData.handle,
        )
        for split in ('val', 'test')
    }
    data_loaders['train'] = DataLoader(
        datasets['train'],
        batch_sampler=RandomChunkedBatchSampler(
            # batch_sampler=SequentialChunkedSampler(
            range(len(indices['train'])),
            batch_size=(
                config.batch_config['train'].batch_size
                or len(indices['train'])
            ),
            chunk_size=config.batch_config['train'].chunk_size,
            drop_last=True,
        ),
        num_workers=config.batch_config['train'].num_workers,
        pin_memory=config.batch_config['train'].pin_memory,
        persistent_workers=config.batch_config['train'].persistent_workers,
        collate_fn=BatchData.handle,
    )
    return _train(config, hgnn_data, data_loaders)


def _train(
    config: TrainerConfig,
    hgnn_data: HGNNReturnT,
    data_loaders: dict[Split, DataLoader],
    return_best_logits=False,
):
    max_iters_per_epoch = None
    if isinstance(config.evaluator_config.epochs_per_eval, float):
        max_iters_per_epoch = int(
            config.evaluator_config.epochs_per_eval *
            len(data_loaders['train'])
        )
        assert max_iters_per_epoch >= 1, f'{len(data_loaders["train"]) = }'
    model = hgnn_data['model'].to(config.device)
    scaler = config.amp and torch.cuda.amp.GradScaler()
    tracker = nf.tracker.SimpleTracker(
        model,
        optimizer=hgnn_data['optimizer'],
        scheduler=hgnn_data.get('scheduler', None),
        **dict(config.tracker_config),
        from_checkpoint=nf.tracker.checkpoint.parse_args(),
    )

    writer = SummaryWriter(
        log_dir=tracker.log_dir, purge_step=tracker.start_epoch
    )

    writer = tracker.register_summary_writer(writer)

    if 'loss' in config.evaluator_config.early_stopping_objective:
        tracker.register_scalar(
            f'{config.evaluator_config.early_stopping_objective}/val',
            nf.tracker.metrics.Loss,
            for_early_stopping=True,
        )
    else:

        class AvoidUnderfittingRatio(nf.tracker.metrics.Ratio):

            _underfitted = True

            def better_than(self, rhs):
                if (
                    AvoidUnderfittingRatio._underfitted is True
                    and config.avoid_underfitting_threshold is not None
                    and self.value < config.avoid_underfitting_threshold
                ):
                    return True
                AvoidUnderfittingRatio._underfitted = False
                return self.value > rhs.value

        tracker.register_scalar(
            f'{config.evaluator_config.early_stopping_objective}/val',
            AvoidUnderfittingRatio,
            for_early_stopping=True,
        )

    for metric in config.evaluator_config.metrics:
        tracker.register_scalar(
            f'{metric}/*', 'ratio'
        )  # XXX: not necessary be ratio here

    writer.add_text(
        'config', nf.strfconfig(config, strformat='markdown'),
        tracker.start_epoch
    )
    nf.dump_config(
        config, os.path.join(tracker.log_dir, 'config.env'), description='full'
    )

    best_logits = None
    for epoch in tracker.range(config.epochs, position=[0, 2]):

        data: dict[str, tuple] = {}
        losses: dict[str, float] = {}
        *data['train'], losses['train'] = _train_step(
            model=model,
            device=config.device,
            data_loader=data_loaders['train'],
            forward_fn=hgnn_data['forward_fn'],
            loss_fn=hgnn_data.get('loss_fn', config.loss_fn),
            optimizer=hgnn_data['optimizer'],
            scheduler=hgnn_data.get('scheduler', None),
            scaler=scaler,
            amp=config.amp and getattr(torch, config.amp),
            max_iters=max_iters_per_epoch,
            grad_max_norm=config.grad_max_norm,
            grad_max_value=config.grad_max_value,
            verbose=config.tracker_config.progress != 'none',
        )

        if (
            config.evaluator_config.epochs_per_eval < 1
            or epoch % config.evaluator_config.epochs_per_eval == 0
        ):
            *data['val'], losses['val'] = _eval_step(
                model=model,
                device=config.device,
                data_loader=data_loaders['val'],
                forward_fn=hgnn_data.get(
                    'eval_forward_fn', hgnn_data['forward_fn']
                ),
                loss_fn=hgnn_data.get('loss_fn', config.loss_fn),
                amp=config.amp and getattr(torch, config.amp),
                verbose=config.tracker_config.progress != 'none',
            )
            *data['test'], losses['test'] = _eval_step(
                model=model,
                device=config.device,
                data_loader=data_loaders['test'],
                forward_fn=hgnn_data.get(
                    'eval_forward_fn', hgnn_data['forward_fn']
                ),
                loss_fn=hgnn_data.get('loss_fn', config.loss_fn),
                amp=config.amp and getattr(torch, config.amp),
                verbose=config.tracker_config.progress != 'none',
            )

            results = config.evaluator_config.eval(
                data['train'],
                data['val'],
                data['test'],
            )
        else:
            results = config.evaluator_config.eval(data['train'])

        for split, metrics in zip(['train', 'val', 'test'], results):
            writer.add_scalar(f'loss/{split}', losses[split], epoch)
            for name, val in metrics.items():
                writer.add_scalar(f'{name}/{split}', val, epoch)

        if return_best_logits and tracker.is_best_epoch(epoch):
            best_logits = {
                'val': data['val'][0],
                'test': data['test'][0],
            }
        if any(math.isnan(l) for l in losses.values()):
            print('NAN detected. aborting.')
            if return_best_logits:
                return tracker, best_logits
            return tracker
        if config.early_breaking is not None:
            cur = tracker.get_best_scalars(no_within_loop_warning=True)
            if config.early_breaking.whether_break(epoch, cur):
                break

    best_metrics = tracker.get_best_scalars(no_within_loop_warning=True)
    if best_metrics is not None:
        writer.add_hparams(
            {'best': config.tracker_config.comment},
            {
                f'best/{name}': v
                for name, v in best_metrics.items()
            },
            run_name='.',
        )
    if return_best_logits:
        return tracker, best_logits

    return tracker
