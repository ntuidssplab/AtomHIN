from __future__ import annotations

import os
from typing import Callable, Mapping, NamedTuple, Sequence

import torch
from dgl import DGLHeteroGraph
from dgl.dataloading import DataLoader
from dgl.heterograph import DGLBlock
from torch import Tensor, nn
from torch.utils.tensorboard.writer import SummaryWriter

import dhgl
import naive_flow as nf
from dhgl import hgget as H
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.dataloading import MockWholeGraphDataLoader

from .base import HGNNReturnT, PredDataT
from .config import TrainerConfig

HGraph = BaseHeteroGraphLike | list[DGLBlock]


class BatchData(NamedTuple):

    Ntype = str
    input_nodes: dict[Ntype, Tensor]
    output_nodes: dict[Ntype, Tensor]
    hblocks: Sequence[DGLBlock] | None
    blocks: Sequence[DGLBlock] | None
    subgraph: DGLHeteroGraph | None

    @classmethod
    def handle(cls, data: tuple):
        input_nodes, output_nodes, hblocks_or_graph, *blocks = data

        blocks = blocks[0] if blocks else None

        if isinstance(hblocks_or_graph, Sequence):
            hblocks = hblocks_or_graph
            subgraph = None
        else:
            subgraph = hblocks_or_graph
            hblocks = None
            assert blocks is None

        return cls(
            input_nodes,
            output_nodes,
            hblocks,
            blocks,
            subgraph,
        )

    @property
    def dstdata(self):
        if self.hblocks is not None:
            return self.hblocks[-1].dstdata
        assert self.subgraph is not None
        return self.subgraph.dstdata

    @property
    def srcdata(self):
        if self.hblocks is not None:
            return self.hblocks[0].srcdata
        assert self.subgraph is not None
        return self.subgraph.srcdata


def train_step(
    model,
    data_loader: DataLoader | MockWholeGraphDataLoader,
    forward_fn: Callable[[HGraph, Mapping[str, Tensor]], Tensor | PredDataT],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer,
    scheduler=None,
    max_iters: int | None = None,
    grad_max_norm: float | None = None,
    grad_max_value: float | None = None,
):

    model.train()

    def batch_train_step(data_loader: DataLoader):

        def _batch_train_step():
            tgt_ntype = list(data_loader.indices.keys())[0]
            assert data_loader.dataset.drop_last
            for i, data in enumerate(map(BatchData.handle, data_loader)):
                sub_labels = data.dstdata['label'][tgt_ntype]
                sub_pred = forward_fn(
                    data.blocks or data.hblocks or data.subgraph,
                    data.srcdata['feat']
                )

                if data.subgraph is not None:
                    # sampler samples subgraph instead of mfgs
                    # redundant nodes need to be trimmed
                    out_indices = data.output_nodes[tgt_ntype]
                    sub_pred = sub_pred[out_indices]
                    sub_labels = sub_labels[out_indices]
                    assert not data.subgraph.dstdata['test_mask'][tgt_ntype][
                        out_indices].any()

                train_loss = loss_fn(sub_pred, sub_labels)
                optimizer.zero_grad()
                train_loss.backward()
                if grad_max_value is not None:
                    nn.utils.clip_grad_value_(
                        model.parameters(), grad_max_value
                    )
                if grad_max_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                sub_logits = (
                    sub_pred
                    if isinstance(sub_pred, Tensor) else sub_pred.logits
                )
                yield (
                    sub_logits.detach().cpu(),
                    sub_labels.cpu(),
                    train_loss.detach().item(),
                )
                if max_iters is not None and i + 1 >= max_iters:
                    return

        logits, labels, losses = zip(*_batch_train_step())
        logits = torch.concat(logits)
        labels = torch.concat(labels)
        return (logits, labels, sum(losses) / len(losses))

    def graph_train_step(hg: BaseHeteroGraphLike, mask: Tensor):
        preds = forward_fn(hg, H.ndata(hg, 'feat'))
        preds = preds[mask]
        label = H.label(hg)[mask]
        optimizer.zero_grad()
        loss = loss_fn(preds, label)
        loss.backward()
        if grad_max_value is not None:
            nn.utils.clip_grad_value_(model.parameters(), grad_max_value)
        if grad_max_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)
        # grads = [
        #     param.grad.detach().flatten() for param in model.parameters()
        #     if param.grad is not None
        # ]
        # # grad_names = sum(grad_names, [])
        # grads = torch.cat(grads)
        # norm = grads.norm()
        # print(f'{grads.min() =},  {grads.max() = }')
        # print(f'{norm = }')
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        logits = preds if isinstance(preds, Tensor) else preds.logits
        return (logits, label, loss.detach().item())

    if isinstance(data_loader, DataLoader):
        return batch_train_step(data_loader)
    assert isinstance(data_loader, MockWholeGraphDataLoader)
    assert max_iters is None
    return graph_train_step(data_loader.hg, data_loader.indices)


@torch.no_grad()
def eval_step(
    model,
    data_loaders: list[DataLoader | MockWholeGraphDataLoader],
    forward_fn: Callable[[HGraph, dict[str, Tensor]], Tensor | PredDataT],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
):

    model.eval()

    def batch_eval_step(data_loader: DataLoader):

        def _batch_eval_step():
            tgt_ntype = list(data_loader.indices.keys())[0]
            for data in map(BatchData.handle, data_loader):
                sub_labels = data.dstdata['label'][tgt_ntype]
                sub_pred = forward_fn(
                    data.blocks or data.hblocks or data.subgraph,
                    data.srcdata['feat']
                )
                batch_size = len(data.output_nodes[tgt_ntype])
                if data.subgraph is not None:
                    # sampler samples subgraph instead of mfgs
                    # redundant nodes need to be trimmed
                    out_indices = data.output_nodes[tgt_ntype]
                    sub_pred = sub_pred[out_indices]
                    sub_labels = sub_labels[out_indices]
                loss = loss_fn(sub_pred, sub_labels)

                sub_logits = (
                    sub_pred
                    if isinstance(sub_pred, Tensor) else sub_pred.logits
                )

                yield (
                    sub_logits.detach().cpu(),
                    sub_labels.cpu(),
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

    cached_preds: dict[BaseHeteroGraphLike, Tensor] = {}

    def graph_eval_step(hg: BaseHeteroGraphLike, mask: Tensor):

        preds = cached_preds.get(hg, None)
        if preds is None:
            preds = forward_fn(hg, H.ndata(hg, 'feat'))
            cached_preds[hg] = preds

        split_preds = preds[mask]
        split_labels = H.label(hg)[mask]

        split_loss = loss_fn(split_preds, split_labels)
        split_logits = (
            split_preds
            if isinstance(split_preds, Tensor) else split_preds.logits
        )
        return (
            split_logits,
            split_labels,
            split_loss.detach().cpu().item(),
        )

    def tem():
        for data_loader in data_loaders:
            if isinstance(data_loader, DataLoader):
                yield batch_eval_step(data_loader)
            else:
                assert isinstance(data_loader, MockWholeGraphDataLoader)
                yield graph_eval_step(data_loader.hg, data_loader.indices)

    return list(tem())


def _set_cv(hg: BaseHeteroGraphLike, config: TrainerConfig):
    # hg = hg.clone() # <-- buggy if unreachable ntype exists
    from sklearn.model_selection import StratifiedKFold
    cv_conf = config.cv_config
    assert cv_conf is not None
    kf = StratifiedKFold(
        n_splits=cv_conf.num_folds, shuffle=True, random_state=cv_conf.seed
    )
    indices = torch.concat([H.index(hg, 'train'), H.index(hg, 'val')], dim=0)
    labels = torch.concat([H.label(hg, 'train'), H.label(hg, 'val')], dim=0)
    train_indices, val_indices = list(kf.split(indices,
                                               labels))[cv_conf.ith_fold]

    def indices_to_mask(indices: torch.Tensor):
        mask = torch.zeros_like(H.mask(hg, 'train'))
        mask[indices] = True
        return mask

    H.tgt_data(hg)['train_mask'] = indices_to_mask(indices[train_indices])
    H.tgt_data(hg)['val_mask'] = indices_to_mask(indices[val_indices])

    return hg


def _init_scheduler(config: TrainerConfig, hgnn_data: HGNNReturnT):
    if config.scheduler_config is not None:
        iters_per_epoch = 1
        if config.batch_config.train.is_in_batch_mode:
            raise NotImplementedError
            # n_samples = len(H.label(hg, 'train'))
            # iters_per_epoch = n_samples // global_conf.batch_config.train.batch_size
        total_iters = iters_per_epoch * config.epochs
        return config.scheduler_config.init(
            hgnn_data['optimizer'], total_iters
        )
    return hgnn_data.get('scheduler', None)


def _hgnn_init_adaption(config: TrainerConfig, return_data) -> HGNNReturnT:
    if isinstance(return_data, tuple):
        # NOTE: backward adaption
        hg, model, optimizer, scheduler, forward_fn, *loss_fn = return_data
        if isinstance(forward_fn, Callable):
            forward_fn = [forward_fn] * 2
        train_forward, eval_forward = forward_fn
        return_data: HGNNReturnT = {
            'hg': hg,
            'model': model,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'forward_fn': train_forward,
            'eval_forward_fn': eval_forward,
        }
        if config.scheduler_config is not None:
            assert scheduler is None
            return_data.pop('scheduler')
        if loss_fn:
            return_data['loss_fn'] = loss_fn[0]
        return return_data
    return return_data


def train(config: TrainerConfig):
    """Trainer for the MODEL"""

    if hasattr(config, 'dataset'):
        hg = config.dataset.load()
    else:
        # backward compatibility
        hg = config.dataset_config.load()

    if config.cv_config is not None:
        hg = _set_cv(hg, config)

    hgnn_data = _hgnn_init_adaption(
        config, config.hgnn_config.init(hg, config)
    )

    hg = hgnn_data.pop('hg')
    scheduler = _init_scheduler(config, hgnn_data)

    if config.batch_config.device_mode == 'gpu':
        hg = hg.to(config.device)
    else:
        # XXX: may introduce large memory overhead
        hg = dhgl.transforms.to_dense(hg)

    data_loaders = {
        'train': (
            config.batch_config.train.data_loader(
                hg, H.index(hg, 'train'), config.hgnn_config.num_layers
            )
        ),
        **{
            split: (
                config.batch_config.eval.data_loader(
                    hg,
                    H.index(hg, split),
                    config.hgnn_config.num_layers,
                )
            )
            for split in ('val', 'test')
        }
    }
    max_iters_per_epoch = None
    if isinstance(config.evaluator_config.epochs_per_eval, float):
        assert config.batch_config.train.is_in_batch_mode, (
            'If the epochs_per_eval is not integer, the training must be in batch mode.'
        )
        max_iters_per_epoch = int(
            config.evaluator_config.epochs_per_eval *
            len(data_loaders['train'])
        )
        assert max_iters_per_epoch >= 1, f'{len(data_loaders["train"]) = }'
    """Device Control"""
    hgnn_data['model'].to(config.device)
    """Init tracker"""
    tracker = nf.tracker.SimpleTracker(
        hgnn_data['model'],
        hgnn_data['optimizer'],
        scheduler,
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

        class AvoidUnderfittedRatio(nf.tracker.metrics.Ratio):

            _underfitted = True

            def better_than(self, rhs):
                if (
                    AvoidUnderfittedRatio._underfitted is True
                    and config.avoid_underfitting_threshold is not None
                    and self.value < config.avoid_underfitting_threshold
                ):
                    return True
                AvoidUnderfittedRatio._underfitted = False
                return self.value > rhs.value

        tracker.register_scalar(
            f'{config.evaluator_config.early_stopping_objective}/val',
            AvoidUnderfittedRatio,
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

    for epoch in tracker.range(config.epochs):

        data: dict[str, tuple] = {}
        losses: dict[str, float] = {}
        *data['train'], losses['train'] = train_step(
            model=hgnn_data['model'],
            data_loader=data_loaders['train'],
            forward_fn=hgnn_data['forward_fn'],
            loss_fn=hgnn_data.get('loss_fn', config.loss_fn),
            optimizer=hgnn_data['optimizer'],
            scheduler=scheduler,
            max_iters=max_iters_per_epoch,
            grad_max_norm=config.grad_max_norm,
            grad_max_value=config.grad_max_value,
        )

        if (
            config.evaluator_config.epochs_per_eval < 1
            or epoch % config.evaluator_config.epochs_per_eval == 0
        ):
            (*data['val'], losses['val']), (*data['test'], losses['test']) =\
                eval_step(
                    model=hgnn_data['model'],
                    data_loaders=(data_loaders['val'], data_loaders['test']),
                    forward_fn=hgnn_data.get('eval_forward_fn', hgnn_data['forward_fn']),
                    loss_fn=hgnn_data.get('loss_fn', config.loss_fn),
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

    best_metrics = tracker.get_best_scalars()
    if best_metrics is not None:
        writer.add_hparams(
            {'best': config.tracker_config.comment},
            {
                f'best/{name}': v
                for name, v in best_metrics.items()
            },
            run_name='.',
        )
    return tracker
