from __future__ import annotations

import os
import warnings
from typing import Callable, Mapping, NamedTuple

import dgl
import naive_flow as nf
import torch
from dgl import DGLHeteroGraph
from dgl.heterograph import DGLBlock
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from dhgl import hgget as H
from dhgl import transforms
from dhgl.data.link_prediction import LinkPredDatasetLike
from dhgl.data.schema import BaseHeteroGraphLike
from dhgl.dataloading import MockWholeGraphDataLoader
from dhgl.type import CEType
from dhgl.utils import gdata as gdata_utils

from .base import HGNNReturnT
from .config import NegativeSamplerConfig, TrainerConfig
from .dataset import LinkPredTaskDataset

HGraph = BaseHeteroGraphLike | list[DGLBlock]


class BatchData(NamedTuple):
    hg: DGLHeteroGraph
    positive_hg: DGLHeteroGraph
    negative_hg: DGLHeteroGraph


def _train_step(
    model,
    data_loader: DataLoader[BatchData],
    decoder: Callable[[HGraph, Mapping[str, Tensor]], Mapping[str, Tensor]],
    forward_fn: Callable[[HGraph, Mapping[str, Tensor]], Mapping[str, Tensor]],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    optimizer,
    scheduler=None,
    max_iters: int | None = None,
    grad_max_norm: float | None = None,
    grad_max_value: float | None = None,
):

    model.train()

    if len(data_loader) != 1:
        raise NotImplementedError
    for data in data_loader:
        data: BatchData
        embeddings = forward_fn(data.hg, H.ndata(data.hg, 'feat'))
        assert isinstance(embeddings, Mapping)

        pos_logits = decoder(data.positive_hg, embeddings)
        neg_logits = decoder(data.negative_hg, embeddings)
        if len(pos_logits) != len(neg_logits):
            # XXX: should use weighted sum instead, or concat before loss
            raise NotImplementedError
        optimizer.zero_grad()

        loss = 0.5 * (
            loss_fn(pos_logits, torch.ones_like(pos_logits)) +
            loss_fn(neg_logits, torch.zeros_like(neg_logits))
        )
        loss.backward()
        if grad_max_value is not None:
            nn.utils.clip_grad_value_(model.parameters(), grad_max_value)
        if grad_max_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

    assert max_iters is None
    return (
        pos_logits,
        neg_logits,
        data.positive_hg,
        data.negative_hg,
        loss.detach().item(),
    )


@torch.no_grad()
def _eval_step(
    model,
    data_loaders: list[DataLoader | MockWholeGraphDataLoader],
    decoder: Callable[[HGraph, Mapping[str, Tensor]], Mapping[str, Tensor]],
    forward_fn: Callable[[HGraph, Mapping[str, Tensor]], Mapping[str, Tensor]],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
):

    model.eval()

    cached_preds: dict[BaseHeteroGraphLike, Tensor] = {}

    def graph_eval_step(data: BatchData):

        embeddings = cached_preds.get(data.hg, None)
        if embeddings is None:
            embeddings = forward_fn(data.hg, H.ndata(data.hg, 'feat'))
            cached_preds[data.hg] = embeddings

        pos_logits = decoder(data.positive_hg, embeddings)
        neg_logits = decoder(data.negative_hg, embeddings)

        pos_w = len(pos_logits) / (len(pos_logits) + len(neg_logits))
        loss = (
            pos_w + loss_fn(pos_logits, torch.ones_like(pos_logits)) +
            (1 - pos_w) * loss_fn(neg_logits, torch.zeros_like(neg_logits))
        )
        # breakpoint()
        return (
            pos_logits,
            neg_logits,
            data.positive_hg,
            data.negative_hg,
            loss.detach().cpu().item(),
        )

    def tem():
        for data_loader in data_loaders:
            if len(data_loader) != 1:
                raise NotImplementedError
            for data in data_loader:
                yield graph_eval_step(data)

    return list(tem())


def _get_dataloaders(
    dataset: LinkPredDatasetLike,
    target_etypes: list[CEType],
    sampler_config: NegativeSamplerConfig,
):

    def get_dataloader(positive_hg: dgl.DGLHeteroGraph, negative_hg=None):
        d = LinkPredTaskDataset(target_etypes, positive_hg, negative_hg)
        return DataLoader(
            d,
            batch_size=len(d),
            collate_fn=lambda data: BatchData(dataset.graph, *data),
        )

    # neg_sampler = dgl.dataloading.negative_sampler.PerSourceUniform(1)
    # neg_sampler_val = NHopNegativeSampler(
    #     [dataset.graph, dataset.val_graph], dataset.target_etypes, n_hops=2,
    #     k=1
    # )
    data_loaders = {}
    train_hg = dataset.vanilla_graph.to(dataset.graph.device)
    data_loaders['train'] = get_dataloader(
        train_hg, sampler_config.get_sampler(dataset, 'train')
    )
    data_loaders['val'] = get_dataloader(
        dataset.val_graph, sampler_config.get_sampler(dataset, 'val')
    )

    if sampler_config.test != 'static':
        warnings.warn('Using a non static negative testing graph')
    data_loaders['test'] = get_dataloader(
        dataset.test_graph, dataset.neg_test_graph if sampler_config.test
        == 'static' else sampler_config.get_sampler(dataset, 'test')
    )
    return data_loaders


def _get_scheduler(config: TrainerConfig, hgnn_data: HGNNReturnT):
    if hgnn_data.get(
        'scheduler', None
    ) is not None and config.scheduler_config is not None:
        raise ValueError(
            'Found scheduler returned in hgnn_config while scheduler_config set.'
        )

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


def _set_cv(dataset: LinkPredDatasetLike, config: TrainerConfig):
    from sklearn.model_selection import KFold
    cv_conf = config.cv_config
    assert cv_conf is not None

    vanilla_data_dict = {
        etype: dataset.vanilla_graph.edges(etype=etype)
        for etype in dataset.vanilla_graph.canonical_etypes
    }
    train_data_dict = {
        etype: dataset.graph.edges(etype=etype)
        for etype in dataset.graph.canonical_etypes
    }
    val_data_dict = {
        etype: dataset.val_graph.edges(etype=etype)
        for etype in dataset.val_graph.canonical_etypes
    }

    kf = KFold(
        n_splits=cv_conf.num_folds, shuffle=True, random_state=cv_conf.seed
    )
    target_edges = {
        etype: (
            dataset.graph.adj_external(etype=etype) +
            dataset.val_graph.adj_external(etype=etype)
        ).coalesce()
        for etype in dataset.target_etypes
    }
    assert all((adj.values() == 1).all() for adj in target_edges.values())
    for etype, adj in target_edges.items():
        indices = adj.indices()
        train, val = list(kf.split(range(indices.shape[1])))[cv_conf.ith_fold]
        train_data_dict[etype] = tuple(indices[:, train])
        vanilla_data_dict[etype] = tuple(indices[:, train])
        val_data_dict[etype] = tuple(indices[:, val])
        if not hasattr(dataset, 'get_inverse_etype'):
            raise NotImplementedError
        etype_inv = dataset.get_inverse_etype(etype)
        train_data_dict[etype_inv] = tuple(indices[:, train])[::-1]
        vanilla_data_dict[etype_inv] = tuple(indices[:, train])[::-1]
        val_data_dict[etype_inv] = tuple(indices[:, val])[::-1]

    dataset._vanilla_graph = transforms.update_graph_structure(
        # XXX: vanilla graph is supposed to be immutable
        # Here this assumes the vanilla_graph is the getter of _vanilla_graph
        dataset.vanilla_graph,
        vanilla_data_dict
    )
    dataset.graph = transforms.update_graph_structure(
        dataset.graph, train_data_dict
    )
    dataset.val_graph = transforms.update_graph_structure(
        dataset.val_graph, val_data_dict
    )
    return dataset


def train(config: TrainerConfig):
    """Trainer for the MODEL"""

    dataset: LinkPredDatasetLike = config.dataset_config.load()

    # dataset = _resplit(dataset, config)
    if config.cv_config is not None:
        dataset = _set_cv(dataset, config)

    hgnn_data: HGNNReturnT = config.hgnn_config.init(dataset, config)
    decoder = config.decoder_config.init(dataset, hgnn_data['optimizer'])
    scheduler = _get_scheduler(config, hgnn_data)

    if hasattr(dataset, 'dense_etypes'):
        dataset.graph = gdata_utils.dense_adjs_to_gdata(
            dataset.graph, dense_etypes=[
                etype for etype in dataset.dense_etypes
                if etype in dataset.graph.canonical_etypes
            ]
        )
        dataset.graph = gdata_utils.to(dataset.graph, config.device)
        # import dhgl
        # print(dhgl.info(dataset.graph))

    dataset.graph = dataset.graph.to(config.device)
    dataset.val_graph = dataset.val_graph.to(config.device)
    dataset.test_graph = dataset.test_graph.to(config.device)
    if dataset.neg_val_graph is not None:
        dataset.neg_val_graph = dataset.neg_val_graph.to(config.device)
    if dataset.neg_test_graph is not None:
        dataset.neg_test_graph = dataset.neg_test_graph.to(config.device)
    if isinstance(decoder, nn.Module):
        decoder = decoder.to(config.device)

    data_loaders = _get_dataloaders(
        dataset, dataset.target_etypes, config.negative_sampler_config
    )
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
    if config.decoder_config.name == 'dist_mul':
        if (
            config.tracker_config.save_end
            or config.tracker_config.save_n_best > 0
            or config.tracker_config.epochs_per_checkpoint > 0
        ):
            raise NotImplementedError(
                'Decoder parameters should also be saved along with checkpoint'
            )
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
        if config.scheduler_config and config.scheduler_config.verbose:
            lrs = scheduler.get_last_lr()
            tracker.pbar.set_description(f'lr={lrs[0]:.2e}, ')
        *data['train'], losses['train'] = _train_step(
            model=hgnn_data['model'],
            data_loader=data_loaders['train'],
            decoder=decoder,
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
                _eval_step(
                    model=hgnn_data['model'],
                    data_loaders=(data_loaders['val'], data_loaders['test']),
                    decoder=decoder,
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

        if config.early_breaking is not None:
            cur = tracker.get_best_scalars(no_within_loop_warning=True)
            if config.early_breaking.whether_break(epoch, cur):
                break

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
