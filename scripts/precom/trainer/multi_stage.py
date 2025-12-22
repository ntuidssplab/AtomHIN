from __future__ import annotations

import gc
import json
import os
import warnings
from typing import Literal, Mapping

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from tqdm import tqdm

import naive_flow as nf
from dhgl import hgget as H
from dhgl.type import Split

from ..lib.precom_config import PrecomDataset
from .base import BatchData, HGNNReturnT
from .config import TrainerConfig
from .data_utils import MemoryDataset, StackDataset, Subset
from .sampler import RandomChunkedBatchSampler, SequentialChunkedSampler
from .trainer import _train


def train_multi_stage(config: TrainerConfig):

    hg, feats = config.dataset_config.load()
    hgnn_data: HGNNReturnT = config.hgnn_config.init(hg, feats, config)

    indices = {split: H.index(hg, split) for split in ('train', 'val', 'test')}
    labels = {split: H.label(hg, split) for split in ('train', 'val', 'test')}
    all_labels = H.label(hg)

    label_dataset = None
    if isinstance(hgnn_data['dataset'], Mapping):
        label_dataset = hgnn_data['dataset']['label_feat']
        dataset = {
            key:
            MemoryDataset(
                d, device='cpu',
                verbose=config.data_loading_config.train.verbose
            )
            for key, d in hgnn_data['dataset'].items()
        }
        dataset = StackDataset(**dataset)
    else:
        dataset = MemoryDataset(
            hgnn_data['dataset'], device='cpu',
            verbose=config.data_loading_config.train.verbose
        )
    datasets = {
        split: StackDataset(idx, Subset(dataset, idx), labels[split])
        for split, idx in indices.items()
    }

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
            range(len(indices['train'])),
            batch_size=config.batch_config.train.batch_size,
            chunk_size=config.batch_config['train'].chunk_size,
            drop_last=True,
        ),
        num_workers=config.batch_config['train'].num_workers,
        pin_memory=config.batch_config['train'].pin_memory,
        persistent_workers=config.batch_config['train'].persistent_workers,
        collate_fn=BatchData.handle,
    )

    log_root_dir = nf.get_global().get(
        'log_root_dir', config.tracker_config.log_root_dir
    )
    if config.multi_stage_config.last_logits_path is not None:
        best_logits = torch.load(config.multi_stage_config.last_logits_path)
        metrics = []
        log_root_dir = nf.tracker.new_time_formatted_log_dir(
            config.tracker_config.comment, log_root_dir
        )
    else:
        tracker, best_logits = _train(
            config, hgnn_data, data_loaders, return_best_logits=True
        )
        log_root_dir = tracker.log_dir
        config = _update_multi_stage_config(
            tracker, best_logits, config, log_root_dir
        )
        metrics = [tracker.get_best_scalars(no_within_loop_warning=True)]
        metrics[0]['stage'] = 0
        del tracker
        if config.multi_stage_config.verbose:
            print(f'stage0: {metrics[0]}')

    for stage in range(config.multi_stage_config.num_stages - 1):
        enhanced_indices, enhanced_labels, pred_proba = _get_enhanced_indices(
            best_logits,
            indices,
            all_labels,
            config.multi_stage_config.threshold,
            config.multi_stage_config.verbose,
        )
        if len(enhanced_indices) == 0:
            warnings.warn('Not confident node found, aborted.')
            break

        if config.multi_stage_config.update_label_feats:
            dataset.datasets['label_feat'] = _update_label_feats_(
                label_dataset,
                pred_proba,
                threshold=(
                    config.multi_stage_config.update_label_feats if isinstance(
                        config.multi_stage_config.update_label_feats, float
                    ) else None
                ),
            )
        enhanced_dataset = StackDataset(
            enhanced_indices, Subset(dataset, enhanced_indices),
            enhanced_labels
        )
        train_batch_size = int(
            config.batch_config.train.batch_size * len(indices['train']) /
            (len(enhanced_indices) + len(indices['train']))
        )
        enhanced_batch_size = config.batch_config.train.batch_size - train_batch_size
        enhanced_loader = DataLoader(
            enhanced_dataset,
            batch_sampler=BatchSampler(
                RandomSampler(range(len(enhanced_indices))),
                batch_size=enhanced_batch_size,
                drop_last=True,
            ),
            # batch_sampler=RandomChunkedBatchSampler(
            #     range(len(enhanced_indices)),
            #     batch_size=enhanced_batch_size,
            #     chunk_size=config.batch_config['train'].chunk_size,
            #     drop_last=True,
            # ),
            num_workers=config.batch_config['train'].num_workers,
            pin_memory=config.batch_config['train'].pin_memory,
            persistent_workers=config.batch_config['train'].persistent_workers,
            collate_fn=BatchData.handle,
        )
        data_loaders['enhanced'] = enhanced_loader
        del hgnn_data['optimizer']
        del hgnn_data['model']
        del hgnn_data
        gc.collect()
        try:
            hgnn_data: HGNNReturnT = config.hgnn_config.init(
                hg, feats, config, require_dataset=False
            )
        except TypeError:
            hgnn_data: HGNNReturnT = config.hgnn_config.init(hg, feats, config)
        with nf.global_params(log_root_dir=log_root_dir):
            tracker, best_logits = _train_multi_stage_step(
                config, hgnn_data, data_loaders,
                pred_proba.max(1)[0]
            )
            metrics.append(tracker.get_best_scalars())
            metrics[-1]['stage'] = stage + 1
            config = _update_multi_stage_config(
                tracker, best_logits, config, log_root_dir=log_root_dir
            )
        del tracker
        if config.multi_stage_config.verbose:
            print(f'stage = {stage + 1},  {metrics[-1]}')
        if config.multi_stage_config.early_breaking is not None:
            if config.multi_stage_config.early_breaking.whether_break(
                stage, metrics[-1]
            ):
                break

    best = metrics[0]

    def better_than(new, cur):
        key = f'{config.evaluator_config.early_stopping_objective}/val'
        if 'loss' in config.evaluator_config.early_stopping_objective:
            return cur[key] < new[key]
        return new[key] > cur[key]

    for met in metrics:
        if better_than(met, best):
            best = met
    with open(
        os.path.join(log_root_dir, 'stage_results.json'), 'w', encoding='utf8'
    ) as fout:
        json.dump(metrics, fout, indent=4)
    return best


def _get_enhanced_indices(
    last_logits: dict[Split, torch.Tensor],
    indices: dict[Split, torch.Tensor],
    labels: torch.Tensor,
    threshold,
    verbose=False,
):

    last_pred = torch.zeros((len(labels), last_logits['test'].shape[1]))
    assert 'train' not in last_logits
    for split, logits in last_logits.items():
        last_pred[indices[split]] = logits.float()
    preds = last_pred.argmax(dim=-1)
    predict_prob = last_pred.softmax(dim=1)
    predict_prob[indices['train']] = torch.eye(predict_prob.shape[-1]
                                               )[labels[indices['train']]]
    val_mask = torch.zeros_like(preds).bool()
    val_mask[indices['val']] = True
    test_mask = torch.zeros_like(preds).bool()
    test_mask[indices['test']] = True
    confident_mask = predict_prob.max(1)[0] > threshold
    enhanced_indices = {
        'val': torch.nonzero(confident_mask & val_mask).squeeze(),
        'test': torch.nonzero(confident_mask & test_mask).squeeze(),
    }
    all_enhanced_indices = torch.cat(
        (enhanced_indices['val'], enhanced_indices['test'])
    )
    enhanced_labels = preds[all_enhanced_indices]
    if verbose:
        print(
            f'confident nodes: {len(all_enhanced_indices)} / {len(indices["train"])}'
            f' = {len(all_enhanced_indices) / len(indices["train"]):.2%}'
        )

        def print_confident_level(split: Literal['val', 'test']):
            val_confident_level = (
                predict_prob[enhanced_indices[split]].argmax(1)
                == labels[enhanced_indices[split]]
            ).sum() / len(enhanced_indices[split])
            confident_ratio = (
                val_confident_level * len(enhanced_indices[split]) /
                len(indices[split])
            )
            print(
                f'\t{split} confident nodes: {len(enhanced_indices[split])} / {len(indices[split])}, '
                f'{split} confident level: {val_confident_level:.2%} ({confident_ratio:.2%})'
            )

        print_confident_level('val')
        print_confident_level('test')
    return all_enhanced_indices, enhanced_labels, predict_prob


def _train_multi_stage_step(
    config: TrainerConfig,
    hgnn_data: HGNNReturnT,
    data_loaders: dict[Split, DataLoader],
    last_pred_prob: torch.Tensor,
    # indices: dict,
):
    train_loader = DataLoader(
        data_loaders['train'].dataset,
        batch_sampler=BatchSampler(
            RandomSampler(range(len(data_loaders['train'].dataset))),
            batch_size=(
                data_loaders['train'].batch_sampler.batch_size -
                data_loaders['enhanced'].batch_sampler.batch_size
            ),
            drop_last=True,
        ),
        # batch_sampler=RandomChunkedBatchSampler(
        #     range(len(data_loaders['train'].dataset)),
        #     batch_size=train_batch_size,
        #     chunk_size=train_batch_size,
        #     drop_last=True,
        # ),
        num_workers=config.batch_config['train'].num_workers,
        pin_memory=config.batch_config['train'].pin_memory,
        persistent_workers=config.batch_config['train'].persistent_workers,
        collate_fn=BatchData.handle,
    )

    class WrapLabel:

        def __init__(
            self, train_label: torch.Tensor, enhanced_label: torch.Tensor,
            enhanced_label_proba: torch.Tensor
        ):
            self.train_label = train_label
            self.enhanced_label = enhanced_label
            self.enhanced_label_proba = enhanced_label_proba
            return

        def to(self, device):
            self.train_label = self.train_label.to(device)
            self.enhanced_label = self.enhanced_label.to(device)
            self.enhanced_label_proba = self.enhanced_label_proba.to(device)
            return self

        def cpu(self):
            return torch.concat([self.train_label, self.enhanced_label])

    class MetaDataLoader:

        def __init__(self, train_loader, enhanced_loader):
            self.train_loader = train_loader
            self.enhanced_loader = enhanced_loader
            assert len(self)
            return

        def __len__(self):
            if len(self.train_loader) > len(self.enhanced_loader):
                warnings.warn(
                    f'Detected too few confident nodes: {len(self.enhanced_loader)}'
                )
            else:
                assert len(self.train_loader) == len(self.enhanced_loader)
            return len(self.train_loader)

        @classmethod
        def _feat_concat(
            cls, feat1: list[torch.Tensor], feat2: list[torch.Tensor]
        ):
            assert len(feat1) == len(feat2)
            return [
                torch.concatenate([x1, x2]) for x1, x2 in zip(feat1, feat2)
            ]

        def _infinite_enhanced_loader(self):
            # For the case len(enhanced_loader) < len(train_loader)
            while True:
                yield from self.enhanced_loader

        def __iter__(self):
            for train_batch, enhanced_batch in zip(
                self.train_loader, self._infinite_enhanced_loader()
            ):
                train_batch: BatchData
                enhanced_batch: BatchData
                if isinstance(train_batch.features, dict):
                    feats = {}
                    for key in train_batch.features:
                        feats[key] = self._feat_concat(
                            train_batch.features[key],
                            enhanced_batch.features[key]
                        )
                else:
                    feats = self._feat_concat(
                        train_batch.features, enhanced_batch.features
                    )
                yield BatchData(
                    torch.concat(
                        [
                            train_batch.batch_indices,
                            enhanced_batch.batch_indices
                        ]
                    ),
                    feats,
                    WrapLabel(
                        train_batch.labels, enhanced_batch.labels,
                        last_pred_prob[enhanced_batch.batch_indices]
                    ),
                )

    def loss_fn_wrap(loss_fn):
        enhanced_batch_size = data_loaders['enhanced'].batch_sampler.batch_size
        train_batch_size = config.batch_config.train.batch_size - enhanced_batch_size
        enhanced_ratio = enhanced_batch_size / config.batch_config.train.batch_size

        def loss_fn_(logits, labels: WrapLabel):
            if not isinstance(labels, WrapLabel):
                return loss_fn(logits, labels)
            loss = loss_fn(logits[:train_batch_size], labels.train_label)
            loss_enhanced = loss_fn(
                logits[train_batch_size:], labels.enhanced_label,
                reduction='none'
            )
            assert enhanced_batch_size == len(labels.enhanced_label)
            loss_enhanced = (loss_enhanced *
                             labels.enhanced_label_proba).mean()
            return loss * (
                1 - enhanced_ratio
            ) + config.multi_stage_config.gamma * enhanced_ratio * loss_enhanced

        return loss_fn_

    hgnn_data = hgnn_data.copy()
    hgnn_data['loss_fn'] = loss_fn_wrap(
        hgnn_data.get('loss_fn', config.loss_fn)
    )
    data_loaders = {
        'train': MetaDataLoader(train_loader, data_loaders['enhanced']),
        'val': data_loaders['val'],
        'test': data_loaders['test'],
    }
    return _train(config, hgnn_data, data_loaders, return_best_logits=True)


def _update_multi_stage_config(
    tracker: nf.tracker.BaseTracker,
    best_logits: dict[Split, torch.Tensor],
    config: TrainerConfig,
    log_root_dir: str,
):
    torch.save(best_logits, os.path.join(tracker.log_dir, 'best_logits.pt'))
    with open(
        os.path.join(tracker.log_dir, 'results.json'), 'w', encoding='utf8'
    ) as fout:
        json.dump(
            tracker.get_best_scalars(no_within_loop_warning=True), fout,
            indent=4
        )
    config_data = config.model_dump()
    config_data['tracker_config']['log_root_dir'] = log_root_dir
    config_data['multi_stage_config']['last_logits_path'] = os.path.join(
        tracker.log_dir, 'best_logits.pt'
    )
    if config.multi_stage_config.update_configs is not None:
        for key, values in config.multi_stage_config.update_configs.items():
            values: list
            if len(values) == 0:
                continue
            d = config_data
            for k in key.split('__')[:-1]:
                d = d[k]
            d[key.split('__')[-1]] = values.pop(0)

            config_data['multi_stage_config']['update_configs'][key] = values

    if config.multi_stage_config.last_logits_path is not None:
        # First round does not count
        config_data['multi_stage_config'][
            'num_stages'] = config.multi_stage_config.num_stages - 1

    config = config.model_validate(config_data)
    return config


def _update_label_feats_(
    label_dataset: PrecomDataset, enhanced_proba: torch.Tensor,
    threshold: float | None = None
):
    from ..lib.precom_config import LabelFeatCollector

    collector = LabelFeatCollector(
        label_dataset.collector.mg,
        label_dataset.collector.adjs,
        custom_ntype_alias=label_dataset.collector._ntype_alias,
        cache_dir=label_dataset.collector.cache_dir,
        cache_idx_dtype=label_dataset.collector.cache_idx_dtype,
        cache_val_dtype=label_dataset.collector.cache_val_dtype,
        verbose=label_dataset.collector.verbose,
        readonly=True,
    )
    tgt_ntype = {mp[-1][-1] for mp in label_dataset.required_mps}
    tgt_ntype = list(tgt_ntype)[0]
    if threshold is not None:
        enhanced_proba = enhanced_proba.clone()
        enhanced_proba[enhanced_proba < threshold] = 0.
    label_dataset.feats = {tgt_ntype: enhanced_proba}
    label_dataset.collector = collector
    list(
        tqdm(
            collector.precompute_features(
                label_dataset.required_mps, label_dataset.feats
            ),
            total=len(label_dataset.required_mps),
            disable=not label_dataset.collector.verbose,
        )
    )
    label_dataset = MemoryDataset(label_dataset, device='cpu')
    collector._cache.set_capacity(0)
    return label_dataset
