from __future__ import annotations

import argparse
import gc
import os
import warnings

import naive_flow as nf
import torch
from psutil import Process
from pydantic import TypeAdapter
from tqdm import tqdm

from dhgl import hgget as H
from dhgl.type import NType
from dhgl.utils.precomputation.adj import row_normalized_adjs
from dhgl.utils.precomputation.feature_collector import (
    FeatureCollector,
    LabelFeatCollector,
    LowRankMatrix,
    _format_memory,
)
from dhgl.utils.precomputation.metagraph import MetaGraph

try:
    from .trainer.config import HeteroDatasetConfig, TrainerConfig
except ImportError as e:
    if 'attempted relateive import' not in str(e.msg):
        raise e
    from dhgl.script_utils import correct_module_path
    e.add_note(
        'You are trying to run the py file directly, please you '
        f'"python -m {correct_module_path(__file__)}" instead'
    )
    raise e

from .lib.precom_dataset import _dict_tranpose
from .models.SRGCN.config import PrecomputationConfig


def _filter_x(x: torch.Tensor, indices: torch.Tensor):
    if isinstance(x, LowRankMatrix):
        raise NotImplementedError
    x = x.to_sparse_csr()
    # CSR format faster to index
    # TODO: could return csr format in collector
    # Although converting seem very fast
    return torch.stack([x[i] for i in indices])


def feature_propagate(
    dataset_config: HeteroDatasetConfig,
    config: PrecomputationConfig,
    num_layers: int,
):
    hg, feats = dataset_config.load()
    if H.index(hg).shape[0] < hg.num_nodes(
        H.tgt_ntype(hg)
    ) and not config.drop_unlabeled:
        warnings.warn(
            'The graph has unlabeled target nodes, but filter-unlabeled is not set. '
            'Setting filter-labeled=true could significantly reduce the storage size.',
            UserWarning
        )

    mg = MetaGraph.from_hg(hg)
    mp_list = mg.metapaths(num_layers, dsttype=H.tgt_ntype(hg))

    if config.drop_unlabeled:
        indices = H.index(hg)
        raise NotImplementedError

    base_adjs = row_normalized_adjs(hg, hg.edata['weight'])
    collector = FeatureCollector(
        mg,
        base_adjs,
        cache_dir=config.store_dir,
        cache_idx_dtype=config.cache_idx_dtype,
        cache_val_dtype=config.cache_val_dtype,
        readonly=False,
        verbose=config.verbose,
    )

    ps = Process()
    with tqdm(total=len(mp_list)) as pbar:
        collector._pbar = pbar
        mps = [mp for mp in mp_list if not collector.has_cache(mp, feats)]
        pbar.update(len(mp_list) - len(mps))
        if len(mps) == 0:
            return

        pbar.write(
            'Collected features will be output to: '
            f'{config.store_dir}\n'
        )

        for mp_id, res in collector.precompute_features(
            mps,
            feats,
        ):
            mp = mps[mp_id]
            src_feat = feats[collector._get_srctype(mp)]
            mp_str = collector.cache_name(mp, src_feat)
            if config.drop_unlabeled:
                x = _filter_x(res, indices)
                collector._save_memmap(
                    os.path.join(config.store_dir, mp_str), x,
                    collector._feat_hash(None)
                )
            pbar.set_description(
                f'{mp_str} collected, '
                f'memory: {_format_memory(ps.memory_info().rss)}'
            )
            pbar.update(1)
            del res
        collector._pbar = None


def label_propagate(
    dataset_config: HeteroDatasetConfig,
    config: PrecomputationConfig,
    num_layers: int,
    batch_size: int,
    device: str = 'cuda',
):
    hg, _ = dataset_config.load()
    if H.index(hg).shape[0] < hg.num_nodes(
        H.tgt_ntype(hg)
    ) and not config.drop_unlabeled:
        warnings.warn(
            'The graph has unlabeled target nodes, but filter-unlabeled is not set. '
            'Setting filter-labeled=true could significantly reduce the storage size.',
            UserWarning
        )
    if config.drop_unlabeled:
        raise NotImplementedError
        # TODO: create a tmpdir in store_dir and use it as cache_dir.
        # Next, save the filtered feat in store_dir.

    train_label = H.label(hg, 'train')
    if not H.is_multi_label(hg):

        train_label = torch.sparse_coo_tensor(
            torch.stack([H.index(hg, 'train'),
                         train_label]), values=torch.ones(len(train_label)),
            size=(hg.num_nodes(H.tgt_ntype(hg)), H.n_classes(hg))
        )
        train_label = train_label.to_dense()

    mg = MetaGraph.from_hg(hg)
    mp_list = mg.metapaths(num_layers, dsttype=H.tgt_ntype(hg))
    mp_list = [mp for mp in mp_list if len(mp) > 1 and mp[0][0] == mp[-1][-1]]
    base_adjs = row_normalized_adjs(hg, hg.edata['weight'])

    collector = LabelFeatCollector(
        mg,
        base_adjs,
        cache_dir=config.store_dir,
        cache_idx_dtype=config.cache_idx_dtype,
        cache_val_dtype=config.cache_val_dtype,
        readonly=False,
        verbose=config.verbose,
    )

    if not all(map(collector.has_diag, mp_list)):
        assert batch_size is not None and batch_size > 0, f'{batch_size = }'
        collector.lpa_diag(mp_list, batch_size=batch_size, device=device)

    ps = Process()
    feats = {H.tgt_ntype(hg): train_label}
    with tqdm(total=len(mp_list)) as pbar:
        collector._pbar = pbar
        mps = [mp for mp in mp_list if not collector.has_cache(mp, feats)]
        pbar.update(len(mp_list) - len(mps))
        if len(mps) == 0:
            return

        pbar.write(
            'Collected features will be output to: '
            f'{config.store_dir}\n'
        )

        for mp_id, res in collector.precompute_features(
            mps,
            feats,
        ):
            mp = mps[mp_id]
            src_feat = feats[collector._get_srctype(mp)]
            mp_str = collector.cache_name(mp, src_feat)
            pbar.set_description(
                f'{mp_str} collected, '
                f'memory: {_format_memory(ps.memory_info().rss)}'
            )
            pbar.update(1)
            del res
        collector._pbar = None


def slot_feature_propagate(
    dataset_config: HeteroDatasetConfig,
    config: PrecomputationConfig,
    num_layers: int,
    only_on_slot: NType | None = None,
):
    if isinstance(dataset_config, dict):
        dataset_config = TypeAdapter(HeteroDatasetConfig
                                     ).validate_python(dataset_config)
    if isinstance(config, dict):
        config = TypeAdapter(PrecomputationConfig).validate_python(config)
    # hg, feats = dataset_config.load(verbose=only_on_slot and False)
    hg, feats = dataset_config.load(verbose=True)
    if H.index(hg).shape[0] < hg.num_nodes(
        H.tgt_ntype(hg)
    ) and not config.drop_unlabeled:
        warnings.warn(
            'The graph has unlabeled target nodes, but filter-unlabeled is not set. '
            'Setting filter-labeled=true could significantly reduce the storage size.',
            UserWarning
        )

    mg = MetaGraph.from_hg(hg)
    mp_list = mg.metapaths(num_layers, dsttype=H.tgt_ntype(hg))
    base_adjs = row_normalized_adjs(hg, hg.edata['weight'])
    assert isinstance(list(feats.values())[0], dict)
    if only_on_slot is not None:
        feats_t = _dict_tranpose(feats)
        slot_i = list(feats_t).index(only_on_slot)
        pbar0 = tqdm([(only_on_slot, feats_t[only_on_slot])], disable=True)
    else:
        slot_i = None
        pbar0 = tqdm(_dict_tranpose(feats).items())
    if config.drop_unlabeled:
        indices = H.index(hg)
        raise NotImplementedError
        # TODO: create a tmpdir in store_dir and use it as cache_dir.
        # Next, save the filtered feat in store_dir.

    collector = FeatureCollector(
        mg,
        base_adjs,
        cache_dir=config.store_dir,
        cache_idx_dtype=config.cache_idx_dtype,
        cache_val_dtype=config.cache_val_dtype,
        readonly=False,
        verbose=config.verbose,
    )
    for slot_ntype, slot_feats in pbar0:
        pbar0.write(
            'Collected features will be output to: '
            f'{config.store_dir}\n'
        )
        pbar0.set_description(f'Collecting for slot: {slot_ntype}')
        gc.collect()

        with tqdm(total=len(mp_list), position=slot_i) as pbar:
            collector._pbar = pbar
            mps = [
                mp for mp in mp_list
                if not collector.has_cache(mp, slot_feats)
            ]
            pbar.update(len(mp_list) - len(mps))
            if len(mps) == 0:
                continue

            ps = Process()
            for mp_id, res in collector.precompute_features(
                mps,
                slot_feats,
                # TODO: add cache size
            ):
                mp = mps[mp_id]
                src_feat = slot_feats[collector._get_srctype(mp)]
                mp_str = collector.cache_name(mp, src_feat)
                if config.drop_unlabeled:
                    raise NotImplementedError
                    x = _filter_x(res, indices)
                    collector._save_memmap(
                        os.path.join(config.store_dir, mp_str), x,
                        collector._feat_hash(src_feat)
                    )
                pbar.set_description(
                    f'Slot: {slot_ntype}, '
                    f'{mp_str} collected, '
                    f'memory: {_format_memory(ps.memory_info().rss)}'
                )
                pbar.update(1)
                del res
            collector._pbar = None
    pbar0.close()


def ls(
    hg,
    feats,
    config: PrecomputationConfig,
    num_layers: int,
):
    if H.index(hg).shape[0] < hg.num_nodes(
        H.tgt_ntype(hg)
    ) and not config.drop_unlabeled:
        warnings.warn(
            'The graph has unlabeled target nodes, but filter-unlabeled is not set. '
            'Setting filter-labeled=true could significantly reduce the storage size.',
            UserWarning
        )

    mg = MetaGraph.from_hg(hg)
    mp_list = mg.metapaths(num_layers, dsttype=H.tgt_ntype(hg))
    base_adjs = row_normalized_adjs(hg, hg.edata['weight'])
    collector = FeatureCollector(
        mg,
        base_adjs,
        cache_dir=config.store_dir,
        cache_idx_dtype=config.cache_idx_dtype,
        cache_val_dtype=config.cache_val_dtype,
        verbose=config.verbose,
    )

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    with open(f'mpls_{timestamp}.txt', 'w', encoding='utf8') as fout:
        if isinstance(list(feats.values())[0], dict):
            for slot_ntype, slot_feats in _dict_tranpose(feats).items():
                for f in collector.ls(mp_list, slot_feats):
                    print(f)
                    print(f, file=fout)
        else:
            for f in collector.ls(mp_list, feats):
                print(f)
                print(f, file=fout)
    print(f'Results dumped to "mpls_{timestamp}.txt"')
    return


def main():
    arg_parser = argparse.ArgumentParser(
        description=('Propagate all metapath-based features within K hops')
    )
    arg_parser.add_argument(
        'env',
        type=str,
        # required=True,
        help='path to the .env file to use as config',
    )
    arg_parser.add_argument(
        '--num-layers',
        '--num_layers',
        '-K',
        type=int,
        required=True,
        help='number of hops',
    )
    arg_parser.add_argument(
        '--check',
        action='store_true',
    )
    arg_parser.add_argument(
        '--slot',
        type=str,
        required=False,
        help='used to manually replace multiprocessing',
    )
    arg_parser.add_argument(
        '--lpa',
        action='store_true',
        help='Propagate label feats',
    )
    arg_parser.add_argument(
        '--ls',
        action='store_true',
    )
    arg_parser.add_argument(
        '--lpa-batch-size',
        type=int,
        required=False,
    )
    # arg_parser.add_argument(
    #     '--filter-unlabeled', action='store_true',
    #     help='Filter unlabeld rows and only store labeled rows'
    # )
    args = arg_parser.parse_args()
    env_path = args.env
    assert os.path.isfile(env_path), 'No env file found'

    data = nf.load_env_file(
        env_path, preset_env_vars={
            '__file__': env_path,
            '__dir__': os.path.dirname(env_path)
        }
    )
    config = TrainerConfig.model_validate_strings(data)
    print(nf.strfconfig(config))
    # TODO: move precomputation_config to TrainerConfig
    if args.check:
        return

    if args.lpa:
        if args.ls:
            raise NotImplementedError
        # assert args.lpa_batch_size is not None
        label_propagate(
            config.dataset_config,
            config.hgnn_config.precomputation_config,
            args.num_layers,
            batch_size=args.lpa_batch_size,
            device='cuda',  #XXX
        )
        return
    hg, feats = config.dataset_config.load()

    if args.ls:
        ls(
            hg,
            feats,
            config.hgnn_config.precomputation_config,
            args.num_layers,
        )
        return
    if isinstance(list(feats.values())[0], dict):
        del hg
        del feats
        slot_feature_propagate(
            config.dataset_config,
            config.hgnn_config.precomputation_config,
            config.hgnn_config.num_layers,
            only_on_slot=args.slot,
        )
        return
    del hg
    del feats
    feature_propagate(
        config.dataset_config, config.hgnn_config.precomputation_config,
        config.hgnn_config.num_layers
    )
    return


if __name__ == '__main__':
    main()
