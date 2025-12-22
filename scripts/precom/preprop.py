from __future__ import annotations

import argparse
import gc
import os

import naive_flow as nf
import psutil
from tqdm import tqdm

import dhgl
from scripts.train.dataset.base import _slot_prop

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


def _format_memory(bytes_val: int) -> str:
    GB = 1024**3
    MB = 1024**2
    KB = 1024

    if bytes_val >= GB:
        return f"{bytes_val / GB:.1f}GB"
    elif bytes_val >= MB:
        return f"{bytes_val / MB:.1f}MB"
    else:
        return f"{bytes_val / KB:.0f}KB"


# def _prop(
#     hg: dhgl.BaseHeteroGraphLike,
#     feats: dict[NType, torch.Tensor],
#     srctype: NType,
#     max_hops,
#     reduce_fn,
#     edge_weights,
# ):
#     """propagate node features to nodes without feature

#     Args:
#         hg (BaseHeteroGraphLike): The heterogeneous graph
#         max_hops (int | None, optional): Number of max hops to propagate. The propagation
#             terminates when either the max_hops reached or all nodes are filled with features.
#             Defaults to 99.
#         reduce_fn (Literal[&#39;mean&#39;, &#39;max&#39;], optional): either mean or max.
#             Setting mean will generally allow different ntypes to have same accumulated weights.
#             Setting max works effectively as union.
#             Defaults to 'mean'.
#         edge_weights (dict[EType, torch.Tensor] | None, optional): Required for weighted graphs.
#         Defaults to None.

#     Returns:
#         The graph with propagated features
#     """

#     max_hops = max_hops or 99

#     srcfeat = feats.pop(srctype)
#     assert not feats

#     with hg.local_scope():
#         if edge_weights is not None:
#             hg.edata['w'] = edge_weights
#         elif set(hg.edata) & {'w', 'weights', 'weight'}:
#             keys = set(hg.edata) & {'w', 'weights', 'weight'}
#             warnings.warn(
#                 f'edge_weights not set while edata[{keys}] detected.'
#             )

#         hs = {srctype: srcfeat}

#         def weighted_union(u, v):

#             def fn(nodes):
#                 indices = nodes.mailbox[u].abs().argmax(dim=1, keepdim=True)
#                 res = torch.take_along_dim(nodes.mailbox[u], indices, dim=1)
#                 return {v: res.squeeze(dim=1)}

#             return fn

#         def get_update_fns(etype):
#             message_fn = (
#                 fn.u_mul_e('x', 'w', 'm')
#                 if 'w' in hg.edges[etype].data else fn.copy_u('x', 'm')
#             )
#             reduce_fn_ = (fn.mean if reduce_fn == 'mean' else weighted_union)
#             return message_fn, reduce_fn_('m', 'h')

#         import psutil
#         ps = psutil.Process()
#         has_feat: dict[NType, bool] = {
#             ntype: ntype in hs
#             for ntype in hg.ntypes
#         }
#         for i in range(max_hops):
#             valid_etypes = [
#                 (s, e, d) for s, e, d, in hg.canonical_etypes
#                 if has_feat[s] and not has_feat[d]
#             ]
#             if len(valid_etypes) == 0:
#                 break
#             print('valid_etypes', valid_etypes)
#             hg.ndata['x'] = hs

#             for dsttype in hg.ntypes:
#                 valid_etypes_ = [
#                     (s, e, d) for s, e, d in hg.canonical_etypes
#                     if has_feat[s] and not has_feat[d] and d == dsttype
#                 ]
#                 if len(valid_etypes_) == 0:
#                     continue
#                 print(f'{valid_etypes_ = }')
#                 # TODO: Request src feat
#                 hg.multi_update_all(
#                     {etype: get_update_fns(etype)
#                      for etype in valid_etypes_}, cross_reducer=reduce_fn
#                 )
#                 has_feat[dsttype] = True
#                 yield dsttype, hg.ndata['h'][dsttype]
#                 # TODO: Clean src & dst feat
#                 print(i, 'memory:', ps.memory_info().rss / 1024**3, 'GB')
#     return


def prepropagate(config: HeteroDatasetConfig):
    assert config.prepropagation_config.mode == 'slot'
    assert config.prepropagation_config.cache_dir is not None, \
        'Cache directory must be specified for prepropagation'

    preprop_conf = config.prepropagation_config
    dataset_conf_obj = config.model_dump()
    dataset_conf_obj.pop('prepropagation_config')
    dataset_conf_obj.pop('exclude_edge_types')

    ps = psutil.Process()
    hg, feat = config.model_validate(dataset_conf_obj).load()
    hg.ndata[dhgl.FEAT] = feat
    cache_sub_dir = config.prepropagation_config._get_cache_subdir(
        hg,
        preprop_conf.max_hops,
        reduce_fn=preprop_conf.reduce_fn,
        edge_weights=hg.edata[dhgl.EWEIGHT],
        cache_dir=preprop_conf.cache_dir,
    )
    print(
        f'Prepropagating dataset {config.name} with cache dir: {cache_sub_dir}'
    )

    with tqdm(hg.ntypes, disable=not preprop_conf.verbose) as pbar:
        for srctype in pbar:
            pbar.set_description(
                f'Propagating feats for {srctype}, memory: {_format_memory(ps.memory_info().rss)}'
            )
            if dhgl.FEAT not in hg.nodes[srctype].data:
                continue
            gc.collect()
            _slot_prop(
                srctype,
                pbar,
                hg=hg,
                max_hops=preprop_conf.max_hops,
                reduce_fn=preprop_conf.reduce_fn,
                edge_weights=hg.edata[dhgl.EWEIGHT],
                cache_dir=cache_sub_dir,
                verbose=preprop_conf.verbose,
            )
    return


def main():
    arg_parser = argparse.ArgumentParser(
        description=(
            'Pre-preproagate the dataset for training, '
            'this is more memory-efficient than the one in training script.'
        )
    )
    arg_parser.add_argument(
        '-e',
        '--env',
        type=str,
        required=True,
        help='path to the .env file to use as config',
    )
    arg_parser.add_argument(
        '--check',
        action='store_true',
    )
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
    print(nf.strfconfig(config.dataset_config))
    if args.check:
        return
    prepropagate(config.dataset_config)
    return


if __name__ == '__main__':
    main()
