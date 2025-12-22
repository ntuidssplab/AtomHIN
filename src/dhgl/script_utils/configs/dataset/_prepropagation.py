from __future__ import annotations

import contextlib
import hashlib
import json
import os
from typing import Literal

import dgl
import torch
from packaging import version
from pydantic import Field, field_validator

import dhgl
from dhgl.schema.preprop import prepropagate
from dhgl.script_utils import BaseConfig

if version.parse(torch.__version__) >= version.parse('2.1.0'):
    from torch._tensor_str import printoptions
else:

    @contextlib.contextmanager
    def printoptions(*args, **kwargs):
        print_options = {
            'precision': torch._tensor_str.PRINT_OPTS.precision,
            'threshold': torch._tensor_str.PRINT_OPTS.threshold,
            'edgeitems': torch._tensor_str.PRINT_OPTS.edgeitems,
            'linewidth': torch._tensor_str.PRINT_OPTS.linewidth,
            'sci_mode': torch._tensor_str.PRINT_OPTS.sci_mode,
        }

        try:
            yield torch.set_printoptions(*args, **kwargs)
        finally:
            torch.set_printoptions(**print_options)


class PrepropagationConfig(BaseConfig):
    mode: Literal['slot'] = Field('slot', exclude=True)
    max_hops: int | None = None
    reduce_fn: Literal['mean', 'max', 'absmax'] = 'absmax'

    slots: list[str] | None = None
    """slots to perform slot-based prepropagation.
    If unset, propagation will applied to all ntypes with features."""
    cache_check_feathash: bool | None = None
    """whether to check the feathash in cache. Default to True.
    This may be useful to turn off in some rare cases where
        user want to experiment with different features.
    """
    cache_dir: str | None = None
    verbose: bool | None = None

    to_sparse_threshold: float | None = Field(None, gt=0, lt=1)

    @field_validator('slots', mode='before')
    @classmethod
    def load_list(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    def propagate(self, hg: dgl.DGLHeteroGraph, return_features: bool = False):
        cache_subdir = self._get_cache_subdir(
            hg,
            self.max_hops,
            self.reduce_fn,
            edge_weights=hg.edata[dhgl.EWEIGHT],
            cache_dir=self.cache_dir,
        )
        if self.slots is not None:
            raise NotImplementedError
        feats = prepropagate(
            hg,
            hg.ndata[dhgl.FEAT],
            edge_weights=hg.edata[dhgl.EWEIGHT],
            max_hops=self.max_hops,
            # TODO: slots
            reduce_fn=self.reduce_fn,
            cache_dir=cache_subdir,
            cache_check_feathash=self.cache_check_feathash,
            to_sparse_threshold=self.to_sparse_threshold,
            return_slots=False,
            verbose=self.verbose,
        )
        if return_features:
            hg.ndata.pop(dhgl.FEAT)
            return hg, feats
        for ntype, feat in feats.items():
            hg.nodes[ntype].data[dhgl.FEAT] = feat
        return hg

    @staticmethod
    def _get_cache_subdir(
        hg: dgl.DGLHeteroGraph,
        max_hops,
        reduce_fn,
        edge_weights,
        cache_dir: str,
    ) -> str | None:

        if cache_dir is None:
            return None
        cache_subdir = os.path.join(
            os.path.expanduser(cache_dir),
            PrepropagationConfig._slot_propagation_hash(
                hg, max_hops, reduce_fn, edge_weights
            )
        )
        os.makedirs(cache_subdir, exist_ok=True)
        return cache_subdir

    @staticmethod
    def _slot_propagation_hash(
        hg: dgl.DGLHeteroGraph, max_hops, reduce_fn, edge_weights=None
    ) -> str:
        with printoptions(profile='default'):
            sha1 = hashlib.sha1()
            sha1.update(str(hg).encode())
            sha1.update(str(edge_weights).encode())
            sha1.update(f'max_hops={max_hops}'.encode())
            sha1.update(f'reduce_fn={reduce_fn}'.encode())
            return sha1.hexdigest()
