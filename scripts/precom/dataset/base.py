from __future__ import annotations

from pprint import pprint

from pydantic import Field

import dhgl
from dhgl import BaseHeteroGraphLike
from dhgl import hgget as H
from dhgl import transforms
from dhgl.script_utils.configs.dataset._feat_types import (
    BasicFeatType,
    FeatTypes,
    RandomFeatType,
)
from dhgl.script_utils.configs.dataset.hetero_dgl_dataset import (
    BaseDatasetConfig as Base,
)


class BaseDatasetConfig(Base):

    non_tgt_feat: None = None
    feat_types: FeatTypes[BasicFeatType | RandomFeatType] | None = None
    # prepropagation_config: PrepropagationConfig | None = None
    exclude_edge_types: list[str] | None = Field(
        None, deprecated='Use MetaGraph.remove_etype instead.'
    )

    def _process_feats(
        self,
        hg: BaseHeteroGraphLike,
        verbose: bool | None = None,
        **kwargs,
    ):
        assert self.non_tgt_feat is None

        hg = self.feat_types.apply_(hg)

        if self.prepropagation_config is not None:
            hg, feat = self.prepropagation_config.propagate(
                hg, return_features=True
            )
        else:
            feat = hg.ndata.pop(dhgl.FEAT)
        if self.exclude_edge_types:
            if len(self.exclude_edge_types) < len(hg.etypes):
                hg = transforms.remove_etypes(
                    hg,
                    self.exclude_edge_types,
                    unreachable_mode='none',
                )
            else:
                assert set(
                    map(hg.to_canonical_etype, self.exclude_edge_types)
                ) == set(hg.canonical_etypes)
                # XXX: This is an UNSAFE workaround, which assume later dhgl.add_self_loop will be called
                # (and the self-loop edge type follow the format {NTYPE}-self)
                hg = transforms.update_graph_structure(
                    hg, {
                        (
                            self._tgt_ntype, f'{self._tgt_ntype}-self', self._tgt_ntype
                        ): ([], [])
                    }, copy_ndata=True, copy_edata=True
                )
                for ntype in hg.ntypes:
                    if ntype == self._tgt_ntype:
                        continue
                    if dhgl.FEAT in hg.nodes[ntype].data:
                        feat.pop(ntype)
            # remove unreachable feat
            for ntype in transforms._get_unreachable_ntypes(
                hg.ntypes, hg.canonical_etypes, H.tgt_ntype(hg)
            ):
                feat.pop(ntype, None)

        if (verbose if verbose is not None else self.verbose):

            def get_dim(x):
                if isinstance(x, dict):
                    return sum(x_.shape[-1] for x_ in x.values())
                return x.shape[-1]

            print('Feature Dim: ', end='')
            pprint({ntype: get_dim(x) for ntype, x in feat.items()})
            print(dhgl.info(hg))
        return hg, feat
