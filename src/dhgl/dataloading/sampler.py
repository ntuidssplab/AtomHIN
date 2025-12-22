from functools import lru_cache
import torch
import dgl
from dgl import backend as _F

from ..type import EType, NType

__all__ = [
    'NeighborSubgraphSampler',
    'NeighborSampler',
    'MultiLayerFullNeighborSubgraphSampler',
    'MultiLayerFullNeighborSampler',
]


@lru_cache(maxsize=1)
def cached_to_homogeneous(hg: dgl.DGLHeteroGraph):
    g = dgl.to_homogeneous(hg)
    return g


def fanout_is_zero(fanout: int | dict[NType, int]):
    if isinstance(fanout, int):
        return fanout == 0
    return all(f == 0 for f in fanout.values())


class NeighborSubgraphSampler(dgl.dataloading.Sampler):
    """This is a sampler similar to `dgl.dataloading.MultiLayerFullNeighborSampler`
    Whereas this sampler samples a subgraph instead of blocks(mfgs)
    """

    def __init__(self, fanouts: list[int] | list[dict[EType, int]], **_kwargs):
        super().__init__()
        self.fanouts = fanouts
        return

    def sample(self, g: dgl.DGLHeteroGraph, indices: dict[str, torch.Tensor]):
        sg = g
        output_nodes = indices
        for fanout in reversed(self.fanouts):

            if fanout_is_zero(fanout):
                continue
            sg = g.sample_neighbors(indices, fanout)
            eid = sg.edata[dgl.EID]
            block = dgl.to_block(sg, indices)
            block.edata[dgl.EID] = eid
            indices = block.srcdata[dgl.NID]
            input_nodes = indices

        csg = dgl.compact_graphs(sg)
        inv_mapping = {
            ntype:
            torch.empty(
                sg.num_nodes(ntype),
                dtype=csg.ndata[dgl.NID][ntype].dtype,
                device=csg.device,
            )
            for ntype in input_nodes
        }
        for ntype in inv_mapping:
            inv_mapping[ntype][
                csg.ndata[dgl.NID][ntype]
            ] = torch.arange(csg.num_nodes(ntype), device=csg.device)

        for ntype, nids in input_nodes.items():
            input_nodes[ntype] = inv_mapping[ntype][nids]
        for ntype, nids in output_nodes.items():
            output_nodes[ntype] = inv_mapping[ntype][nids]
        return input_nodes, output_nodes, csg


class MultiLayerFullNeighborSubgraphSampler(NeighborSubgraphSampler):
    """This is a sampler similar to `dgl.dataloading.MultiLayerFullNeighborSampler`
    Whereas this sampler samples a subgraph instead of blocks(mfgs)
    """

    def __init__(self, num_layers: int, **kwargs):
        super().__init__(fanouts=[-1] * num_layers, **kwargs)
        return


class NeighborSampler(dgl.dataloading.NeighborSampler):

    def __init__(
        self,
        fanouts,
        homo_block: bool = False,
        ref_hg: dgl.DGLHeteroGraph = None,
        **kwargs,
    ):
        super().__init__(
            fanouts,
            **kwargs,
        )
        self.homo_block = homo_block
        self.ref_hg = ref_hg
        return

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []

        if self.fused:
            # pylint: disable=no-member
            cpu = _F.device_type(g.device) == 'cpu'
            if isinstance(seed_nodes, dict):
                for ntype in list(seed_nodes.keys()):
                    if not cpu:
                        break
                    cpu = (
                        cpu
                        and _F.device_type(seed_nodes[ntype].device) == 'cpu'
                    )
            else:
                cpu = cpu and _F.device_type(seed_nodes.device) == 'cpu'
            if cpu and isinstance(
                g, dgl.DGLGraph
            ) and _F.backend_name == 'pytorch':
                raise NotImplementedError('fused==True is not supported')

        prev_fanout = None
        for fanout in reversed(self.fanouts):
            if not fanout_is_zero(fanout):
                # NOTE: if fanout is 0, the default behavior would create block without edges.
                # Here, fanout==0 stops the neighborhood sampling
                frontier = g.sample_neighbors(
                    seed_nodes,
                    fanout,
                    edge_dir=self.edge_dir,
                    prob=self.prob,
                    replace=self.replace,
                    output_device=self.output_device,
                    exclude_edges=exclude_eids,
                )
                eid = frontier.edata[dgl.EID]
            if (
                prev_fanout is not None and fanout_is_zero(prev_fanout)
                and fanout_is_zero(fanout)
            ):
                # avoid recreating same blocks
                prev_fanout = fanout
                blocks.insert(0, block)
                continue
            block = dgl.to_block(frontier, seed_nodes)
            block.edata[dgl.EID] = eid
            seed_nodes = block.srcdata[dgl.NID]
            prev_fanout = fanout
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks

    def sample_homo_blocks(
        self,
        g: dgl.DGLGraph,
        seed_nodes: dict[str, torch.Tensor],
        exclude_eids,
    ):

        hg = self.ref_hg

        def _hetero_ids_to_homo(hetero_nids: dict[NType, torch.Tensor]):
            ntype = list(hetero_nids)[0]
            ntype_id = hg.get_ntype_id(ntype)
            offset = sum(
                hg.num_nodes(ntype)
                for _, ntype in zip(range(ntype_id), hg.ntypes)
            )
            return hetero_nids[ntype] + offset

        input_nodes, output_nodes, blocks = super().sample(
            g, _hetero_ids_to_homo(seed_nodes), exclude_eids
        )
        output_device = input_nodes.device
        input_nodes = input_nodes.to(g.device)
        output_nodes = output_nodes.to(g.device)

        def homo_ids_to_hetero(homo_ids: torch.Tensor):
            ntype_ids = g.ndata[dgl.NTYPE][homo_ids]
            hetero_nids = g.ndata[dgl.NID][homo_ids]

            return {
                ntype: hetero_nids[ntype_ids == ntype_id]
                for ntype_id, ntype in enumerate(hg.ntypes)
            }

        in_hnids = homo_ids_to_hetero(input_nodes)

        forward_mapping = torch.concatenate(
            [
                torch.where(g.ndata[dgl.NTYPE][input_nodes] == ntype_id
                            )[0].to(blocks[0].device)
                for ntype_id in range(len(hg.ntypes))
            ]
        )
        in_hids_flat = torch.concatenate(list(in_hnids.values()))
        inv = torch.empty_like(in_hids_flat)
        inv[forward_mapping] = torch.arange(len(inv), device=inv.device)
        # assert torch.equal(in_hids_flat[inv], g.ndata[dgl.NID][input_nodes])
        # assert torch.equal(
        #     in_hids_flat, g.ndata[dgl.NID][input_nodes][forward_mapping]
        # )
        # assert torch.equal(
        #     input_nodes, blocks[0].srcdata[dgl.NID]
        # ), f'{blocks[0].srcdata[dgl.NID] = }'
        out_hnids = homo_ids_to_hetero(output_nodes)
        blocks[0].srcdata['homo->hetero'] = forward_mapping.to(output_device)
        blocks[0].srcdata['hetero->homo'] = inv.to(output_device)
        hblock = self._get_lazy_hetero_feature_block(
            in_hnids, out_hnids, output_device
        )
        in_hnids = {
            ntype: nids.to(output_device)
            for ntype, nids in in_hnids.items()
        }
        out_hnids = {
            ntype: nids.to(output_device)
            for ntype, nids in out_hnids.items()
        }
        return in_hnids, out_hnids, [hblock], blocks

    def _get_lazy_hetero_feature_block(
        self,
        input_nodes: dict[NType, torch.Tensor],
        output_nodes: dict[NType, torch.Tensor],
        output_device,
    ):
        assert self.ref_hg is not None
        hg = self.ref_hg
        hblock = dgl.create_block(
            {etype: ([], [])
             for etype in hg.canonical_etypes},
            num_src_nodes={
                ntype: len(hnids)
                for ntype, hnids in input_nodes.items()
            },
            num_dst_nodes={
                ntype: len(hnids)
                for ntype, hnids in output_nodes.items()
            },
            device=output_device,
        )
        assert all(
            key.startswith('_') for key in hg.edata
        ), 'currently only accept graph without edata'

        for ntype in hg.ntypes:
            # XXX
            src_feats = ['feat']
            dst_feats = ['label']
            hblock.srcnodes[ntype].data.update(
                {
                    k:
                    ndata[input_nodes[ntype].to(hg.device)].to(output_device)
                    for k, ndata in hg.nodes[ntype].data.items()
                    if k in src_feats
                }
            )
            hblock.dstnodes[ntype].data.update(
                {
                    k:
                    ndata[output_nodes[ntype].to(hg.device)].to(output_device)
                    for k, ndata in hg.nodes[ntype].data.items()
                    if k in dst_feats
                }
            )
        return hblock

    def sample(
        self,
        g: dgl.DGLHeteroGraph,
        seed_nodes: dict[str, torch.Tensor],
        exclude_eids=None,
    ):
        if self.homo_block is not True:
            return super().sample(g, seed_nodes, exclude_eids)

        return self.sample_homo_blocks(g, seed_nodes, exclude_eids)

    # def indices_dict_to_homo_indices(hg: dgl.DGLHeteroGraph, indices_dict):
    #     from itertools import accumulate
    #     id_offsets = list(
    #         accumulate(
    #             (hg.number_of_nodes(ntype) for ntype in hg.ntypes), initial=0
    #         )
    #     )
    #     indices_list = []
    #     for type_id, ntype in enumerate(hg.ntypes):
    #         if ntype in indices_dict:
    #             indices_list.append(indices_dict[ntype] + id_offsets[type_id])

    #     return torch.concat(indices_list)

    # def sample(
    #     self,
    #     g: dgl.DGLHeteroGraph,
    #     seed_nodes: dict[str, torch.Tensor],
    #     exclude_eids=None,
    # ):
    #     hg = g
    #     g = dgl.to_homogeneous(hg)
    #     from itertools import accumulate
    #     id_offsets = list(
    #         accumulate(
    #             (hg.number_of_nodes(ntype) for ntype in hg.ntypes), initial=0
    #         )
    #     )
    #     indices_list = []
    #     for type_id, ntype in enumerate(hg.ntypes):
    #         if ntype in seed_nodes:
    #             indices_list.append(seed_nodes[ntype] + id_offsets[type_id])

    #     # seed_nodes = indices_dict_to_homo_indices(hg, seed_nodes)
    #     indices = torch.concat(indices_list)
    #     input_nodes, output_nodes, blocks = super().sample(
    #         g, indices, exclude_eids=exclude_eids
    #     )

    #     def mapper():
    #         for ntype_id, ntype in enumerate(hg.ntypes):
    #             r = torch.stack(
    #                 [
    #                     torch.full(
    #                         (hg.num_nodes(ntype), ), ntype_id,
    #                         device=indices.device
    #                     ),
    #                     torch.arange(
    #                         hg.num_nodes(ntype), device=indices.device
    #                     ),
    #                 ]
    #             ).T
    #             yield r

    #     id_mapper = torch.concat(list(mapper()))

    #     # for ntype in hg.ntypes:
    #     #     blocks[0]
    #     torch.set_printoptions(threshold=1000000, linewidth=100000)
    #     print(blocks[0].ndata[dgl.NTYPE]['_N'].cpu())
    #     # print(blocks[0])
    #     # for i in range(len(hg.ntypes)):
    #     #     print((blocks[0].ndata[dgl.NTYPE]['_N'] == i).sum())
    #     raise
    #     return


class MultiLayerFullNeighborSampler(NeighborSampler):

    def __init__(
        self,
        num_layers: int,
        homo_block: bool = False,
        ref_hg: dgl.DGLGraph = None,
        **kwargs,
    ):
        """

        Args:
            num_layers (int)
            homo_block (bool, optional): Whether to sample homogenenous block.
                If ture, the sampler will return both hetero-blocks and homo-blocks.

        """
        super().__init__([-1] * num_layers, homo_block, ref_hg, **kwargs)
        return
