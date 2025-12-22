import warnings
from typing import Literal, Sequence
import torch
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dgl.dataloading import DataLoader
from dgl.heterograph import DGLBlock
from dgl import DGLHeteroGraph
import dgl
from ... import dataloading
from ... import hgget as H
from ...type import EType


class _BaseBatchConfig(BaseSettings):

    model_config = SettingsConfigDict(env_nested_delimiter='__', frozen=True)

    name: str

    _shuffle: bool

    _drop_last: bool

    _mini_batch_mode: bool = True

    device_mode: Literal['uva', 'gpu'] = Field(exclude=True)

    batch_size: int

    def _init_sampler(self, num_layers: int, **kwargs):
        raise NotImplementedError

    def data_loader(
        self, hg: DGLHeteroGraph, indices: torch.Tensor, n_hops: int
    ) -> Sequence[tuple[torch.Tensor, torch.Tensor, list[DGLBlock]]]:

        if self.device_mode == 'gpu':
            assert hg.device == torch.device('cuda:0'), (
                '"gpu" is a recommended mode and it requires user to put the graph to cuda '
                'beforehand'
            )

        if not self.is_in_batch_mode:
            return dataloading.MockWholeGraphDataLoader(
                hg,
                {H.tgt_ntype(hg): indices.cuda()},
            )

        sampler = self._init_sampler(n_hops, ref_hg=hg)  # pylint: disable=no-member
        graph = dgl.to_homogeneous(hg) if isinstance(
            sampler, dataloading.NeighborSampler
        ) and sampler.homo_block else hg
        return DataLoader(
            graph,
            {H.tgt_ntype(hg): indices.cuda()},
            sampler,
            device='cuda:0',
            batch_size=self.batch_size,
            shuffle=self._shuffle,
            drop_last=self._drop_last,
            use_uva=self.device_mode == 'uva',
            num_workers=0,
        )

    @property
    def is_in_batch_mode(self):
        return self._mini_batch_mode


class _BaseNeighborSamplerConfig(_BaseBatchConfig):

    fanouts: dict[EType, list[int]] | list[dict[EType, int]
                                           ] | list[int] | None = Field(None)
    """fanouts passed to samplers. The length of fanouts must match the num_layers of GNN models.

    If 

    """

    def _init_sampler(self, num_layers: int, **kwargs):
        raise NotImplementedError

    @model_validator(mode='before')
    @classmethod
    def _preprocess_fanouts(cls, data: dict[str, str]):
        # pylint: disable=import-outside-toplevel
        fanouts = data.get('fanouts', None)
        if fanouts is None:
            return data
        import json
        if isinstance(fanouts, dict):
            import numpy as np
            fanouts = {etype: json.loads(v) for etype, v in fanouts.items()}
            lengths = list(map(len, fanouts.values()))
            assert all(
                length == lengths[0] for length in lengths
            ), (f'expect all fanouts have equal length, but got {fanouts = }')
            fanouts_mat = np.array(list(fanouts.values()))
            fanouts = [
                {
                    etype: fanouts_mat[i, l]
                    for i, etype in enumerate(fanouts)
                } for l in range(lengths[0])
            ]
            data['fanouts'] = fanouts
        if isinstance(fanouts, str):
            data['fanouts'] = json.loads(fanouts)
        return data

    @model_validator(mode='after')
    def _check(self):
        if 'full' in self.name:
            #pylint: disable=not-an-iterable
            assert self.fanouts is None or all(
                fo == -1 for fo in self.fanouts
            ), (
                f'the name is set as {self.name} while the fanouts is set {self.fanouts}'
            )
        else:
            assert self.fanouts is not None, (
                f'the name is set as {self.name} while the fanouts is set {self.fanouts}'
            )
        return self


# def _assert_fanouts_by_etype(
#     fanouts: list[dict[EType, int]] | list[int] | None
# ):
#     assert fanouts is not None
#     assert fanouts and isinstance(fanouts[0], dict), (
#         'fanouts is expected to set by edge type for heterogeneous subgraph/blocks, '
#         f'while got {fanouts = }. '
#         # 'One can set use "__all__" as key for all unspecified edge types E.g. {"__all__": 10}'
#     )


class _BaseBlockSamplerConfig(_BaseNeighborSamplerConfig):

    name: Literal['full', 'full_block', 'block'] = 'full_block'

    block_type: Literal['homo', 'hetero'] = 'hetero'

    def _init_sampler(self, num_layers: int, **kwargs):
        fanouts = self.fanouts or [-1] * num_layers
        assert len(fanouts) == num_layers, (
            f'the {fanouts = }, while {num_layers = }'
        )
        return dataloading.NeighborSampler(
            fanouts, homo_block=(self.block_type == 'homo'), **kwargs
        )

    @model_validator(mode='before')
    @classmethod
    def _rename(cls, data: dict[str, str]):
        if data['name'] == 'full':
            warnings.warn(
                'Used of name "full" is ambiguous and has been deprecated, '
                'used "full_block" instead'
            )
            data['name'] = 'full_block'
        return data

    @model_validator(mode='after')
    def _check(self):
        if self.block_type == 'hetero':
            # _assert_fanouts_by_etype(self.fanouts)
            pass
        else:
            assert self.block_type == 'homo'
            assert self.fanouts and isinstance(self.fanouts[0], int), (
                f'The block_type is set as "homo" while fanouts = {self.fanouts}'
            )
        return self


class _TrainBlockBatchConfig(_BaseBlockSamplerConfig):
    _shuffle: bool = True
    _drop_last: bool = True


class _EvalBlockBatchConfig(_BaseBlockSamplerConfig):
    _shuffle: bool = False
    _drop_last: bool = False


class _BaseSubgraphSamplerConfig(_BaseNeighborSamplerConfig):

    name: Literal['full_subgraph', 'subgraph'] = 'full_subgraph'

    def _init_sampler(self, num_layers: int, **kwargs):
        fanouts = self.fanouts or [-1] * num_layers
        assert len(fanouts) == num_layers, (
            f'the {fanouts = }, while {num_layers = }'
        )
        return dataloading.NeighborSubgraphSampler(fanouts, **kwargs)

    # @model_validator(mode='after')
    # def _check(self):
    #     _assert_fanouts_by_etype(self.fanouts)
    #     return self


class _TrainSubgraphBatchConfig(_BaseSubgraphSamplerConfig):
    _shuffle: bool = True
    _drop_last: bool = True


class _EvalSubgraphBatchConfig(_BaseSubgraphSamplerConfig):
    _shuffle: bool = False
    _drop_last: bool = False


class _DummyWholeGraphSamplerConfig(_BaseBatchConfig):
    """This serves as an interface alignment for whole graph
    learning (without mini-batch)
    """
    model_config = SettingsConfigDict(env_nested_delimiter='__', frozen=True)

    name: Literal['dummy_whole_graph', 'whole_graph'] = 'whole_graph'

    _mini_batch_mode: bool = False
    # _shuffle: bool
    # _drop_last: bool

    device_mode: Literal['gpu'] = Field(exclude=True)

    batch_size: Literal[0, '0'] = Field(0, exclude=True)

    def _init_sampler(self, num_layers: int, **kwargs):
        raise NotImplementedError


_TrainBatchConfigs = _TrainBlockBatchConfig\
        | _TrainSubgraphBatchConfig | _DummyWholeGraphSamplerConfig
_EvalBatchConfigs = _EvalBlockBatchConfig\
        | _EvalSubgraphBatchConfig | _DummyWholeGraphSamplerConfig


class BatchConfig(BaseSettings):
    model_config = SettingsConfigDict(env_nested_delimiter='__', frozen=True)

    train: _TrainBatchConfigs = Field(discriminator='name')
    eval: _EvalBatchConfigs = Field(discriminator='name')

    device_mode: Literal['uva', 'gpu'] = 'gpu'
    """
    TLDR: Try "gpu" first. If out-of-memory, try "uva".

    Consider use "gpu" first if memory consumption is not an issue.
    Use "uva" instead if out-of-memory occurred, and notice that do not put graph and data to cuda
    beforehand (since the data will be copied to cuda by data loader)
    """

    @model_validator(mode='before')
    @classmethod
    def propagate_device_mode(cls, data: dict[str, str]):

        device_mode = data.get('device_mode', 'gpu')
        assert 'train' in data, 'Field "train" is missing'
        assert 'eval' in data, 'Field "eval" is missing'
        if 'device_mode' not in data['train']:
            data['train']['device_mode'] = device_mode
        if 'device_mode' not in data['eval']:
            data['eval']['device_mode'] = device_mode
        return data
