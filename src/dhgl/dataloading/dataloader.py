import torch
from dgl import DGLHeteroGraph


class MockWholeGraphDataLoader:
    """This is the mock data loader that does not work like actual data loader
    but serve for interface alignment.
    """

    hg: DGLHeteroGraph
    indices: torch.Tensor

    def __init__(
        self,
        graph: DGLHeteroGraph,
        indices: dict,
        *_,
        **__,
    ):
        self.hg = graph
        assert len(indices) == 1
        self.indices = list(indices.values())[0]
        return

    def __len__(self):
        return 1
