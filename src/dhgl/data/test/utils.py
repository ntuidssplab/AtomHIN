import torch
from dgl import DGLHeteroGraph


def dgl_equal(g1: DGLHeteroGraph, g2: DGLHeteroGraph):

    assert set(g1.ntypes) == set(g2.ntypes)
    assert set(g2.etypes) == set(g2.etypes)

    for ntype in g1.ntypes:
        ndata1: dict = g1.nodes[ntype].data
        ndata2: dict = g2.nodes[ntype].data

        assert set(ndata1.keys()) == set(ndata2.keys())

        for data_key in ndata1.keys():

            print(f'Checking {ntype}:{data_key}', end='\t')

            data1, data2 = ndata1[data_key], ndata2[data_key]
            assert torch.is_tensor(data1)
            assert torch.is_tensor(data2)
            assert isinstance(data1, torch.Tensor)
            assert isinstance(data2, torch.Tensor)

            assert not (data1.is_sparse ^ data2.is_sparse)
            if data1.is_sparse:
                data1 = data1.to_dense()
                data2 = data2.to_dense()

            assert torch.equal(data1, data2)
            print('OK')

    return True
