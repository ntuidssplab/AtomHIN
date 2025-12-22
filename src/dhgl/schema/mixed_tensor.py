from __future__ import annotations

import torch


def linear(input: MixedTensor, weight: torch.Tensor, bias=None):
    if input._dim != 1:
        return NotImplemented
    dim = input._value.shape[input._dim]
    r1 = torch.nn.functional.linear(input._value, weight[:, :dim], bias)
    r2 = torch.nn.functional.linear(input._value2, weight[:, dim:], None)
    return r1 + r2


HANDLED_FUNCTIONS = {torch.nn.functional.linear: linear}


class MixedTensor:
    """A wrapper to combine dense and sparse tensors without concatenating them.

    This class enables handling of mixed-format features (dense + sparse) as a
    single tensor-like object. It is primarily designed for use in models where
    the first operation is a linear projection, e.g. ``torch.nn.Linear``.

    With this design, existing models do not need to be modified to accept
    mixed-format inputs, as long as the first layer is linear. Currently,
    only ``torch.nn.Linear`` is supported.

    In the linear case, the operation

        y = x A^T + b

    decomposes into

        y = x_dense A_1^T + x_sparse A_2^T + b

    where ``x_dense`` and ``x_sparse`` are the respective dense and sparse
    feature components, and ``A_1``, ``A_2`` are partitions of the weight
    matrix corresponding to each component.

    Note:
        This is an experimental and tricky solution. Support is limited to
        linear projections and may not generalize to other operators.
    """

    def __init__(self, value: torch.Tensor, value2: torch.Tensor, dim):
        self._value = value
        self._value2 = value2
        self._dim = dim
        sum_dim = value.shape[dim] + value2.shape[dim]
        self._shape = torch.Size(
            [sum_dim if i == dim else d for i, d in enumerate(value.shape)]
        )

    def __repr__(self):
        return 'MixedTensor(value1={}, value2={})'.format(
            self._value, self._value2
        )

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._value.dtype

    @property
    def device(self):
        return self._value.device

    def to_dense(self, *args, **kwargs):
        return torch.concatenate(
            [
                self._value.to_dense(*args, **kwargs),
                self._value2.to_dense(*args, **kwargs)
            ], dim=self._dim
        )

    def cuda(self, *args, **kwargs):
        self._value = self._value.cuda(*args, **kwargs)
        self._value2 = self._value2.cuda(*args, **kwargs)
        return self

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func in HANDLED_FUNCTIONS:
            return HANDLED_FUNCTIONS[func](*args, **kwargs)
        return NotImplemented
        # print(f'{func} called')
        # args = [a._value if isinstance(a, MixedTensor) else a for a in args]
        # return func(*args, **kwargs)
