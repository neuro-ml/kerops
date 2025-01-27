from math import ceil

import torch
from triton import next_power_of_2

from ...kernels.linear import _ReLULinearAdd
from ...settings import ConfigurableArg, confexc, configure


@confexc(KeyError)
def ilp(in_channels):
    return {16: 8, 32: 8, 64: 4, 128: 4}[in_channels]


@configure(
    num_warps=4,
    D_block=16,
    ILP=lambda x: ilp(x.shape[1]),
)
def ReLULinearAdd(
    x,
    weight,
    add_other,
    *,
    num_warps: ConfigurableArg,
    D_block: ConfigurableArg,
    ILP: ConfigurableArg,
):
    in_channels = x.shape[1]
    out_channels = add_other.shape[1]
    numel = x.numel()

    assert x.ndim == add_other.ndim == 5
    assert list(x.shape[2:]) == list(add_other.shape[2:])
    assert x.shape[0] == add_other.shape[0]
    assert in_channels == next_power_of_2(in_channels)
    assert out_channels == next_power_of_2(out_channels)
    assert list(weight.shape) == [in_channels, out_channels]
    assert x.dtype == weight.dtype == add_other.dtype == torch.float16
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert add_other.is_contiguous(memory_format=torch.channels_last_3d)

    numel_no_channels = numel // in_channels
    grid_size = ceil(numel_no_channels / (D_block * ILP))

    output = torch.empty_like(add_other)

    _ReLULinearAdd[(grid_size,)](
        x, weight, add_other, output, numel_no_channels, in_channels, out_channels, D_block, ILP, num_warps=num_warps
    )

    return output
