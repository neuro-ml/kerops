from math import ceil

import torch
from triton import next_power_of_2

from ...kernels.linear import _LinBReLULinAdd
from ...settings import ConfigurableArg, confexc, configure


@confexc(KeyError)
def dblock(channels):
    # '64: 32' removed, fix bwd func
    return {16: 32, 32: 16}[channels]


@confexc(KeyError)
def ilp(channels):
    # '64: 8' removed, fix bwd func
    return {16: 1, 32: 2}[channels]


@configure(
    num_warps=4,
    D_block=lambda x: dblock(x.shape[1]),
    ILP=lambda x: ilp(x.shape[1]),
)
def LinBReLULinAdd(
    x,
    weight_up,
    weight_down,
    bias,
    add_other,
    *,
    num_warps: ConfigurableArg,
    D_block: ConfigurableArg,
    ILP: ConfigurableArg,
):
    in_channels = x.shape[1]
    hidden_channels = 2 * in_channels
    numel = x.numel()

    assert x.ndim == add_other.ndim == 5
    assert list(x.shape) == list(add_other.shape)
    assert in_channels == next_power_of_2(in_channels)
    assert list(weight_up.shape) == [in_channels, hidden_channels]
    assert list(weight_down.shape) == [hidden_channels, in_channels]
    assert list(bias.shape) == [hidden_channels]
    assert x.dtype == weight_up.dtype == weight_down.dtype == bias.dtype == add_other.dtype == torch.float16
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert add_other.is_contiguous(memory_format=torch.channels_last_3d)

    numel_no_channels = numel // in_channels
    grid_size = ceil(numel_no_channels / (D_block * ILP))

    output = torch.empty_like(x)

    _LinBReLULinAdd[(grid_size,)](
        x,
        weight_up,
        weight_down,
        bias,
        add_other,
        output,
        numel_no_channels,
        in_channels,
        hidden_channels,
        D_block,
        ILP,
        num_warps=num_warps,
    )

    return output
