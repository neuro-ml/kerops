from math import ceil

import torch
from triton import next_power_of_2

from ...kernels.linear import _LinBReLULinBackward
from ...settings import ConfigurableArg, confexc, configure


@confexc(KeyError)
def ilp(channels):
    return {16: 8, 32: 9}[channels]


@configure(
    num_warps=2,
    D_block=32,
    ILP=lambda x: ilp(x.shape[1]),
)
def LinBReLULinBackward(
    x,
    grad,
    weight_up,
    weight_down,
    bias,
    *,
    num_warps: ConfigurableArg,
    D_block: ConfigurableArg,
    ILP: ConfigurableArg,
):
    in_channels = x.shape[1]
    hidden_channels = 2 * in_channels
    numel = grad.numel()

    assert grad.ndim == x.ndim == 5
    assert list(x.shape) == list(grad.shape)
    assert in_channels == next_power_of_2(in_channels)
    assert list(weight_up.shape) == [in_channels, hidden_channels]
    assert list(weight_down.shape) == [hidden_channels, in_channels]
    assert list(bias.shape) == [hidden_channels]
    assert x.dtype == grad.dtype == weight_up.dtype == weight_down.dtype == bias.dtype == torch.float16
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert grad.is_contiguous(memory_format=torch.channels_last_3d)

    numel_no_channels = numel // in_channels
    grid_size = ceil(numel_no_channels / (D_block * ILP))

    x_grad = torch.empty_like(x)
    weight_up_grad = torch.zeros([grid_size, in_channels, hidden_channels], dtype=torch.float16, device='cuda')
    weight_down_grad = torch.zeros([grid_size, hidden_channels, in_channels], dtype=torch.float16, device='cuda')
    bias_grad = torch.zeros([grid_size, hidden_channels], dtype=torch.float16, device='cuda')

    _LinBReLULinBackward[(grid_size,)](
        x,
        grad,
        x_grad,
        weight_up,
        weight_down,
        bias,
        weight_up_grad,
        weight_down_grad,
        bias_grad,
        numel_no_channels,
        in_channels,
        hidden_channels,
        D_block,
        ILP,
        num_warps=num_warps,
    )

    return x_grad, weight_up_grad.sum(dim=0), weight_down_grad.sum(dim=0), bias_grad.sum(dim=0)
