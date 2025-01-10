from math import ceil

import torch
from triton import next_power_of_2

from ..kernels.linear import _LinBReLULinBackward
from ..settings import ConfigurableArg, configure


def ilp(channels):
    return {16: 8, 32: 9}[channels]


@configure(
    _num_warps=2,
    D_block=32,
    _ILP=lambda x: ilp(x.shape[1]),
)
def LinBReLULinBackward(
    input,
    grad,
    weight_up,
    weight_down,
    bias,
    *,
    _num_warps: ConfigurableArg,
    D_block: ConfigurableArg,
    _ILP: ConfigurableArg,
):
    in_channels = input.shape[1]
    hidden_channels = weight_up.shape[1]
    numel = grad.numel()

    assert grad.ndim == input.ndim == 5
    assert list(grad.shape) == list(input.shape)

    assert in_channels == next_power_of_2(in_channels)
    assert in_channels * 2 == hidden_channels
    assert list(weight_up.shape) == [in_channels, hidden_channels]
    assert list(weight_down.shape) == [hidden_channels, in_channels]
    assert list(bias.shape) == [hidden_channels]
    assert grad.dtype == input.dtype == weight_up.dtype == weight_down.dtype == bias.dtype == torch.float16
    assert grad.is_contiguous(memory_format=torch.channels_last_3d)
    assert input.is_contiguous(memory_format=torch.channels_last_3d)

    numel_no_channels = numel // in_channels

    grid_size = ceil(numel_no_channels / (D_block * _ILP))

    bsize, _, H, W, D = grad.shape
    input_grad = torch.empty(
        [bsize, in_channels, H, W, D],
        dtype=grad.dtype,
        device=grad.device,
        memory_format=torch.channels_last_3d,
    )
    weight_up_grad = torch.zeros([grid_size, in_channels, hidden_channels], dtype=torch.float32, device='cuda')
    weight_down_grad = torch.zeros([grid_size, hidden_channels, in_channels], dtype=torch.float32, device='cuda')
    bias_grad = torch.zeros([grid_size, hidden_channels], dtype=torch.float32, device='cuda')

    _LinBReLULinBackward[(grid_size,)](
        input,
        grad,
        input_grad,
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
        _ILP,
        num_warps=_num_warps,
    )

    return input_grad, weight_up_grad.sum(dim=0), weight_down_grad.sum(dim=0), bias_grad.sum(dim=0)
