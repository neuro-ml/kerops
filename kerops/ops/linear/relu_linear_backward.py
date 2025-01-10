from math import ceil

import torch
from triton import next_power_of_2

from ..kernels.linear import _ReLULinearAddBackward
from ..settings import ConfigurableArg, configure


def warps(in_channels):
    return {16: 4, 32: 8, 64: 8, 128: 8}[in_channels]


def dblock(in_channels):
    return {16: 16, 32: 32, 64: 32, 128: 32}[in_channels]


@configure(
    _num_warps=lambda weight: warps(weight.shape[0]),
    D_block=lambda weight: dblock(weight.shape[0]),
    _ILP=16,
)
def ReLULinearBackward(
    input,
    grad,
    weight,
    *,
    _num_warps: ConfigurableArg,
    D_block: ConfigurableArg,
    _ILP: ConfigurableArg,
):
    in_channels = weight.shape[0]
    out_channels = grad.shape[1]
    numel = grad.numel()

    assert grad.ndim == input.ndim == 5
    assert list(grad.shape[2:]) == list(input.shape[2:])
    assert grad.shape[0] == input.shape[0]
    assert grad.shape[1] == out_channels
    assert input.shape[1] == in_channels
    assert in_channels == next_power_of_2(in_channels)
    assert out_channels == next_power_of_2(out_channels)
    assert list(weight.shape) == [in_channels, out_channels]
    assert grad.dtype == weight.dtype == torch.float16
    assert grad.is_contiguous(memory_format=torch.channels_last_3d)
    assert input.is_contiguous(memory_format=torch.channels_last_3d)

    numel_no_channels = numel // out_channels

    grid_size = ceil(numel_no_channels / (D_block * _ILP))

    bsize, _, H, W, D = grad.shape
    input_grad = torch.empty(
        [bsize, in_channels, H, W, D],
        dtype=grad.dtype,
        device=grad.device,
        memory_format=torch.channels_last_3d,
    )
    weight_grad = torch.zeros([grid_size, in_channels, out_channels], dtype=torch.float16, device='cuda')

    _ReLULinearAddBackward[(grid_size,)](
        input,
        grad,
        input_grad,
        weight,
        weight_grad,
        numel_no_channels,
        in_channels,
        out_channels,
        D_block,
        _ILP,
        num_warps=_num_warps,
    )

    return input_grad, weight_grad.sum(dim=0)
