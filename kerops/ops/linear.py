from math import ceil

import torch
from triton import next_power_of_2

from ..kernels.linear import _LinBReLULinAdd, _LinBReLULinBackward, _ReLULinearAdd, _ReLULinearAddBackward
from ..settings import ConfigurableArg, configure


def fwd_warps(in_channels):
    return {16: 2, 32: 2, 64: 1, 128: 1}[in_channels]


def fwd_ilp(in_channels):
    return {16: 8, 32: 8, 64: 4, 128: 4}[in_channels]


def bwd_warps(in_channels):
    return {16: 4, 32: 8, 64: 8, 128: 8}[in_channels]


def bwd_dblock(in_channels):
    return {16: 16, 32: 32, 64: 32, 128: 32}[in_channels]


@configure(
    _num_warps=lambda weight: fwd_warps(weight.shape[0]),
    D_block=16,
    _ILP=lambda weight: fwd_ilp(weight.shape[0]),
)
def ReLULinearAdd(
    x,
    weight,
    add_other,
    *,
    _num_warps: ConfigurableArg,
    D_block: ConfigurableArg,
    _ILP: ConfigurableArg,
):
    in_channels = x.shape[1]
    out_channels = weight.shape[1]
    numel = x.numel()

    assert in_channels >= 16 and out_channels >= 16

    assert x.ndim == add_other.ndim == 5
    assert list(x.shape[2:]) == list(add_other.shape[2:])
    assert x.shape[0] == add_other.shape[0]
    assert add_other.shape[1] == out_channels
    assert in_channels == next_power_of_2(in_channels)
    assert out_channels == next_power_of_2(out_channels)
    assert list(weight.shape) == [in_channels, out_channels]
    assert x.dtype == weight.dtype == add_other.dtype == torch.float16
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert add_other.is_contiguous(memory_format=torch.channels_last_3d)

    numel_no_channels = numel // in_channels

    grid_size = ceil(numel_no_channels / (D_block * _ILP))

    bsize, _, H, W, D = x.shape
    output = torch.empty_like(add_other)

    _ReLULinearAdd[(grid_size,)](
        x,
        weight,
        add_other,
        output,
        numel_no_channels,
        in_channels,
        out_channels,
        D_block,
        _ILP,
    )

    return output


@configure(
    _num_warps=lambda weight: bwd_warps(weight.shape[0]),
    D_block=lambda weight: bwd_dblock(weight.shape[0]),
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


def LinBReLULinAdd(
    x,
    weight_up,
    weight_down,
    bias,
    add_other,
    *,
    _num_warps=2,
    D_block=16,
    _ILP=8,
):
    in_channels = x.shape[1]
    hidden_channels = weight_up.shape[1]
    numel = x.numel()

    assert in_channels >= 16
    assert in_channels * 2 == hidden_channels

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

    grid_size = ceil(numel_no_channels / (D_block * _ILP))

    bsize, _, H, W, D = x.shape
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
        _ILP,
    )

    return output


def LinBReLULinBackward(
    input,
    grad,
    weight_up,
    weight_down,
    bias,
    *,
    _num_warps=8,
    D_block=32,
    _ILP=16,
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
