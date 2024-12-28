from math import ceil

import torch
from triton import next_power_of_2

from ..kernels.linear import _ReLULinearAdd, _ReLULinearAddBackward
from ._settings import configure, ConfigurableArg


def configure_linear(in_channels):
    # num_warps, D_block, ILP
    HARDCODED_CONFIG = {
        16: [[2, 16, 8], [4, 16, 16]],
        32: [[2, 16, 8], [8, 32, 16]],
        64: [[1, 16, 4], [8, 32, 16]],
        128: [[1, 16, 4], [8, 32, 16]],
    }

    return HARDCODED_CONFIG.get(in_channels, None)


@configure(
    _num_warps=lambda weight: configure_linear(weight.shape[0])[0][0],
    D_block=lambda weight: configure_linear(weight.shape[0])[0][1],
    _ILP=lambda weight: configure_linear(weight.shape[0])[0][2],
)
def ReLULinearAdd(
    x,
    weight,
    add_other,
    *,
    _num_warps: ConfigurableArg=2,
    D_block: ConfigurableArg=16,
    _ILP: ConfigurableArg=8,
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
    _num_warps=lambda weight: configure_linear(weight.shape[0])[1][0],
    D_block=lambda weight: configure_linear(weight.shape[0])[1][1],
    _ILP=lambda weight: configure_linear(weight.shape[0])[1][2],
)
def ReLULinearBackward(
    input,
    grad,
    weight,
    *,
    _num_warps: ConfigurableArg=8,
    D_block: ConfigurableArg=32,
    _ILP: ConfigurableArg=16,
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
