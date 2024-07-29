from functools import reduce
from math import ceil, floor, log2

import torch
from triton import next_power_of_2

from ..kernels.bnrelu import _ApplyBNReLU_cl3d_backward_impl, _ApplyBNReLU_cl3d_impl
from ._settings import settings_wrapper


@settings_wrapper
def ApplyBNReLU(x, weight, bias, _l1_cache_bytes, _num_warps):
    num_channels = x.shape[1]
    numel = x.numel()
    assert x.ndim == 5
    assert num_channels == next_power_of_2(num_channels)
    assert x.dtype == torch.float16
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert weight.numel() == bias.numel() == num_channels

    MAX_SIZE = _l1_cache_bytes // x.element_size()  # 32768 for fp16
    numel_no_channels = reduce(lambda x, y: x * y, [s if idx != 1 else 1 for idx, s in enumerate(x.shape)], 1)
    other = min(MAX_SIZE // num_channels, numel_no_channels)
    other = int(2 ** (floor(log2(other))))
    BLOCK_SIZE = num_channels * other
    grid_size = ceil(numel / BLOCK_SIZE)

    output = torch.empty_like(x)

    _ApplyBNReLU_cl3d_impl[(grid_size,)](
        x,
        output,
        weight,
        bias,
        numel_no_channels,
        BLOCK_SIZE=BLOCK_SIZE,
        num_channels=num_channels,
        block_other=other,
        num_warps=_num_warps,
    )
    return output


@settings_wrapper
def ApplyBNReLUBackward(x, weight, bias, grad, _l1_cache_bytes, _num_warps):
    num_channels = x.shape[1]
    numel = x.numel()
    assert x.ndim == 5
    assert x.shape == grad.shape
    assert num_channels == next_power_of_2(num_channels)
    assert x.dtype == grad.dtype == torch.float16
    assert weight.dtype == bias.dtype == torch.float32
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert grad.is_contiguous(memory_format=torch.channels_last_3d)
    assert weight.numel() == bias.numel() == num_channels

    MAX_SIZE = _l1_cache_bytes // x.element_size()  # 32768 for fp16
    numel_no_channels = reduce(lambda x, y: x * y, [s if idx != 1 else 1 for idx, s in enumerate(x.shape)], 1)
    other = min(MAX_SIZE // num_channels, numel_no_channels)
    other = int(2 ** (floor(log2(other))))
    BLOCK_SIZE = num_channels * other
    grid_size = ceil(numel / BLOCK_SIZE)

    outgrad = torch.empty_like(x)
    weight_grad = torch.zeros_like(weight)
    bias_grad = torch.zeros_like(bias)

    _ApplyBNReLU_cl3d_backward_impl[(grid_size,)](
        x,
        weight,
        bias,
        grad,
        outgrad,
        weight_grad,
        bias_grad,
        numel_no_channels,
        BLOCK_SIZE=BLOCK_SIZE,
        num_channels=num_channels,
        block_other=other,
        num_warps=_num_warps,
    )
    return outgrad, weight_grad, bias_grad
