from functools import reduce
from math import ceil, floor, log2

import torch
from triton import next_power_of_2

from ..kernels.addition import _AddStats_cl3d_backward_impl, _AddStats_cl3d_impl
from ._settings import settings_wrapper


@settings_wrapper
def AddStats(x, y, _l1_cache_bytes, _num_warps, inplace=False):
    num_channels = x.shape[1]
    numel = x.numel()
    assert x.shape == y.shape
    assert x.ndim == y.ndim == 5
    assert num_channels == next_power_of_2(num_channels)
    assert x.dtype == y.dtype == torch.float16
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert y.is_contiguous(memory_format=torch.channels_last_3d)

    MAX_SIZE = _l1_cache_bytes // x.element_size()  # 32768 for fp16
    numel_no_channels = reduce(lambda x, y: x * y, [s if idx != 1 else 1 for idx, s in enumerate(x.shape)], 1)
    other = min(MAX_SIZE // num_channels, numel_no_channels)
    other = int(2 ** (floor(log2(other))))
    BLOCK_SIZE = num_channels * other
    grid_size = ceil(numel / BLOCK_SIZE)

    if inplace:
        output = x
    else:
        output = torch.empty_like(x)
    mean = torch.zeros(num_channels, dtype=torch.float32, device=x.device)
    sqmean = torch.zeros(num_channels, dtype=torch.float32, device=x.device)

    _AddStats_cl3d_impl[(grid_size,)](
        x,
        y,
        output,
        mean,
        sqmean,
        numel,
        numel_no_channels,
        BLOCK_SIZE=BLOCK_SIZE,
        num_channels=num_channels,
        block_other=other,
        num_warps=_num_warps,
    )
    return output, mean, sqmean


@settings_wrapper
def AddStatsBackward(add_grad, mean_grad, sqmean_grad, add_result, _l1_cache_bytes, _num_warps):
    num_channels = add_grad.shape[1]
    numel = add_grad.numel()
    assert add_result.shape == add_grad.shape
    assert add_grad.ndim == add_result.ndim == 5
    assert num_channels == next_power_of_2(num_channels)
    assert add_grad.dtype == add_result.dtype == torch.float16
    assert mean_grad.numel() == sqmean_grad.numel() == num_channels
    assert add_grad.is_contiguous(memory_format=torch.channels_last_3d)
    assert add_result.is_contiguous(memory_format=torch.channels_last_3d)

    MAX_SIZE = _l1_cache_bytes // add_grad.element_size()  # 32768 for fp16
    numel_no_channels = reduce(lambda x, y: x * y, [s if idx != 1 else 1 for idx, s in enumerate(add_grad.shape)], 1)
    other = min(MAX_SIZE // num_channels, numel_no_channels)
    other = int(2 ** (floor(log2(other))))
    BLOCK_SIZE = num_channels * other
    grid_size = ceil(numel / BLOCK_SIZE)

    output_grad = torch.empty_like(add_grad)

    _AddStats_cl3d_backward_impl[(grid_size,)](
        add_grad,
        mean_grad,
        sqmean_grad,
        add_result,
        output_grad,
        numel,
        numel_no_channels,
        BLOCK_SIZE=BLOCK_SIZE,
        num_channels=num_channels,
        block_other=other,
        num_warps=_num_warps,
    )
    return output_grad
