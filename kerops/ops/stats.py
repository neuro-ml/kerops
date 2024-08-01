from functools import reduce
from math import ceil, floor, log2

import torch
from triton import next_power_of_2

from ..kernels.stats import _Stats_cl3d_backward_impl, _Stats_cl3d_impl
from ._settings import configure, get_l1_cache, ConfigurableArg


@configure(
    _l1_cache_bytes=lambda: get_l1_cache(),
    _num_warps=lambda: 4
)
def Stats(x, *, _l1_cache_bytes: ConfigurableArg, _num_warps: ConfigurableArg):
    num_channels = x.shape[1]
    numel = x.numel()
    assert x.ndim == 5
    assert num_channels == next_power_of_2(num_channels)
    assert x.dtype == torch.float16
    assert x.is_contiguous(memory_format=torch.channels_last_3d)

    MAX_SIZE = _l1_cache_bytes // x.element_size()  # 32768 for fp16
    numel_no_channels = reduce(lambda x, y: x * y, [s if idx != 1 else 1 for idx, s in enumerate(x.shape)], 1)
    other = min(MAX_SIZE // num_channels, numel_no_channels)
    other = int(2 ** (floor(log2(other))))
    BLOCK_SIZE = num_channels * other
    grid_size = ceil(numel / BLOCK_SIZE)

    mean = torch.zeros(num_channels, dtype=torch.float32, device=x.device)
    sqmean = torch.zeros(num_channels, dtype=torch.float32, device=x.device)

    _Stats_cl3d_impl[(grid_size,)](x, mean, sqmean, numel_no_channels, num_channels, other, num_warps=_num_warps)
    return mean, sqmean


@configure(
    _l1_cache_bytes=lambda: get_l1_cache(),
    _num_warps=lambda: 4
)
def StatsBackward(x, mean_grad, sqmean_grad, *, _l1_cache_bytes: ConfigurableArg, _num_warps: ConfigurableArg):
    num_channels = x.shape[1]
    numel = x.numel()
    assert x.ndim == 5
    assert num_channels == next_power_of_2(num_channels)
    assert x.dtype == torch.float16
    assert mean_grad.numel() == sqmean_grad.numel() == num_channels
    assert x.is_contiguous(memory_format=torch.channels_last_3d)

    MAX_SIZE = _l1_cache_bytes // x.element_size()  # 32768 for fp16
    numel_no_channels = reduce(lambda x, y: x * y, [s if idx != 1 else 1 for idx, s in enumerate(x.shape)], 1)
    other = min(MAX_SIZE // num_channels, numel_no_channels)
    other = int(2 ** (floor(log2(other))))
    BLOCK_SIZE = num_channels * other
    grid_size = ceil(numel / BLOCK_SIZE)

    output_grad = torch.empty_like(x)

    _Stats_cl3d_backward_impl[(grid_size,)](
        x,
        mean_grad,
        sqmean_grad,
        output_grad,
        numel_no_channels,
        num_channels=num_channels,
        block_other=other,
        num_warps=_num_warps,
    )
    return output_grad
