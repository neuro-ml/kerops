from math import ceil, floor, log2

import torch

from ..kernels.quantization import _DequantUint8Window_impl, _QuantUint8Window_impl
from ._settings import ConfigurableArg, configure, get_l1_cache


@configure(_num_warps=4, _l1_cache_bytes=get_l1_cache)
def QuantUint8Window(x, window, *, _num_warps: ConfigurableArg, _l1_cache_bytes: ConfigurableArg):
    numel = x.numel()
    MAX_SIZE = _l1_cache_bytes // (2 * x.element_size())
    BLOCK_SIZE = min(MAX_SIZE, numel)
    BLOCK_SIZE = int(2 ** (floor(log2(BLOCK_SIZE))))

    output = torch.empty_like(x, dtype=torch.uint8)
    numblocks = ceil(numel / BLOCK_SIZE)

    _QuantUint8Window_impl[(numblocks,)](x, output, numel, window, BLOCK_SIZE, num_warps=_num_warps)
    return output


@configure(_num_warps=4, _l1_cache_bytes=get_l1_cache)
def DequantUint8Window(x, init_dtype, window, _num_warps: ConfigurableArg, _l1_cache_bytes: ConfigurableArg):
    numel = x.numel()
    output = torch.empty_like(x, dtype=init_dtype)

    MAX_SIZE = _l1_cache_bytes // (2 * output.element_size())
    BLOCK_SIZE = min(MAX_SIZE, numel)
    BLOCK_SIZE = int(2 ** (floor(log2(BLOCK_SIZE))))

    numblocks = ceil(numel / BLOCK_SIZE)

    _DequantUint8Window_impl[(numblocks,)](x, output, numel, window, BLOCK_SIZE, num_warps=_num_warps)
    return output
