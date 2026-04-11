from math import floor, log2

import torch

from ..kernels.quantization import _DequantUint8Window_impl, _QuantUint8Window_impl
from ..settings import ConfArg, StaticKernelConfig, ConfiguredFunction
from ..utils import cdiv


quant_config = StaticKernelConfig(
    l1_cache_bytes=65536,
    num_warps=4
)


@ConfiguredFunction.configure(quant_config)
def QuantUint8Window(x, window, *, l1_cache_bytes: ConfArg, num_warps: ConfArg):
    numel = x.numel()
    MAX_SIZE = l1_cache_bytes // (2 * x.element_size())
    BLOCK_SIZE = min(MAX_SIZE, numel)
    BLOCK_SIZE = int(2 ** (floor(log2(BLOCK_SIZE))))

    output = torch.empty_like(x, dtype=torch.uint8)
    numblocks = cdiv(numel, BLOCK_SIZE)

    _QuantUint8Window_impl[(numblocks,)](x, output, numel, window, BLOCK_SIZE, num_warps=num_warps)
    return output


@ConfiguredFunction.configure(quant_config)
def DequantUint8Window(x, init_dtype, window, l1_cache_bytes: ConfArg, num_warps: ConfArg):
    numel = x.numel()
    output = torch.empty_like(x, dtype=init_dtype)

    MAX_SIZE = l1_cache_bytes // (2 * output.element_size())
    BLOCK_SIZE = min(MAX_SIZE, numel)
    BLOCK_SIZE = int(2 ** (floor(log2(BLOCK_SIZE))))

    numblocks = cdiv(numel, BLOCK_SIZE)

    _DequantUint8Window_impl[(numblocks,)](x, output, numel, window, BLOCK_SIZE, num_warps=num_warps)
    return output
