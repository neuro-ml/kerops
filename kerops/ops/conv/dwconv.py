from math import ceil

import torch
from triton import language as tl, next_power_of_2

from ...kernels.dw_conv import _DWConv_cl3d_impl
from ...settings import ConfigurableArg, confexc, configure


@confexc(KeyError)
def warps(channels):
    return {8: 1, 16: 2, 32: 2, 64: 2, 128: 4}[channels]


@confexc(KeyError)
def dblock(channels):
    return {8: 32, 16: 32, 32: 16, 64: 8, 128: 8}[channels]


@configure(
    ACCTYPE='float32',
    num_warps=lambda x: warps(x.shape[1]),
    D_block=lambda x: dblock(x.shape[1]),
)
def DWConv(x, weight, *, ACCTYPE: ConfigurableArg, num_warps: ConfigurableArg, D_block: ConfigurableArg):
    channels = x.shape[1]

    assert x.ndim == 5
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert x.dtype == weight.dtype == torch.float16
    assert channels == next_power_of_2(channels)
    assert list(weight.shape) == [3, 3, 3, channels]
    assert D_block == next_power_of_2(D_block)

    ACCTYPE = {'float32': tl.float32, 'float16': tl.float16}[ACCTYPE]

    bsize, _, H, W, D = x.shape
    batch_stride, _, H_stride, W_stride, _ = x.stride()

    output = torch.empty_like(x)

    H_grid = ceil(H / 2)
    W_grid = ceil(W / 2)
    D_grid = ceil(D / D_block)
    grid = (H_grid, W_grid, D_grid)

    for unbatched_x, unbatched_y in zip(x, output):
        _DWConv_cl3d_impl[grid](
            unbatched_x,
            weight,
            unbatched_y,
            H,
            W,
            D,
            H_stride,
            W_stride,
            ACCTYPE,
            channels,
            D_block,
            num_warps=num_warps,
        )

    return output
