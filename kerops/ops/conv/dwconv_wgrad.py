from math import ceil

import torch
from triton import language as tl, next_power_of_2

from ...kernels.dw_conv import _DWConv_wgrad_cl3d_impl
from ...settings import ConfigurableArg, configure, confexc


@confexc(KeyError)
def warps(channels):
    return {8: 1, 16: 1, 32: 1, 64: 1, 128: 2}[channels]


@confexc(KeyError)
def dblock(channels):
    return {8: 32, 16: 32, 32: 16, 64: 8, 128: 8}[channels]


@confexc(KeyError)
def ilp(channels):
    return {8: 1, 16: 1, 32: 2, 64: 3, 128: 3}[channels]


@configure(
    ACCTYPE='float32',
    _num_warps=lambda x: warps(x.shape[1]),
    D_block=lambda x: dblock(x.shape[1]),
    ILP=lambda x: ilp(x.shape[1]),
)
def DWConvWGRAD(
    x, grad, *, ACCTYPE: ConfigurableArg, _num_warps: ConfigurableArg, D_block: ConfigurableArg, ILP: ConfigurableArg
):
    channels = x.shape[1]

    assert x.ndim == grad.ndim == 5
    assert x.shape == grad.shape
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert grad.is_contiguous(memory_format=torch.channels_last_3d)
    assert x.dtype == grad.dtype == torch.float16
    assert channels == next_power_of_2(channels)
    assert D_block == next_power_of_2(D_block)

    ACCTYPE = {'float32': tl.float32, 'float16': tl.float16}[ACCTYPE]

    bsize, _, H, W, D = x.shape
    batch_stride, _, H_stride, W_stride, _ = x.stride()

    H_grid = ceil(H / (2 * ILP))
    W_grid = ceil(W / 2)
    D_grid = ceil(D / D_block)
    grid = (H_grid, W_grid, D_grid)

    grad_w = torch.zeros([bsize, H_grid * W_grid * D_grid, 3, 3, 3, channels], device=x.device, dtype=torch.float16)
    WD_grid = W_grid * D_grid  # TODO: mb implement in another way

    for unbatched_x, unbatched_grad, unbatched_grad_w in zip(x, grad, grad_w):
        _DWConv_wgrad_cl3d_impl[grid](
            unbatched_grad,
            unbatched_x,
            unbatched_grad_w,
            H,
            W,
            D,
            H_stride,
            W_stride,
            ACCTYPE,
            channels,
            D_block,
            WD_grid,
            D_grid,
            H_grid,
            ILP,
            num_warps=_num_warps,
        )

    grad_w = grad_w.sum(dim=(0, 1))

    return grad_w
