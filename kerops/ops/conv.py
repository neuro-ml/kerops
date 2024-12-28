from math import ceil

import torch
from triton import language as tl, next_power_of_2

from ..kernels.dw_conv import _DWConv_cl3d_impl, _DWConv_wgrad_cl3d_impl
from ._settings import ConfigurableArg, configure


def configure_dwconv(channels):
    """
    Hardcoded, benchmarked on RTX 3090, mb should be generated automatically
    H, W, D = [350, 350, 128]

    channels: [[num_warps, D_block], [num_warps, D_block]]  one for fwd another for bwd
    """

    """
    TODO
    More geeky solution is to compare performances with respect to splitting axis D
    to N * D_block with padding
    """

    HARDCODED_CONFIG = {
        8: [[1, 32], [1, 32]],
        16: [[2, 32], [1, 32]],
        32: [[2, 16], [1, 32]],
        64: [[2, 8], [1, 16]],
        128: [[4, 8], [2, 16]],
    }

    return HARDCODED_CONFIG.get(channels, None)


@configure(
    ACCTYPE='float32',
    _num_warps=lambda weight: configure_dwconv(weight.shape[-1])[0][0],
    D_block=lambda weight: configure_dwconv(weight.shape[-1])[0][1],
)
def DWConv(
    x, weight, *, ACCTYPE: ConfigurableArg = 'float32', _num_warps: ConfigurableArg = 2, D_block: ConfigurableArg = 32
):
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
            num_warps=_num_warps,
        )

    return output


@configure(
    ACCTYPE='float32',
    _num_warps=lambda x: configure_dwconv(x.shape[1])[1][0],
    D_block=lambda x: configure_dwconv(x.shape[1])[1][1],
)
def DWConvWGRAD(
    x, grad, *, ACCTYPE: ConfigurableArg = 'float32', _num_warps: ConfigurableArg = 2, D_block: ConfigurableArg = 32
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

    H_grid = ceil(H / 2)
    W_grid = ceil(W / 2)
    D_grid = ceil(D / D_block)
    grid = (H_grid, W_grid * D_grid)

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
            num_warps=_num_warps,
        )

    grad_w = torch.flip(grad_w.sum(dim=(0, 1)), dims=(2,))

    return grad_w
