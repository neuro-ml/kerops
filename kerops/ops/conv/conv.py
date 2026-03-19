import numpy as np
import torch
from triton import language as tl, next_power_of_2

from ...kernels.conv import _Conv_cl3d_impl_V6, _ApplyBNReLUConvStats_cl3d_impl
from ...settings import ConfArg, configure, confexc
from ...utils import cdiv


@confexc(KeyError)
def num_warps(in_channels, out_channels):
    return {
        (16, 16): 4,
        (16, 32): 4,
        (32, 16): 4,
        (32, 32): 2,
        (32, 64): 2,
        (64, 32): 2,
        (64, 64): 4,
        (64, 128): 2,
        (128, 64): 2,
        (128, 128): 4,
    }[(in_channels, out_channels)]


@confexc(KeyError)
def d_block(in_channels, out_channels):
    return {
        (16, 16): 64,
        (16, 32): 64,
        (32, 16): 64,
        (32, 32): 32,
        (32, 64): 32,
        (64, 32): 32,
        (64, 64): 32,
        (64, 128): 16,
        (128, 64): 16,
        (128, 128): 16,
    }[(in_channels, out_channels)]


@confexc(KeyError)
def cin_block(in_channels, out_channels):
    return {
        (16, 16): 16,
        (16, 32): 16,
        (32, 16): 16,
        (32, 32): 16,
        (32, 64): 16,
        (64, 32): 16,
        (64, 64): 32,
        (64, 128): 16,
        (128, 64): 16,
        (128, 128): 16,
    }[(in_channels, out_channels)]


@confexc(KeyError)
def weight_major(in_channels, out_channels):
    return {
        (16, 16): False,
        (16, 32): True,
        (32, 16): False,
        (32, 32): False,
        (32, 64): True,
        (64, 32): False,
        (64, 64): True,
        (64, 128): True,
        (128, 64): True,
        (128, 128): True,
    }[(in_channels, out_channels)]


@configure(
    ACCTYPE='float32',
    num_warps=lambda weight: num_warps(*weight.shape[-2:]),
    D_BLOCK=lambda weight: d_block(*weight.shape[-2:]),
    CIN_BLOCK=lambda weight: cin_block(*weight.shape[-2:]),
    LOAD_WEIGHT_FIRST=True,
    WEIGHT_MAJOR=lambda weight: weight_major(*weight.shape[-2:]),
)
def Conv3d(x, weight, *, ACCTYPE: ConfArg, num_warps: ConfArg, D_BLOCK: ConfArg, CIN_BLOCK: ConfArg, LOAD_WEIGHT_FIRST: ConfArg, WEIGHT_MAJOR: ConfArg):
    assert x.device == weight.device
    assert x.is_cuda

    assert x.ndim == 5
    assert weight.ndim == 5
    bsize, x_channels, H, W, D = x.shape
    in_channels, out_channels = weight.shape[-2:]
    assert x_channels == next_power_of_2(x_channels)
    assert list(weight.shape) == [3, 3, 3, in_channels, out_channels]  # 3, 3, 3, in_channels, out_channels
    assert in_channels == x_channels
    assert out_channels == next_power_of_2(out_channels)

    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert weight.is_contiguous()

    assert x.dtype == weight.dtype == torch.float16
    
    assert D_BLOCK == next_power_of_2(D_BLOCK)
    assert CIN_BLOCK == next_power_of_2(CIN_BLOCK)
    assert CIN_BLOCK <= in_channels
    assert ACCTYPE in ('float16', 'float32')

    ACCTYPE = {'float32': tl.float32, 'float16': tl.float16}[ACCTYPE]
    output = torch.empty([bsize, H, W, D, out_channels], device=x.device, dtype=x.dtype, layout=x.layout).permute(0, -1, 1, 2, 3)
    grid = (cdiv(W, 2), cdiv(H, 2), cdiv(D, D_BLOCK) * bsize)

    _Conv_cl3d_impl_V6[grid](
        x,
        weight,
        output,
        H,
        W,
        D,
        D_BLOCK=D_BLOCK,
        ACCTYPE=ACCTYPE,
        IN_CHANNELS=in_channels,
        OUT_CHANNELS=out_channels,
        CIN_BLOCK=CIN_BLOCK,
        LOAD_WEIGHT_FIRST=LOAD_WEIGHT_FIRST,
        WEIGHT_MAJOR=WEIGHT_MAJOR,
        num_warps=num_warps,
    )

    return output


@configure(
    ACCTYPE='float32',
    num_warps=lambda weight: num_warps(*weight.shape[-2:]),
    D_BLOCK=lambda weight: d_block(*weight.shape[-2:]),
    CIN_BLOCK=lambda weight: cin_block(*weight.shape[-2:]),
)
def ApplyBNReLUConv3dStats(x, bn_weight, bn_bias, weight, *, ACCTYPE: ConfArg, num_warps: ConfArg, D_BLOCK: ConfArg, CIN_BLOCK: ConfArg):
    assert x.device == weight.device == bn_weight.device == bn_bias.device
    assert x.is_cuda

    assert x.ndim == 5
    assert weight.ndim == 5
    assert bn_weight.ndim == bn_bias.ndim == 1
    bsize, x_channels, H, W, D = x.shape
    in_channels, out_channels = weight.shape[-2:]
    assert x_channels == next_power_of_2(x_channels)
    assert list(weight.shape) == [3, 3, 3, in_channels, out_channels]  # 3, 3, 3, in_channels, out_channels
    assert in_channels == x_channels == bn_weight.numel() == bn_bias.numel()
    assert out_channels == next_power_of_2(out_channels)

    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert weight.is_contiguous()

    assert x.dtype == weight.dtype == torch.float16
    assert bn_weight.dtype == bn_bias.dtype == torch.float32
    
    assert D_BLOCK == next_power_of_2(D_BLOCK)
    assert CIN_BLOCK == next_power_of_2(CIN_BLOCK)
    assert CIN_BLOCK <= in_channels
    assert ACCTYPE in ('float16', 'float32')

    ACCTYPE = {'float32': tl.float32, 'float16': tl.float16}[ACCTYPE]
    output = torch.empty([bsize, H, W, D, out_channels], device=x.device, dtype=x.dtype, layout=x.layout).permute(0, -1, 1, 2, 3)
    grid = (cdiv(W, 2), cdiv(H, 2), cdiv(D, D_BLOCK))
    mean = torch.zeros([bsize, np.prod(grid), out_channels], device=x.device, dtype=torch.float32)
    sqmean = torch.zeros([bsize, np.prod(grid), out_channels], device=x.device, dtype=torch.float32)

    numel_no_channels = bsize * H * W * D

    for unbatched_x, unbatched_y, unbatched_mean, unbatched_sqmean in zip(x, output, mean, sqmean):
        _ApplyBNReLUConvStats_cl3d_impl[grid](
            unbatched_x,
            bn_weight,
            bn_bias,
            weight,
            unbatched_y,
            unbatched_mean,
            unbatched_sqmean,
            H,
            W,
            D,
            D_BLOCK=D_BLOCK,
            ACCTYPE=ACCTYPE,
            IN_CHANNELS=in_channels,
            OUT_CHANNELS=out_channels,
            CIN_BLOCK=CIN_BLOCK,
            num_warps=num_warps,
        )

    return output, mean.sum(dim=(0, 1)) / numel_no_channels, sqmean.sum(dim=(0, 1)) / numel_no_channels
