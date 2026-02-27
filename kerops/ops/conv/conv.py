import torch
from triton import language as tl, next_power_of_2

from ...kernels.conv import _Conv_cl3d_impl_V5, _ApplyBNReLUConv_cl3d_impl
from ...settings import ConfigurableArg, configure, confexc
from ...utils import cdiv


@confexc(KeyError)
def num_warps(in_channels, out_channels):
    return {
        (16, 16): 4,
        (16, 32): 4,
        (32, 16): 4,
        (32, 32): 2,
        (32, 64): 4,
        (64, 32): 1,
        (64, 64): 2,
        (64, 128): 4,
        (128, 64): 4,
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
        (32, 16): 32,
        (32, 32): 16,
        (32, 64): 16,
        (64, 32): 16,
        (64, 64): 16,
        (64, 128): 16,
        (128, 64): 16,
        (128, 128): 16,
    }[(in_channels, out_channels)]


@configure(
    ACCTYPE='float32',
    num_warps=lambda weight: num_warps(*weight.shape[-2:]),
    D_BLOCK=lambda weight: d_block(*weight.shape[-2:]),
    CIN_BLOCK=lambda weight: cin_block(*weight.shape[-2:]),
)
def Conv3d(x, weight, *, ACCTYPE: ConfigurableArg, num_warps: ConfigurableArg, D_BLOCK: ConfigurableArg, CIN_BLOCK: ConfigurableArg):
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
    grid = (cdiv(W, 2), cdiv(H, 2), cdiv(D, D_BLOCK))

    for unbatched_x, unbatched_y in zip(x, output):
        _Conv_cl3d_impl_V5[grid](
            unbatched_x,
            weight,
            unbatched_y,
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

    return output


@configure(
    ACCTYPE='float32',
    num_warps=lambda weight: num_warps(*weight.shape[-2:]),
    D_BLOCK=lambda weight: d_block(*weight.shape[-2:]),
    CIN_BLOCK=lambda weight: cin_block(*weight.shape[-2:]),
)
def ApplyBNReLUConv3d(x, bn_weight, bn_bias, weight, *, ACCTYPE: ConfigurableArg, num_warps: ConfigurableArg, D_BLOCK: ConfigurableArg, CIN_BLOCK: ConfigurableArg):
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

    for unbatched_x, unbatched_y in zip(x, output):
        _ApplyBNReLUConv_cl3d_impl[grid](
            unbatched_x,
            bn_weight,
            bn_bias,
            weight,
            unbatched_y,
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

    return output
