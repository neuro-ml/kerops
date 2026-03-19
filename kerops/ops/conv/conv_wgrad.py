import torch
from triton import language as tl, next_power_of_2

from ...kernels.conv import _Conv_wgrad_cl3d_impl_V2
from ...settings import ConfArg, configure, confexc
from ...utils import cdiv


@confexc(KeyError)
def num_warps(in_channels, out_channels):
    # BE AWARE of swap_grad_with_input - it permutes grad with x
    return {
        (16, 16): 1,
        (16, 32): 2,
        (32, 16): 2,
        (32, 32): 2,
        (32, 64): 4,
        (64, 32): 4,
        (64, 64): 2,
        (64, 128): 4,
        (128, 64): 4,
        (128, 128): 4,
    }[(in_channels, out_channels)]


@confexc(KeyError)
def CIN_BLOCK(in_channels, out_channels):
    # BE AWARE of swap_grad_with_input - it permutes grad with x
    return {
        (16, 16): 16,
        (16, 32): 16,
        (32, 16): 16,
        (32, 32): 32,
        (32, 64): 16,
        (64, 32): 16,
        (64, 64): 32,
        (64, 128): 32,
        (128, 64): 32,
        (128, 128): 16,
    }[(in_channels, out_channels)]


@confexc(KeyError)
def COUT_BLOCK(in_channels, out_channels):
    # BE AWARE of swap_grad_with_input - it permutes grad with x
    return {
        (16, 16): 16,
        (16, 32): 32,
        (32, 16): 32,
        (32, 32): 32,
        (32, 64): 64,
        (64, 32): 64,
        (64, 64): 64,
        (64, 128): 128,
        (128, 64): 128,
        (128, 128): 128,
    }[(in_channels, out_channels)]


# TODO: looks like this trick can be fixed with a better algorithm.
def swap_grad_with_input(in_channels, out_channels):
    return out_channels < in_channels


@configure(
    ACCTYPE='float32',
    num_warps=lambda grad, x: num_warps(x.shape[1], grad.shape[1]),
    REDUCTION_FACTOR=32,
    CIN_BLOCK=lambda grad, x: CIN_BLOCK(x.shape[1], grad.shape[1]),
    COUT_BLOCK=lambda grad, x: COUT_BLOCK(x.shape[1], grad.shape[1]),
    D_BLOCK=32,
    SWAP_GRAD_WITH_INPUT=lambda grad, x: swap_grad_with_input(x.shape[1], grad.shape[1])
)
def Conv3dWgrad(grad, x, *, D_BLOCK: ConfArg, ACCTYPE: ConfArg, num_warps: ConfArg, REDUCTION_FACTOR: ConfArg, CIN_BLOCK: ConfArg, COUT_BLOCK: ConfArg, SWAP_GRAD_WITH_INPUT: ConfArg):
    if SWAP_GRAD_WITH_INPUT:
        grad, x = x, grad

    assert x.device == grad.device
    assert x.is_cuda

    assert x.ndim == grad.ndim == 5
    xbsize, in_channels, xH, xW, xD = x.shape
    gbsize, out_channels, gH, gW, gD = grad.shape
    assert in_channels == next_power_of_2(in_channels)
    assert out_channels == next_power_of_2(out_channels)
    assert [xbsize, xH, xW, xD] == [gbsize, gH, gW, gD]

    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert grad.is_contiguous(memory_format=torch.channels_last_3d)

    assert x.dtype == grad.dtype == torch.float16

    assert D_BLOCK == next_power_of_2(D_BLOCK)
    assert ACCTYPE in ('float16', 'float32')
    assert CIN_BLOCK == next_power_of_2(CIN_BLOCK)
    assert CIN_BLOCK <= in_channels
    assert COUT_BLOCK == next_power_of_2(COUT_BLOCK)
    assert COUT_BLOCK <= out_channels
    assert isinstance(REDUCTION_FACTOR, int)

    ACCTYPE = {'float32': tl.float32, 'float16': tl.float16}[ACCTYPE]
    num_buffers = cdiv(xW, REDUCTION_FACTOR * 2) * cdiv(xH, REDUCTION_FACTOR * 2) * cdiv(xD, D_BLOCK * REDUCTION_FACTOR) * xbsize
    weight_grad = torch.zeros([num_buffers, 3, 3, 3, in_channels, out_channels], device=x.device, dtype=torch.float32)
    grid = (cdiv(xW, 2) * cdiv(out_channels, COUT_BLOCK), cdiv(xH, 2), cdiv(xD, D_BLOCK) * xbsize)

    _Conv_wgrad_cl3d_impl_V2[grid](
        grad,
        x,
        weight_grad,
        xH,
        xW,
        xD,
        num_buffers,
        IN_CHANNELS=in_channels,
        OUT_CHANNELS=out_channels,
        ACCTYPE=ACCTYPE,
        D_BLOCK=D_BLOCK,
        CIN_BLOCK=CIN_BLOCK,
        COUT_BLOCK=COUT_BLOCK,
        num_warps=num_warps,
    )
    
    weight_grad = weight_grad.sum(dim=0).to(torch.float16)

    if SWAP_GRAD_WITH_INPUT:
        weight_grad = weight_grad.permute(0, 1, 2, 4, 3)
        weight_grad = torch.flip(weight_grad, dims=(0, 1, 2))
        weight_grad = weight_grad.contiguous()

    return weight_grad
