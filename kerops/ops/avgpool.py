from functools import reduce
from math import ceil

import torch
from triton import next_power_of_2

from ..kernels.avgpool import _AvgPoolCeilStats_cl3d_backward_impl, _AvgPoolCeilStats_cl3d_impl
from ._settings import ConfigurableArg, configure, get_l1_cache


@configure(
    _l1_cache_bytes=get_l1_cache,
    _num_warps=2,
)
def AvgPoolCeilStats(x, *, _l1_cache_bytes: ConfigurableArg, _num_warps: ConfigurableArg):
    num_channels = x.shape[1]
    input_d = x.shape[-1]
    MAX_SIZE = _l1_cache_bytes // x.element_size()  # 32768 for fp16

    assert input_d * num_channels <= MAX_SIZE
    assert num_channels == next_power_of_2(num_channels)
    assert x.ndim == 5
    assert x.dtype == torch.float16
    assert x.is_contiguous(memory_format=torch.channels_last_3d)

    BLOCK_SIZE = next_power_of_2(input_d * num_channels)
    almost_half_d = BLOCK_SIZE // (2 * num_channels)

    out_shape = [x.shape[0]] + [ceil(sh / 2) for sh in x.shape[2:]] + [x.shape[1]]
    output = torch.empty(out_shape, dtype=torch.float16, device=x.device).permute(0, 4, 1, 2, 3)

    grid_batch, _, grid_H, grid_W, _ = output.shape

    mean = torch.zeros(num_channels, device=x.device, dtype=torch.float32)
    sqmean = torch.zeros(num_channels, device=x.device, dtype=torch.float32)
    numel_no_channels_output = reduce(
        lambda x, y: x * y, [s if idx != 1 else 1 for idx, s in enumerate(output.shape)], 1
    )

    _AvgPoolCeilStats_cl3d_impl[(grid_batch, grid_H, grid_W)](
        X_ptr=x,
        Out_ptr=output,
        Mean_ptr=mean,
        Sqmean_ptr=sqmean,
        h_input=x.shape[2],
        w_input=x.shape[3],
        d_input=x.shape[4],
        d_output=output.shape[-1],
        batch_stride_input=x.stride(0),
        H_stride_input=x.stride(2),
        W_stride_input=x.stride(3),
        batch_stride_output=output.stride(0),
        H_stride_output=output.stride(2),
        W_stride_output=output.stride(3),
        numel_no_channels_output=numel_no_channels_output,
        num_channels=num_channels,
        almost_half_d=almost_half_d,
        num_warps=_num_warps,
    )
    return output, mean, sqmean


@configure(_l1_cache_bytes=get_l1_cache, _num_warps=4)
def AvgPoolCeilStatsBackward(
    inpgrad,
    meangrad,
    sqmeangrad,
    output,
    outgrad_shape,
    *,
    _l1_cache_bytes: ConfigurableArg,
    _num_warps: ConfigurableArg,
):
    MAX_SIZE = _l1_cache_bytes // inpgrad.element_size()  # 32768 for fp16
    bsize, num_channels, h_outgrad, w_outgrad, d_outgrad = outgrad_shape
    d_inpgrad = inpgrad.shape[-1]

    assert d_outgrad * num_channels <= MAX_SIZE
    assert inpgrad.shape[1] == num_channels
    assert inpgrad.shape == output.shape
    assert meangrad.numel() == sqmeangrad.numel() == num_channels
    assert inpgrad.ndim == 5
    assert inpgrad.is_contiguous(memory_format=torch.channels_last_3d)
    assert output.is_contiguous(memory_format=torch.channels_last_3d)
    assert num_channels == next_power_of_2(num_channels)
    assert inpgrad.dtype == output.dtype == torch.float16
    assert meangrad.dtype == sqmeangrad.dtype == torch.float32

    BLOCK_SIZE = next_power_of_2(d_outgrad * num_channels)
    almost_half_d = BLOCK_SIZE // (2 * num_channels)

    outgrad_shape = [bsize, h_outgrad, w_outgrad, d_outgrad, num_channels]
    outgrad = torch.zeros(outgrad_shape, dtype=torch.float16, device=inpgrad.device).permute(0, 4, 1, 2, 3)

    grid_batch, _, grid_H, grid_W, _ = inpgrad.shape

    numel_no_channels_inpgrad = reduce(
        lambda x, y: x * y, [s if idx != 1 else 1 for idx, s in enumerate(inpgrad.shape)], 1
    )

    _AvgPoolCeilStats_cl3d_backward_impl[(grid_batch, grid_H, grid_W)](
        Inpgrad_ptr=inpgrad,
        Outgrad_ptr=outgrad,
        Output_ptr=output,
        Meangrad_ptr=meangrad,
        Sqmeangrad_ptr=sqmeangrad,
        h_outgrad=h_outgrad,
        w_outgrad=w_outgrad,
        d_outgrad=d_outgrad,
        d_inpgrad=d_inpgrad,
        batch_stride_outgrad=outgrad.stride(0),
        H_stride_outgrad=outgrad.stride(2),
        W_stride_outgrad=outgrad.stride(3),
        batch_stride_inpgrad=inpgrad.stride(0),
        H_stride_inpgrad=inpgrad.stride(2),
        W_stride_inpgrad=inpgrad.stride(3),
        numel_no_channels_inpgrad=numel_no_channels_inpgrad,
        num_channels=num_channels,
        almost_half_d=almost_half_d,
        num_warps=_num_warps,
    )
    return outgrad
