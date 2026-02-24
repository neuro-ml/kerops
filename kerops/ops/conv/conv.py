import torch
from triton import language as tl, next_power_of_2

from ...kernels.conv import _Conv_cl3d_impl_V5
from ...settings import ConfigurableArg, configure
from ...utils import cdiv


@configure(
    ACCTYPE='float32',
    num_warps=4,
    D_BLOCK=64,
    CIN_BLOCK=16,
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
