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
    channels, H, W, D = x.shape[1:]

    assert x.ndim == 5
    assert x.is_contiguous(memory_format=torch.channels_last_3d)
    assert x.dtype == weight.dtype == torch.float16
    assert channels == next_power_of_2(channels)
    assert list(weight.shape) == [3, 3, 3, channels, channels]  # 3, 3, 3, out_channels, in_channels
    assert D_BLOCK == next_power_of_2(D_BLOCK)

    ACCTYPE = {'float32': tl.float32, 'float16': tl.float16}[ACCTYPE]
    output = torch.empty_like(x)
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
            CHANNELS=channels,
            CIN_BLOCK=CIN_BLOCK,
            num_warps=num_warps,
        )

    return output
