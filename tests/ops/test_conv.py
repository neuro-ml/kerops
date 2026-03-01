import math

import torch
from torch import nn
from torch.nn import functional as F

from kerops.ops.conv import Conv3d
from kerops.utils import allclose_two_stage


def test_conv(bsize, conv_in_channels, conv_out_channels, other_1, other_2, other_3):
    if not (conv_in_channels == conv_out_channels or conv_in_channels * 2 == conv_out_channels or conv_in_channels == 2 * conv_out_channels):
        return
    
    torch.manual_seed(322)
    x = torch.randn(bsize, conv_in_channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).to(
        memory_format=torch.channels_last_3d
    )

    weight = torch.empty(conv_out_channels, conv_in_channels, 3, 3, 3, device='cuda', dtype=torch.float16)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    weight_kernel = weight.permute(2, 3, 4, 1, 0).contiguous()

    out_standard = F.conv3d(x, weight, None, padding=(1, 1, 1), stride=(1, 1, 1))
    out_check = Conv3d(x, weight_kernel)

    assert allclose_two_stage(
        out_standard,
        out_check,
        rtol_strict=1e-4,
        atol_strict=1e-3,
        rtol_narrow=1e-3,
        atol_narrow=2e-3,
        debug_info='print',
    )

    torch.cuda.empty_cache()
