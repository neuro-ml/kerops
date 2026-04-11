import math

import torch
from torch import nn
from torch.nn import functional as F

from kerops.ops.conv import DWConv, DWConvWGRAD
from kerops.utils import allclose_two_stage, weight_grad_similarity


def test_dwconv(bsize, channels, other_1, other_2, other_3):
    if channels < 8:
        return

    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).to(
        memory_format=torch.channels_last_3d
    )

    weight = torch.empty(channels, 1, 3, 3, 3, device='cuda', dtype=torch.float16)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    weight_ker = weight[:, 0].permute(1, 2, 3, 0).contiguous()

    out_standard = F.conv3d(x, weight, None, padding=(1, 1, 1), stride=(1, 1, 1), groups=channels)
    out_check = DWConv(x, weight_ker)

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


# TODO: how to sample to make weight's gradient adequate?
def test_dwconv_wgrad(bsize, channels, other_1, other_2, other_3):
    # DWConv from torch is EXTREAMLY slow
    if channels < 8 or bsize > 3 or other_1 > 53 or other_2 > 53 or other_3 > 53:
        return

    torch.manual_seed(322)
    x = torch.randn(bsize, channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).to(
        memory_format=torch.channels_last_3d
    )
    grad = torch.randn(bsize, channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).to(
        memory_format=torch.channels_last_3d
    )

    weight = torch.empty(channels, 1, 3, 3, 3, device='cuda', dtype=torch.float32)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    weight = weight[:, 0].permute(1, 2, 3, 0).contiguous()
    weight.requires_grad_(True)

    grad_w_check = DWConvWGRAD(x, grad).to(torch.float32)

    with torch.amp.autocast('cuda'):
        out = F.conv3d(
            x,
            weight.permute(3, 0, 1, 2)[:, None].contiguous(),
            None,
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            groups=channels,
        )

    out.backward(grad)

    assert weight_grad_similarity(
        weight.grad,
        grad_w_check,
        rtol_cos=1e-4,
        atol_cos=1e-4,
        rtol_len=1e-3 * bsize,
        atol_len=1e-3,
        debug_info='print',
    )

    torch.cuda.empty_cache()
