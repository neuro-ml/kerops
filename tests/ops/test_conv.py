import torch
from torch import nn
from torch.nn import functional as F

from kerops.ops.conv import DWConv, DWConvWGRAD
from kerops.utils import allclose_two_stage


def test_dwconv(bsize, channels, other_1, other_2, other_3):
    if channels >= 8:
        torch.manual_seed(322)
        x = torch.randn(bsize, channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).to(
            memory_format=torch.channels_last_3d
        )

        weight_standard = nn.Conv3d(channels, channels, 3, groups=channels).to('cuda').weight.to(torch.float16)
        weight_check = weight_standard[:, 0].permute(1, 2, 3, 0).contiguous()

        with torch.inference_mode():
            out_standard = F.conv3d(x, weight_standard, None, padding=(1, 1, 1), stride=(1, 1, 1), groups=channels)
            out_check = DWConv(x, weight_check)

        assert allclose_two_stage(
            out_standard,
            out_check,
            rtol_strict=1e-4,
            atol_strict=1e-3,
            rtol_narrow=1e-4,
            atol_narrow=2e-3,
            debug_info='print',
        )

        torch.cuda.empty_cache()


# TODO: how to sample to make weight's gradient adequate?
def test_dwconv_wgrad(bsize, channels, other_1, other_2, other_3):
    if channels >= 8 and other_1 < 53 and other_2 < 53 and other_3 < 53:
        torch.manual_seed(322)
        x = (
            torch.randn(bsize, channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).to(
                memory_format=torch.channels_last_3d
            )
            / 5
        )
        grad = (
            torch.randn(bsize, channels, other_1, other_2, other_3, device='cuda', dtype=torch.float16).to(
                memory_format=torch.channels_last_3d
            )
            / 5
        )
        weight_standard = torch.randn(channels, 1, 3, 3, 3, device='cuda', dtype=torch.float16)

        with torch.inference_mode():
            _, grad_w, _ = torch.ops.aten.convolution_backward(
                grad,
                x,
                weight_standard,
                [0],  # bias_sizes
                [1, 1, 1],  # stride
                [1, 1, 1],  # padding
                [1, 1, 1],  # dilation
                False,  # transposed
                [0, 0, 0],  # output padding
                channels,  # groups!
                [False, True, False],  # output_mask - grad_inpt, grad_weight, grad_bias
            )
            grad_w = grad_w[:, 0].permute(1, 2, 3, 0).contiguous()

            grad_w_check = DWConvWGRAD(x, grad)

        assert allclose_two_stage(
            grad_w,
            grad_w_check,
            rtol_strict=1e-2,
            atol_strict=bsize * 1e-2,
            rtol_narrow=1e-2,
            atol_narrow=bsize * 2e-2,
            debug_info='print',
        )

        torch.cuda.empty_cache()
