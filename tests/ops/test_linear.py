import torch
from torch import nn
from torch.nn import functional as F

from kerops.ops.linear import LinBReLULinAdd, ReLULinearAdd, ReLULinearBackward
from kerops.utils import allclose_two_stage


def test_relu_linear_add(bsize, other_1, other_2, other_3, channels_out):
    torch.manual_seed(322)
    x = torch.randn(bsize, 32, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    other = torch.randn(bsize, channels_out, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    weight = nn.Conv3d(32, channels_out, 1, bias=False).weight.data.to('cuda').to(torch.float16)

    with torch.inference_mode():
        out_1 = F.conv3d(F.relu(x), weight, None, stride=(1, 1, 1), padding=(0, 0, 0)) + other
        out_2 = ReLULinearAdd(x, weight[:, :, 0, 0, 0].permute(1, 0).contiguous(), other)

    assert allclose_two_stage(
        out_1, out_2, atol_strict=1e-4, rtol_strict=1e-4, rtol_narrow=1e-3, atol_narrow=1e-3, debug_info='print'
    )


def test_relu_linear_add_backward(bsize, other_1, other_2, other_3, channels_out):
    if other_1 < 53 and other_2 < 53 and other_3 < 53:
        torch.manual_seed(322)
        x = (
            torch.randn(bsize, 32, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
                memory_format=torch.channels_last_3d
            )
            / 5
        )
        weight_x = nn.Conv3d(32, channels_out, 1, bias=False).weight.data.to('cuda').to(torch.float16)
        x.requires_grad_(True)
        weight_x.requires_grad_(True)

        grad = (
            torch.randn(bsize, channels_out, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
                memory_format=torch.channels_last_3d
            )
            / 5
        )

        out_1 = F.conv3d(F.relu(x), weight_x, None, stride=(1, 1, 1), padding=(0, 0, 0))
        out_1.backward(grad)

        grad_x_check, grad_weight_x_check = ReLULinearBackward(
            x, grad, weight_x[:, :, 0, 0, 0].permute(1, 0).contiguous()
        )
        grad_weight_x_check = grad_weight_x_check.permute(1, 0).contiguous()[:, :, None, None, None]

        assert allclose_two_stage(
            x.grad,
            grad_x_check,
            atol_strict=1e-4,
            rtol_strict=1e-4,
            rtol_narrow=1e-3,
            atol_narrow=1e-3,
            debug_info='print',
        )
        assert allclose_two_stage(
            weight_x.grad,
            grad_weight_x_check,
            atol_strict=1e-2,
            rtol_strict=bsize * 1e-2,
            rtol_narrow=1e-2,
            atol_narrow=bsize * 2e-2,
            debug_info='print',
        )


def test_linbrelulinadd(bsize, channels, other_1, other_2, other_3):
    torch.manual_seed(322)
    if channels in [16, 32, 64]:
        x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
            memory_format=torch.channels_last_3d
        )
        add_other = torch.randn_like(x)
        weight_up = (
            nn.Conv3d(channels, 2 * channels, 1, bias=False, device='cuda')
            .weight[:, :, 0, 0, 0]
            .permute(1, 0)
            .contiguous()
            .to(torch.float16)
        )
        weight_down = (
            nn.Conv3d(2 * channels, channels, 1, bias=False, device='cuda')
            .weight[:, :, 0, 0, 0]
            .permute(1, 0)
            .contiguous()
            .to(torch.float16)
        )
        bias = torch.randn(2 * channels, dtype=torch.float16, device='cuda')

        out = LinBReLULinAdd(x, weight_up, weight_down, bias, add_other)

        out_base = F.conv3d(x, weight_up.permute(1, 0)[:, :, None, None, None], bias)
        out_base = F.relu(out_base)
        out_base = F.conv3d(out_base, weight_down.permute(1, 0)[:, :, None, None, None], None)
        out_base += add_other

        assert allclose_two_stage(
            out_base,
            out,
            rtol_strict=1e-4,
            atol_strict=5e-4,
            rtol_narrow=1e-4,
            atol_narrow=2e-3,
            debug_info='print',
        )
