import math

import torch
from torch import nn
from torch.nn import functional as F

from kerops.ops.linear import LinBReLULinAdd, LinBReLULinBackward, ReLULinearAdd, ReLULinearBackward
from kerops.utils import allclose_two_stage, weight_grad_similarity


def test_relu_linear_add(bsize, other_1, other_2, other_3, channels_out):
    torch.manual_seed(322)
    x = torch.randn(bsize, 32, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    other = torch.randn(bsize, channels_out, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    weight = torch.empty(32, channels_out, device='cuda', dtype=torch.float16)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

    out_2 = ReLULinearAdd(x, weight, other)

    with torch.inference_mode():
        out_1 = F.conv3d(
            F.relu(x), weight.permute(1, 0)[:, :, None, None, None], None, stride=(1, 1, 1), padding=(0, 0, 0)
        )
        out_1 += other

    assert allclose_two_stage(
        out_1, out_2, atol_strict=1e-4, rtol_strict=1e-4, rtol_narrow=1e-3, atol_narrow=1e-3, debug_info='print'
    )


def test_relu_linear_add_backward(bsize, other_1, other_2, other_3, channels_out):
    torch.manual_seed(322)
    x = torch.randn(bsize, 32, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    x.requires_grad_(True)

    weight_x = torch.empty(32, channels_out, device='cuda', dtype=torch.float32)
    nn.init.kaiming_uniform_(weight_x, a=math.sqrt(5))
    weight_x.requires_grad_(True)

    grad = torch.randn(bsize, channels_out, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )

    with torch.amp.autocast('cuda'):
        out_1 = F.conv3d(
            F.relu(x), weight_x.permute(1, 0)[:, :, None, None, None], None, stride=(1, 1, 1), padding=(0, 0, 0)
        )

    out_1.backward(grad)

    grad_x_check, grad_weight_x_check = ReLULinearBackward(x, grad, weight_x.to(torch.float16))
    grad_weight_x_check = grad_weight_x_check.to(torch.float32)

    assert allclose_two_stage(
        x.grad,
        grad_x_check,
        atol_strict=1e-4,
        rtol_strict=1e-4,
        rtol_narrow=1e-3,
        atol_narrow=1e-3,
        debug_info='print',
    )

    assert weight_grad_similarity(
        weight_x.grad,
        grad_weight_x_check,
        rtol_cos=1e-4,
        atol_cos=1e-4,
        rtol_len=1e-3 * bsize,
        atol_len=1e-3,
        debug_info='print',
    )


def test_linbrelulinadd(bsize, channels, other_1, other_2, other_3):
    if channels not in [16, 32, 64]:
        return

    torch.manual_seed(322)

    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    add_other = torch.randn_like(x)

    weight_up = torch.randn(channels, channels * 2, dtype=torch.float32, device='cuda')
    nn.init.kaiming_uniform_(weight_up, a=math.sqrt(5))

    weight_down = torch.randn(channels * 2, channels, dtype=torch.float32, device='cuda')
    nn.init.kaiming_uniform_(weight_down, a=math.sqrt(5))

    bias = torch.randn(2 * channels, dtype=torch.float32, device='cuda')

    out = LinBReLULinAdd(
        x, weight_up.to(torch.float16), weight_down.to(torch.float16), bias.to(torch.float16), add_other
    )

    with torch.amp.autocast('cuda'), torch.inference_mode():
        out_base = F.conv3d(x, weight_up.permute(1, 0)[:, :, None, None, None], bias)
        out_base = F.relu(out_base)
        out_base = F.conv3d(out_base, weight_down.permute(1, 0)[:, :, None, None, None], None)
        out_base += add_other

    assert allclose_two_stage(
        out_base,
        out,
        rtol_strict=1e-4,
        atol_strict=5e-4,
        rtol_narrow=1e-3,
        atol_narrow=1e-3,
        debug_info='print',
    )


def test_linbrelulin_backward(bsize, channels, other_1, other_2, other_3):
    if channels not in [16, 32]:
        return

    torch.manual_seed(322)

    x = torch.randn(bsize, channels, other_1, other_2, other_3, dtype=torch.float16, device='cuda').to(
        memory_format=torch.channels_last_3d
    )
    grad = torch.randn_like(x)
    x.requires_grad_(True)

    weight_up = torch.randn(channels, channels * 2, dtype=torch.float32, device='cuda')
    nn.init.kaiming_uniform_(weight_up, a=math.sqrt(5))
    weight_up.requires_grad_(True)

    weight_down = torch.randn(channels * 2, channels, dtype=torch.float32, device='cuda')
    nn.init.kaiming_uniform_(weight_down, a=math.sqrt(5))
    weight_down.requires_grad_(True)

    bias = torch.randn(2 * channels, dtype=torch.float32, device='cuda')
    bias.requires_grad_(True)

    input_grad, weight_up_grad, weight_down_grad, bias_grad = LinBReLULinBackward(
        x,
        grad,
        weight_up.to(torch.float16),
        weight_down.to(torch.float16),
        bias.to(torch.float16),
    )

    with torch.amp.autocast('cuda'):
        x_ = F.conv3d(x, weight_up.permute(1, 0)[:, :, None, None, None], bias, stride=1, padding=0)
        x_ = F.relu(x_)
        x_ = F.conv3d(x_, weight_down.permute(1, 0)[:, :, None, None, None], None, stride=1, padding=0)

    x_.backward(grad)

    assert weight_grad_similarity(
        weight_down.grad,
        weight_down_grad,
        rtol_cos=1e-4,
        atol_cos=1e-4,
        rtol_len=1e-3 * bsize,
        atol_len=1e-3,
        debug_info='print',
    )

    assert weight_grad_similarity(
        weight_up.grad,
        weight_up_grad,
        rtol_cos=1e-4,
        atol_cos=1e-4,
        rtol_len=2e-3 * bsize,
        atol_len=1e-3,
        debug_info='print',
    )

    assert weight_grad_similarity(
        weight_up.grad,
        weight_up_grad,
        rtol_cos=1e-3,
        atol_cos=1e-3,
        rtol_len=2e-3 * bsize,
        atol_len=1e-3,
        debug_info='print',
    )

    allclose = allclose_two_stage(
        x.grad,
        input_grad,
        rtol_strict=1e-4,
        atol_strict=1e-4,
        rtol_narrow=1e-3,
        atol_narrow=1e-3,
        debug_info='print',
    )

    # Because of relu non-differentiability some voxels may significatnly differ
    # Check these voxels manually
    if not allclose:
        soft_map = torch.abs(x.grad - input_grad) < 1e-3 + 1e-3 * torch.abs(input_grad)
        not_soft_map_vox = torch.all((~soft_map), dim=1)

        b, h, w, d = torch.where(not_soft_map_vox)

        with torch.cuda.amp.autocast(), torch.inference_mode():
            for inp, grd, inp_grad in zip(x[b, :, h, w, d], grad[b, :, h, w, d], input_grad[b, :, h, w, d]):
                linup = inp @ weight_up + bias

                linup_relu_grad = weight_down @ grd
                linup_grad = linup_relu_grad * (linup > 0)
                base_grd = weight_up @ linup_grad

                assert torch.allclose(base_grd, inp_grad, atol=1e-3, rtol=1e-3)
